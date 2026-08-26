// Baseline tiled multi-head attention, zero gating, no persistence.
// Math matches traces/gen_golden.py: scores = QK^T / sqrt(dh), softmax rows,
// out = softmax @ V, FP32 throughout.
//
// Cross mode: causal temporal prefix — key span grows to (frame_idx+1)*T,
// K/V/map passed at FULL buffer bases (per-frame launches only).
//
// Compile with -DATTN_STANDALONE to get the Part-2 CLI binary; without it,
// this file only provides the kernel + launcher for linking.

#include "attn_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA FAIL %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

// Remap mode: caller pre-offsets Q/Out (and KVMap) to one frame and launches
// with grid.z == 1; K/V/Arena stay at their buffer bases and rows are resolved
// through KVMap. Baseline mode: full-trace tensors, identity addressing.
// Cross mode: causal temporal prefix — key span grows to (frame_idx+1)*T,
// K/V/map passed at FULL buffer bases (per-frame launches only).
__global__ void attn_forward_kernel(const float* __restrict__ Q,
                                    const float* __restrict__ K,
                                    const float* __restrict__ V,
                                    const float* __restrict__ Arena,
                                    const long* __restrict__ KVMap,
                                    float* __restrict__ Out,
                                    int T, int D, int dh, int remap,
                                    int cross, int frame_idx) {
    const int head = blockIdx.y;
    const size_t foff = remap ? 0 : (size_t)blockIdx.z * T;
    const long span = cross ? (long)(frame_idx + 1) * T : (long)T;
    const size_t koff = cross ? 0 : foff;
    const size_t lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * ATTENTION_WARPS + warp;
    if (row >= T) return;

    const size_t row_base = (foff + row) * D + (size_t)head * dh;

    // lane-local slice of the query vector, loaded once
    float qreg[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        qreg[i] = Q[row_base + lane * 4 + i];

    float m = -INFINITY;  // running max of scores
    float l = 0.f;        // running softmax denominator
    float oacc[4] = {0.f, 0.f, 0.f, 0.f};

    extern __shared__ float smem[];
    float* k_smem = smem;                        // [KTILE][dh]
    float* s_smem = smem + ATTENTION_KTILE * dh; // [WARPS][KTILE]

    const int num_key_tiles =
        (int)((span + ATTENTION_KTILE - 1) / ATTENTION_KTILE);
    for (int t = 0; t < num_key_tiles; ++t) {
        const int j0 = t * ATTENTION_KTILE;
        const int valid = min(ATTENTION_KTILE, (int)span - j0);

        for (int idx = threadIdx.x; idx < valid * dh;
             idx += ATTENTION_WARPS * 32) {
            const int jj = idx / dh;
            const int dd = idx % dh;
            const int j_abs = j0 + jj;
            const long mapped =
                KVMap ? KVMap[(cross ? 0 : foff) + j_abs] : -1L;
            const float* src =
                (mapped >= 0)
                    ? Arena + (size_t)mapped * 2 * D + head * dh
                    : K + (koff + j_abs) * D + head * dh;
            k_smem[jj * dh + dd] = src[dd];
        }
        __syncthreads();

        // tile scores: full dot product per key via butterfly reduction
        for (int j = 0; j < valid; ++j) {
            float partial = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                partial += qreg[i] * k_smem[j * dh + lane * 4 + i];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor_sync(0xffffffffu, partial, off);
            s_smem[warp * ATTENTION_KTILE + j] = partial * rsqrtf((float)dh);
        }

        // online softmax update for this tile
        float tile_max = -INFINITY;
        for (int j = 0; j < valid; ++j)
            tile_max = fmaxf(tile_max, s_smem[warp * ATTENTION_KTILE + j]);
        const float m_new = fmaxf(m, tile_max);
        const float scale = expf(m - m_new);  // exp(-inf)=0 handles first tile

        float l_add = 0.f;
        for (int j = 0; j < valid; ++j) {
            const float p = expf(s_smem[warp * ATTENTION_KTILE + j] - m_new);
            s_smem[warp * ATTENTION_KTILE + j] = p;
            l_add += p;
        }
        l = l * scale + l_add;
#pragma unroll
        for (int i = 0; i < 4; ++i) oacc[i] *= scale;

        // accumulate V; consecutive lanes hit consecutive 16B chunks
        for (int j = 0; j < valid; ++j) {
            const float p = s_smem[warp * ATTENTION_KTILE + j];
            const int j_abs = j0 + j;
            const long mapped =
                KVMap ? KVMap[(cross ? 0 : foff) + j_abs] : -1L;
            const float* vrow =
                (mapped >= 0)
                    ? Arena + (size_t)mapped * 2 * D + D + head * dh
                    : V + (koff + j_abs) * D + head * dh;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                oacc[i] += p * vrow[lane * 4 + i];
        }
        m = m_new;
        __syncthreads();  // all warps done with k_smem before next overwrite
    }

    const float inv_l = 1.f / l;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        Out[row_base + lane * 4 + i] = oacc[i] * inv_l;
}

void attn_forward_launcher(const float* d_q, const float* d_k,
                           const float* d_v, const float* d_arena,
                           const long* d_kvmap, float* d_out,
                           int frames, int T, int D, int heads,
                           bool remap) {
    const int dh = D / heads;
    const size_t smem =
        (ATTENTION_KTILE * dh + ATTENTION_WARPS * ATTENTION_KTILE) *
        sizeof(float);

    if (remap) {
        // Per-frame intra-frame launches (legacy path).
        dim3 grid((unsigned)((T + ATTENTION_WARPS - 1) / ATTENTION_WARPS),
                  (unsigned)heads, 1u);
        for (int f = 0; f < frames; ++f) {
            attn_forward_kernel<<<grid, ATTENTION_WARPS * 32, smem>>>(
                d_q + (size_t)f * T * D, d_k + (size_t)f * T * D,
                d_v + (size_t)f * T * D, d_arena,
                d_kvmap + (size_t)f * T, d_out + (size_t)f * T * D,
                T, D, dh, 1, 0, f);
        }
    } else {
        dim3 grid((unsigned)((T + ATTENTION_WARPS - 1) / ATTENTION_WARPS),
                  (unsigned)heads, (unsigned)frames);
        attn_forward_kernel<<<grid, ATTENTION_WARPS * 32, smem>>>(
            d_q, d_k, d_v, nullptr, nullptr, d_out, T, D, dh, 0, 0, 0);
    }
}

void attn_forward_single_frame(const float* d_q_f, const float* d_k,
                               const float* d_v, const float* d_arena,
                               const long* d_kvmap_f, float* d_out_f,
                               int T, int D, int heads, int cross,
                               int frame_idx) {
    const int dh = D / heads;
    const size_t smem =
        (ATTENTION_KTILE * dh + ATTENTION_WARPS * ATTENTION_KTILE) *
        sizeof(float);
    dim3 grid((unsigned)((T + ATTENTION_WARPS - 1) / ATTENTION_WARPS),
              (unsigned)heads, 1u);
    attn_forward_kernel<<<grid, ATTENTION_WARPS * 32, smem>>>(
        d_q_f, d_k, d_v, d_arena, d_kvmap_f, d_out_f, T, D, dh, 1, cross,
        frame_idx);
}

#ifdef ATTN_STANDALONE

static std::vector<float> read_bin(const std::string& path, size_t expect_elems,
                                   const char* what) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s (%s)\n", path.c_str(), what); exit(1); }
    std::vector<float> buf(expect_elems);
    if (fread(buf.data(), sizeof(float), expect_elems, f) != expect_elems) {
        fprintf(stderr, "short read on %s (%s)\n", path.c_str(), what);
        exit(1);
    }
    fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    std::string q_path, k_path, v_path, out_path;
    long frames = 8, tokens = 1560, hidden = 1536, heads = 12;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--q") q_path = next();
        else if (a == "--k") k_path = next();
        else if (a == "--v") v_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--frames") frames = atol(next().c_str());
        else if (a == "--tokens") tokens = atol(next().c_str());
        else if (a == "--hidden") hidden = atol(next().c_str());
        else if (a == "--heads") heads = atol(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    const size_t elems = (size_t)frames * tokens * hidden;

    std::vector<float> h_q = read_bin(q_path, elems, "q");
    std::vector<float> h_k = read_bin(k_path, elems, "k");
    std::vector<float> h_v = read_bin(v_path, elems, "v");

    float *d_q, *d_k, *d_v, *d_out;
    cuda_check(cudaMalloc(&d_q, elems * sizeof(float)), "cudaMalloc d_q");
    cuda_check(cudaMalloc(&d_k, elems * sizeof(float)), "cudaMalloc d_k");
    cuda_check(cudaMalloc(&d_v, elems * sizeof(float)), "cudaMalloc d_v");
    cuda_check(cudaMalloc(&d_out, elems * sizeof(float)), "cudaMalloc d_out");
    cuda_check(cudaMemcpy(d_q, h_q.data(), elems * sizeof(float),
                          cudaMemcpyHostToDevice), "H2D q");
    cuda_check(cudaMemcpy(d_k, h_k.data(), elems * sizeof(float),
                          cudaMemcpyHostToDevice), "H2D k");
    cuda_check(cudaMemcpy(d_v, h_v.data(), elems * sizeof(float),
                          cudaMemcpyHostToDevice), "H2D v");

    attn_forward_launcher(d_q, d_k, d_v, nullptr, nullptr, d_out,
                          (int)frames, (int)tokens, (int)hidden, (int)heads,
                          false);
    cuda_check(cudaGetLastError(), "kernel launch");
    cuda_check(cudaDeviceSynchronize(), "kernel run");

    std::vector<float> h_out(elems);
    cuda_check(cudaMemcpy(h_out.data(), d_out, elems * sizeof(float),
                          cudaMemcpyDeviceToHost), "D2H out");

    std::filesystem::create_directories(
        std::filesystem::path(out_path).parent_path());
    FILE* f = fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s for write\n", out_path.c_str()); return 1; }
    fwrite(h_out.data(), sizeof(float), elems, f);
    fclose(f);

    printf("gated_attention[baseline]: frames=%ld heads=%ld T=%ld D=%ld "
           "-> %s (%zu bytes)\n",
           frames, heads, tokens, hidden, out_path.c_str(),
           elems * sizeof(float));

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    cudaDeviceReset();
    return 0;
}
#endif  // ATTN_STANDALONE
