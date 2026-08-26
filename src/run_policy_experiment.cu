// // Phase 4/6 driver: runs the tiled attention kernel under one of three gating
// policies (none | uniform | motion), wiring TileGatingPolicy selections to
// KVTileManager's persistence arena on identical traces.
//
// Per frame (persist policies): policy selects tiles -> manager stages their
// K/V rows into the arena -> single persisting window over staged bytes ->
// one remapped kernel launch -> arena_end_step(). The kernel stays
// policy-blind; it only sees a row map.

#include "attn_kernel.h"
#include "tile_gating_policy.h"
#include "kv_tile_manager.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA FAIL %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static std::vector<float> read_bin(const std::string& path, size_t elems,
                                   const char* what) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s (%s)\n", path.c_str(), what); exit(1); }
    std::vector<float> buf(elems);
    if (fread(buf.data(), sizeof(float), elems, f) != elems) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
    }
    fclose(f);
    return buf;
}

// motion_trace CSV: frame_id,tile_id,motion_score,is_static,start,count[,reuse]
static std::vector<std::vector<float>> read_motion_csv(
    const std::string& path, int frames, int num_tiles,
    std::vector<std::vector<char>>* reuse_out) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "cannot open motion csv %s\n", path.c_str()); exit(1); }
    std::vector<std::vector<float>> scores(frames,
                                           std::vector<float>(num_tiles, 0.f));
    if (reuse_out)
        reuse_out->assign(frames, std::vector<char>(num_tiles, 0));
    char line[256];
    fgets(line, sizeof line, f);  // header
    while (fgets(line, sizeof line, f)) {
        int fr, tid, is_static, start, count, reuse = 0;
        float score;
        int got = sscanf(line, "%d,%d,%f,%d,%d,%d,%d", &fr, &tid, &score,
                         &is_static, &start, &count, &reuse);
        if ((got == 6 || got == 7) && fr >= 0 && fr < frames &&
            tid >= 0 && tid < num_tiles) {
            scores[fr][tid] = score;
            if (got == 7 && reuse_out) (*reuse_out)[fr][tid] = (char)reuse;
        }
    }
    fclose(f);
    return scores;
}

int main(int argc, char** argv) {
    std::string policy_name, q_path, k_path, v_path, csv_path, out_path;
    std::string attn_mode = "intra";
    int streaming = 0;
    long frames = 8, tokens = 1560, hidden = 1536, heads = 12;
    long tile_tokens = 16, set_aside = 2162688;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--policy") policy_name = next();
        else if (a == "--attn-mode") attn_mode = next();
        else if (a == "--streaming") streaming = 1;
        else if (a == "--q") q_path = next();
        else if (a == "--k") k_path = next();
        else if (a == "--v") v_path = next();
        else if (a == "--motion-csv") csv_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--frames") frames = atol(next().c_str());
        else if (a == "--tokens") tokens = atol(next().c_str());
        else if (a == "--hidden") hidden = atol(next().c_str());
        else if (a == "--heads") heads = atol(next().c_str());
        else if (a == "--tile-tokens") tile_tokens = atol(next().c_str());
        else if (a == "--set-aside-bytes") set_aside = atol(next().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    const bool xframe = (attn_mode == "cross");

    const long num_tiles = (tokens + tile_tokens - 1) / tile_tokens;
    const size_t elems = (size_t)frames * tokens * hidden;

    std::unique_ptr<TileGatingPolicy> policy;
    if (policy_name == "none") policy.reset(new NoPersistencePolicy());
    else if (policy_name == "uniform") policy.reset(new UniformWindowPolicy());
    else if (policy_name == "recency") policy.reset(new RecencyWindowPolicy());
    else if (policy_name == "motion") policy.reset(new MotionGatedPolicy());
    else if (policy_name == "reuse_weighted")
        policy.reset(new ReuseWeightedPolicy((int)num_tiles, (int)frames));
    else { fprintf(stderr, "unknown policy %s\n", policy_name.c_str()); return 1; }

    std::vector<float> h_q = read_bin(q_path, elems, "q");
    std::vector<float> h_k = read_bin(k_path, elems, "k");
    std::vector<float> h_v = read_bin(v_path, elems, "v");
    const bool persist = policy_name != "none";
    std::vector<std::vector<char>> reuse_flags;
    std::vector<std::vector<float>> scores;
    if (persist)
        scores = read_motion_csv(csv_path, (int)frames, (int)num_tiles,
                                 streaming ? &reuse_flags : nullptr);
    // frame-major flattened scores for cross-mode candidate ranking
    std::vector<float> scores_flat;
    for (const auto& s : scores)
        scores_flat.insert(scores_flat.end(), s.begin(), s.end());
    // NOTE: trace rows are immutable offline artifacts, so a resident sticky
    // slot never needs restaging; restage volume is purely admission churn
    // (release -> re-admit), which is exactly what policy quality controls.

    float *d_q, *d_k, *d_v, *d_out;
    cuda_check(cudaMalloc(&d_q, elems * 4), "malloc q");
    cuda_check(cudaMalloc(&d_k, elems * 4), "malloc k");
    cuda_check(cudaMalloc(&d_v, elems * 4), "malloc v");
    cuda_check(cudaMalloc(&d_out, elems * 4), "malloc out");
    cuda_check(cudaMemcpy(d_q, h_q.data(), elems * 4, cudaMemcpyHostToDevice), "H2D q");
    cuda_check(cudaMemcpy(d_k, h_k.data(), elems * 4, cudaMemcpyHostToDevice), "H2D k");
    cuda_check(cudaMemcpy(d_v, h_v.data(), elems * 4, cudaMemcpyHostToDevice), "H2D v");

    KVTileManager mgr((size_t)set_aside, (size_t)set_aside);
    long* d_kvmap = nullptr;
    std::vector<long> h_map;
    if (persist) {
        cuda_check(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                      (size_t)set_aside), "set limit");
        // fp32 tile bytes: tokens*D*4(K)+tokens*D*4(V); slots from budget.
        const size_t tile_bytes = (size_t)tile_tokens * hidden * 4 * 2;
        const long slots = (long)(set_aside / tile_bytes);
        if (mgr.init() && mgr.arena_init((size_t)set_aside,
                                         (size_t)hidden * 2)) {
            h_map.assign((size_t)frames * tokens, -1L);
            cuda_check(cudaMalloc((void**)&d_kvmap,
                                  h_map.size() * sizeof(long)), "malloc map");
            printf("policy=%s slots=%ld tile_bytes=%zu set_aside=%ld\n",
                   policy->name(), slots, tile_bytes, set_aside);

            std::vector<long> ids((size_t)frames * num_tiles);
            std::vector<char> prev_selected;
            if (streaming) mgr.streaming_begin();
            for (long f = 0; f < frames; ++f) {
                int count = 0;
                if (xframe) {
                    // candidates: frames 0..f, flattened frame-major
                    policy->select(scores_flat.data(),
                                   (int)((f + 1) * num_tiles), (int)slots,
                                   ids.data(), &count);
                } else {
                    policy->select(scores[f].data(), (int)num_tiles,
                                   (int)slots, ids.data(), &count);
                }

                // Streaming: release slots of tiles that dropped out of the
                // selection before admitting new ones.
                std::vector<char> selected((size_t)((f + 1) * num_tiles), 0);
                for (int i = 0; i < count; ++i)
                    selected[(size_t)ids[i]] = 1;
                if (streaming && f > 0) {
                    const long prev_len = (long)prev_selected.size();
                    for (long gid = 0; gid < prev_len; ++gid) {
                        if (!prev_selected[gid]) continue;
                        const bool still =
                            gid < (long)selected.size() && selected[gid];
                        if (still) continue;
                        const long src_f = xframe ? gid / num_tiles : f;
                        const long tile = xframe ? gid % num_tiles : gid;
                        const long t0 = tile * tile_tokens;
                        const long c = std::min(tile_tokens, tokens - t0);
                        for (long r = 0; r < c; ++r)
                            mgr.streaming_release(src_f * tokens + t0 + r);
                    }
                }

                mgr.arena_begin_step();
                // This step's map describes only the currently pinned set:
                // reset prefix entries, then repopulate from stage results.
                // (Streaming's slot table persists across steps; the map is
                // rebuilt fresh every step either way.)
                std::fill(h_map.begin(), h_map.begin() + (f + 1) * tokens,
                          -1L);
                int staged_tiles = 0;
                for (int i = 0; i < count; ++i) {
                    const long gid = ids[i];
                    const long src_f = xframe ? gid / num_tiles : f;
                    const long tile = xframe ? gid % num_tiles : gid;
                    const long t0 = tile * tile_tokens;
                    const long c = std::min(tile_tokens, tokens - t0);
                    // One arena slot = one token's K|V row; stage per row so
                    // slot layout matches the kernel's map semantics.
                    bool all = true;
                    for (long r = 0; r < c && all; ++r) {
                        const long row_id = src_f * tokens + t0 + r;
                        if (streaming) {
                            long slot = -1;
                            bool restaged = false;
                            // rows are immutable: changed=false always
                            all = mgr.streaming_stage(
                                row_id,
                                d_k + ((size_t)row_id) * hidden,
                                (size_t)hidden * 4,
                                d_v + ((size_t)row_id) * hidden,
                                (size_t)hidden * 4, false, &slot,
                                &restaged);
                            if (all) h_map[(size_t)row_id] = slot;
                        } else {
                            long row = -1;
                            all = mgr.arena_stage(
                                d_k + ((size_t)row_id) * hidden,
                                (size_t)hidden * 4,
                                d_v + ((size_t)row_id) * hidden,
                                (size_t)hidden * 4, &row);
                            if (all)
                                h_map[(size_t)row_id] = row;
                        }
                    }
                    if (all) ++staged_tiles;
                }
                // Upload map: full prefix under cross (absolute indexing),
                // per-frame slice otherwise.
                const size_t map_elems =
                    xframe ? (size_t)((f + 1) * tokens) : (size_t)tokens;
                long* map_dst =
                    xframe ? d_kvmap : d_kvmap + (size_t)f * tokens;
                const long* map_src = xframe
                                          ? h_map.data()
                                          : h_map.data() + (size_t)f * tokens;
                cuda_check(cudaMemcpy(map_dst, map_src,
                                      map_elems * sizeof(long),
                                      cudaMemcpyHostToDevice), "H2D map");
                if (streaming)
                    mgr.streaming_apply_window();
                else
                    mgr.arena_apply_window();
                attn_forward_single_frame(
                    d_q + (size_t)f * tokens * hidden,
                    xframe ? d_k : d_k + (size_t)f * tokens * hidden,
                    xframe ? d_v : d_v + (size_t)f * tokens * hidden,
                    (const float*)mgr.arena_device_ptr(),
                    xframe ? d_kvmap : d_kvmap + (size_t)f * tokens,
                    d_out + (size_t)f * tokens * hidden,
                    (int)tokens, (int)hidden, (int)heads, xframe ? 1 : 0,
                    (int)f);
                cuda_check(cudaDeviceSynchronize(), "frame run");
                if (streaming)
                    prev_selected = selected;
                else
                    mgr.arena_end_step();
                if (f == 0) {
                    printf("frame 0 selected tiles:");
                    for (int i = 0; i < count; ++i) {
                        const long gid = ids[i];
                        const long sf = xframe ? gid / num_tiles : f;
                        const long tl = xframe ? gid % num_tiles : gid;
                        printf(" %ld(f%ld,t%ld,score %.3f)", gid, sf, tl,
                               scores[sf][tl]);
                    }
                    printf("\n");
                }
            }
            if (streaming) {
                mgr.streaming_end();
                printf("streamed staged_bytes=%zu\n",
                       mgr.streaming_staged_bytes());
            }
        } else {
            fprintf(stderr, "manager init failed\n");
            return 1;
        }
    } else {
        // Per-frame launches even without persistence, so Nsight sees an
        // identical launch structure across policies (no cross-frame L2
        // warmth advantage for the baseline).
        for (long f = 0; f < frames; ++f) {
            attn_forward_single_frame(
                d_q + (size_t)f * tokens * hidden,
                xframe ? d_k : d_k + (size_t)f * tokens * hidden,
                xframe ? d_v : d_v + (size_t)f * tokens * hidden,
                nullptr, nullptr,
                d_out + (size_t)f * tokens * hidden,
                (int)tokens, (int)hidden, (int)heads, xframe ? 1 : 0, (int)f);
        }
        cuda_check(cudaGetLastError(), "launch");
        cuda_check(cudaDeviceSynchronize(), "run");
    }

    std::vector<float> h_out(elems);
    cuda_check(cudaMemcpy(h_out.data(), d_out, elems * 4,
                          cudaMemcpyDeviceToHost), "D2H out");

    std::filesystem::create_directories(
        std::filesystem::path(out_path).parent_path());
    FILE* fo = fopen(out_path.c_str(), "wb");
    fwrite(h_out.data(), 4, elems, fo);
    fclose(fo);
    printf("policy=%s -> %s (%zu bytes)\n", policy->name(), out_path.c_str(),
           elems * 4);

    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_out);
    if (d_kvmap) cudaFree(d_kvmap);
    cudaDeviceReset();
    return 0;
}
