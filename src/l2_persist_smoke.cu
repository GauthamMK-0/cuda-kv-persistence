#include <cstdio>
#include <cstdint>

__global__ void touch_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += 1.0f;
}

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            printf("FAIL: %s -> %s (%s:%d)\n", #call,                          \
                   cudaGetErrorString(e), __FILE__, __LINE__);                 \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s | cc %d.%d\n", prop.name, prop.major, prop.minor);
    printf("l2CacheSize             = %zu B\n", (size_t)prop.l2CacheSize);
    printf("persistingL2CacheMaxSize= %zu B\n", (size_t)prop.persistingL2CacheMaxSize);

    size_t want = prop.persistingL2CacheMaxSize;
    CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, want));

    size_t got = 0;
    CHECK(cudaDeviceGetLimit(&got, cudaLimitPersistingL2CacheSize));
    printf("set cudaLimitPersistingL2CacheSize = %zu B, readback = %zu B\n", want, got);
    if (got != want) {
        printf("FAIL: limit readback mismatch\n");
        return 1;
    }

    const int N = 256 * 1024;
    const size_t bytes = N * sizeof(float);
    float* d_buf = nullptr;
    CHECK(cudaMalloc(&d_buf, bytes));
    CHECK(cudaMemset(d_buf, 0, bytes));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    size_t window = bytes < prop.accessPolicyMaxWindowSize ? bytes : prop.accessPolicyMaxWindowSize;
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = d_buf;
    attr.accessPolicyWindow.num_bytes = window;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    CHECK(cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr));
    printf("accessPolicyWindow applied: %zu B flagged persisting on stream\n", window);

    touch_kernel<<<(N + 255) / 256, 256, 0, stream>>>(d_buf, N);
    CHECK(cudaGetLastError());
    CHECK(cudaStreamSynchronize(stream));

    float host0 = 0.0f;
    CHECK(cudaMemcpy(&host0, d_buf, sizeof(float), cudaMemcpyDeviceToHost));
    if (host0 != 1.0f) {
        printf("FAIL: data integrity after persisted-window kernel (buf[0]=%f)\n", host0);
        return 1;
    }
    printf("kernel executed against persisting window; data verified\n");

    cudaStreamAttrValue reset{};
    reset.accessPolicyWindow.num_bytes = 0;
    reset.accessPolicyWindow.hitRatio = 0.f;
    reset.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    reset.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &reset);
    cudaCtxResetPersistingL2Cache();
    CHECK(cudaFree(d_buf));
    CHECK(cudaStreamDestroy(stream));

    size_t after_reset = 0;
    CHECK(cudaDeviceGetLimit(&after_reset, cudaLimitPersistingL2CacheSize));
    printf("post-reset persisting limit = %zu B\n", after_reset);

    printf("PASS: L2 persistence mechanism operational on this hardware\n");
    return 0;
}
