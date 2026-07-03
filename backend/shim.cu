#define TE_BUILD_DLL
#include "shim.h"

#include <cuda_runtime.h>
#include <cstdio>

// Per-thread last-error message so concurrent Go callers don't clobber each other.
static thread_local char g_last_error[512] = "";

#define TE_CHECK(call)                                                        \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            snprintf(g_last_error, sizeof(g_last_error), "%s:%d: %s: %s",     \
                     __FILE__, __LINE__, #call, cudaGetErrorString(err_));    \
            return (int)err_;                                                 \
        }                                                                     \
    } while (0)

extern "C" {

int te_last_error(char* buf, int buf_len) {
    snprintf(buf, buf_len, "%s", g_last_error);
    return 0;
}

int te_init(int device) {
    TE_CHECK(cudaSetDevice(device));
    TE_CHECK(cudaFree(nullptr)); // force context creation now, not lazily
    return 0;
}

int te_device_info(char* buf, int buf_len) {
    int dev = 0;
    TE_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    TE_CHECK(cudaGetDeviceProperties(&p, dev));
    snprintf(buf, buf_len, "%s (sm_%d%d, %.1f GiB, %d SMs)", p.name, p.major,
             p.minor, p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
             p.multiProcessorCount);
    return 0;
}

__global__ void vector_add_kernel(const float* a, const float* b, float* out,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int te_smoke_vector_add(const float* a, const float* b, float* out, int n) {
    size_t bytes = (size_t)n * sizeof(float);
    float *da = nullptr, *db = nullptr, *dout = nullptr;
    TE_CHECK(cudaMalloc(&da, bytes));
    TE_CHECK(cudaMalloc(&db, bytes));
    TE_CHECK(cudaMalloc(&dout, bytes));
    TE_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(da, db, dout, n);
    TE_CHECK(cudaGetLastError());

    TE_CHECK(cudaMemcpy(out, dout, bytes, cudaMemcpyDeviceToHost));
    TE_CHECK(cudaFree(da));
    TE_CHECK(cudaFree(db));
    TE_CHECK(cudaFree(dout));
    return 0;
}

} // extern "C"
