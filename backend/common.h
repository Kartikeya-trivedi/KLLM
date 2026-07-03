// Shared error plumbing for the CUDA backend. The per-thread message buffer
// is defined in shim.cu; every C-ABI function fills it on failure.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>

extern thread_local char g_te_last_error[512];

#define TE_CHECK(call)                                                          \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            snprintf(g_te_last_error, sizeof(g_te_last_error), "%s:%d: %s: %s", \
                     __FILE__, __LINE__, #call, cudaGetErrorString(err_));      \
            return (int)err_;                                                   \
        }                                                                       \
    } while (0)

#define TE_CHECK_CUBLAS(call)                                                   \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            snprintf(g_te_last_error, sizeof(g_te_last_error),                  \
                     "%s:%d: %s: cublas status %d", __FILE__, __LINE__, #call,  \
                     (int)st_);                                                 \
            return TE_ERR_CUBLAS;                                               \
        }                                                                       \
    } while (0)

// Fill the error buffer and return `code`.
#define TE_FAIL(code, ...)                                                      \
    do {                                                                        \
        snprintf(g_te_last_error, sizeof(g_te_last_error), __VA_ARGS__);        \
        return (code);                                                          \
    } while (0)
