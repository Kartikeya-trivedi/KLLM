// C-ABI boundary between the Go engine and the CUDA C++ backend.
// Keep this narrow: small control data crosses, big tensors stay on-device.
#pragma once

#include <stdint.h>

#if defined(_WIN32) && defined(TE_BUILD_DLL)
#define TE_API __declspec(dllexport)
#elif defined(_WIN32)
#define TE_API __declspec(dllimport)
#else
#define TE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// All functions return 0 on success or a cudaError_t code; on failure
// te_last_error copies a human-readable message for the calling thread into
// buf (truncated to buf_len).
TE_API int te_last_error(char* buf, int buf_len);

// Create the CUDA context on the given device. Call once before anything else.
TE_API int te_init(int device);

// Write a one-line device description into buf (truncated to buf_len).
TE_API int te_device_info(char* buf, int buf_len);

// Walking-skeleton smoke test: out[i] = a[i] + b[i] computed on the GPU.
// Proves DLL load, context, H2D/D2H copies, and a kernel launch from Go.
TE_API int te_smoke_vector_add(const float* a, const float* b, float* out, int n);

#ifdef __cplusplus
}
#endif
