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

// All functions return 0 on success, a positive cudaError_t code for CUDA
// failures, or a negative TE_ERR_* code for logic errors; on failure
// te_last_error copies a human-readable message for the calling thread into
// buf (truncated to buf_len).
#define TE_ERR_STATE   (-1)  // wrong call order / missing weights
#define TE_ERR_ARG     (-2)  // bad argument
#define TE_ERR_CUBLAS  (-3)  // cuBLAS failure

TE_API int te_last_error(char* buf, int buf_len);

// Create the CUDA context on the given device. Call once before anything else.
TE_API int te_init(int device);

// Write a one-line device description into buf (truncated to buf_len).
TE_API int te_device_info(char* buf, int buf_len);

// Walking-skeleton smoke test: out[i] = a[i] + b[i] computed on the GPU.
TE_API int te_smoke_vector_add(const float* a, const float* b, float* out, int n);

// ---- Model lifecycle -------------------------------------------------------

// Field order/types must match engine/backend.ModelConfig exactly (all
// 8-byte fields; the struct crosses the boundary as a raw pointer).
typedef struct TeModelConfig {
    int64_t hidden;
    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t intermediate;
    int64_t vocab;
    int64_t max_seq;
    // MoE (all 0 for dense models)
    int64_t n_experts;         // 0 = dense MLP
    int64_t top_k;             // experts per token
    int64_t moe_intermediate;  // per-expert FFN width
    int64_t router_mode;       // 0 = softmax top-k renorm (Mixtral/Qwen),
                               // 1 = sigmoid + expert-bias norm (Sarvam/DSv3)
    double rope_theta;
    double rms_eps;
} TeModelConfig;

// Allocate a model (weights arrive via te_model_load_tensor; call
// te_model_finalize before te_forward). One model per process for now.
TE_API int te_model_create(const TeModelConfig* cfg);

// Upload one fp32 weight tensor by its HF name. numel floats are copied to
// the device; the host buffer may be reused afterwards.
TE_API int te_model_load_tensor(const char* name, const float* data, int64_t numel);

// Validate that every weight the config requires has arrived; allocate KV
// cache and scratch. After this the model is immutable and ready to run.
TE_API int te_model_finalize(void);

// Run the forward pass for n tokens starting at absolute position pos
// (pos == current KV length; prefill passes n>1, decode passes n=1).
// Writes vocab-size logits for the LAST token into logits_out (host buffer).
TE_API int te_forward(const int32_t* tokens, int64_t n, int64_t pos, float* logits_out);

// Drop all cached KV (start a fresh sequence).
TE_API int te_reset_kv(void);

// Toggle fused kernels (residual-add + RMSNorm). Default on; the unfused
// path exists so speedups can be measured honestly in one binary.
TE_API int te_set_fusion(int64_t enabled);

// ---- Debug taps (correctness harness only) ---------------------------------
// When enabled, each te_forward captures host copies of: embedding output,
// every layer's residual-stream output, and the final-norm output
// (n_tokens x hidden floats each, matching HF output_hidden_states layout).

TE_API int te_debug_set(int64_t enabled);
TE_API int64_t te_debug_count(void);              // entries captured by last forward
TE_API int64_t te_debug_size(int64_t idx);        // numel of entry idx (or <0)
TE_API int te_debug_read(int64_t idx, float* out, int64_t numel);

#ifdef __cplusplus
}
#endif
