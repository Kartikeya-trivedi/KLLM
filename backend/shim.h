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
    // Paged KV
    int64_t kv_block_size;  // tokens per KV block (e.g. 16)
    int64_t kv_num_blocks;  // pool size; VRAM = layers*2*num_blocks*block*kv_dim*4B
    // Architecture family
    int64_t arch;             // 0 = llama (RMSNorm w, SiLU), 1 = gemma3
                              //   (RMSNorm (1+w), embed*sqrt(hidden), GELU-tanh,
                              //    qk-norm, sandwich norms)
    int64_t sliding_window;   // >0: sliding-attention layers use this window
    int64_t sliding_pattern;  // layer l is sliding iff (l+1) % pattern != 0
    double rope_theta;        // full-attention (global) layers
    double rope_local_theta;  // sliding layers (0 = same as rope_theta)
    double query_scalar;      // attn scale = 1/sqrt(query_scalar); 0 = head_dim
    double rms_eps;
} TeModelConfig;

// Allocate a model (weights arrive via te_model_load_tensor; call
// te_model_finalize before te_forward). One model per process for now.
TE_API int te_model_create(const TeModelConfig* cfg);

// Upload one fp32 weight tensor by its HF name. numel floats are copied to
// the device; the host buffer may be reused afterwards.
TE_API int te_model_load_tensor(const char* name, const float* data, int64_t numel);

// Upload one W4 group-quantized projection weight (registered under the
// plain ".weight" name; the forward pass dispatches to the dequant-fused
// matmul for it). q: [out][in/2] packed nibbles (even col low), value =
// (nibble - 8) * scales[out][in/group].
TE_API int te_model_load_tensor_w4(const char* name, const uint8_t* q,
                                   const float* scales, int64_t out_dim,
                                   int64_t in_dim, int64_t group);

// Microbenchmark: time Y[n,m] = X[n,k] x W[m,k]^T over iters iterations.
// mode 0 = fp32 cuBLAS, 1 = W4 dequant-fused kernel. Writes avg ms per
// iteration to ms_out. Standalone (own allocations); requires te_init only.
TE_API int te_bench_matmul(int64_t m, int64_t k, int64_t n, int64_t iters,
                           int64_t mode, double* ms_out);

// Optional, before finalize: explicit per-layer sliding flags (1 = sliding,
// 0 = full attention), length n_layers. Overrides the (l+1)%pattern formula —
// authoritative when the checkpoint ships layer_types.
TE_API int te_model_set_layer_sliding(const int32_t* sliding, int64_t n);

// Validate that every weight the config requires has arrived; allocate KV
// cache and scratch. After this the model is immutable and ready to run.
TE_API int te_model_finalize(void);

// Maximum sequences per forward_step batch (bounds host/device staging).
#define TE_MAX_BATCH_SEQS 128

// Run one forward step for a batch of sequences. The backend is stateless
// about sequences: Go owns block tables and positions.
//   tokens          concatenated new tokens for all sequences
//   n_tokens[s]     new-token count of sequence s (prefill >1, decode ==1)
//   pos[s]          absolute start position of sequence s's new tokens
//   block_tables    [n_seqs][max_blocks_per_seq] physical KV block ids;
//                   entries beyond a sequence's needs are ignored
//   logits_out      [n_seqs][vocab] logits of each sequence's LAST new token
TE_API int te_forward_batch(int64_t n_seqs, const int32_t* tokens,
                            const int32_t* n_tokens, const int32_t* pos,
                            const int32_t* block_tables,
                            int64_t max_blocks_per_seq, float* logits_out);

// Toggle fused kernels (residual-add + RMSNorm). Default on; the unfused
// path exists so speedups can be measured honestly in one binary.
TE_API int te_set_fusion(int64_t enabled);

// Select kernel-optimization attempt versions (-1 leaves one unchanged).
// w4: 0 naive, 1 coalesced+reduction, 2 vectorized (default).
// attn: 0 thread-per-(token,head), 1 block-parallel (default).
TE_API int te_set_kernels(int64_t w4_version, int64_t attn_version);

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
