// Model state + forward pass. Phase 0: fp32 everywhere, batch=1, contiguous
// KV cache, cuBLAS SGEMM for matmuls, naive elementwise/attention kernels.
// Correctness over speed — kernels get optimized once the whole system is
// green against the HF reference dumps.
#define TE_BUILD_DLL
#include "shim.h"
#include "common.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct DeviceTensor {
    float* d = nullptr;
    int64_t numel = 0;
};

// W4 group-quantized weight: packed nibbles + per-group scales on device.
struct QuantTensor {
    uint8_t* q = nullptr;    // [out][in/2], even col in low nibble
    float* scales = nullptr; // [out][in/group]
    int64_t out_dim = 0, in_dim = 0, group = 0;
};

struct Model {
    TeModelConfig c{};
    int64_t q_dim = 0;   // n_heads * head_dim
    int64_t kv_dim = 0;  // n_kv_heads * head_dim
    bool finalized = false;

    std::unordered_map<std::string, DeviceTensor> w;
    std::unordered_map<std::string, QuantTensor> qw;

    // Paged KV pool: [n_layers][2 (k,v)][num_blocks][block_size][kv_dim].
    // Go owns block tables + free list; the backend is sequence-stateless.
    float* kv = nullptr;

    // Scratch, sized for max_seq concatenated tokens in flight.
    float *x = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
    float *attn = nullptr, *proj = nullptr, *mlp_out = nullptr;
    float *ff_gate = nullptr, *ff_up = nullptr;

    // MoE scratch (allocated only when n_experts > 0); rows = max_seq*top_k.
    float* d_router = nullptr;      // [max_seq, n_experts]
    int32_t* d_topk_idx = nullptr;  // [max_seq, top_k]
    float* d_topk_w = nullptr;      // [max_seq, top_k]
    int32_t* d_perm_tok = nullptr;  // [rows] source/dest token per permuted row
    float* d_perm_w = nullptr;      // [rows] routing weight per permuted row
    float* d_xg = nullptr;          // [rows, hidden] gathered inputs
    float* d_moe_gate = nullptr;    // [rows, moe_intermediate]
    float* d_moe_up = nullptr;      // [rows, moe_intermediate]
    float* d_moe_down = nullptr;    // [rows, hidden]
    float *d_last_hidden = nullptr;  // [TE_MAX_BATCH_SEQS, hidden]
    float* d_logits = nullptr;       // [TE_MAX_BATCH_SEQS, vocab]
    int32_t* d_tokens = nullptr;     // per concat token
    int32_t* d_positions = nullptr;  // per concat token: absolute position
    int32_t* d_seq_ids = nullptr;    // per concat token: owning sequence
    int32_t* d_tables = nullptr;     // [TE_MAX_BATCH_SEQS, max_blocks_per_seq_cap]
    int32_t* d_last_idx = nullptr;   // per seq: concat row of its last token
    int64_t tables_cap = 0;          // allocated entries in d_tables

    cublasHandle_t cublas = nullptr;
    bool fused = true;  // te_set_fusion

    // Explicit per-layer sliding flags (te_model_set_layer_sliding);
    // empty = derive from (l+1) % sliding_pattern.
    std::vector<int32_t> layer_sliding;

    bool debug = false;
    std::vector<std::vector<float>> dbg;  // host copies of residual stream
};

Model* g_model = nullptr;

// ---- Kernels ----------------------------------------------------------------

__global__ void embed_gather_kernel(float* out, const float* embed,
                                    const int32_t* ids, int64_t hidden) {
    int64_t t = blockIdx.x;
    const float* src = embed + (int64_t)ids[t] * hidden;
    float* dst = out + t * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// One block per row; blockDim must be a power of two. plus_one selects the
// Gemma parameterization out = x * rsqrt(ms+eps) * (1 + w) (weights stored
// zero-centered) vs the Llama out = x * rsqrt(ms+eps) * w. Safe in-place
// (out == in): each element is read and written by the same thread.
__global__ void rmsnorm_kernel(float* out, const float* in, const float* w,
                               int64_t hidden, float eps, int plus_one) {
    extern __shared__ float sh[];
    const float* row = in + (int64_t)blockIdx.x * hidden;
    float* orow = out + (int64_t)blockIdx.x * hidden;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = row[i];
        ss += v * v;
    }
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hidden + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float wi = plus_one ? 1.f + w[i] : w[i];
        orow[i] = row[i] * scale * wi;
    }
}

__global__ void scale_kernel(float* x, int64_t n, float s) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        x[i] *= s;
    }
}

// gate = gelu_tanh(gate) * up (Gemma's gelu_pytorch_tanh approximation).
__global__ void gelu_tanh_mul_kernel(float* gate, const float* up, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float x = gate[i];
        float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        gate[i] = 0.5f * x * (1.f + t) * up[i];
    }
}

// Phase 1 fusion: x += res, then out = rmsnorm(x) * w, one pass instead of
// an add kernel plus a norm kernel (saves one full read+write of the
// residual stream per site). One block per token.
__global__ void add_rmsnorm_kernel(float* x, const float* res, float* out,
                                   const float* w, int64_t hidden, float eps) {
    extern __shared__ float sh[];
    float* xrow = x + (int64_t)blockIdx.x * hidden;
    const float* rrow = res + (int64_t)blockIdx.x * hidden;
    float* orow = out + (int64_t)blockIdx.x * hidden;
    float ss = 0.f;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = xrow[i] + rrow[i];
        xrow[i] = v;
        ss += v * v;
    }
    sh[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float scale = rsqrtf(sh[0] / (float)hidden + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
        orow[i] = xrow[i] * scale * w[i];
}

// HF Llama RoPE (rotate_half): pairs (j, j+d/2) share inv_freq[j].
// Angles computed in double to track the fp32 CPU reference closely.
// positions[] gives each concatenated token its absolute position.
__global__ void rope_kernel(float* t, const int32_t* positions,
                            int64_t n_heads, int64_t head_dim, double theta) {
    int64_t token = blockIdx.x;
    int64_t half = head_dim / 2;
    for (int64_t idx = threadIdx.x; idx < n_heads * half; idx += blockDim.x) {
        int64_t h = idx / half, j = idx % half;
        double freq = pow(theta, -2.0 * (double)j / (double)head_dim);
        double ang = (double)positions[token] * freq;
        float c = (float)cos(ang), s = (float)sin(ang);
        float* base = t + token * (n_heads * head_dim) + h * head_dim;
        float a = base[j], b = base[j + half];
        base[j] = a * c - b * s;
        base[j + half] = b * c + a * s;
    }
}

// Paged append: token t's KV rows land in its sequence's block table slot.
__global__ void kv_append_paged_kernel(float* kpool, float* vpool,
                                       const float* k, const float* v,
                                       const int32_t* positions,
                                       const int32_t* seq_ids,
                                       const int32_t* tables, int64_t mbps,
                                       int64_t block_size, int64_t kv_dim) {
    int64_t t = blockIdx.x;
    int64_t p = positions[t];
    int32_t phys = tables[(int64_t)seq_ids[t] * mbps + p / block_size];
    int64_t dst = ((int64_t)phys * block_size + p % block_size) * kv_dim;
    for (int64_t i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        kpool[dst + i] = k[t * kv_dim + i];
        vpool[dst + i] = v[t * kv_dim + i];
    }
}

// Deliberately naive paged causal attention: one thread per (query token,
// head), gathering K/V through the block table. Fine at lab scale; the
// optimized tiled kernel is the deferred endgame.
#define TE_ATTN_MAX_SEQ 4096
#define TE_ATTN_MAX_HEAD_DIM 256

__global__ void attn_paged_kernel(float* out, const float* q,
                                  const float* kpool, const float* vpool,
                                  const int32_t* positions,
                                  const int32_t* seq_ids,
                                  const int32_t* tables, int64_t mbps,
                                  int64_t block_size, int64_t n_heads,
                                  int64_t n_kv, int64_t head_dim,
                                  int64_t kv_dim, float scale,
                                  int64_t window) {
    int64_t t = blockIdx.x, h = blockIdx.y;
    int64_t q_dim = n_heads * head_dim;
    int64_t kvh = h / (n_heads / n_kv);
    const float* qv = q + t * q_dim + h * head_dim;
    const int32_t* table = tables + (int64_t)seq_ids[t] * mbps;
    int64_t len = positions[t] + 1;  // causal: attend to positions [0, len)
    int64_t start = 0;               // sliding layers see only the last `window`
    if (window > 0 && len > window) start = len - window;

    float sc[TE_ATTN_MAX_SEQ];
    float maxs = -1e30f;
    for (int64_t p = start; p < len; p++) {
        int64_t off = ((int64_t)table[p / block_size] * block_size + p % block_size) * kv_dim;
        const float* kv_row = kpool + off + kvh * head_dim;
        float dot = 0.f;
        for (int64_t d = 0; d < head_dim; d++) dot += qv[d] * kv_row[d];
        sc[p - start] = dot * scale;
        if (sc[p - start] > maxs) maxs = sc[p - start];
    }
    float sum = 0.f;
    for (int64_t p = start; p < len; p++) {
        sc[p - start] = expf(sc[p - start] - maxs);
        sum += sc[p - start];
    }
    float acc[TE_ATTN_MAX_HEAD_DIM];
    for (int64_t d = 0; d < head_dim; d++) acc[d] = 0.f;
    for (int64_t p = start; p < len; p++) {
        int64_t off = ((int64_t)table[p / block_size] * block_size + p % block_size) * kv_dim;
        const float* v_row = vpool + off + kvh * head_dim;
        float wgt = sc[p - start] / sum;
        for (int64_t d = 0; d < head_dim; d++) acc[d] += wgt * v_row[d];
    }
    float* orow = out + t * q_dim + h * head_dim;
    for (int64_t d = 0; d < head_dim; d++) orow[d] = acc[d];
}

// W4 dequant-fused matmul: Y[n,m] = X[n,k] x dequant(Q,S)[m,k]^T.
// Naive one-thread-per-output-element version — correctness first; the
// tiled/tensor-core version is the deferred kernel-optimization endgame.
__global__ void matmul_w4_kernel(float* Y, const float* X, const uint8_t* Q,
                                 const float* S, int64_t m, int64_t n,
                                 int64_t k, int64_t group) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * m) return;
    int64_t t = idx / m, r = idx % m;
    const uint8_t* qrow = Q + r * (k / 2);
    const float* srow = S + r * (k / group);
    const float* xrow = X + t * k;
    float acc = 0.f;
    for (int64_t j = 0; j < k; j += 2) {
        uint8_t byte = qrow[j >> 1];
        float s = srow[j / group];  // group size is even: j, j+1 share a group
        acc += (float)((int)(byte & 0xF) - 8) * s * xrow[j];
        acc += (float)((int)(byte >> 4) - 8) * s * xrow[j + 1];
    }
    Y[t * m + r] = acc;
}

// Gather selected rows (each sequence's last token) before the LM head.
__global__ void gather_rows_kernel(float* out, const float* in,
                                   const int32_t* idx, int64_t hidden) {
    int64_t r = blockIdx.x;
    const float* src = in + (int64_t)idx[r] * hidden;
    float* dst = out + r * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// MoE routing: one thread per token. mode 0 = softmax top-k renormalized
// (Mixtral/Qwen); mode 1 = sigmoid scores, selection by score+bias, weights
// from unbiased scores normalized over the selection (Sarvam/DSv3 family).
#define TE_MAX_EXPERTS 64

__global__ void route_kernel(int32_t* topk_idx, float* topk_w,
                             const float* logits, int64_t n, int64_t n_exp,
                             int64_t k, int64_t mode, const float* bias) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const float* row = logits + t * n_exp;
    float score[TE_MAX_EXPERTS], metric[TE_MAX_EXPERTS];
    if (mode == 0) {
        float mx = row[0];
        for (int64_t e = 1; e < n_exp; e++) mx = fmaxf(mx, row[e]);
        float sum = 0.f;
        for (int64_t e = 0; e < n_exp; e++) {
            score[e] = expf(row[e] - mx);
            sum += score[e];
        }
        for (int64_t e = 0; e < n_exp; e++) {
            score[e] /= sum;
            metric[e] = score[e];
        }
    } else {
        for (int64_t e = 0; e < n_exp; e++) {
            score[e] = 1.f / (1.f + expf(-row[e]));
            metric[e] = score[e] + bias[e];
        }
    }
    bool used[TE_MAX_EXPERTS] = {false};
    float wsum = 0.f;
    for (int64_t i = 0; i < k; i++) {
        int best = -1;
        for (int64_t e = 0; e < n_exp; e++)
            if (!used[e] && (best < 0 || metric[e] > metric[best])) best = (int)e;
        used[best] = true;
        topk_idx[t * k + i] = best;
        topk_w[t * k + i] = score[best];
        wsum += score[best];
    }
    for (int64_t i = 0; i < k; i++) topk_w[t * k + i] /= wsum;
}

// Un-permute: out[token] += w * rows[r] for each permuted row. Multiple
// experts hit the same token, hence atomics (fp32 atomicAdd; ~1e-7
// nondeterminism, inside test tolerances).
__global__ void scatter_add_kernel(float* out, const float* rows,
                                   const int32_t* perm_tok,
                                   const float* perm_w, int64_t hidden) {
    int64_t r = blockIdx.x;
    float* dst = out + (int64_t)perm_tok[r] * hidden;
    const float* src = rows + r * hidden;
    float w = perm_w[r];
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
        atomicAdd(&dst[i], w * src[i]);
}

__global__ void silu_mul_kernel(float* gate, const float* up, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        float g = gate[i];
        gate[i] = (g / (1.f + expf(-g))) * up[i];
    }
}

__global__ void add_inplace_kernel(float* x, const float* y, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        x[i] += y[i];
    }
}

// ---- Helpers ------------------------------------------------------------------

// Y[n,m] (row-major) = X[n,k] (row-major) x W[m,k]^T, HF weight convention.
// Column-major reading: Y'[m,n] = op_T(W'[k,m]) x X'[k,n].
int gemm_rowmajor(cublasHandle_t h, const float* W, const float* X, float* Y,
                  int64_t m, int64_t n, int64_t k) {
    const float one = 1.f, zero = 0.f;
    TE_CHECK_CUBLAS(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, (int)m, (int)n,
                                (int)k, &one, W, (int)k, X, (int)k, &zero, Y,
                                (int)m));
    return 0;
}

const DeviceTensor* find(Model* m, const std::string& name) {
    auto it = m->w.find(name);
    return it == m->w.end() ? nullptr : &it->second;
}

int require(Model* m, const std::string& name, int64_t numel,
            const float** out) {
    const DeviceTensor* t = find(m, name);
    if (!t) TE_FAIL(TE_ERR_STATE, "missing weight %s", name.c_str());
    if (t->numel != numel)
        TE_FAIL(TE_ERR_STATE, "weight %s has %lld elements, expected %lld",
                name.c_str(), (long long)t->numel, (long long)numel);
    if (out) *out = t->d;
    return 0;
}

std::string lname(int64_t layer, const char* suffix) {
    return "model.layers." + std::to_string(layer) + "." + suffix;
}

// Finalize-time check: projection present either dense or quantized.
int require_proj(Model* m, const std::string& name, int64_t out_dim, int64_t in_dim) {
    auto qi = m->qw.find(name);
    if (qi != m->qw.end()) {
        if (qi->second.out_dim != out_dim || qi->second.in_dim != in_dim)
            TE_FAIL(TE_ERR_STATE, "quant weight %s has wrong shape", name.c_str());
        return 0;
    }
    return require(m, name, out_dim * in_dim, nullptr);
}

int debug_capture(Model* m, const float* dev, int64_t numel) {
    if (!m->debug) return 0;
    std::vector<float> host(numel);
    TE_CHECK(cudaMemcpy(host.data(), dev, numel * sizeof(float),
                        cudaMemcpyDeviceToHost));
    m->dbg.push_back(std::move(host));
    return 0;
}

constexpr int kBlock = 256;

int grid_for(int64_t n) {
    int64_t g = (n + kBlock - 1) / kBlock;
    return (int)(g > 4096 ? 4096 : g);
}

// x += res, then xn = rmsnorm(x, w). Fused (1 kernel) or unfused (2), per
// the te_set_fusion toggle — identical math either way. Llama-path only.
void add_norm(Model* m, int64_t n, float* res, const float* w) {
    const TeModelConfig& c = m->c;
    if (m->fused) {
        add_rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
            m->x, res, m->xn, w, c.hidden, (float)c.rms_eps);
    } else {
        add_inplace_kernel<<<grid_for(n * c.hidden), kBlock>>>(m->x, res, n * c.hidden);
        rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
            m->xn, m->x, w, c.hidden, (float)c.rms_eps, 0);
    }
}

// Gemma-style norm: out = rmsnorm_{1+w}(in). Rows of width `width`.
void norm_g(int64_t rows, float* out, const float* in, const float* w,
            int64_t width, float eps) {
    rmsnorm_kernel<<<(int)rows, kBlock, kBlock * sizeof(float)>>>(out, in, w, width, eps, 1);
}

bool layer_is_sliding(const Model* m, int64_t l) {
    const TeModelConfig& c = m->c;
    if (c.sliding_window <= 0) return false;
    if (!m->layer_sliding.empty()) return m->layer_sliding[l] != 0;
    return c.sliding_pattern > 0 && ((l + 1) % c.sliding_pattern != 0);
}

// Projection matmul with per-weight dispatch: W4 dequant-fused kernel if the
// weight arrived quantized, cuBLAS SGEMM otherwise.
int mm(Model* m, const std::string& name, const float* X, float* Y,
       int64_t out_dim, int64_t in_dim, int64_t n) {
    auto qi = m->qw.find(name);
    if (qi != m->qw.end()) {
        const QuantTensor& t = qi->second;
        if (t.out_dim != out_dim || t.in_dim != in_dim)
            TE_FAIL(TE_ERR_STATE, "quant weight %s is [%lld,%lld], expected [%lld,%lld]",
                    name.c_str(), (long long)t.out_dim, (long long)t.in_dim,
                    (long long)out_dim, (long long)in_dim);
        matmul_w4_kernel<<<grid_for(n * out_dim), kBlock>>>(
            Y, X, t.q, t.scales, out_dim, n, in_dim, t.group);
        return 0;
    }
    const float* wptr;
    int rc = require(m, name, out_dim * in_dim, &wptr);
    if (rc) return rc;
    return gemm_rowmajor(m->cublas, wptr, X, Y, out_dim, n, in_dim);
}

// MoE FFN for layer l: route -> permute tokens by expert -> per-expert
// segment GEMMs ("grouped GEMM" v0: a loop of segment GEMMs; the fused
// grouped kernel is deferred optimization work) -> weighted un-permute.
// Reads m->xn (post-norm hidden), writes m->mlp_out.
int moe_forward(Model* m, int64_t l, int64_t n) {
    const TeModelConfig& c = m->c;
    int rc;

    // Router logits + top-k on device.
    const float* w_gate;
    if ((rc = require(m, lname(l, "block_sparse_moe.gate.weight"), c.n_experts * c.hidden, &w_gate))) return rc;
    if ((rc = gemm_rowmajor(m->cublas, w_gate, m->xn, m->d_router, c.n_experts, n, c.hidden))) return rc;
    const float* bias = nullptr;
    if (c.router_mode == 1)
        if ((rc = require(m, lname(l, "block_sparse_moe.gate.e_score_correction_bias"), c.n_experts, &bias))) return rc;
    route_kernel<<<grid_for(n), kBlock>>>(m->d_topk_idx, m->d_topk_w,
                                          m->d_router, n, c.n_experts, c.top_k,
                                          c.router_mode, bias);

    // Host builds the expert-sorted permutation (small: n*top_k entries).
    int64_t rows = n * c.top_k;
    std::vector<int32_t> h_idx(rows);
    std::vector<float> h_w(rows);
    TE_CHECK(cudaMemcpy(h_idx.data(), m->d_topk_idx, rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
    TE_CHECK(cudaMemcpy(h_w.data(), m->d_topk_w, rows * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int32_t> perm_tok(rows);
    std::vector<float> perm_w(rows);
    std::vector<int64_t> seg_size(c.n_experts, 0);
    int64_t r = 0;
    for (int64_t e = 0; e < c.n_experts; e++) {
        for (int64_t t = 0; t < n; t++)
            for (int64_t i = 0; i < c.top_k; i++)
                if (h_idx[t * c.top_k + i] == e) {
                    perm_tok[r] = (int32_t)t;
                    perm_w[r] = h_w[t * c.top_k + i];
                    r++;
                    seg_size[e]++;
                }
    }
    TE_CHECK(cudaMemcpy(m->d_perm_tok, perm_tok.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(m->d_perm_w, perm_w.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    gather_rows_kernel<<<(int)rows, kBlock>>>(m->d_xg, m->xn, m->d_perm_tok, c.hidden);

    // Per-expert segment GEMMs over the permuted rows.
    int64_t off = 0;
    for (int64_t e = 0; e < c.n_experts; e++) {
        int64_t ns = seg_size[e];
        if (ns == 0) continue;
        std::string epre = "block_sparse_moe.experts." + std::to_string(e) + ".";
        const float* X = m->d_xg + off * c.hidden;
        float* g = m->d_moe_gate + off * c.moe_intermediate;
        float* u = m->d_moe_up + off * c.moe_intermediate;
        float* d = m->d_moe_down + off * c.hidden;
        if ((rc = mm(m, lname(l, (epre + "w1.weight").c_str()), X, g, c.moe_intermediate, c.hidden, ns))) return rc;
        if ((rc = mm(m, lname(l, (epre + "w3.weight").c_str()), X, u, c.moe_intermediate, c.hidden, ns))) return rc;
        silu_mul_kernel<<<grid_for(ns * c.moe_intermediate), kBlock>>>(g, u, ns * c.moe_intermediate);
        if ((rc = mm(m, lname(l, (epre + "w2.weight").c_str()), g, d, c.hidden, c.moe_intermediate, ns))) return rc;
        off += ns;
    }

    // Weighted un-permute into the residual contribution.
    TE_CHECK(cudaMemset(m->mlp_out, 0, n * c.hidden * sizeof(float)));
    scatter_add_kernel<<<(int)rows, kBlock>>>(m->mlp_out, m->d_moe_down,
                                              m->d_perm_tok, m->d_perm_w, c.hidden);
    return 0;
}

}  // namespace

// ---- C-ABI ------------------------------------------------------------------

extern "C" {

int te_model_create(const TeModelConfig* cfg) {
    if (g_model) TE_FAIL(TE_ERR_STATE, "model already created");
    if (!cfg) TE_FAIL(TE_ERR_ARG, "null config");
    if (cfg->n_experts != 0) {
        if (cfg->n_experts < 0 || cfg->n_experts > TE_MAX_EXPERTS)
            TE_FAIL(TE_ERR_ARG, "n_experts %lld out of range (max %d)",
                    (long long)cfg->n_experts, TE_MAX_EXPERTS);
        if (cfg->top_k <= 0 || cfg->top_k > cfg->n_experts)
            TE_FAIL(TE_ERR_ARG, "top_k %lld invalid for %lld experts",
                    (long long)cfg->top_k, (long long)cfg->n_experts);
        if (cfg->moe_intermediate <= 0)
            TE_FAIL(TE_ERR_ARG, "moe_intermediate must be positive");
        if (cfg->router_mode != 0 && cfg->router_mode != 1)
            TE_FAIL(TE_ERR_ARG, "router_mode must be 0 (softmax) or 1 (sigmoid+bias)");
    }
    if (cfg->hidden <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->head_dim <= 0 || cfg->vocab <= 0 ||
        cfg->max_seq <= 0 || cfg->intermediate <= 0)
        TE_FAIL(TE_ERR_ARG, "non-positive config field");
    if (cfg->n_heads % cfg->n_kv_heads != 0)
        TE_FAIL(TE_ERR_ARG, "n_heads %% n_kv_heads != 0");
    if (cfg->max_seq > TE_ATTN_MAX_SEQ || cfg->head_dim > TE_ATTN_MAX_HEAD_DIM)
        TE_FAIL(TE_ERR_ARG, "max_seq/head_dim exceed naive-attention limits");
    if (cfg->kv_block_size <= 0 || cfg->kv_num_blocks <= 0)
        TE_FAIL(TE_ERR_ARG, "kv_block_size/kv_num_blocks must be positive");
    if (cfg->arch != 0 && cfg->arch != 1)
        TE_FAIL(TE_ERR_ARG, "arch must be 0 (llama) or 1 (gemma3)");
    if (cfg->sliding_window > 0 && cfg->sliding_pattern <= 0)
        TE_FAIL(TE_ERR_ARG, "sliding_window set but sliding_pattern is not");

    Model* m = new Model();
    m->c = *cfg;
    m->q_dim = cfg->n_heads * cfg->head_dim;
    m->kv_dim = cfg->n_kv_heads * cfg->head_dim;
    g_model = m;
    return 0;
}

int te_model_load_tensor(const char* name, const float* data, int64_t numel) {
    Model* m = g_model;
    if (!m) TE_FAIL(TE_ERR_STATE, "no model");
    if (m->finalized) TE_FAIL(TE_ERR_STATE, "model already finalized");
    if (!name || !data || numel <= 0) TE_FAIL(TE_ERR_ARG, "bad tensor args");
    if (m->w.count(name)) TE_FAIL(TE_ERR_ARG, "duplicate tensor %s", name);

    DeviceTensor t;
    t.numel = numel;
    TE_CHECK(cudaMalloc(&t.d, numel * sizeof(float)));
    cudaError_t err = cudaMemcpy(t.d, data, numel * sizeof(float),
                                 cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(t.d);
        TE_FAIL((int)err, "upload %s: %s", name, cudaGetErrorString(err));
    }
    m->w[name] = t;
    return 0;
}

int te_model_load_tensor_w4(const char* name, const uint8_t* q,
                            const float* scales, int64_t out_dim,
                            int64_t in_dim, int64_t group) {
    Model* m = g_model;
    if (!m) TE_FAIL(TE_ERR_STATE, "no model");
    if (m->finalized) TE_FAIL(TE_ERR_STATE, "model already finalized");
    if (!name || !q || !scales || out_dim <= 0 || in_dim <= 0 || group <= 0)
        TE_FAIL(TE_ERR_ARG, "bad w4 tensor args");
    if (in_dim % 2 != 0 || in_dim % group != 0 || group % 2 != 0)
        TE_FAIL(TE_ERR_ARG, "w4 %s: in_dim %lld / group %lld must be even and divide",
                name, (long long)in_dim, (long long)group);
    if (m->w.count(name) || m->qw.count(name))
        TE_FAIL(TE_ERR_ARG, "duplicate tensor %s", name);

    QuantTensor t;
    t.out_dim = out_dim;
    t.in_dim = in_dim;
    t.group = group;
    TE_CHECK(cudaMalloc(&t.q, out_dim * in_dim / 2));
    TE_CHECK(cudaMalloc(&t.scales, out_dim * (in_dim / group) * sizeof(float)));
    cudaError_t err = cudaMemcpy(t.q, q, out_dim * in_dim / 2, cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(t.scales, scales, out_dim * (in_dim / group) * sizeof(float),
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(t.q);
        cudaFree(t.scales);
        TE_FAIL((int)err, "upload w4 %s: %s", name, cudaGetErrorString(err));
    }
    m->qw[name] = t;
    return 0;
}

int te_bench_matmul(int64_t m_dim, int64_t k, int64_t n, int64_t iters,
                    int64_t mode, double* ms_out) {
    if (m_dim <= 0 || k <= 0 || n <= 0 || iters <= 0 || !ms_out)
        TE_FAIL(TE_ERR_ARG, "bad bench args");
    int64_t group = (k % 128 == 0) ? 128 : 32;
    if (k % group != 0 || k % 2 != 0) TE_FAIL(TE_ERR_ARG, "k must be even");

    float *X = nullptr, *Y = nullptr, *W = nullptr, *S = nullptr;
    uint8_t* Q = nullptr;
    cublasHandle_t handle = nullptr;
    TE_CHECK(cudaMalloc(&X, n * k * sizeof(float)));
    TE_CHECK(cudaMalloc(&Y, n * m_dim * sizeof(float)));
    TE_CHECK(cudaMemset(X, 0, n * k * sizeof(float)));
    if (mode == 0) {
        TE_CHECK(cudaMalloc(&W, m_dim * k * sizeof(float)));
        TE_CHECK(cudaMemset(W, 0, m_dim * k * sizeof(float)));
        TE_CHECK_CUBLAS(cublasCreate(&handle));
    } else {
        TE_CHECK(cudaMalloc(&Q, m_dim * k / 2));
        TE_CHECK(cudaMalloc(&S, m_dim * (k / group) * sizeof(float)));
        TE_CHECK(cudaMemset(Q, 0x88, m_dim * k / 2));
        TE_CHECK(cudaMemset(S, 0, m_dim * (k / group) * sizeof(float)));
    }

    const float one = 1.f, zero = 0.f;
    auto run_once = [&]() -> int {
        if (mode == 0) {
            TE_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)m_dim, (int)n, (int)k, &one, W,
                                        (int)k, X, (int)k, &zero, Y, (int)m_dim));
        } else {
            matmul_w4_kernel<<<grid_for(n * m_dim), kBlock>>>(Y, X, Q, S, m_dim,
                                                              n, k, group);
        }
        return 0;
    };

    int rc;
    for (int i = 0; i < 3; i++)
        if ((rc = run_once())) return rc;  // warmup
    TE_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    TE_CHECK(cudaEventCreate(&start));
    TE_CHECK(cudaEventCreate(&stop));
    TE_CHECK(cudaEventRecord(start));
    for (int64_t i = 0; i < iters; i++)
        if ((rc = run_once())) return rc;
    TE_CHECK(cudaEventRecord(stop));
    TE_CHECK(cudaEventSynchronize(stop));
    float ms = 0.f;
    TE_CHECK(cudaEventElapsedTime(&ms, start, stop));
    *ms_out = (double)ms / (double)iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(X);
    cudaFree(Y);
    if (W) cudaFree(W);
    if (Q) cudaFree(Q);
    if (S) cudaFree(S);
    if (handle) cublasDestroy(handle);
    return 0;
}

int te_model_finalize(void) {
    Model* m = g_model;
    if (!m) TE_FAIL(TE_ERR_STATE, "no model");
    if (m->finalized) TE_FAIL(TE_ERR_STATE, "already finalized");
    const TeModelConfig& c = m->c;

    // Validate presence + shape of everything the forward pass will touch.
    int rc;
    if ((rc = require(m, "model.embed_tokens.weight", c.vocab * c.hidden, nullptr))) return rc;
    if ((rc = require(m, "model.norm.weight", c.hidden, nullptr))) return rc;
    if ((rc = require(m, "lm_head.weight", c.vocab * c.hidden, nullptr))) return rc;
    for (int64_t l = 0; l < c.n_layers; l++) {
        if ((rc = require(m, lname(l, "input_layernorm.weight"), c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "post_attention_layernorm.weight"), c.hidden, nullptr))) return rc;
        if (c.arch == 1) {  // gemma3: sandwich norms + per-head qk-norm
            if ((rc = require(m, lname(l, "pre_feedforward_layernorm.weight"), c.hidden, nullptr))) return rc;
            if ((rc = require(m, lname(l, "post_feedforward_layernorm.weight"), c.hidden, nullptr))) return rc;
            if ((rc = require(m, lname(l, "self_attn.q_norm.weight"), c.head_dim, nullptr))) return rc;
            if ((rc = require(m, lname(l, "self_attn.k_norm.weight"), c.head_dim, nullptr))) return rc;
        }
        if ((rc = require_proj(m, lname(l, "self_attn.q_proj.weight"), m->q_dim, c.hidden))) return rc;
        if ((rc = require_proj(m, lname(l, "self_attn.k_proj.weight"), m->kv_dim, c.hidden))) return rc;
        if ((rc = require_proj(m, lname(l, "self_attn.v_proj.weight"), m->kv_dim, c.hidden))) return rc;
        if ((rc = require_proj(m, lname(l, "self_attn.o_proj.weight"), c.hidden, m->q_dim))) return rc;
        if (c.n_experts == 0) {
            if ((rc = require_proj(m, lname(l, "mlp.gate_proj.weight"), c.intermediate, c.hidden))) return rc;
            if ((rc = require_proj(m, lname(l, "mlp.up_proj.weight"), c.intermediate, c.hidden))) return rc;
            if ((rc = require_proj(m, lname(l, "mlp.down_proj.weight"), c.hidden, c.intermediate))) return rc;
        } else {
            if ((rc = require(m, lname(l, "block_sparse_moe.gate.weight"), c.n_experts * c.hidden, nullptr))) return rc;
            if (c.router_mode == 1)
                if ((rc = require(m, lname(l, "block_sparse_moe.gate.e_score_correction_bias"), c.n_experts, nullptr))) return rc;
            for (int64_t e = 0; e < c.n_experts; e++) {
                std::string epre = "block_sparse_moe.experts." + std::to_string(e) + ".";
                if ((rc = require_proj(m, lname(l, (epre + "w1.weight").c_str()), c.moe_intermediate, c.hidden))) return rc;
                if ((rc = require_proj(m, lname(l, (epre + "w2.weight").c_str()), c.hidden, c.moe_intermediate))) return rc;
                if ((rc = require_proj(m, lname(l, (epre + "w3.weight").c_str()), c.moe_intermediate, c.hidden))) return rc;
            }
        }
    }

    TE_CHECK(cudaMalloc(&m->kv, c.n_layers * 2 * c.kv_num_blocks *
                                    c.kv_block_size * m->kv_dim * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->x, c.max_seq * c.hidden * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->xn, c.max_seq * c.hidden * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->proj, c.max_seq * c.hidden * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->q, c.max_seq * m->q_dim * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->k, c.max_seq * m->kv_dim * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->v, c.max_seq * m->kv_dim * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->attn, c.max_seq * m->q_dim * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->mlp_out, c.max_seq * c.hidden * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->ff_gate, c.max_seq * c.intermediate * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->ff_up, c.max_seq * c.intermediate * sizeof(float)));
    if (c.n_experts > 0) {
        int64_t rows = c.max_seq * c.top_k;
        TE_CHECK(cudaMalloc(&m->d_router, c.max_seq * c.n_experts * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_topk_idx, c.max_seq * c.top_k * sizeof(int32_t)));
        TE_CHECK(cudaMalloc(&m->d_topk_w, c.max_seq * c.top_k * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_perm_tok, rows * sizeof(int32_t)));
        TE_CHECK(cudaMalloc(&m->d_perm_w, rows * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_xg, rows * c.hidden * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_moe_gate, rows * c.moe_intermediate * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_moe_up, rows * c.moe_intermediate * sizeof(float)));
        TE_CHECK(cudaMalloc(&m->d_moe_down, rows * c.hidden * sizeof(float)));
    }
    TE_CHECK(cudaMalloc(&m->d_last_hidden, TE_MAX_BATCH_SEQS * c.hidden * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->d_logits, TE_MAX_BATCH_SEQS * c.vocab * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->d_tokens, c.max_seq * sizeof(int32_t)));
    TE_CHECK(cudaMalloc(&m->d_positions, c.max_seq * sizeof(int32_t)));
    TE_CHECK(cudaMalloc(&m->d_seq_ids, c.max_seq * sizeof(int32_t)));
    TE_CHECK(cudaMalloc(&m->d_last_idx, TE_MAX_BATCH_SEQS * sizeof(int32_t)));
    m->tables_cap = TE_MAX_BATCH_SEQS *
                    ((c.max_seq + c.kv_block_size - 1) / c.kv_block_size);
    TE_CHECK(cudaMalloc(&m->d_tables, m->tables_cap * sizeof(int32_t)));
    TE_CHECK_CUBLAS(cublasCreate(&m->cublas));

    m->finalized = true;
    return 0;
}

int te_model_set_layer_sliding(const int32_t* sliding, int64_t n) {
    Model* m = g_model;
    if (!m) TE_FAIL(TE_ERR_STATE, "no model");
    if (m->finalized) TE_FAIL(TE_ERR_STATE, "model already finalized");
    if (!sliding || n != m->c.n_layers)
        TE_FAIL(TE_ERR_ARG, "layer flags length %lld != n_layers %lld",
                (long long)n, (long long)m->c.n_layers);
    m->layer_sliding.assign(sliding, sliding + n);
    return 0;
}

int te_set_fusion(int64_t enabled) {
    if (!g_model) TE_FAIL(TE_ERR_STATE, "no model");
    g_model->fused = enabled != 0;
    return 0;
}

int te_debug_set(int64_t enabled) {
    if (!g_model) TE_FAIL(TE_ERR_STATE, "no model");
    g_model->debug = enabled != 0;
    if (!enabled) g_model->dbg.clear();
    return 0;
}

int64_t te_debug_count(void) { return g_model ? (int64_t)g_model->dbg.size() : -1; }

int64_t te_debug_size(int64_t idx) {
    if (!g_model || idx < 0 || idx >= (int64_t)g_model->dbg.size()) return -1;
    return (int64_t)g_model->dbg[idx].size();
}

int te_debug_read(int64_t idx, float* out, int64_t numel) {
    if (!g_model) TE_FAIL(TE_ERR_STATE, "no model");
    if (idx < 0 || idx >= (int64_t)g_model->dbg.size())
        TE_FAIL(TE_ERR_ARG, "debug index %lld out of range", (long long)idx);
    const std::vector<float>& e = g_model->dbg[idx];
    if (numel != (int64_t)e.size())
        TE_FAIL(TE_ERR_ARG, "debug entry %lld has %lld floats, caller wants %lld",
                (long long)idx, (long long)e.size(), (long long)numel);
    memcpy(out, e.data(), numel * sizeof(float));
    return 0;
}

int te_forward_batch(int64_t n_seqs, const int32_t* tokens,
                     const int32_t* n_tokens, const int32_t* pos,
                     const int32_t* block_tables, int64_t max_blocks_per_seq,
                     float* logits_out) {
    Model* m = g_model;
    if (!m || !m->finalized) TE_FAIL(TE_ERR_STATE, "model not ready");
    if (n_seqs <= 0 || n_seqs > TE_MAX_BATCH_SEQS)
        TE_FAIL(TE_ERR_ARG, "n_seqs %lld out of range", (long long)n_seqs);
    if (!tokens || !n_tokens || !pos || !block_tables || !logits_out ||
        max_blocks_per_seq <= 0)
        TE_FAIL(TE_ERR_ARG, "bad forward args");

    const TeModelConfig& c = m->c;
    const int64_t bs = c.kv_block_size;
    const int64_t pool_stride = c.kv_num_blocks * bs * m->kv_dim;

    // Flatten per-sequence descriptors into per-token position/owner arrays
    // and validate block tables cover every touched slot.
    std::vector<int32_t> h_positions, h_seq_ids;
    std::vector<int32_t> h_last_idx(n_seqs);
    int64_t n = 0;
    for (int64_t s = 0; s < n_seqs; s++) {
        if (n_tokens[s] <= 0) TE_FAIL(TE_ERR_ARG, "seq %lld has no tokens", (long long)s);
        int64_t end = (int64_t)pos[s] + n_tokens[s];
        if (pos[s] < 0 || end > c.max_seq)
            TE_FAIL(TE_ERR_ARG, "seq %lld range [%d,%lld) exceeds max_seq",
                    (long long)s, pos[s], (long long)end);
        int64_t need_blocks = (end + bs - 1) / bs;
        if (need_blocks > max_blocks_per_seq)
            TE_FAIL(TE_ERR_ARG, "seq %lld needs %lld blocks, table has %lld",
                    (long long)s, (long long)need_blocks, (long long)max_blocks_per_seq);
        for (int64_t b = 0; b < need_blocks; b++) {
            int32_t phys = block_tables[s * max_blocks_per_seq + b];
            if (phys < 0 || phys >= c.kv_num_blocks)
                TE_FAIL(TE_ERR_ARG, "seq %lld block %lld: bad physical id %d",
                        (long long)s, (long long)b, phys);
        }
        for (int64_t t = 0; t < n_tokens[s]; t++) {
            h_positions.push_back((int32_t)(pos[s] + t));
            h_seq_ids.push_back((int32_t)s);
        }
        n += n_tokens[s];
        h_last_idx[s] = (int32_t)(n - 1);
    }
    if (n > c.max_seq)
        TE_FAIL(TE_ERR_ARG, "batch has %lld tokens, scratch holds %lld",
                (long long)n, (long long)c.max_seq);
    if (n_seqs * max_blocks_per_seq > m->tables_cap)
        TE_FAIL(TE_ERR_ARG, "block tables exceed staging capacity");

    m->dbg.clear();
    int rc;

    TE_CHECK(cudaMemcpy(m->d_tokens, tokens, n * sizeof(int32_t), cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(m->d_positions, h_positions.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(m->d_seq_ids, h_seq_ids.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(m->d_last_idx, h_last_idx.data(), n_seqs * sizeof(int32_t), cudaMemcpyHostToDevice));
    TE_CHECK(cudaMemcpy(m->d_tables, block_tables,
                        n_seqs * max_blocks_per_seq * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

    const float* w_embed;
    if ((rc = require(m, "model.embed_tokens.weight", c.vocab * c.hidden, &w_embed))) return rc;
    embed_gather_kernel<<<(int)n, kBlock>>>(m->x, w_embed, m->d_tokens, c.hidden);
    if (c.arch == 1)  // gemma scales embeddings by sqrt(hidden)
        scale_kernel<<<grid_for(n * c.hidden), kBlock>>>(m->x, n * c.hidden, sqrtf((float)c.hidden));
    if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;

    const float attn_scale =
        1.f / sqrtf(c.query_scalar > 0 ? (float)c.query_scalar : (float)c.head_dim);
    const float eps = (float)c.rms_eps;

    if (c.arch == 0) {
        // Llama family. The residual add is deferred and fused into the NEXT
        // norm: layer l's MLP output is added to x at the top of layer l+1
        // (or at the final norm). Debug captures move with it, preserving HF
        // hidden_states order.
        float* pending = nullptr;

        for (int64_t l = 0; l < c.n_layers; l++) {
            const float *w_in_ln, *w_post_ln;
            if ((rc = require(m, lname(l, "input_layernorm.weight"), c.hidden, &w_in_ln))) return rc;
            if ((rc = require(m, lname(l, "post_attention_layernorm.weight"), c.hidden, &w_post_ln))) return rc;
            bool sliding = layer_is_sliding(m, l);
            double theta = (sliding && c.rope_local_theta > 0) ? c.rope_local_theta : c.rope_theta;
            int64_t window = sliding ? c.sliding_window : 0;

            // Attention block
            if (pending) {
                add_norm(m, n, pending, w_in_ln);  // x += prev MLP out; xn = norm(x)
                if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;  // out of layer l-1
            } else {
                rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
                    m->xn, m->x, w_in_ln, c.hidden, eps, 0);
            }
            if ((rc = mm(m, lname(l, "self_attn.q_proj.weight"), m->xn, m->q, m->q_dim, c.hidden, n))) return rc;
            if ((rc = mm(m, lname(l, "self_attn.k_proj.weight"), m->xn, m->k, m->kv_dim, c.hidden, n))) return rc;
            if ((rc = mm(m, lname(l, "self_attn.v_proj.weight"), m->xn, m->v, m->kv_dim, c.hidden, n))) return rc;
            rope_kernel<<<(int)n, kBlock>>>(m->q, m->d_positions, c.n_heads, c.head_dim, theta);
            rope_kernel<<<(int)n, kBlock>>>(m->k, m->d_positions, c.n_kv_heads, c.head_dim, theta);

            float* kpool = m->kv + (l * 2 + 0) * pool_stride;
            float* vpool = m->kv + (l * 2 + 1) * pool_stride;
            kv_append_paged_kernel<<<(int)n, kBlock>>>(
                kpool, vpool, m->k, m->v, m->d_positions, m->d_seq_ids,
                m->d_tables, max_blocks_per_seq, bs, m->kv_dim);

            dim3 attn_grid((unsigned)n, (unsigned)c.n_heads);
            attn_paged_kernel<<<attn_grid, 1>>>(
                m->attn, m->q, kpool, vpool, m->d_positions, m->d_seq_ids,
                m->d_tables, max_blocks_per_seq, bs, c.n_heads, c.n_kv_heads,
                c.head_dim, m->kv_dim, attn_scale, window);

            if ((rc = mm(m, lname(l, "self_attn.o_proj.weight"), m->attn, m->proj, c.hidden, m->q_dim, n))) return rc;
            add_norm(m, n, m->proj, w_post_ln);  // x += attn proj; xn = norm(x)

            // FFN block (output deferred into the next norm via `pending`):
            // dense SwiGLU or MoE depending on the model.
            if (c.n_experts > 0) {
                if ((rc = moe_forward(m, l, n))) return rc;
            } else {
                if ((rc = mm(m, lname(l, "mlp.gate_proj.weight"), m->xn, m->ff_gate, c.intermediate, c.hidden, n))) return rc;
                if ((rc = mm(m, lname(l, "mlp.up_proj.weight"), m->xn, m->ff_up, c.intermediate, c.hidden, n))) return rc;
                silu_mul_kernel<<<grid_for(n * c.intermediate), kBlock>>>(m->ff_gate, m->ff_up, n * c.intermediate);
                if ((rc = mm(m, lname(l, "mlp.down_proj.weight"), m->ff_gate, m->mlp_out, c.hidden, c.intermediate, n))) return rc;
            }
            pending = m->mlp_out;
        }

        const float* w_norm;
        if ((rc = require(m, "model.norm.weight", c.hidden, &w_norm))) return rc;
        add_norm(m, n, pending, w_norm);  // x = last layer's output; xn = final norm
        if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;
        if ((rc = debug_capture(m, m->xn, n * c.hidden))) return rc;
    } else {
        // Gemma 3: (1+w) norms, per-head qk-norm before RoPE, sandwich norms
        // around attention and MLP outputs, GELU-tanh MLP, sliding-window
        // layers with their own rope theta. Eager residual adds (no fusion).
        for (int64_t l = 0; l < c.n_layers; l++) {
            const float *w_in_ln, *w_post_attn, *w_pre_ffw, *w_post_ffw, *w_qn, *w_kn;
            if ((rc = require(m, lname(l, "input_layernorm.weight"), c.hidden, &w_in_ln))) return rc;
            if ((rc = require(m, lname(l, "post_attention_layernorm.weight"), c.hidden, &w_post_attn))) return rc;
            if ((rc = require(m, lname(l, "pre_feedforward_layernorm.weight"), c.hidden, &w_pre_ffw))) return rc;
            if ((rc = require(m, lname(l, "post_feedforward_layernorm.weight"), c.hidden, &w_post_ffw))) return rc;
            if ((rc = require(m, lname(l, "self_attn.q_norm.weight"), c.head_dim, &w_qn))) return rc;
            if ((rc = require(m, lname(l, "self_attn.k_norm.weight"), c.head_dim, &w_kn))) return rc;
            bool sliding = layer_is_sliding(m, l);
            double theta = (sliding && c.rope_local_theta > 0) ? c.rope_local_theta : c.rope_theta;
            int64_t window = sliding ? c.sliding_window : 0;

            // Attention: norm -> qkv -> per-head qk-norm -> rope -> attend
            norm_g(n, m->xn, m->x, w_in_ln, c.hidden, eps);
            if ((rc = mm(m, lname(l, "self_attn.q_proj.weight"), m->xn, m->q, m->q_dim, c.hidden, n))) return rc;
            if ((rc = mm(m, lname(l, "self_attn.k_proj.weight"), m->xn, m->k, m->kv_dim, c.hidden, n))) return rc;
            if ((rc = mm(m, lname(l, "self_attn.v_proj.weight"), m->xn, m->v, m->kv_dim, c.hidden, n))) return rc;
            norm_g(n * c.n_heads, m->q, m->q, w_qn, c.head_dim, eps);
            norm_g(n * c.n_kv_heads, m->k, m->k, w_kn, c.head_dim, eps);
            rope_kernel<<<(int)n, kBlock>>>(m->q, m->d_positions, c.n_heads, c.head_dim, theta);
            rope_kernel<<<(int)n, kBlock>>>(m->k, m->d_positions, c.n_kv_heads, c.head_dim, theta);

            float* kpool = m->kv + (l * 2 + 0) * pool_stride;
            float* vpool = m->kv + (l * 2 + 1) * pool_stride;
            kv_append_paged_kernel<<<(int)n, kBlock>>>(
                kpool, vpool, m->k, m->v, m->d_positions, m->d_seq_ids,
                m->d_tables, max_blocks_per_seq, bs, m->kv_dim);
            dim3 attn_grid((unsigned)n, (unsigned)c.n_heads);
            attn_paged_kernel<<<attn_grid, 1>>>(
                m->attn, m->q, kpool, vpool, m->d_positions, m->d_seq_ids,
                m->d_tables, max_blocks_per_seq, bs, c.n_heads, c.n_kv_heads,
                c.head_dim, m->kv_dim, attn_scale, window);

            // x += post_attention_layernorm(o_proj(attn))
            if ((rc = mm(m, lname(l, "self_attn.o_proj.weight"), m->attn, m->proj, c.hidden, m->q_dim, n))) return rc;
            norm_g(n, m->xn, m->proj, w_post_attn, c.hidden, eps);
            add_inplace_kernel<<<grid_for(n * c.hidden), kBlock>>>(m->x, m->xn, n * c.hidden);

            // x += post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))
            norm_g(n, m->xn, m->x, w_pre_ffw, c.hidden, eps);
            if ((rc = mm(m, lname(l, "mlp.gate_proj.weight"), m->xn, m->ff_gate, c.intermediate, c.hidden, n))) return rc;
            if ((rc = mm(m, lname(l, "mlp.up_proj.weight"), m->xn, m->ff_up, c.intermediate, c.hidden, n))) return rc;
            gelu_tanh_mul_kernel<<<grid_for(n * c.intermediate), kBlock>>>(m->ff_gate, m->ff_up, n * c.intermediate);
            if ((rc = mm(m, lname(l, "mlp.down_proj.weight"), m->ff_gate, m->mlp_out, c.hidden, c.intermediate, n))) return rc;
            norm_g(n, m->proj, m->mlp_out, w_post_ffw, c.hidden, eps);
            add_inplace_kernel<<<grid_for(n * c.hidden), kBlock>>>(m->x, m->proj, n * c.hidden);

            if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;
        }

        const float* w_norm;
        if ((rc = require(m, "model.norm.weight", c.hidden, &w_norm))) return rc;
        norm_g(n, m->xn, m->x, w_norm, c.hidden, eps);
        if ((rc = debug_capture(m, m->xn, n * c.hidden))) return rc;
    }

    const float* w_lm;
    if ((rc = require(m, "lm_head.weight", c.vocab * c.hidden, &w_lm))) return rc;

    // Logits for each sequence's last token only.
    gather_rows_kernel<<<(int)n_seqs, kBlock>>>(m->d_last_hidden, m->xn,
                                                m->d_last_idx, c.hidden);
    if ((rc = gemm_rowmajor(m->cublas, w_lm, m->d_last_hidden, m->d_logits,
                            c.vocab, n_seqs, c.hidden))) return rc;

    TE_CHECK(cudaGetLastError());
    TE_CHECK(cudaMemcpy(logits_out, m->d_logits,
                        n_seqs * c.vocab * sizeof(float),
                        cudaMemcpyDeviceToHost));
    return 0;
}

}  // extern "C"
