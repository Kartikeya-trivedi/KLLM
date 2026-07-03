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

struct Model {
    TeModelConfig c{};
    int64_t q_dim = 0;   // n_heads * head_dim
    int64_t kv_dim = 0;  // n_kv_heads * head_dim
    bool finalized = false;

    std::unordered_map<std::string, DeviceTensor> w;

    // Contiguous KV cache: [n_layers][2 (k,v)][max_seq][kv_dim]
    float* kv = nullptr;
    int64_t kv_len = 0;  // tokens currently cached

    // Scratch, sized for max_seq tokens in flight.
    float *x = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr;
    float *attn = nullptr, *proj = nullptr, *mlp_out = nullptr;
    float *ff_gate = nullptr, *ff_up = nullptr;
    float* d_logits = nullptr;
    int32_t* d_tokens = nullptr;

    cublasHandle_t cublas = nullptr;
    bool fused = true;  // te_set_fusion

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

// One block per token; blockDim must be a power of two.
__global__ void rmsnorm_kernel(float* out, const float* in, const float* w,
                               int64_t hidden, float eps) {
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
    for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x)
        orow[i] = row[i] * scale * w[i];
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
__global__ void rope_kernel(float* t, int64_t n_heads, int64_t head_dim,
                            int64_t pos0, double theta) {
    int64_t token = blockIdx.x;
    int64_t half = head_dim / 2;
    for (int64_t idx = threadIdx.x; idx < n_heads * half; idx += blockDim.x) {
        int64_t h = idx / half, j = idx % half;
        double freq = pow(theta, -2.0 * (double)j / (double)head_dim);
        double ang = (double)(pos0 + token) * freq;
        float c = (float)cos(ang), s = (float)sin(ang);
        float* base = t + token * (n_heads * head_dim) + h * head_dim;
        float a = base[j], b = base[j + half];
        base[j] = a * c - b * s;
        base[j + half] = b * c + a * s;
    }
}

__global__ void kv_append_kernel(float* kcache, float* vcache, const float* k,
                                 const float* v, int64_t pos0, int64_t kv_dim) {
    int64_t t = blockIdx.x;
    for (int64_t i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        kcache[(pos0 + t) * kv_dim + i] = k[t * kv_dim + i];
        vcache[(pos0 + t) * kv_dim + i] = v[t * kv_dim + i];
    }
}

// Deliberately naive causal attention: one thread per (query token, head).
// Fine at lab scale; replaced by a paged, parallel kernel later.
#define TE_ATTN_MAX_SEQ 4096
#define TE_ATTN_MAX_HEAD_DIM 256

__global__ void attn_naive_kernel(float* out, const float* q,
                                  const float* kcache, const float* vcache,
                                  int64_t pos0, int64_t n_heads, int64_t n_kv,
                                  int64_t head_dim, int64_t kv_dim,
                                  float scale) {
    int64_t t = blockIdx.x, h = blockIdx.y;
    int64_t q_dim = n_heads * head_dim;
    int64_t kvh = h / (n_heads / n_kv);
    const float* qv = q + t * q_dim + h * head_dim;
    int64_t len = pos0 + t + 1;  // causal: attend to positions [0, len)

    float sc[TE_ATTN_MAX_SEQ];
    float maxs = -1e30f;
    for (int64_t p = 0; p < len; p++) {
        const float* kv_row = kcache + p * kv_dim + kvh * head_dim;
        float dot = 0.f;
        for (int64_t d = 0; d < head_dim; d++) dot += qv[d] * kv_row[d];
        sc[p] = dot * scale;
        if (sc[p] > maxs) maxs = sc[p];
    }
    float sum = 0.f;
    for (int64_t p = 0; p < len; p++) {
        sc[p] = expf(sc[p] - maxs);
        sum += sc[p];
    }
    float acc[TE_ATTN_MAX_HEAD_DIM];
    for (int64_t d = 0; d < head_dim; d++) acc[d] = 0.f;
    for (int64_t p = 0; p < len; p++) {
        const float* v_row = vcache + p * kv_dim + kvh * head_dim;
        float wgt = sc[p] / sum;
        for (int64_t d = 0; d < head_dim; d++) acc[d] += wgt * v_row[d];
    }
    float* orow = out + t * q_dim + h * head_dim;
    for (int64_t d = 0; d < head_dim; d++) orow[d] = acc[d];
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
// the te_set_fusion toggle — identical math either way.
void add_norm(Model* m, int64_t n, float* res, const float* w) {
    const TeModelConfig& c = m->c;
    if (m->fused) {
        add_rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
            m->x, res, m->xn, w, c.hidden, (float)c.rms_eps);
    } else {
        add_inplace_kernel<<<grid_for(n * c.hidden), kBlock>>>(m->x, res, n * c.hidden);
        rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
            m->xn, m->x, w, c.hidden, (float)c.rms_eps);
    }
}

}  // namespace

// ---- C-ABI ------------------------------------------------------------------

extern "C" {

int te_model_create(const TeModelConfig* cfg) {
    if (g_model) TE_FAIL(TE_ERR_STATE, "model already created");
    if (!cfg) TE_FAIL(TE_ERR_ARG, "null config");
    if (cfg->n_experts != 0)
        TE_FAIL(TE_ERR_ARG, "MoE not implemented yet (n_experts=%lld)",
                (long long)cfg->n_experts);
    if (cfg->hidden <= 0 || cfg->n_layers <= 0 || cfg->n_heads <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->head_dim <= 0 || cfg->vocab <= 0 ||
        cfg->max_seq <= 0 || cfg->intermediate <= 0)
        TE_FAIL(TE_ERR_ARG, "non-positive config field");
    if (cfg->n_heads % cfg->n_kv_heads != 0)
        TE_FAIL(TE_ERR_ARG, "n_heads %% n_kv_heads != 0");
    if (cfg->max_seq > TE_ATTN_MAX_SEQ || cfg->head_dim > TE_ATTN_MAX_HEAD_DIM)
        TE_FAIL(TE_ERR_ARG, "max_seq/head_dim exceed naive-attention limits");

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
        if ((rc = require(m, lname(l, "self_attn.q_proj.weight"), m->q_dim * c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "self_attn.k_proj.weight"), m->kv_dim * c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "self_attn.v_proj.weight"), m->kv_dim * c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "self_attn.o_proj.weight"), c.hidden * m->q_dim, nullptr))) return rc;
        if ((rc = require(m, lname(l, "mlp.gate_proj.weight"), c.intermediate * c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "mlp.up_proj.weight"), c.intermediate * c.hidden, nullptr))) return rc;
        if ((rc = require(m, lname(l, "mlp.down_proj.weight"), c.hidden * c.intermediate, nullptr))) return rc;
    }

    TE_CHECK(cudaMalloc(&m->kv, c.n_layers * 2 * c.max_seq * m->kv_dim * sizeof(float)));
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
    TE_CHECK(cudaMalloc(&m->d_logits, c.vocab * sizeof(float)));
    TE_CHECK(cudaMalloc(&m->d_tokens, c.max_seq * sizeof(int32_t)));
    TE_CHECK_CUBLAS(cublasCreate(&m->cublas));

    m->finalized = true;
    m->kv_len = 0;
    return 0;
}

int te_reset_kv(void) {
    if (!g_model) TE_FAIL(TE_ERR_STATE, "no model");
    g_model->kv_len = 0;
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

int te_forward(const int32_t* tokens, int64_t n, int64_t pos,
               float* logits_out) {
    Model* m = g_model;
    if (!m || !m->finalized) TE_FAIL(TE_ERR_STATE, "model not ready");
    if (!tokens || n <= 0 || !logits_out) TE_FAIL(TE_ERR_ARG, "bad forward args");
    if (pos != m->kv_len)
        TE_FAIL(TE_ERR_ARG, "pos %lld != cached length %lld (missed reset?)",
                (long long)pos, (long long)m->kv_len);
    if (pos + n > m->c.max_seq)
        TE_FAIL(TE_ERR_ARG, "pos+n %lld exceeds max_seq %lld",
                (long long)(pos + n), (long long)m->c.max_seq);

    const TeModelConfig& c = m->c;
    const int64_t layer_stride = 2 * c.max_seq * m->kv_dim;
    m->dbg.clear();
    int rc;

    TE_CHECK(cudaMemcpy(m->d_tokens, tokens, n * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

    const float* w_embed;
    if ((rc = require(m, "model.embed_tokens.weight", c.vocab * c.hidden, &w_embed))) return rc;
    embed_gather_kernel<<<(int)n, kBlock>>>(m->x, w_embed, m->d_tokens, c.hidden);
    if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;

    // The residual add is deferred and fused into the NEXT norm: layer l's
    // MLP output is added to x at the top of layer l+1 (or at the final
    // norm). Debug captures move with it, preserving HF hidden_states order.
    float* pending = nullptr;

    for (int64_t l = 0; l < c.n_layers; l++) {
        const float *w_in_ln, *w_post_ln, *w_q, *w_k, *w_v, *w_o, *w_gate, *w_up, *w_down;
        if ((rc = require(m, lname(l, "input_layernorm.weight"), c.hidden, &w_in_ln))) return rc;
        if ((rc = require(m, lname(l, "post_attention_layernorm.weight"), c.hidden, &w_post_ln))) return rc;
        if ((rc = require(m, lname(l, "self_attn.q_proj.weight"), m->q_dim * c.hidden, &w_q))) return rc;
        if ((rc = require(m, lname(l, "self_attn.k_proj.weight"), m->kv_dim * c.hidden, &w_k))) return rc;
        if ((rc = require(m, lname(l, "self_attn.v_proj.weight"), m->kv_dim * c.hidden, &w_v))) return rc;
        if ((rc = require(m, lname(l, "self_attn.o_proj.weight"), c.hidden * m->q_dim, &w_o))) return rc;
        if ((rc = require(m, lname(l, "mlp.gate_proj.weight"), c.intermediate * c.hidden, &w_gate))) return rc;
        if ((rc = require(m, lname(l, "mlp.up_proj.weight"), c.intermediate * c.hidden, &w_up))) return rc;
        if ((rc = require(m, lname(l, "mlp.down_proj.weight"), c.hidden * c.intermediate, &w_down))) return rc;

        // Attention block
        if (pending) {
            add_norm(m, n, pending, w_in_ln);  // x += prev MLP out; xn = norm(x)
            if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;  // out of layer l-1
        } else {
            rmsnorm_kernel<<<(int)n, kBlock, kBlock * sizeof(float)>>>(
                m->xn, m->x, w_in_ln, c.hidden, (float)c.rms_eps);
        }
        if ((rc = gemm_rowmajor(m->cublas, w_q, m->xn, m->q, m->q_dim, n, c.hidden))) return rc;
        if ((rc = gemm_rowmajor(m->cublas, w_k, m->xn, m->k, m->kv_dim, n, c.hidden))) return rc;
        if ((rc = gemm_rowmajor(m->cublas, w_v, m->xn, m->v, m->kv_dim, n, c.hidden))) return rc;
        rope_kernel<<<(int)n, kBlock>>>(m->q, c.n_heads, c.head_dim, pos, c.rope_theta);
        rope_kernel<<<(int)n, kBlock>>>(m->k, c.n_kv_heads, c.head_dim, pos, c.rope_theta);

        float* kcache = m->kv + l * layer_stride;
        float* vcache = kcache + c.max_seq * m->kv_dim;
        kv_append_kernel<<<(int)n, kBlock>>>(kcache, vcache, m->k, m->v, pos, m->kv_dim);

        dim3 attn_grid((unsigned)n, (unsigned)c.n_heads);
        attn_naive_kernel<<<attn_grid, 1>>>(
            m->attn, m->q, kcache, vcache, pos, c.n_heads, c.n_kv_heads,
            c.head_dim, m->kv_dim, 1.f / sqrtf((float)c.head_dim));

        if ((rc = gemm_rowmajor(m->cublas, w_o, m->attn, m->proj, c.hidden, n, m->q_dim))) return rc;
        add_norm(m, n, m->proj, w_post_ln);  // x += attn proj; xn = norm(x)

        // MLP block (output deferred into the next norm via `pending`)
        if ((rc = gemm_rowmajor(m->cublas, w_gate, m->xn, m->ff_gate, c.intermediate, n, c.hidden))) return rc;
        if ((rc = gemm_rowmajor(m->cublas, w_up, m->xn, m->ff_up, c.intermediate, n, c.hidden))) return rc;
        silu_mul_kernel<<<grid_for(n * c.intermediate), kBlock>>>(m->ff_gate, m->ff_up, n * c.intermediate);
        if ((rc = gemm_rowmajor(m->cublas, w_down, m->ff_gate, m->mlp_out, c.hidden, n, c.intermediate))) return rc;
        pending = m->mlp_out;
    }

    const float *w_norm, *w_lm;
    if ((rc = require(m, "model.norm.weight", c.hidden, &w_norm))) return rc;
    if ((rc = require(m, "lm_head.weight", c.vocab * c.hidden, &w_lm))) return rc;
    add_norm(m, n, pending, w_norm);  // x = last layer's output; xn = final norm
    if ((rc = debug_capture(m, m->x, n * c.hidden))) return rc;
    if ((rc = debug_capture(m, m->xn, n * c.hidden))) return rc;

    // Logits for the last token only.
    if ((rc = gemm_rowmajor(m->cublas, w_lm, m->xn + (n - 1) * c.hidden,
                            m->d_logits, c.vocab, 1, c.hidden))) return rc;

    TE_CHECK(cudaGetLastError());
    TE_CHECK(cudaMemcpy(logits_out, m->d_logits, c.vocab * sizeof(float),
                        cudaMemcpyDeviceToHost));
    m->kv_len = pos + n;
    return 0;
}

}  // extern "C"
