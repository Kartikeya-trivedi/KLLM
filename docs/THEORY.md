# Theory — the concepts behind the engine

This is the "why" behind each part of kllm: what problem each phase solves
and the reasoning that makes the solution the right one. It assumes you know
what a transformer is but not how serving one efficiently works. Every
concept here maps to code and to a phase in [PLAN.md](PLAN.md); the measured
results are in [JOURNAL.md](JOURNAL.md).

---

## 1. Autoregressive decoding, and why it has two phases

A decoder LLM generates one token at a time. To produce token *t+1* it runs
a forward pass over the sequence so far and takes the highest-scoring token
from the output distribution (greedy sampling; temperature/top-p are
variations). That token is appended and the process repeats. This is
**autoregression** — each output depends on all previous ones, so generation
is inherently sequential.

Serving splits naturally into two phases with very different performance
characteristics:

- **Prefill**: process the whole prompt at once. Many tokens go through the
  network in parallel — this is *compute-bound* (big matmuls, the GPU is
  busy). It produces the first output token; its latency is **TTFT** (time
  to first token).
- **Decode**: generate the rest, one token per forward pass. Each step
  processes a single token — this is *memory-bound* (tiny matmuls, the GPU
  mostly waits on weight reads). The per-token latency is **ITL**
  (inter-token latency); its inverse is tokens/second.

Almost all of the interesting engineering is about making the memory-bound
decode phase fast, because that's where most of the wall-clock goes for
long generations.

*In kllm:* `Prefill` runs the prompt through `te_forward_batch` with many
tokens; each decode step calls it with one. `cmd/bench` measures TTFT and
ITL separately for exactly this reason.

---

## 2. The KV cache — the central data structure of inference

Attention at position *t* needs the **keys and values** of every earlier
position. Naively, each new token would recompute K and V for the entire
sequence — O(t) work that grows every step. Instead we compute each token's
K and V once and **cache** them; each new token computes only its own Q, K,
V and attends over the cached K/V. This turns per-step attention cost from
"recompute everything" into "compute one token, read the cache."

The cache is the reason decode is memory-bound and the reason long contexts
are expensive: its size is

```
2 (K and V) × n_layers × n_kv_heads × head_dim × seq_len × dtype_bytes
```

per sequence. For a 30B-class model at a few thousand tokens this is
gigabytes — often more than the weights of the active experts. Managing this
memory well is the difference between serving 4 sequences and 40.

*In kllm:* K/V are written once per token by the append kernel and read by
the attention kernel every step. Phase 0 used a simple contiguous cache;
Phase 2 replaced it with paging (below).

---

## 3. Paged KV cache — memory that doesn't fragment

The obvious KV layout is one contiguous buffer per sequence, sized to the
maximum context length. This wastes enormous memory: a sequence that only
uses 50 tokens still reserves space for thousands, and you can't pack the
leftovers. It's the same problem an operating system has with contiguous
process memory — and the same fix applies: **paging** (this is the idea
PagedAttention/vLLM introduced to inference).

Carve GPU memory into fixed-size **blocks** (e.g. 16 tokens of KV each). Each
sequence gets a **block table** — a list of physical block ids in logical
order, exactly like a page table. A sequence grows by grabbing another block
from a shared free list; when it finishes it returns all its blocks. Memory
waste is bounded by at most one partly-filled block per sequence, instead of
"max length minus actual length" per sequence.

The attention and KV-append kernels take the block table and an indirection:
logical position *p* lives in physical block `table[p / block_size]` at
offset `p % block_size`. One shared pool serves however many sequences
actually fit.

*In kllm:* the block pool is one big `cudaMalloc` in the backend; the
allocator, free list, and per-sequence block tables live in Go
([`engine/kv`](../engine/kv/kv.go)) — **Go decides placement, CUDA never
allocates per sequence.** This clean split is exactly what the narrow ABI
was for. See ARCHITECTURE.md for the division of labor.

---

## 4. Continuous batching — the throughput multiplier

Because decode is memory-bound, a single sequence leaves the GPU almost
idle: you pay to read the weights from memory and then use them for one
token's worth of math. If you run *B* sequences together, you read the same
weights **once** and reuse them for *B* tokens — throughput scales with batch
size at almost no extra latency, until you hit a compute or memory limit.

But requests don't arrive together and don't finish together. **Static**
batching (wait for a full batch, run it to completion) wastes the GPU: short
requests in the batch finish early and their slots sit idle until the whole
batch is done, and late arrivals wait for the next batch.

**Continuous (in-flight) batching** fixes this by re-forming the batch every
single decode step. A finished sequence leaves immediately and frees its
slot; a newly arrived request joins on the next step. The batch is a
constantly-churning set, not a fixed group. Prefill of a new request and
decode of running requests can even ride in the same step.

The scheduler must budget **two** resources per step, not one:
- **batch slots** (how many sequences the kernels handle at once), and
- **tokens per step** (prefills contribute many tokens; the backend's
  scratch is finite).

Getting only the first is a classic bug — a burst of long prompts overflows
the token budget. (kllm hit exactly this; see the Phase 3 note in the
journal.)

*In kllm:* one scheduler goroutine owns the GPU; requests arrive on a
channel, get admitted under both budgets, and stream tokens back on
per-request channels ([`engine/scheduler.go`](../engine/scheduler.go)).
Measured 20× throughput from batching before the curve flattens at the batch
cap.

---

## 5. GEMM and the roofline — knowing what you're bound by

Nearly all the compute in a transformer is **GEMM** (general matrix
multiply): the Q/K/V/O projections and the MLP are all matmuls. Don't
hand-write these — vendor libraries (cuBLAS) and tensor-core template
libraries (CUTLASS) are at or near the hardware limit. The engine's job is
to *feed* them well and to write the *non*-GEMM kernels (norm, RoPE,
activation, attention, dequant).

Whether an operation can go faster is decided by the **roofline model**. Every
kernel moves some bytes and does some FLOPs; its **arithmetic intensity** is
FLOPs ÷ bytes. Plot achievable performance against intensity and you get two
regimes:

- Low intensity → **memory-bound**: performance is capped by bandwidth
  (bytes/sec). Doing less math doesn't help; moving fewer bytes does.
- High intensity → **compute-bound**: capped by FLOP throughput. Moving
  fewer bytes doesn't help; doing less math (or using tensor cores) does.

Decode-time matmuls have a batch dimension of 1, so they read a big weight
matrix to do a tiny amount of math — deeply memory-bound. This single fact
drives the two biggest optimizations below: **quantization** (move fewer
weight bytes) and **fusion + CUDA graphs** (remove per-launch overhead that
dominates when each kernel is tiny).

*In kllm:* profile before optimizing. The Phase 1 note measured that on the
tiny model, decode is *launch-overhead-bound* (weights are trivially small),
so the win there is fusion. The Phase 4 note measured that at real
30B-weight shapes, the same matmul is *bandwidth-bound*, so the win there is
quantization. Same operation, opposite bottleneck — the roofline tells you
which.

---

## 6. Kernel fusion — stop paying for memory round-trips

A GPU kernel launch reads its inputs from global memory and writes its
outputs back. If you run "add the residual" and then "RMSNorm" as two
kernels, the intermediate is written to memory and immediately read back —
wasted bandwidth, plus two launch overheads. **Fusing** them into one kernel
computes the whole thing in registers/shared memory and touches global
memory once. For memory-bound elementwise ops this is close to a free win,
and it also removes a kernel launch (which matters more than you'd think when
each kernel is tiny and launches are ~microseconds each, especially on
Windows' WDDM driver model).

*In kllm:* `add_rmsnorm_kernel` fuses the residual add into the following
norm, and the forward pass defers each block's residual add into the next
norm so the fusion applies everywhere. A runtime toggle keeps an unfused path
so the speedup can be measured honestly — and the correctness suite validates
*both* paths against the oracle.

The endgame version of this idea is **CUDA graphs**: capture the entire
decode step (dozens of kernel launches) once and replay it as a single unit,
paying per-launch and per-FFI-call overhead one time instead of every step.

---

## 7. Weight quantization (W4) — move fewer bytes

Since decode is bandwidth-bound on weight reads, the highest-leverage
optimization is to **make the weights smaller**. Store each weight in 4 bits
instead of 16, and you read ~4× fewer bytes per token — up to ~4× faster
decode, if the kernel is written well.

The scheme kllm uses is **group-wise symmetric int4**. For a weight matrix,
split each row into groups of *G* consecutive values (e.g. 128). Per group,
find `scale = max|w| / 7` and store each weight as a 4-bit integer
`q = round(w / scale)` in `[-8, 7]`. Two nibbles pack into one byte. At
compute time the kernel **dequantizes on the fly** — `w ≈ q × scale` — and
multiplies, fused into the matmul so the fp16/fp32 weights never exist in
memory. Group-wise scales (rather than one scale per matrix) keep accuracy
high because they adapt to local weight magnitude; this is the family AWQ /
GPTQ / compressed-tensors belong to.

The subtlety is **validating** it. "Close to the fp32 output" conflates two
different things: kernel bugs and quantization error. kllm separates them:
the quantizer emits a *second* checkpoint with the weights **dequantized
back to fp32**, and the oracle runs HF on *that*. If the W4 engine matches
those dumps to floating-point tolerance, the kernel provably computes exactly
`dequant(q, scale) · x` — any remaining difference from the original model is
quantization error, which is a separate, honest measurement.

*In kllm:* `tools/quantize_w4.py` produces both checkpoints; the backend's
`matmul_w4_kernel` does dequant-fused matmul; `e2ew4` proves exactness. The
naive kernel currently *loses* to cuBLAS because it reads the packed weights
uncoalesced and re-reads the input per thread — that gap (measured in
wbench) is precisely the deferred kernel-optimization work, now with a clear
target.

---

## 8. Mixture of Experts (MoE) — conditional compute

A dense FFN runs every token through the same big weight matrices. **MoE**
replaces the single FFN with *E* smaller "expert" FFNs plus a **router**, and
sends each token to only its top *k* experts (e.g. 2 of many). The model has
a large *total* parameter count but only activates a small *fraction* per
token — you get the capacity of a big model at the compute cost of a small
one. This is why modern 30B-class models are frequently MoE (Mixtral,
Qwen3-MoE, DeepSeek, Sarvam).

There are two routing families, and a general engine must handle both:

- **Softmax top-k, renormalized** (Mixtral, Qwen): softmax over all expert
  logits, take the top *k*, renormalize those *k* weights to sum to 1.
- **Sigmoid + expert-bias selection** (DeepSeek-V3, Sarvam): sigmoid each
  logit into an independent score; **select** the top *k* by score *plus a
  per-expert bias* (a load-balancing term); **weight** them using the
  *unbiased* scores normalized over the selection. Selecting and weighting
  use different quantities — an easy detail to get wrong.

Mechanically, MoE turns one matmul into a routing problem: which tokens go to
which experts, then a matmul per expert over a *variable* number of tokens.
The standard dataflow is **permute → grouped-GEMM → un-permute**: sort tokens
by assigned expert so each expert's tokens are contiguous, run one GEMM per
expert-segment (a "grouped GEMM"), then scatter results back to the original
token order, weighted by the routing weights. Because multiple experts write
to the same token, the un-permute uses atomic adds.

*In kllm:* one implementation handles both families via a `router_mode`
switch. The routing kernel, the expert-sorted permutation, per-expert segment
GEMMs, and the weighted scatter-add are all in `model.cu`. Because expert
GEMMs go through the same matmul dispatch as dense weights, **INT4 experts
came for free** — quantize the expert FFNs, keep the router in fp32. This is
the Sarvam-30B-INT4 serving path in miniature. The softmax family is
validated against HF Mixtral; the sigmoid family against a standalone numpy
reference (HF has no matching class), both to the same tolerances.

---

## 9. Speculative decoding (planned) — break the sequential bottleneck

Decode is sequential and memory-bound: one token per expensive weight read.
**Speculative decoding** uses a small, fast *draft* model to propose several
tokens ahead, then the big *target* model **verifies all of them in a single
forward pass** (verification is parallel over the proposed tokens, like
prefill). Accepted tokens are kept; on the first rejection you fall back to
the target's own token. When the draft is often right, you get several tokens
per target forward pass — a real speedup with *no* change to the output
distribution (the accept/reject rule is exact).

The orchestration — running draft and target, comparing distributions,
managing the accept/reject loop and the KV rollback on rejection — is
control-flow-heavy and lives naturally in Go; only the verify forward pass is
a kernel. This is the stretch phase; it's listed here to close the loop on
"how do you make sequential generation less sequential."

---

## Where it all points

The through-line: **inference is a memory-movement problem wearing a
compute costume.** The KV cache, paged memory, continuous batching,
quantization, fusion, and CUDA graphs are all, at bottom, about moving fewer
bytes or moving them less often. Get the correctness spine and the memory
model right first (Phases 0–3), then the kernels are where the remaining
speed lives (Phases 4–6) — which is exactly the order this project builds in,
and why kernel optimization is the deliberate endgame rather than the
starting point.
