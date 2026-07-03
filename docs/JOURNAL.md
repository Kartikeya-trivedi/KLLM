# Build Journal

Running log of the end-to-end build (Phases 0-5 of [PLAN.md](PLAN.md)),
executed on the GTX 1650 lab box (sm_75, 4 GiB, 16 SMs, no tensor cores).
Strategy: every phase's engine code and kernels are built and
correctness-validated here against HF oracles on tiny test models; moving to
the 3x A6000 box and 30B checkpoints is then a redeploy + requantize, not new
code. Kernels are written naive-but-correct — **kernel optimization is the
deliberately-deferred final focus** once the whole system is green.

## Environment

- Windows 11, Go 1.26.2, CUDA 12.8 (nvcc + MSVC 2022 as host compiler),
  Python 3.12, torch 2.5.1+cu121, transformers 4.48.1, numpy 2.2.6
- No cgo: the local MinGW is 32-bit, unusable for windows/amd64 cgo. The Go
  side crosses the C-ABI via `syscall.LoadDLL` — validated the plan's
  "narrow boundary" decision immediately. Linux/A6000 gets a cgo loader
  behind the same `engine/backend` interface.
- Triton note: Triton has no official Windows support, so the
  "Triton-prototype-then-port" step is skipped on this box; kernels are
  written directly in CUDA and validated against the HF dumps (the second,
  stronger stage of the two-stage validation).

## Cloud GPU lab (Modal) — DONE

Modal (modal.com) is authenticated on this box (profile
`kartikeyatrivedi4oct2004`). Verified end-to-end: `modal run
tools/modal_lab.py::gpu_smoke` → **A10G, 23 GB, compute_cap 8.6 = sm_86, the
exact Ampere arch of the target A6000 box**. A CUDA 12.8-devel + Go 1.26
image is built and cached in the workspace; `build_and_test` ships the repo,
builds `libtoyengine.so -arch=sm_86`, builds the Go engine with the Linux
cgo loader, and runs the suite on the GPU. Consequence for the plan: the
"A6000 box" steps (tensor-core arch validation, 30B checkpoints, Linux cgo
path) are unblocked via Modal — used as per-phase gates, not inner-loop
iteration (GPU-seconds cost money; the 1650 stays the inner loop).
Windows gotcha: set `PYTHONIOENCODING=utf-8` or the Modal CLI crashes
rendering ✓ glyphs ('charmap' codec error).

## Walking skeleton (pre-Phase 0) — DONE

Go → nvcc-built DLL → CUDA context → vector-add kernel → verified in Go
(`cmd/smoke`, max abs error 0 over 1M floats). C-ABI error model: every
function returns a cudaError_t-style code; `te_last_error` copies a
per-thread message into a caller buffer (no C-string lifetime games).

## Phase 0 — Correctness spine — DONE

**Result: green on the first full run.** `go test ./engine/e2e/`: all 5
prompts x 16 greedy steps produce exactly HF's tokens (80/80), per-step
logits max abs diff < 1e-3, per-layer activations max abs diff < 1e-4.
`go run ./cmd/generate --ids "1 17 42 100 7"` reproduces the HF continuation
token-for-token. What made it work first try (worth remembering — these are
the classic divergence sources, each pinned down before running):

1. **cuBLAS layout trick derived once, used everywhere.** HF weights are
   `[out, in]` row-major; a row-major view of column-major cuBLAS gives
   `Y[n,out] = sgemm(OP_T, OP_N, m=out, n=ntok, k=in, W lda=in, X ldb=in,
   Y ldc=out)`.
2. **HF RoPE convention exactly**: rotate_half pairs `(j, j+d/2)` sharing
   `inv_freq[j] = theta^(-2j/d)`, angles computed in double before casting.
3. **HF `hidden_states` layout**: `[0]` = embedding, `[j]` = *input* to
   layer j, last = final-norm output (not the raw last-layer residual). The
   debug-tap mapping in the test accounts for this off-by-one trap.
4. **Strict shim validation**: `te_model_finalize` checks every expected
   weight name and element count before anything runs; `te_forward` rejects
   `pos != kv_len` (catches missed KV resets at the boundary instead of as
   silent garbage).

The correctness harness is now the regression net for every later phase:
any kernel change that breaks numerics fails `engine/e2e` immediately.

## Phase 1 — First fused kernel + metrics + roofline — DONE

Built `add_rmsnorm_kernel` (residual add fused into the next RMSNorm) with a
runtime toggle (`te_set_fusion`) so fused/unfused A/B runs in one binary, and
`cmd/bench` for TTFT/ITL/tok-s. The forward loop now defers each MLP
residual add into the next norm (`pending` pointer), halving norm-site
kernel launches; debug captures moved with it so the HF per-layer diff still
passes bit-for-bit semantics. **e2e suite validates both paths** (fused=true
and fused=false subtests) — the fusion is proven equal, not assumed.

Measured (GTX 1650, tiny 2-layer model, prompt 32, 256 decode steps,
interleaved best-of-8, aggregate-timed):

| config  | ITL avg | decode tok/s |
|---------|---------|--------------|
| unfused | ~570 µs | ~1750        |
| fused   | ~545 µs | ~1830        |

**Roofline position (the actual lesson):** the tiny model moves ~0.56 MB of
weights per decode token; at the 1650's ~192 GB/s that is ~3 µs of traffic —
under 1% of the ~550 µs measured ITL. This workload is **launch/overhead
bound, nowhere near the bandwidth roof**: ~30 kernel+GEMM launches per step
at ~10-20 µs each (Windows WDDM submission makes launches especially
expensive) accounts for essentially the whole ITL. Fusion removed 5
launches/step → the observed ~3-5%. Consequences recorded for later phases:
(a) at 30B/48-layer scale, decode becomes truly bandwidth-bound (~15 GB of
W4 weights/token ≈ 20 ms at A6000's 768 GB/s) and fusion/quantization pay
proportionally; (b) the real per-launch fix is CUDA graphs (planned at the
boundary since day one); (c) per-sample Windows timing quantizes at ~0.5 ms
— bench measures in aggregate and divides.

Tooling note: Nsight Compute (`ncu` 2025.1) is installed on this box —
reserved for the kernel-optimization endgame; Nsight Systems is not.

Also fixed en route: shim return codes are C `int` (32-bit) — the Go side
must truncate `uintptr` to `int32` before sign-reading or negative TE_ERR_*
codes print as huge unsigned values.

## Phase 2 — Paged KV — DONE

Clean break at the boundary: single-sequence `te_forward` (backend-tracked
position) replaced by a **stateless `te_forward_batch`** — per-sequence
token spans, absolute positions, and block tables all supplied by Go. The
backend holds only the physical pool
(`[layers][2][num_blocks][block_size][kv_dim]`, one cudaMalloc); Go's
`engine/kv` owns the free list and per-sequence tables (LIFO allocator,
Reserve/Commit/Release). Attention and KV-append gather through the block
table; RoPE takes a per-token position array (mixed prefill+decode batches
have non-uniform positions).

Division of labor that fell out nicely: **Go decides placement, CUDA never
allocates per-sequence**; the backend validates every table slot it will
touch before launching (bad physical ids fail fast at the boundary instead
of corrupting the pool).

Correctness gates:
- `TestMatchesHFReference` re-passes through the paged path (both fusion
  modes) — same 5x16 tokens, same tolerances.
- New `TestPagedBatchMatchesSolo`: two prompts of different lengths run
  jointly — one batched forward per step, shared pool, sequences retiring
  as they finish (a small preview of continuous batching). Both streams
  match HF token-for-token.

Capacity math (the point of paging): contiguous KV needed
`n_seqs x max_seq x layers x 2 x kv_dim` up front — at lab scale
(max_seq 256) that is 1 MiB per sequence slot regardless of use; paged, a
5-token prompt holds one 16-token block per layer-pair (~4 KiB) and the same
pool serves however many sequences actually fit. At 30B scale (48 layers,
kv_dim 1024, 4K context) contiguous is ~1.6 GiB per *slot* while paged
fragmentation waste is bounded by block_size-1 tokens per sequence.

## Phase 3 — Scheduler + continuous batching + HTTP/SSE — DONE

Go's payoff phase. One batching goroutine owns the GPU; requests arrive on a
channel, get admitted into free batch slots, and every step's batch is
formed fresh: prompts (prefill) for new sequences, one token for running
ones, sequences retiring at EOS/max-tokens and freeing their KV blocks
immediately. Two admission constraints enforced per step: batch slots
(`max-batch`) and the backend's **per-step token budget** (scratch size) —
prefills that don't fit wait; decodes never starve (1 token each, budget ≥
batch size). `server` wraps it in `POST /v1/generate` (SSE token stream) +
`/healthz`; `cmd/loadgen` drives it with mixed-length concurrent requests.

Found by the load generator, fixed, lesson recorded: the first scheduler
capped *sequences* per batch but not *tokens* — 32 concurrent prefills
overflowed the 256-token scratch. Continuous batching must budget both
dimensions (vLLM's `max_num_batched_tokens` exists for exactly this reason).
Also: Windows reserves port 8080 (WinNAT excluded range) — bind errors, not
firewall prompts; the server defaults are fine, the demo runs on 8177.

Correctness gate: `TestSchedulerMatchesHF` — all 5 reference prompts
submitted concurrently through the scheduler (max batch 4 < 5, so queueing +
mixed prefill/decode batches are exercised) reproduce HF's tokens exactly.

Throughput (tiny model, GTX 1650, 64 requests/level, 32 new tokens each,
prompts 4-48 tokens, HTTP end-to-end):

| conc | req/s | tok/s  | TTFT p50 | TTFT p99 | total p99 |
|------|-------|--------|----------|----------|-----------|
| 1    | 36    | 1,106  | 2.4 ms   | 7.0 ms   | 59 ms     |
| 4    | 178   | 5,518  | 2.2 ms   | 2.9 ms   | 28 ms     |
| 16   | 540   | 16,978 | 2.7 ms   | 3.8 ms   | 35 ms     |
| 32   | 741   | 22,833 | 5.1 ms   | 11.3 ms  | 43 ms     |
| 64   | 742   | 23,147 | 18.2 ms  | 51.7 ms  | 85 ms     |

20x throughput from batching before the curve flattens at the max-batch cap
(32) — beyond it only queueing latency grows. Exactly the shape continuous
batching should produce: the GPU step cost is near-flat in batch size at
this scale (launch-bound), so batching is almost free throughput.

## Phase 4 — W4 group-quant weights + dequant-fused matmul — DONE (correctness; speed deferred by design)

Pipeline: `tools/quantize_w4.py` (numpy) group-quantizes all 14 projection
matrices (symmetric int4, group 32 here / 128 at scale, packed nibbles +
fp32 scales as `<base>.qweight`/`<base>.scales`); embeddings, lm_head, norms
stay fp32. The Go loader recognizes the pair and ships it across a new
`te_model_load_tensor_w4`; the forward pass dispatches per weight — W4
dequant-fused kernel if quantized, cuBLAS if dense (mixed checkpoints just
work). Triton prototyping was skipped deliberately: Triton has no Windows
support, and the two-sided oracle below is a stronger validator anyway.

**Validation trick worth keeping: the quantizer emits a second, DEQUANTIZED
fp32 checkpoint, and HF runs on that.** The W4 engine must match those dumps
to fp tolerance (per-layer 2e-4, logits 2e-3, tokens exact — all pass). This
proves the kernel computes exactly `dequant(Q,S) @ x`, separating kernel
bugs from quantization error. Quality-vs-fp32 is then a separate, honest
question: on this ultra-quant-sensitive random tiny model, 2 of 5 greedy
continuations remain identical to fp32; real 30B checkpoints tolerate W4
far better (that measurement belongs to the 30B phase on Modal).

Microbenchmark (`cmd/wbench`, decode shape n=1, GTX 1650):

| weight (MxK) | fp32 cuBLAS | naive W4 | W4 effective GB/s |
|--------------|-------------|----------|-------------------|
| 4096x4096    | 0.60 ms (113 GB/s) | 1.52 ms | 5.9 |
| 11008x4096   | 1.61 ms (112 GB/s) | 12.9 ms | 1.9 |
| 4096x11008   | 1.66 ms (109 GB/s) | 2.76 ms | 8.7 |

The brutal, useful result: cuBLAS fp32 already sits at the bandwidth roof,
while the naive W4 kernel — one thread per output element, adjacent threads
reading rows 2 KB apart (uncoalesced), X re-read per thread, no shared
memory — throws away the 8x byte advantage and LOSES. The information-
theoretic target is ~fp32/8 per matmul; the gap (3-50x vs that target) is
precisely the deferred kernel-optimization work, now with a measured
baseline, a clear mechanism (coalescing + reuse + occupancy), and `ncu`
available to drive it. This is the roofline lesson of Phase 1 inverted: at
these shapes the bytes DO dominate, so the kernel is finally the thing that
matters.

## Phase 0 — original goals

Goal: Go greedy-decodes `testmodels/tiny-llama` (2-layer, hidden 64, GQA 4/2,
vocab 512, ~140K params, seeded random weights, HF-loadable) and matches
`tools/gen_reference.py` dumps exactly (top-1 tokens) and numerically
(logits + per-layer activations).

Oracle: 5 raw-id prompts x 16 greedy steps dumped on CPU fp32
(`refdumps/tiny-llama`): per-step final logits, per-layer hidden states,
token ids. HF `hidden_states` layout note: `[0]` = embedding output, `[i]` =
input to layer i, last entry = **final-norm output** (not the raw last-layer
residual) — the Go test's layer mapping accounts for this.

Design decisions:
- Everything fp32 on GPU for Phase 0 (the tiny model is fp32; removes dtype
  conversion as a confounder while validating math).
- Batch=1, contiguous KV `[layers, 2, max_seq, kv_dim]`; paged KV replaces
  this in Phase 2.
- cuBLAS SGEMM for all matmuls. HF weights are `[out, in]` row-major; with
  cuBLAS column-major convention this is
  `sgemm(OP_T, OP_N, m=out, n=ntok, k=in, W lda=in, X ldb=in, Y ldc=out)`
  giving row-major `[ntok, out]` — derived once, used everywhere.
- Naive kernels: embedding gather, RMSNorm (block/token + shared-mem
  reduction), HF-convention RoPE (rotate_half pairing j and j+d/2), KV
  append, causal attention (one thread per (token, head) — 4 GiB tiny-model
  scale makes this fine; optimization comes later), SiLU-mul, residual add.
- Sampling: logits copied to host, greedy argmax in Go.
- Debug hooks in the shim (`te_debug_*`): capture embedding output, each
  layer's residual output, and the final-norm output on the device and read
  them back for the per-layer diff test — this is the divergence
  binary-search tool the plan calls for.
