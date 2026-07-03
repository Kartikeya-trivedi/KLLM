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

## Phase 5 — MoE routing + grouped expert GEMM — DONE

One MoE implementation, two routing families behind a `router_mode` switch:
- **mode 0, softmax top-k renorm** (Mixtral/Qwen): softmax over all experts,
  take top-k, renormalize. Oracle: HF `MixtralForCausalLM` on a tiny
  4-expert/top-2 random model (`tools/make_test_moe.py`) — per-layer,
  logits, and tokens all match (`e2emoe`).
- **mode 1, sigmoid + expert-bias** (Sarvam/DeepSeek-V3 family): sigmoid
  scores, top-k selected by score+bias, weights from the *unbiased* scores
  normalized over the selection. HF has no class pairing this router with
  plain Llama attention, so the oracle is a standalone numpy forward
  (`tools/gen_reference_numpy_moe.py`) — same dump format, same harness
  (`e2emoesig`). Passes at the same tolerances.

Dataflow (the real grouped-GEMM structure, naive v0): router GEMM → top-k
kernel on device → host builds the expert-sorted permutation (one small D2H
per layer — a deliberate correctness-phase concession) → `gather_rows` →
**per-expert segment GEMMs** (a loop over experts with variable row counts —
the fused grouped kernel is deferred optimization work) → SwiGLU → weighted
`scatter_add` un-permute (fp32 atomics; ~1e-7 nondeterminism, inside
tolerance).

Because expert GEMMs go through the same `mm()` dispatch as dense weights,
**INT4 MoE fell out for free**: `quantize_w4.py` now also quantizes
w1/w2/w3 (router gate stays fp32), and `e2emoew4` validates the quantized
expert path against HF on the dequantized twin — exact within fp tolerance.
That is the Sarvam-30B-INT4 serving path in miniature: sigmoid routing ✓,
quantized experts ✓, continuous batching + paged KV underneath ✓.

Correctness harness refactored into `engine/oracle` (shared by the four
model-variant suites; each variant is its own test package because the
backend allows one model per process).

## Modal A10G gate — DONE (Linux cgo + real Ampere, first run green)

`engine/backend/backend_cgo.go` (build-tagged linux) links
`libtoyengine.so` via cgo — same `impl` interface as the Windows syscall
loader, batch flattening shared in `backend.go`. `modal run
tools/modal_lab.py::build_and_test` ships the repo (incl. test models +
refdumps) into the CUDA 12.8 container, builds the backend `-arch=sm_86`,
and runs everything on an **A10 (sm_86, 22 GiB, 72 SMs)**:

- All 7 test packages pass, first run: dense-vs-HF, paged batch, scheduler,
  W4-vs-dequant, MoE softmax-vs-HF, MoE sigmoid-vs-numpy, INT4-MoE.
- Smoke: max abs error 0. wbench on A10: cuBLAS fp32 at ~500 GB/s;
  naive W4 kernel 13-35 GB/s effective (0.20-0.53x) — same optimization
  target, now measured on the real target architecture.

The engine is now proven portable: Windows/syscall/sm_75 for the inner loop,
Linux/cgo/sm_86 for the target. Gotcha for the record: Modal image `.env()`
does not expand `$PATH` — set the full explicit PATH or lose /usr/bin.

## What remains — kernel optimization (the deliberate endgame)

Phases 0-5 are complete and gated. The system is done; the kernels are
naive on purpose. The optimization backlog, each with a measured baseline
and a correctness net that will catch any regression:

1. **W4 dequant-matmul** (biggest lever): coalesce Q access, stage X in
   shared memory, warp-level reduction, k-split for n=1 occupancy. Target:
   beat cuBLAS fp32 by ~4-8x in bytes terms (A10 baseline: 0.2x).
2. **Paged attention**: one thread per (token, head) → warp/block per query
   with shared-mem K/V tiles and online softmax (FA2-style, GQA-aware).
3. **Fused grouped-GEMM MoE**: kill the per-layer host round-trip (device
   prefix-sum permutation), single grouped kernel over expert segments.
4. **CUDA graphs**: capture the decode step, replay per step — the
   launch-overhead fix the Phase 1 roofline note predicted.
5. `ncu` on this box (and Modal for sm_86 numbers) drives all of it.

30B-class checkpoints (Gemma/Qwen/Sarvam) become a loader exercise
(bf16→fp32 conversion or an fp16 compute path, sharded index already
supported) + Modal A10G/A100 for VRAM — no new engine architecture.

## Observability + experiment tracking — DONE

Added a dependency-free metrics layer (`engine/metrics`): counters, gauges,
histograms, rendered as Prometheus text exposition and as a JSON snapshot.
The scheduler instruments every step — TTFT (submit→first token), ITL
(inter-token), batch size, running/queued sequences, KV utilization, token
and request counters, and an EWMA of aggregate decode tok/s. The server
serves `GET /metrics` (Grafana-scrapeable) and `GET /stats.json` (the
browser UI polls it for live server-wide tok/s). Verified live: a loadgen
burst produced a fully-populated `/metrics` (48 requests, 1110 tokens, ~7.7K
tok/s aggregate EWMA, populated TTFT/ITL/batch histograms).

For experiment tracking specifically (the W&B ask), `cmd/bench` gained
`--json` and `tools/wandb_bench.py` runs it across configs and logs
`decode_tok_s`/TTFT/ITL to Weights & Biases, tagged by kernel version — so
each kernel-optimization iteration is a tracked run and the speedups chart
themselves. It degrades gracefully (runs the benches and prints numbers when
wandb isn't installed), and has a `--serve-url` live mode that streams a
running server's tok/s to W&B. Why not a native Go→W&B logger: W&B has no
first-class Go SDK, so the honest split is **Prometheus for live serving
metrics (Go-native) + W&B via the Python tools for experiment tracking**
(the official SDK), which is also where the rest of the offline tooling
already lives.

## Real model on a single A100 (Modal) — DONE

First run of the engine on a **real** model and a **datacenter GPU**, not the
toy: `modal run tools/modal_lab.py::bench_model` on a single **A100-SXM4-40GB
(sm_80, 108 SMs)**. Pipeline: download **TinyLlama-1.1B-Chat** → convert
bf16→fp32 in the engine's layout (`tools/convert_hf.py`, 1.10B params, 201
tensors) → build backend `-arch=sm_80` → validate vs HuggingFace → benchmark.

TinyLlama was chosen because the engine implements *plain* Llama (RMSNorm,
rotate_half RoPE, SiLU-gated MLP, GQA, no attention bias, no qk-norm) and
TinyLlama matches that exactly — same tensor names, no bias. The converter
refuses archs that need kernel changes (Gemma embedding-scale/softcap/qk-norm,
Qwen QKV-bias) with a clear error rather than emitting silent garbage.

Results:
- **Correctness: 16/16 greedy tokens identical to HuggingFace** on the real
  1.1B model. The oracle isn't just for the toy — the engine is provably
  right on a real checkpoint. (HF ran fp32 on the same A100; transformers
  5.13 needed the `torch_dtype`→`dtype` kwarg fix.)
- **Single-stream decode: 48.3 tok/s** (TTFT 20.9 ms, ITL 20.7 ms),
  `cmd/bench`, prompt 64, 128 steps.
- **Aggregate (HTTP + continuous batching, `cmd/loadgen`):**

  | concurrency | tok/s | TTFT p50 | TTFT p99 |
  |------------:|------:|---------:|---------:|
  | 1  | 81   | 16 ms  | 20 ms  |
  | 8  | 366  | 56 ms  | 64 ms  |
  | 32 | 1,211 | 117 ms | 198 ms |

  ~15× throughput from batching (81→1211 tok/s) on the real model — the
  Phase-3 continuous-batching curve, reproduced on A100 at 1.1B scale.

Reading against the [Gemma reference](#external-reference-gemma-inference-speed):
48 tok/s single-stream sits inside the hosted-**Gemma-3-27B** provider band
(15–67 tok/s) — but for a 25× *smaller* model on *naive* kernels, so per
parameter we're far off the frontier. That's the honest baseline the kernel
endgame optimizes against, now measured on the target architecture with a
real model. Config note: TinyLlama is fp32 here (4.4 GB) and fits the 40 GB
A100 comfortably; a real 30B would need the W4 path (Phase 4) to fit, which
is exactly why that phase exists. HF downloads warned `no HF_TOKEN`
(fine for public TinyLlama; gated Llama/Gemma would need a Modal secret).

## Gemma 3 + Sarvam on A100 — DONE (the plan's target models, real checkpoints)

**Sarvam-1 (2B, sarvamai/sarvam-1)** — plain-Llama arch, so it ran with zero
engine changes: **16/16 greedy tokens match HF**, 21.8 tok/s single-stream,
37 → 577 tok/s aggregate (conc 1 → 32) on A100. First run OOM'd and taught a
real serving lesson: KV pool size must be computed from model dims (28
layers x kv_dim 1024 made a fixed 8192-block pool a 30 GiB allocation);
`bench_model` now sizes the pool from config with an 8 GiB cap.

**Gemma 3 1B (unsloth/gemma-3-1b-it)** — needed genuine architecture work:
(1+w)-parameterized RMSNorm, embedding scaling by sqrt(hidden), GELU-tanh,
per-head qk-norm, sandwich norms (4 per layer), sliding-window attention
(512) with per-layer rope theta (10k local / 1M global), decoupled head_dim
(256 != hidden/heads), query_pre_attn_scalar. Validated the same way as
everything else: a tiny random Gemma3 (HF-loadable) + per-layer oracle
(`validate_gemma` on Modal) — then the real model: **16/16 tokens match HF**
on A100, 15.1 tok/s single-stream (naive kernels; the 262K-vocab lm_head
GEMM alone reads 1.2 GB/token in fp32).

Two debugging lessons the per-layer oracle made cheap:
1. It localized the divergence to *the first full-attention layer* in one
   read, which unmasked a transformers-5 config change: top-level
   `rope_theta`/`rope_local_base_freq` are gone, replaced by a nested
   `rope_parameters` dict per layer type. The Go parser fell back to the 10k
   default — sliding layers *accidentally correct*, global layers wrong.
2. `layer_types` beats formulas: the sliding/full assignment is now passed
   explicitly across the ABI (`te_model_set_layer_sliding`), with the
   converter asking transformers' own runtime config for the authoritative
   list rather than re-deriving it from `sliding_window_pattern`.

## Kernel-optimization loop — attempts, measured (graph: docs/assets/kernel_progress.png)

The deferred endgame, run as a loop: change one kernel → full oracle suite
must stay green (all 7 model-variant suites now run against the NEW kernels
by default) → measure → log ([bench/kernel_attempts.json](../bench/kernel_attempts.json))
→ plot (`tools/plot_kernels.py`). Kernel versions stay runtime-selectable
(`te_set_kernels`) so every attempt remains reproducible in one binary.

| # | attempt | W4 4096x4096 (1650) | W4 11008x4096 (1650) | tiny-model tok/s (seq~512) |
|---|---------|--------------------:|---------------------:|---------------------------:|
| 0 | naive baseline | 5.6 GB/s (0.37x fp32) | 1.9 GB/s (0.13x) | 916 |
| 1 | W4 coalesced + shared reduction | **26.0 GB/s (1.73x fp32)** | **29.5 GB/s (1.95x)** | 914 |
| 2 | W4 vectorized uint32 loads | 23.2 GB/s (1.54x) | 22.1 GB/s (1.46x) | 910 |
| 3 | parallel paged attention | — | — | **1958 (2.14x)** |

Readings:
- **Attempt 1 crosses the line that matters**: the W4 kernel now beats cuBLAS
  fp32 (1.7–2.0x) instead of losing 3–8x — a 4.7–15.6x kernel-level jump —
  by fixing exactly what the Phase-4 baseline diagnosed (uncoalesced Q rows,
  no X reuse, one thread per output). Block-per-row + byte-strided warp
  reads + shared-mem reduction.
- **Attempt 2 is an honest regression on Turing**: uint32 vectorized loads
  measured *slower* than v1 on the 1650 (fewer, fatter iterations hide
  latency worse at this occupancy). Logged, kept selectable, re-judged on
  Ampere below.
- **Attempt 3 is the big end-to-end win**: block-parallel attention (256
  threads/[token,head] with shared-softmax) doubled decode throughput at
  seq~512 — the 1-thread-per-(token,head) baseline had become the dominant
  cost as sequences grew.
- Headroom stays honest: v1's ~29 GB/s is still ~4x under the 1650's
  achievable bandwidth (cuBLAS reaches ~114 GB/s) — next levers are
  shared-memory X staging, multiple rows per block, and float4 X loads.

**Ampere results (A10 sm_86 wbench; A100 model-level; full suite green on
both):**

| kernel | 4096x4096 (A10) | 11008x4096 (A10) |
|--------|----------------:|-----------------:|
| fp32 cuBLAS | 490 GB/s | 502 GB/s |
| w4 naive | 13 GB/s (0.20x) | 35 GB/s (0.53x) |
| w4 v1 coalesced | 155 GB/s (2.39x) | 164 GB/s (2.46x) |
| w4 v2 vectorized | **171 GB/s (2.64x)** | **181 GB/s (2.71x)** |

- **v2 wins on Ampere** (2.64–2.71x vs fp32) after losing on Turing — the
  arch-dependent flip is exactly why versions stay selectable and measured
  per GPU rather than assumed.
- **Model-level, A100, TinyLlama-1.1B fp32** (attention kernel is what
  applies to an fp32 model): single-stream **48.3 → 144.9 tok/s (3.0x)**,
  ITL 20.7 → 6.9 ms (effective weight bandwidth ~213 → ~638 GB/s);
  aggregate at conc 32: 1211 → **1804 tok/s**; correctness still 16/16 vs HF.
- Single-stream 145 tok/s for a 1.1B fp32 now clears the hosted
  Gemma-3-27B provider band (15–67 tok/s) instead of merely touching it —
  and the remaining fp32→W4 lever (4x fewer weight bytes at 2.6x higher
  kernel efficiency) is measured and waiting.

## External reference — Gemma inference speed

Collected to calibrate what "fast" means for a 30B-class model, so the
engine's tok/s numbers have real targets (not just tiny-model figures).
**Two very different numbers matter and are constantly conflated:**

- **Single-stream decode** (one request, tokens/sec the user feels) — for
  **Gemma 3 27B** across hosted API providers this is roughly **15–67 tok/s**
  (Amazon ~67, Parasail ~40, Novita ~26, DeepInfra ~17, Nebius FP8 ~15),
  a 3.4× spread; TTFT ~1.2–11.8 s. This is the memory-bound decode regime —
  the number our per-request `ITL`/`decode_tok_s` should be compared to.
- **Aggregate throughput** (all concurrent requests summed, via continuous
  batching) — Gemma 3 4B ~**3,976 tok/s** on a single A100 40GB (vLLM),
  Gemma 3 12B ~2.4K tok/s, and Gemma 3 27B ~**22K tok/s** on a 4×H100 node.
  This is what our server-wide EWMA tok/s (`/metrics`) is the analogue of.

Takeaways for this project: (1) single-stream 30B decode being only tens of
tok/s is exactly the bandwidth-bound story from the Phase 4 roofline note —
which is *why* W4 quantization is the headline lever. (2) The 100–1000×
gap between single-stream and aggregate is continuous batching doing its job
(Phase 3). (3) Provider spread of 3–5× at the same model = the kernel/serving
stack is most of the performance, which is the whole thesis of building one.

Sources: [Artificial Analysis — Gemma 3 27B providers](https://artificialanalysis.ai/models/gemma-3-27b/providers),
[DatabaseMart A100 40GB vLLM benchmark](https://www.databasemart.com/blog/vllm-gpu-benchmark-a100-40gb),
[HF gemma-3-27b-it tok/s discussion](https://huggingface.co/google/gemma-3-27b-it/discussions/39),
[Google Cloud — Gemma 3 on Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/announcing-gemma-3-on-vertex-ai/).

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
