# 5 · The GPU performance model

This chapter is the quantitative core: the model that predicts, before any
profiler runs, which of compute, memory bandwidth, or launch overhead
bounds a kernel — and therefore which optimizations can possibly work.
Every measured result in kllm's journal is an instance of it.

## 5.1 The roofline model

A kernel does $F$ FLOPs and moves $Q$ bytes to/from global memory. Its
**arithmetic intensity** is $I = F/Q$ (FLOP/byte). Hardware supplies peak
compute $\pi$ (FLOP/s) and peak bandwidth $\beta$ (B/s). Runtime is bounded
by both requirements:

$$t \;\ge\; \max\!\Big(\frac{F}{\pi},\ \frac{Q}{\beta}\Big) \qquad\Longleftrightarrow\qquad \text{attainable FLOP/s} \;=\; \min(\pi,\ \beta \cdot I)$$

Plotted on log-log axes (performance vs $I$), this is a roof: a bandwidth
slope $\beta I$ meeting a compute ceiling $\pi$ at the **ridge point**
$I^* = \pi/\beta$. Everything to the left is memory-bound (more FLOPs are
free; fewer bytes help); everything right is compute-bound (the reverse).

Ridge points for the machines in this project (fp32 CUDA-core compute):

| GPU | $\pi$ (fp32) | $\beta$ | $I^*$ |
|---|---:|---:|---:|
| GTX 1650 | ~3 TFLOP/s | ~128 GB/s | ~23 FLOP/B |
| A10 | ~31 TFLOP/s | 600 GB/s | ~52 |
| A100 40GB | 19.5 TFLOP/s (156 TF/s TF32 tensor) | 1555 GB/s | ~12 (fp32) / ~100 (tensor) |

## 5.2 Intensity of the operations we run

**GEMM** $C_{M\times N} = A_{M\times K} B_{K\times N}$: $F = 2MNK$,
$Q \ge (MK + KN + MN)b$. For square-ish shapes $I \approx \tfrac{2}{3b}\min(M,N,K)$
— intensity grows with the *smallest* dimension. Prefill ($M = $ hundreds
of tokens) sails past any ridge point: compute-bound, tensor-core
territory, cuBLAS's home game. **Never hand-write this one.**

**GEMV** (decode matmul, $N{=}1$): $F = 2MK$, $Q \approx MKb$ (the weight
matrix dominates). $I = 2/b$ — **0.5 FLOP/byte at fp32**, 4 at int4.
Fifty-times-left of every ridge point: purely bandwidth-bound, and the
*only* lever is bytes (quantization) and achieved-vs-peak bandwidth (this
chapter). This single number is why decode tok/s ≈ bandwidth ÷ model bytes.

**Attention (decode, one head)**: score+aggregate over $T$ cached
positions: $F \approx 4Td_h$, $Q \approx 2Td_hb$ → $I = 2/b$ again.
Memory-bound on KV reads; the FlashAttention insight (§5.7) is about not
adding *extra* traffic beyond the mandatory KV read.

**Elementwise ops** (norms, activations, residual adds): $I \le 1$ —
maximally memory-bound; their optimization is *fusion* (§5.6): eliminate
whole read-write round trips, since the math is free.

## 5.3 Achieved bandwidth: the coalescing contract

Peak $\beta$ assumes perfect access patterns. DRAM transacts in 32-byte
sectors and the L1/L2 fabric in 128-byte lines; a warp (32 threads) issuing
a load produces as many transactions as distinct sectors it touches.

- **Coalesced:** thread $i$ reads address $\text{base} + i\cdot w$ for
  small $w$ — one or a few transactions serve all 32 threads. Effective
  bandwidth ≈ peak.
- **Strided/scattered:** thread $i$ reads $\text{base} + i \cdot \text{stride}$
  with a large stride — up to 32 transactions, each mostly wasted bytes.
  Effective bandwidth = peak ÷ (waste factor), easily 10–30×.

This is the single most consequential rule in GPU kernel writing, and
kllm's W4 kernel history is a controlled experiment on it. The naive
kernel put *one thread per output row*, so simultaneous threads read
different 2 KB-apart rows: ~2–6 GB/s effective on a 128 GB/s part
(**4–6% of peak**). Attempt 1 put a *block* per row with threads striding
consecutive bytes — a warp reads 32 contiguous bytes — and effective
bandwidth rose 5–15×, flipping the kernel from 0.13× to 1.95× of cuBLAS.
Same math, same bytes, different *order*.

Corollaries: structure-of-arrays beats array-of-structs; vectorized loads
(`uint32`/`float4`) cut instruction count and widen per-thread transactions
(kllm's attempt 2 — a *win on Ampere, loss on Turing*, §5.9); and the
matrix layout should serve the kernel, not the file format.

## 5.4 The execution model: occupancy and latency hiding

A GPU hides its ~400-cycle DRAM latency not with caches but with
**parallelism**: an SM holds many resident warps and switches among them
zero-cost every cycle; while warp A waits on memory, warps B..Z issue.
**Occupancy** = resident warps ÷ hardware maximum, limited by each block's
register and shared-memory appetite. The requirement is not 100% — it's
*enough concurrent memory requests to saturate the memory system*
(Little's law again: outstanding bytes = bandwidth × latency, ~600 KB in
flight on an A100). Two ways kllm kernels have violated it:

- The naive W4 kernel at $N{=}1$ launched only $M$ threads total (4–11K)
  — tens of warps across 108 SMs, nowhere near enough outstanding loads.
  The block-per-row version launches $M$ *blocks* of 256 — thousands of
  warps.
- The naive attention kernel launched **one thread** per (token, head):
  a decode step of batch 1 with 22 heads ran 22 threads on the whole GPU.
  Its replacement runs a 256-thread block per (token, head).

The reduction idiom that block-per-row kernels need — tree-reduce partial
sums in shared memory in $\log_2(\text{blockDim})$ steps with
`__syncthreads()` between levels — appears in kllm's norms, softmax, and
matmul kernels; its cost is trivial next to the loads it organizes.

## 5.5 Launch overhead and CUDA graphs

Every kernel launch costs microseconds of CPU-side driver work (worse on
Windows' WDDM, which batches submissions). A decode step is a *parade* of
small kernels — kllm's 2-layer toy ran ~30 launches/step, and at 26–48
layers real models run hundreds. When per-kernel *work* is tens of
microseconds, launch overhead is a first-order term:

$$t_{step} \approx n_{launch}\,c_{launch} + \sum_i \max(F_i/\pi,\ Q_i/\beta)$$

kllm's Phase-1 roofline note measured the toy model as ~99% launch-bound —
the correct conclusion was *fusion helps a little; the real fix is
submission amortization*. **CUDA graphs** capture the entire step's launch
DAG once and replay it as one submission, converting $n_{launch}c_{launch}$
into $\approx c_{replay}$. The engine was architected for this from day one
(one `forward_step` ABI call per step; no host round-trips inside the
step... except MoE's routing copy, the known graph-breaker to fix). Still
unimplemented; it is the top remaining item at small-batch decode.

## 5.6 Fusion: arithmetic is free left of the ridge

For memory-bound ops, runtime = bytes ÷ bandwidth, so merging two kernels
that share data eliminates a full write+read of the intermediate:
residual-add + RMSNorm fused saves one round-trip of the hidden state per
call site (kllm measured ~3–5% end-to-end on the launch-bound toy — small
there, proportionally larger where bytes dominate). Dequantization fused
*into* the matmul (Ch. 6) is the same principle with higher stakes: the
fp32 weights never exist in memory at all. The general rule: **count the
global-memory round trips of an activation through a block; every one you
remove is pure profit** (the logical extreme is FlashAttention).

## 5.7 Case study: attention kernels and online softmax

Materializing the $T\times T$ score matrix costs $O(T^2)$ memory traffic —
fine at decode ($1\times T$), fatal at prefill for long $T$.
**FlashAttention** (Dao et al.) computes exact attention in $O(T)$ extra
memory by tiling K/V through shared memory and maintaining a *streaming*
softmax: carrying running $(m, s, o)$ = (max so far, exp-sum so far,
weighted output so far) and rescaling on each new tile —

$$m' = \max(m, m_{tile}),\quad s' = s\,e^{m-m'} + s_{tile}e^{m_{tile}-m'},\quad o' = o\,e^{m-m'} + o_{tile}e^{m_{tile}-m'}$$

— an exact regrouping of the stable softmax of §1.3, not an approximation.
kllm's current parallel decode kernel is the *shared-memory* (not
streaming) version: scores for one query live in SMEM, two block-wide
reductions, then a coalesced weighted-V pass where adjacent threads read
adjacent columns. That rewrite alone was worth 3.0× end-to-end on A100
(ITL 20.7 → 6.9 ms) because it fixed *both* §5.3 (coalescing of V) and
§5.4 (22 threads → thousands). The FlashAttention-style tiled kernel
remains the endgame for prefill and very long contexts.

## 5.8 Measurement discipline

The model above picks candidates; only measurement confirms. Rules kllm
follows, each earned: time on-device with CUDA events or in aggregate
(Windows' clock quantizes at ~0.5 ms — per-call host timing lies); warm up
before timing (first launches pay JIT/allocation costs); interleave A/B
runs (thermal drift); report effective GB/s next to the hardware roof so
"fast" has a denominator; keep every kernel version runtime-selectable so
regressions stay reproducible; and **never accept a speedup from an
unvalidated kernel** — kllm's oracle suite runs against the new kernels by
default, so a wrong optimization fails CI, not production.

## 5.9 Microarchitecture is a variable, not a constant

kllm's attempt-2 kernel (vectorized `uint32` loads, 8 weights/thread/iter)
measured **slower** than the byte-strided version on Turing (GTX 1650) and
**fastest** on Ampere (A10: 171–181 GB/s, 2.6–2.7× cuBLAS fp32). Plausible
mechanism: fewer, fatter iterations per thread reduce the number of
independent outstanding loads, which hurts more on the smaller machine's
latency-hiding budget; Ampere's doubled L1/SMEM bandwidth and larger
in-flight capacity favor the wider accesses. The transferable lesson is
methodological, not mechanistic: **optimization conclusions carry a GPU
architecture suffix**, so the attempt log records (kernel, GPU) pairs and
the dispatcher keeps both versions.

## Where this lives in kllm

- Rooflines in practice: journal Phase 1 (launch-bound toy), Phase 4
  (bandwidth-bound W4 baseline), kernel-loop entries (both GPUs);
  `cmd/wbench` prints effective GB/s against cuBLAS.
- §5.3/§5.4 in code: `matmul_w4_kernel` (anti-pattern, kept as baseline),
  `matmul_w4_v1/v2_kernel`, `attn_paged_par_kernel`, all in
  `backend/model.cu`; `te_set_kernels` keeps attempts selectable.
- Measurement: `te_bench_matmul` (CUDA events), `cmd/bench`
  (aggregate-timed), `bench/kernel_attempts.json` +
  `docs/assets/kernel_progress.png`.

## Reading list

1. Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model*, CACM 2009.
2. NVIDIA, *CUDA C++ Best Practices Guide* — coalescing, occupancy, and the memory-optimization chapters are the ground truth for §5.3–5.4.
3. Dao et al., *FlashAttention* (2022) and *FlashAttention-2* (2023) — read the online-softmax recurrence derivation.
4. Milakov & Gimelshein, *Online Normalizer Calculation for Softmax*, 2018 — the streaming softmax on its own.
5. Volkov, *Understanding Latency Hiding on GPUs* (PhD thesis, 2016) — the deep version of §5.4.
