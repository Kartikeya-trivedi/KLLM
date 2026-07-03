# 6 · Weight quantization

## 6.1 Why: the bandwidth bound, again

Chapter 2 established decode's law: tok/s ≤ bandwidth ÷ bytes-per-token,
and Chapter 5 established that at $N{=}1$ the matmul's intensity is $2/b$
FLOP/byte — the *only* lever on the memory side is $b$, bytes per weight.
Quantization is the direct attack: store weights in 4 bits instead of 16/32
and the mandatory per-token read shrinks 4–8×, with the same factor
available (in principle) as decode speedup. "W4A16" names the scheme:
**W**eights 4-bit, **A**ctivations 16-bit — activations stay high-precision
because (i) they're tiny compared to weights, (ii) they're where outliers
live, and (iii) fp16 tensor cores want fp16 inputs. (kllm's lab spine is
fp32 activations — W4A32 strictly — same theory.)

## 6.2 Number formats: what a bit buys

A $k$-bit format spends its bits on *range* (exponent) vs *resolution*
(mantissa):

| format | bits (s/e/m) | max ≈ | relative step | role |
|---|---|---:|---:|---|
| fp32 | 1/8/23 | $3.4\times10^{38}$ | $1.2\times10^{-7}$ | reference/accumulation |
| fp16 | 1/5/10 | 65504 | $9.8\times10^{-4}$ | activations; overflow-prone reductions |
| bf16 | 1/8/7 | $3.4\times10^{38}$ | $7.8\times10^{-3}$ | fp32's range, 8 bits fewer — training/native checkpoints |
| int8 | 8 (fixed) | ±127·scale | uniform | weights & (with care) activations |
| int4 | 4 (fixed) | ±7·scale | uniform | weights, with grouping (this chapter) |

Integer formats have *uniform* absolute step and *no* range flexibility:
everything depends on choosing the scale that maps reals onto the 16 int4
levels. That choice is the whole game.

## 6.3 Uniform affine quantization, and why symmetric

Quantize real $w$ with scale $s$ and zero-point $z$:

$$q = \mathrm{clamp}\!\big(\mathrm{round}(w/s) + z,\; q_{min},\, q_{max}\big), \qquad \hat w = s\,(q - z)$$

**Symmetric** quantization fixes $z$ at the range center (int4:
levels $-8..7$, $\hat w = s\,q$) — one multiply to dequantize, no
zero-point bookkeeping in the kernel's inner loop. Asymmetric earns its
extra parameter only for one-sided distributions (post-ReLU activations);
weight matrices are near-zero-mean, so symmetric costs almost nothing and
is what kllm implements: $s = \max|w|/7$, $q = \mathrm{clamp}(\mathrm{round}(w/s), -8, 7)$.

**Error model.** Rounding error is $\approx$ uniform on $[-s/2, s/2]$:
mean-square error $s^2/12$. With $s = \max|w|/7$ over a group, per-weight
RMS error $\approx \max|w|/24$. Two consequences: (i) error scales with
the *local dynamic range* — one outlier in a group inflates $s$ and drowns
its neighbors' precision (the motivation for grouping, next); (ii) the
signal-to-quantization-noise ratio improves ~6 dB per bit — int4 is
aggressive, and works for LLM weights because (a) weight distributions
are heavy at zero, (b) each output sums thousands of independently-rounded
terms whose errors partially cancel ($\mathbb{E}$ of the dot-product error
grows as $\sqrt{K}$ while the signal grows as $K$), and (c) networks are
trained with enough redundancy to absorb small perturbations.

## 6.4 Group-wise scaling: localizing the dynamic range

One scale per *tensor* lets a single large weight ruin the whole matrix;
one scale per *element* is no compression. The operating point the field
converged on: one scale per **group** of $G$ consecutive in-dimension
weights within a row (G = 32–128):

- storage: $K/2$ bytes packed nibbles + $(K/G)\cdot 2\text{–}4$ bytes of
  scales per row → effective bits/weight $= 4 + 32/G$ (fp32 scales) —
  4.25–5 bits;
- error: each group's $s$ tracks its *local* $\max|w|$, so an outlier
  poisons only its 32–128 neighbors;
- kernels: scales load once per group per row — negligible traffic, and
  group boundaries align with vectorized loads when $G \bmod 8 = 0$.

kllm packs two nibbles per byte (even column low), stores levels as
$q + 8 \in [0,15]$, fp32 scales, $G{=}32$ (toy) / 128 (real-scale) — the
same family as AWQ/GPTQ "group quantization."

**Beyond round-to-nearest.** kllm quantizes RTN (round-to-nearest), which
is the baseline the sophisticated methods improve on: **GPTQ** rounds
columns sequentially, using a Hessian approximation (from calibration
activations) to compensate each rounding error in the not-yet-quantized
columns; **AWQ** observes that a small fraction of weight *channels*
matter far more (those meeting large activations) and rescales
channels before RTN so the important ones get finer effective resolution.
Both are offline, calibration-driven, and orthogonal to the *serving*
format — an engine that runs group-quantized RTN runs GPTQ/AWQ output too.
What kllm keeps out of scope for now: activation quantization (W8A8 —
needs outlier handling à la SmoothQuant), fp8 (no such tensor cores on
Ampere), and KV-cache quantization (same math, applied to §3's bytes).

## 6.5 The dequant-fused kernel

The naive deployment — dequantize the matrix to fp16 in memory, then GEMM —
*reads and writes the big fp32/fp16 tensor anyway*, forfeiting the entire
bandwidth win. The correct form fuses dequantization into the matmul's
inner loop:

$$y_r = \sum_{g} s_{r,g} \sum_{j \in g} q_{r,j}\, x_j$$

Weights cross the memory bus packed (4.25 bits effective) and are expanded
to real values *in registers*: unpack nibble → subtract 8 → multiply by the
group's scale → FMA with $x_j$. Per §5.2 this kernel's intensity is
~4 FLOP/byte — still memory-bound, so its ideal runtime is (packed bytes ÷
bandwidth), i.e. **~8× faster than the fp32 GEMV**, and every shortfall
from that is kernel quality, not physics. kllm's measured ladder on Ampere:
naive 13–35 GB/s effective → coalesced 155–164 → vectorized 171–181 GB/s =
2.6–2.7× *faster than cuBLAS fp32* while moving 8× fewer bytes — with the
gap to the ~500 GB/s cuBLAS-achieved roof documented as remaining headroom.

## 6.6 Validation: separating kernel bugs from quantization error

"Quantized output is close to fp32 output" conflates two error sources —
the *intended* quantization error and *unintended* kernel bugs — and at
int4 the intended error is large enough to hide real bugs inside a loose
tolerance. kllm's methodology splits them exactly:

1. **Kernel exactness.** The quantizer emits a second checkpoint with
   $\hat W = s\,q$ materialized in fp32. The reference implementation runs
   on $\hat W$; the engine runs the packed $(q, s)$. If the engine matches
   *those* dumps to fp tolerance (per-layer $2{\times}10^{-4}$, exact
   tokens), the kernel provably computes $\hat W x$ — bit-for-bit the right
   function. Any bug in packing, nibble order, scale indexing, or group
   boundaries breaks this immediately.
2. **Quantization quality** is then a separate, honest measurement against
   the original model — perplexity or task metrics at scale; on kllm's toy
   (a worst case: random weights have no redundancy) 2 of 5 greedy
   continuations survive int4 unchanged, quoted as a property of the toy,
   not the method.

This two-checkpoint trick generalizes: it's the right template for
validating *any* lossy-format kernel (fp8, sparsity, KV quantization).

## Where this lives in kllm

- Quantizer: `tools/quantize_w4.py` — symmetric RTN, groupwise, packed
  nibbles + fp32 scales, **emits the dequantized twin**; covers dense
  projections and MoE expert FFNs (router stays fp32).
- Kernels: `matmul_w4_kernel` (baseline) → `_v1` → `_v2` in
  `backend/model.cu`; dispatch in `w4_matmul` / `mm`; loader path
  `te_model_load_tensor_w4` + `engine.uploadWeights`.
- Validation: `engine/e2ew4` and `engine/e2emoew4` (the INT4-MoE case) —
  both implement §6.6 point 1.
- Measurements: `cmd/wbench`, `bench/kernel_attempts.json`, journal
  Phase 4 + kernel-loop entries.

## Reading list

1. Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, 2018 — the affine-quantization foundations.
2. Dettmers et al., *LLM.int8()*, 2022 — the outlier phenomenon that shapes all LLM quantization.
3. Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022.
4. Lin et al., *AWQ: Activation-aware Weight Quantization*, 2023.
5. Xiao et al., *SmoothQuant*, 2022 — what W8A8 requires; the boundary kllm hasn't crossed.
6. Dettmers & Zettlemoyer, *The case for 4-bit precision: k-bit Inference Scaling Laws*, 2022 — why 4 bits is the sweet spot at fixed memory.
