# 2 · Autoregressive decoding: prefill, decode, and the KV cache

## 2.1 The factorization that defines everything

A language model defines a distribution over token sequences via the chain
rule:

$$P(u_1, \dots, u_T) = \prod_{t=1}^{T} P(u_t \mid u_1, \dots, u_{t-1})$$

The transformer is trained to model exactly the conditional
$P(u_t \mid u_{<t})$ — that is what the causal mask enforces. Generation
inverts this: given a prompt $u_{1..n}$, repeatedly (i) evaluate the model
to get $P(\cdot \mid u_{<t})$ as a logit vector, (ii) select $u_t$ from it,
(iii) append and repeat. Two structural facts follow immediately, and
between them they generate this entire field:

1. **Generation is inherently sequential.** $u_{t+1}$'s distribution
   depends on the *sampled value* of $u_t$. No amount of hardware
   parallelism removes this data dependence (speculative decoding, §2.6, is
   the one principled loophole).
2. **The prompt is not sequential.** All conditionals over *given* tokens
   can be evaluated in one parallel pass, because teacher-forced inputs are
   known in advance.

Fact 2 splits inference into two phases with opposite performance
characters.

## 2.2 Prefill vs decode: one model, two workloads

**Prefill** runs the prompt's $n$ tokens through the network at once. Every
matmul has an $n$-row activation matrix: GEMMs with real work in them,
$O(n)$ arithmetic intensity (Ch. 5), tensor-core-friendly,
**compute-bound** for realistic $n$. Its user-visible cost is **TTFT**
(time to first token).

**Decode** produces one token per pass. Every matmul is a matrix–*vector*
product: for each weight matrix, read $\text{out}\times\text{in}$ weights
to do one multiply-add per weight. Arithmetic intensity $\approx$ 2 FLOPs
per weight byte read — two orders of magnitude below the compute/bandwidth
ratio of any modern GPU — so decode is **memory-bandwidth-bound**: its speed
is (weights + KV bytes touched per token) ÷ (achieved bandwidth). Its
user-visible cost is **ITL** (inter-token latency); $1/\text{ITL}$ is
single-stream tokens/sec.

The asymptotic statement worth memorizing: for a dense model with $N$
parameters at batch 1,

$$\text{ITL} \gtrsim \frac{N \cdot (\text{bytes/param})}{\text{memory bandwidth}}$$

This lower bound is the *entire motivation* for quantization (shrink
bytes/param — Ch. 6) and batching (amortize the read over many tokens —
Ch. 4). Example: a 30B model at fp16 on 768 GB/s reads 60 GB/token →
≥ 78 ms/token → ≤ 13 tok/s, no matter how good the kernels are. At int4:
≤ 52 tok/s. Measured hosted Gemma-3-27B single-stream speeds (15–67 tok/s)
are exactly this arithmetic plus implementation quality.

## 2.3 The KV cache: derivation and correctness

Naively, generating token $t{+}1$ means re-running the whole network on all
$t$ tokens: $O(t)$ GEMM work per token, $O(T^2)$ total — plus $O(t^2)$
attention. The KV cache eliminates the re-run, and its correctness is worth
stating as a claim with a proof, because paging, windowing, and speculative
rollback all lean on it.

**Claim.** For a causal transformer, the keys and values of position $p$,
at every layer, are functions of $u_{1..p}$ only, and therefore never change
as the sequence grows.

**Proof sketch (induction over layers).** At layer 0, $x_p^{(0)} = E_{u_p}$
depends only on $u_p$. Inductively assume $x_p^{(\ell)}$ depends only on
$u_{1..p}$. Layer $\ell$'s attention at position $p$ reads
$\{k_j^{(\ell)}, v_j^{(\ell)} : j \le p\}$ — by the causal mask, nothing at
positions $> p$ — and each of those is a projection of $x_j^{(\ell)}$,
$j \le p$. The MLP is position-wise. Hence $x_p^{(\ell+1)}$ depends only on
$u_{1..p}$. Since $k_p^{(\ell)}, v_p^{(\ell)}$ are projections of
$x_p^{(\ell)}$, they are fixed once $u_{1..p}$ is fixed. $\blacksquare$

So the engine computes each position's K and V **exactly once**, stores
them (post-RoPE — the rotation is position-of-birth, §1.5), and each new
token computes only its own $q, k, v$ and attends over the stored cache.
Per-token cost drops from "recompute the prefix" to "one token's worth of
GEMMs + read the cache":

| | without cache | with cache |
|---|---|---|
| GEMM work for token $t$ | $O(t \cdot N)$ | $O(N)$ |
| attention work for token $t$ | $O(t^2 d)$ | $O(t\,d)$ |
| extra memory | — | $O(L \cdot t \cdot H_{kv} d_h)$ ×2 |

The cache converts compute into memory — the field's recurring trade. Its
size (derived properly in Ch. 3) is the binding constraint on concurrency,
and *reading* it is a growing share of decode bandwidth as context grows:
at long context the KV bytes/token rival the weight bytes/token, which is
why GQA (§1.4), windowing (§1.7), and KV quantization exist.

**Invalidation discipline.** The cache is append-only per sequence. The
only mutations are: append (new token), truncate (speculative rejection,
§2.6), release (sequence ends). kllm's backend goes further and is
*stateless* about sequences — position and placement come from the caller
each step — so correctness reduces to the caller's bookkeeping, which is
plain Go code testable without a GPU.

## 2.4 Prefill and decode unified: one batched step

kllm's `forward_step` takes, per sequence, a span of new tokens at a start
position. A prompt is a span of length $n$ at position 0; a decode step is a
span of length 1 at position $t$. Mixed batches (one sequence prefilling
while others decode) are then not a special case but the *general* case:
concatenate all spans, run the network once, gather each sequence's
last-position logits. The per-token position array (not a scalar offset)
is what makes RoPE and causal masking correct in mixed batches — every
token carries its own absolute position.

## 2.5 Selecting the next token

The model yields logits $z \in \mathbb{R}^V$; selection turns them into a
token.

- **Greedy:** $u_t = \arg\max_i z_i$. Deterministic, and *the* correctness
  instrument: greedy against a reference implementation must match
  token-for-token, making it the only sampling mode kllm needs so far.
- **Temperature:** sample from $\mathrm{softmax}(z/\tau)$. $\tau \to 0$
  recovers greedy; $\tau > 1$ flattens.
- **Top-k / nucleus (top-p):** truncate the distribution to the $k$ most
  probable tokens / the smallest set with cumulative mass $\ge p$, then
  renormalize and sample — both are variance-control devices against the
  long tail of a 100K+-entry softmax.

A numerically relevant detail: argmax stability. Two implementations that
agree to $10^{-3}$ in logits can still disagree on argmax when the top two
logits are within that tolerance; kllm's tolerance choices (logits
$10^{-3}$, per-layer $10^{-4}$) plus exact-token assertions have held
empirically across all validated models, but ties are the failure mode to
expect first when tolerances loosen.

## 2.6 Speculative decoding: parallelizing the unparallelizable

The one principled attack on fact 1 of §2.1 (Leviathan et al. 2023; Chen et
al. 2023). Use a cheap **draft** model $q$ to propose $\gamma$ tokens
$\hat u_1 \dots \hat u_\gamma$ autoregressively, then run the expensive
**target** model $p$ *once* over all of them — verification is
teacher-forced, i.e. prefill-shaped, so it costs approximately one decode
step of the target regardless of $\gamma$. Accept a prefix of the proposals
via a per-token rule:

- accept $\hat u_i$ with probability $\min\!\big(1,\; p(\hat u_i)/q(\hat u_i)\big)$;
- on the first rejection at position $i$, sample the replacement from the
  **residual distribution** $\mathrm{norm}\big(\max(0,\, p(\cdot) - q(\cdot))\big)$
  and discard the rest;
- if all $\gamma$ survive, take one bonus token from $p$'s last
  distribution.

**Why the output distribution is exactly $p$** (sketch, one token): the
probability that token $x$ is emitted is

$$\underbrace{q(x)\min\!\Big(1,\tfrac{p(x)}{q(x)}\Big)}_{\text{proposed and accepted}} + \underbrace{\Big(\textstyle\sum_y q(y)\big(1 - \min(1,\tfrac{p(y)}{q(y)})\big)\Big)}_{\text{rejection probability}}\cdot\, \mathrm{norm}\big(\max(0, p - q)\big)(x)$$

The first term is $\min(p(x), q(x))$; the rejection mass equals
$\sum_y \max(0, q(y) - p(y)) = \sum_y \max(0, p(y)-q(y))$, which is exactly
the residual's normalizer, so the second term is $\max(0, p(x) - q(x))$.
Sum: $\min(p,q) + \max(0, p-q) = p(x)$. $\blacksquare$ — acceleration with
**zero** change to outputs.

Expected tokens per target pass with per-token acceptance rate $\alpha$:
$\mathbb{E} = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ (accepted prefix + 1),
so $\alpha = 0.8, \gamma = 4$ gives ≈ 3.4 target-tokens per target-read —
directly multiplying the bandwidth bound of §2.2. The systems costs: run
two models, manage two KV caches, and **truncate both caches on rejection**
(the append-only cache gains its one rollback operation). This is kllm's
planned Phase 6; the orchestration (draft/target interleaving,
accept/reject, rollback) is scheduler-side Go, and only the batched verify
is a kernel concern.

## Where this lives in kllm

- Prefill/decode duality: `engine/engine.go` (`Sequence.Forward` — prompt
  span vs single-token span), `backend/model.cu` `te_forward_batch`
  (per-token position/sequence-id arrays; last-token logits gather).
- Cache bookkeeping (append/commit/release, caller-side): `engine/kv`,
  exercised by `TestPagedBatchMatchesSolo` and the scheduler suite.
- Greedy selection + the exact-token oracle: `engine.Argmax`,
  `engine/oracle`.
- Measured numbers behind §2.2's bound: journal — TinyLlama-1.1B fp32 on
  A100, ITL 20.7 ms naive → 6.9 ms after the attention kernel rewrite
  (effective 213 → 638 GB/s against a ~1.5 TB/s roof).

## Reading list

1. Pope et al., *Efficiently Scaling Transformer Inference*, 2022 — the canonical prefill/decode + memory-bound analysis.
2. Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, 2023 — the acceptance rule and proof.
3. Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
4. Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2019 — nucleus sampling, why truncation.
5. Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 — MQA, the first KV-size attack.
