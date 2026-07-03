# 1 · The transformer forward pass

The decoder-only transformer is a stack of $L$ identical blocks operating on
a **residual stream**: a sequence of vectors $x_t \in \mathbb{R}^d$, one per
token, that every block reads from and writes small updates into. Everything
an inference engine does — caching, batching, quantizing, kernel fusion — is
an optimization of this one computation, so it has to be understood exactly,
not approximately. Getting a single convention wrong (a norm
parameterization, a rotation pairing, an off-by-one in a mask) produces a
model that runs fine and is silently wrong, which is why kllm's whole
methodology is diffing every layer against a reference implementation.

## 1.1 Token embeddings

Input is a sequence of token ids $(u_1, \dots, u_T)$, $u_t \in \{0,\dots,V{-}1\}$,
produced by a tokenizer (in kllm the tokenizer is deliberately out of scope;
the engine consumes raw ids). The embedding matrix $E \in \mathbb{R}^{V\times d}$
maps ids to vectors:

$$x_t^{(0)} = E_{u_t}$$

Two consequential details:

- **Tying.** Many models reuse $E$ as the output projection ("tied
  embeddings", §1.8), halving that memory. Gemma ties by default; the
  engine materializes an explicit `lm_head` at conversion so the forward
  pass has a uniform shape.
- **Scaling (Gemma).** Gemma multiplies embeddings by $\sqrt{d}$:
  $x^{(0)}_t = \sqrt{d}\,E_{u_t}$. This is inherited from the original
  Transformer (Vaswani et al.), where tied input/output matrices need the
  input side scaled up because the embedding entries are initialized with
  variance $1/d$. Llama-family models dropped it. It's exactly the kind of
  one-line difference that shifts every downstream activation.

Embedding lookup is a **gather**: memory-bound, trivially parallel, no math.

## 1.2 Normalization: RMSNorm and its two parameterizations

Each block normalizes its input before using it. Modern decoders use
**RMSNorm** (Zhang & Sennrich, 2019) rather than LayerNorm: it drops the
mean-centering and bias, keeping only the scale — empirically as good,
strictly cheaper, and one fewer reduction:

$$\mathrm{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \cdot w_i$$

$\varepsilon$ (typically $10^{-6}$) guards the degenerate all-zeros input.
The intuition: the residual stream's magnitude grows as layers add into it;
normalization re-projects onto (roughly) the unit hypersphere so each
block's sublayers see inputs of controlled scale, which is what makes
100-layer stacks trainable — and, at inference, is why final logits have
stable magnitude regardless of depth.

**The $(1+w)$ trap.** Gemma stores its norm weights *zero-centered* and
computes $x \cdot \mathrm{rms}^{-1} \cdot (1 + w)$. A freshly initialized
Gemma norm has $w = 0$ (identity), where Llama has $w = 1$. Load Gemma
weights into a Llama-parameterized norm and every activation is scaled by
$w$ instead of $1{+}w$ — close enough to look plausible, wrong enough to
destroy the model. kllm's RMSNorm kernel takes a `plus_one` flag.

**Placement grammar.** Where the norms sit defines the block family:

- *Pre-norm (Llama):* $x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x))$,
  then $x \leftarrow x + \mathrm{MLP}(\mathrm{Norm}(x))$. Two norms/block.
- *Sandwich (Gemma 2/3):* the sublayer **output** is also normalized before
  the residual add:
  $x \leftarrow x + \mathrm{Norm}_{post}(\mathrm{Attn}(\mathrm{Norm}_{pre}(x)))$,
  likewise for the MLP. Four norms/block. The post-norms tame the magnitude
  of what gets added into the stream — a stability device for deep, wide
  models that the engine must replicate exactly.

Numerically, the reduction $\sum x_j^2$ should be accumulated in fp32 even
when activations are fp16/bf16 (kllm computes everything in fp32, so this is
moot until a low-precision activation path exists).

## 1.3 Attention: the scaled dot-product core

Attention lets position $t$ construct its update as a data-dependent convex
combination of information at positions $\le t$. Per head:

1. Project the normalized stream into queries, keys, values:
   $q_t = x_t W_Q^\top$, $k_t = x_t W_K^\top$, $v_t = x_t W_V^\top$, each in
   $\mathbb{R}^{d_h}$.
2. Score every earlier position: $s_{t,p} = \dfrac{q_t \cdot k_p}{\sqrt{d_s}}$
   for $p \le t$.
3. Normalize scores to a distribution:
   $a_{t,p} = \mathrm{softmax}_p(s_{t,\cdot})$.
4. Aggregate: $o_t = \sum_{p \le t} a_{t,p}\, v_p$.
5. Concatenate heads and project back: $y_t = [o_t^{(1)} \| \cdots \| o_t^{(H)}]\, W_O^\top$.

**Why $\sqrt{d_s}$.** If $q, k$ have i.i.d. components with unit variance,
$q\cdot k$ has variance $d_h$; dividing by $\sqrt{d_h}$ keeps scores $O(1)$
so softmax doesn't saturate into a one-hot (killing gradients in training
and making the distribution needle-sharp at inference). The scale
denominator $d_s$ is *usually* $d_h$ but is a free parameter: Gemma 3
exposes it as `query_pre_attn_scalar`. Treat it as config, never derive it.

**Causality.** $s_{t,p} = -\infty$ for $p > t$ (equivalently: the loop stops
at $t$). This is what makes the model autoregressive, and — crucially for
Chapter 2 — what makes $k_p, v_p$ *immutable once computed*: nothing about
position $p$'s key/value depends on any later token.

**Softmax stability.** $e^{s}$ overflows fp32 near $s \approx 88.7$. The
standard identity

$$\mathrm{softmax}(s)_p = \frac{e^{s_p - m}}{\sum_j e^{s_j - m}}, \qquad m = \max_j s_j$$

is exact (numerator and denominator both scaled by $e^{-m}$) and bounds
every exponent by 0. Every attention kernel computes the running or global
max first; FlashAttention's "online softmax" (Ch. 5) is the streaming
version of this same identity.

## 1.4 MHA → MQA → GQA: shrinking the KV footprint

In classic multi-head attention (MHA), all $H$ heads have private $K,V$
projections. At inference this is pure cost: the KV cache (Ch. 2–3) scales
with the number of *KV heads*, and cache size — not compute — is the
binding constraint at long context.

- **MQA** (Shazeer, 2019): one shared $K,V$ head for all queries. Cache
  shrinks $H\times$; quality measurably dips.
- **GQA** (Ainslie et al., 2023): $H_{kv}$ KV heads with $H/H_{kv}$ query
  heads sharing each — the accepted sweet spot. Query head $h$ reads KV head
  $\lfloor h / (H/H_{kv}) \rfloor$.

GQA is *free* at inference time (it's a training-time choice), but the
engine must honor the head-group mapping and — a real-world trap — must not
assume $d = H \cdot d_h$: Gemma 3 1B has $d = 1152$, $H = 4$, $d_h = 256$,
so $H\,d_h = 1024 \ne d$ and $W_Q \in \mathbb{R}^{1024 \times 1152}$,
$W_O \in \mathbb{R}^{1152\times 1024}$. Keep $q_{\dim} = H d_h$ and
$kv_{\dim} = H_{kv} d_h$ as first-class quantities.

**QK-norm (Gemma 3, Qwen 3).** Each head's $q$ and $k$ vectors are
RMSNormalized (over $d_h$, with their own learned $(1{+}w)$ weights) before
rotation. This bounds $\|q\|\|k\|$ and hence the score magnitude — a
softmax-saturation guard that replaced Gemma 2's soft-capping. Order
matters: **norm, then RoPE** (rotation is an isometry, so norm-then-rotate
preserves the normalization; rotate-then-norm would not be what training
did).

## 1.5 RoPE: rotary position embeddings, derived

Attention as defined is permutation-equivariant — it has no notion of
position. RoPE (Su et al., 2021) injects *relative* position through
rotation, and deriving it makes the implementation constraints obvious.

Pair up the head dimensions: treat $(q_{2j}, q_{2j+1})$ as a complex number
$\tilde q_j$. Assign frequency $\theta_j = \Theta^{-2j/d_h}$ to pair $j$
(a geometric ladder from $1$ down to $\approx \Theta^{-1}$; $\Theta$ is the
"base", classically $10^4$). Rotate each pair by (position × frequency):

$$\tilde q_j^{(t)} = \tilde q_j\, e^{i t\theta_j}, \qquad \tilde k_j^{(p)} = \tilde k_j\, e^{i p\theta_j}$$

The attention score contribution of pair $j$ is the real inner product,
which for complex numbers is $\mathrm{Re}(\tilde q \overline{\tilde k})$:

$$\mathrm{Re}\!\left(\tilde q_j e^{it\theta_j}\, \overline{\tilde k_j e^{ip\theta_j}}\right) = \mathrm{Re}\!\left(\tilde q_j \overline{\tilde k_j}\, e^{i(t-p)\theta_j}\right)$$

The absolute positions cancel: **scores depend only on the offset $t - p$**.
That is the entire point — a per-token, cache-friendly operation (each
token's $q,k$ rotated once, by its own position) that yields relative
attention. It is also why the KV cache stores *rotated* keys: $k_p$'s
rotation is fixed at birth and never has to be revisited.

The multi-scale frequency ladder gives high-frequency pairs (small $j$)
sensitivity to adjacent-token offsets and low-frequency pairs sensitivity to
document-scale offsets. Extending context beyond training length is done by
manipulating this ladder (raising $\Theta$, "NTK" scaling, or **linear
scaling** — dividing positions by a factor, which kllm deliberately refuses
until implemented rather than silently ignoring).

**The convention trap.** Two layouts exist for "pairs":

- *Interleaved:* pairs are $(x_0,x_1), (x_2,x_3), \dots$ — the paper's
  presentation, used by GPT-NeoX-style implementations.
- *Half-split (`rotate_half`):* pairs are $(x_j, x_{j+d_h/2})$ — what
  HuggingFace Llama/Gemma actually do:
  $x' = x\cos + \mathrm{rotate\_half}(x)\sin$ with
  $\mathrm{rotate\_half}(x) = [-x_{d_h/2:},\, x_{:d_h/2}]$.

Both are valid RoPEs; they are **not interchangeable** for a given
checkpoint, because training baked one in. Componentwise, half-split is:

$$x'_j = x_j \cos(t\theta_j) - x_{j+d_h/2}\sin(t\theta_j), \qquad x'_{j+d_h/2} = x_{j+d_h/2}\cos(t\theta_j) + x_j\sin(t\theta_j)$$

**Dual bases (Gemma 3).** Sliding-window layers (§1.7) use
$\Theta = 10^4$; full-attention layers use $\Theta = 10^6$. The larger base
flattens the frequency ladder so low-frequency pairs still discriminate at
32K+ offsets — needed only where attention can actually reach that far.
kllm learned the hard way (journal: the transformers-5 `rope_parameters`
migration) that these values must come from the checkpoint's runtime config,
not from defaults.

## 1.6 The gated MLP

The block's second sublayer is a position-wise two-layer network holding
roughly $2/3$ of a dense model's parameters. Modern decoders use the
**gated** variant (GLU family, Shazeer 2020):

$$\mathrm{MLP}(x) = W_{down}\,\big(\phi(x W_{gate}^\top) \odot x W_{up}^\top\big)$$

with $W_{gate}, W_{up} \in \mathbb{R}^{f\times d}$,
$W_{down} \in \mathbb{R}^{d\times f}$. The elementwise product lets one
branch gate the other — a multiplicative interaction a plain MLP lacks.
Activation $\phi$:

- **SwiGLU / SiLU** (Llama family): $\phi(z) = z\,\sigma(z) = z/(1+e^{-z})$.
- **GELU-tanh** (Gemma): $\phi(z) = \tfrac{z}{2}\big(1 + \tanh(\sqrt{2/\pi}\,(z + 0.044715 z^3))\big)$ —
  the tanh *approximation* of $z\,\Phi(z)$ specifically, constant
  $\sqrt{2/\pi} = 0.7978845608\ldots$; using exact-erf GELU against a
  checkpoint trained with the approximation is another silent-wrongness
  source.

Inference-wise the MLP is three GEMMs and one cheap elementwise kernel; at
decode (one token) each GEMM is a matrix–vector product bound entirely by
reading its weights (Ch. 5).

## 1.7 Attention locality: sliding windows and hybrid stacks

Full causal attention costs $O(T)$ work and cache-reads per token per layer.
**Sliding-window attention** (Mistral; Gemma 2/3) restricts layer $\ell$'s
attention to the last $W$ positions: $p \in [\max(0, t{-}W{+}1),\, t]$.
Information still propagates beyond $W$ through depth — after $\ell$ local
layers a token can be influenced by positions up to $\ell W$ back
(receptive-field growth, exactly as in dilated CNNs) — but *retrieval* of
distant exact content degrades, so hybrids interleave: **Gemma 3 uses 5
sliding layers per 1 full layer** ($W{=}512$ at 1B/4B scale). Consequences
the engine must implement:

- per-layer masks (window start $\max(0, t{-}W{+}1)$ — inclusive-of-self,
  an easy off-by-one),
- per-layer RoPE base (§1.5),
- and, at scale, per-layer KV *retention* policy — a sliding layer only
  ever reads the last $W$ entries, so its cache can be a ring buffer of
  size $W$ (kllm keeps full KV and windows the reads; the ring-buffer
  optimization is future memory work).

The layer assignment (`layer_types`) must be taken from the checkpoint
verbatim: the (l+1) mod N formula differs across library versions —
kllm passes explicit per-layer flags across its ABI for exactly this reason.

## 1.8 The LM head and the residual-stream picture

After the last block, a final RMSNorm and a projection to vocabulary logits:

$$z = \mathrm{Norm}(x_T)\, W_{lm}^\top \in \mathbb{R}^{V}$$

Only the **last position's** logits are needed for generation, so a good
engine computes the LM head for one row, not $T$ — at $V = 262144$ (Gemma 3)
this projection alone reads $V d \approx 1.2$ GB of fp32 weights per token
and can dominate small-model decode.

The useful mental model of the whole stack, due to the mechanistic-
interpretability literature: the residual stream is a **shared memory bus**
of width $d$; each attention head and MLP reads from it (through the norms),
computes, and *adds* its result back. Layers communicate only through this
bus. This picture explains why per-layer activation diffs (kllm's oracle)
localize bugs so well: the bus state after block $\ell$ is a complete
summary of everything up to $\ell$ — the first block whose output diverges
contains the bug, full stop.

## 1.9 Parameter and FLOP accounting

Per block (dense, GQA):
$$\underbrace{d\,H d_h + 2\,d\,H_{kv} d_h + H d_h\, d}_{\text{attention}} \;+\; \underbrace{3\,d f}_{\text{gated MLP}} \;+\; \text{(norm vectors, negligible)}$$

plus $2 V d$ for embeddings + head (or $Vd$ tied). A forward pass over one
token does $\approx 2 \times (\text{params touched})$ FLOPs (one multiply +
one add per weight), a fact used constantly for roofline estimates (Ch. 5):
**per decoded token, a dense model does about $2N$ FLOPs and — absent
batching — reads all $N$ weights once.** That sentence is the seed of every
optimization in the rest of these notes.

## Where this lives in kllm

- The whole pass: `backend/model.cu` — `te_forward_batch` (Llama branch and
  Gemma 3 branch), with kernels `rmsnorm_kernel` (with `plus_one`),
  `rope_kernel` (half-split, double-precision angles), `attn_paged_*`
  (causal + windowed, GQA mapping, configurable scale), `silu_mul_kernel`,
  `gelu_tanh_mul_kernel`, `scale_kernel` (embedding scaling).
- Convention validation: `engine/oracle` diffs per-layer activations
  against HF at $10^{-4}$; the journal documents two real bugs it caught
  (hidden-states layout; the rope-theta config migration).
- Architecture plumbing: `models/config.go` (`rope_parameters`,
  `layer_types`, `query_pre_attn_scalar`), `tools/convert_hf.py`.

## Reading list

1. Vaswani et al., *Attention Is All You Need*, 2017 — the origin; read for the residual+norm skeleton and the $\sqrt{d}$ scalings.
2. Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019.
3. Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 — read §3 for the derivation reproduced above.
4. Shazeer, *GLU Variants Improve Transformer*, 2020.
5. Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, 2023.
6. Gemma 3 technical report, 2025 — hybrid attention, qk-norm, dual rope bases.
7. Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021 — the residual-stream-as-bus picture.
