# 7 · Mixture of Experts

## 7.1 Conditional computation: decoupling parameters from FLOPs

In a dense transformer, every parameter participates in every token —
capacity and per-token compute are locked together. **Mixture of Experts**
breaks the lock in the FFN (where ~2/3 of parameters live): replace the
single MLP with $E$ parallel expert MLPs plus a learned **router** that
sends each token to only $k \ll E$ of them:

$$y = \sum_{e \in \mathcal{T}_k(x)} g_e(x)\, \mathrm{MLP}_e(x)$$

Total parameters scale with $E$; per-token FLOPs scale with $k$. A
"30B-A3B" model (30B total, ~3B active) prices like a 3B model in compute
while representing like something much larger — which is why the 30B-class
open models kllm targets (Qwen3-MoE, Sarvam-30B-class, Mixtral lineage)
are MoE. The inference catch: **all $E$ experts' weights must be resident**
(any token may route anywhere), so MoE trades FLOPs for memory capacity —
and at batch 1 decode, where weights-read is the bound (§2.2), the
*bandwidth* cost per token is only the $k$ *activated* experts' weights:
MoE is bandwidth-priced at "active params," making it unusually
decode-friendly. Batched, this advantage erodes: with enough concurrent
tokens, *every* expert has customers each step, and total weight traffic
approaches dense-of-total-size.

## 7.2 Routing, family one: softmax top-k (Mixtral / Qwen lineage)

A linear router $r = x W_{router}^\top \in \mathbb{R}^E$ scores experts;
then:

$$p = \mathrm{softmax}(r), \qquad \mathcal{T}_k = \mathrm{top\text{-}k}(p), \qquad g_e = \frac{p_e}{\sum_{e' \in \mathcal{T}_k} p_{e'}}\ \ (e \in \mathcal{T}_k)$$

Softmax over **all** experts, select the top $k$, **renormalize** the
selected weights to sum to 1 (Mixtral's `norm_topk_prob` behavior; some
models skip the renorm — a checkpoint-specific flag worth treating as
config). Selection is monotone in the raw logits, so top-k can be computed
on logits or probabilities interchangeably; the *weights* cannot.

## 7.3 Routing, family two: sigmoid + bias-corrected selection (DeepSeek-V3 / Sarvam lineage)

The softmax couples experts (scores must compete for one unit of mass).
The newer family scores each expert *independently* with a sigmoid, and —
the subtle part — uses **different quantities for selection and for
weighting**:

$$\sigma_e = \mathrm{sigmoid}(r_e), \qquad \mathcal{T}_k = \mathrm{top\text{-}k}(\sigma_e + b_e), \qquad g_e = \frac{\sigma_e}{\sum_{e' \in \mathcal{T}_k} \sigma_{e'}}$$

$b_e$ is a per-expert **bias used only for selection** (never in the
output weights), adjusted during training to balance load (§7.4). Getting
this wrong in either direction — selecting without the bias, or weighting
*with* it — produces a model that mostly works and is quietly wrong: the
class of bug only a reference oracle catches, and the reason kllm validates
this family against an independent reference implementation. Some
checkpoints add a routed-scaling factor multiplying $g$; treat as config.

An engine that serves both families needs exactly one switch: the routing
kernel (scores → selection metric → weights). Everything downstream —
permutation, expert GEMMs, combination — is family-agnostic.

## 7.4 Load balancing: why routers need help

Routing is a **congestion game** that training must referee: left alone,
routers collapse (a few experts win early, get more gradient, win harder),
strangling both quality (dead experts = wasted capacity) and systems
behavior (one hot expert serializes the whole layer — in expert-parallel
training/serving, the load imbalance is a straggler problem). Mechanisms,
in historical order:

- **Auxiliary losses** (Shazeer 2017; Switch): penalize
  $E \sum_e f_e \bar p_e$ (fraction routed × mean router prob) — pushes
  toward uniform utilization but perturbs the main objective.
- **Capacity factors** (GShard/Switch): hard cap tokens/expert; overflow
  is dropped or overflows to the next choice — a training-time device that
  inference engines mostly ignore (serve what the router says).
- **Aux-loss-free bias** (DeepSeek-V3): the $b_e$ of §7.3 — a controller,
  not a loss: after each batch, nudge $b_e$ down for overloaded experts
  and up for starved ones. Balancing pressure lives entirely in
  *selection*, leaving weights and gradients clean. This is why the bias
  exists in the checkpoint and why it participates in selection only.

At inference the biases are frozen constants — but the *statistics* they
shaped matter operationally: expected tokens/expert per batch ≈
$\text{batch tokens} \times k / E$, with variance that determines how
ragged the grouped GEMM (§7.5) runs.

## 7.5 The dataflow: permute → grouped GEMM → un-permute

Per MoE layer, the token batch (size $n$) becomes $nk$ (token, expert,
weight) assignments. The computational problem: run $E$ *different* weight
matrices over $E$ *variable-size* disjoint row sets. The standard dataflow:

1. **Route:** small GEMM ($n \times E$) + top-k kernel → per-token expert
   ids and weights.
2. **Permute (gather):** sort assignment rows by expert so each expert's
   inputs are contiguous; build the inverse mapping. (Sorting by a small
   key = counting sort: histogram → prefix-sum offsets → scatter.)
3. **Grouped GEMM:** for each expert with $n_e > 0$ rows, its gated-MLP
   GEMMs over rows $[o_e, o_e + n_e)$. Naive = a loop of $E$ GEMM calls
   (kllm's v0 — correct, launch-heavy); production = a single **grouped
   GEMM** kernel (CUTLASS grouped mode) taking an array of problem shapes,
   eliminating per-expert launch overhead and evening out raggedness.
4. **Un-permute (weighted scatter):** route each output row back to its
   token, multiplied by $g$; multiple experts add into the same token —
   either atomics (kllm: fp32 `atomicAdd`, ~1e-7 nondeterminism, inside
   test tolerance) or a second sort by token for determinism.

Decode-batch-1 degenerates gracefully: $n{=}1$, $k$ assignments, $k$ tiny
GEMV-like expert calls — bandwidth-priced at active params, per §7.1.

**Quantization composes.** Expert FFNs are ordinary matrices; group-wise
int4 (Ch. 6) applies per expert with the router kept in fp32 (it's tiny
and decisions are sensitive to its precision). In kllm this composition
cost zero new code — expert GEMMs dispatch through the same
quant-aware matmul as dense weights — and is validated by the same
dequantized-twin oracle. This is the Sarvam-30B-INT4 serving path in
miniature: sigmoid routing + INT4 experts + paged KV + continuous batching.

## 7.6 Systems costs an engine must budget

- **Memory:** all $E$ experts resident (§7.1) — int4 quantization is what
  makes 30B-total MoE fit prosumer/single-GPU VRAM.
- **A host round-trip (kllm's current concession):** building the
  permutation on the host costs a device→host copy of the top-k arrays per
  MoE layer per step — small bytes, but a synchronization that also breaks
  CUDA-graph capture (§5.5). The fix is a device-side counting sort;
  it's the top MoE item on the optimization list.
- **Raggedness:** per-expert row counts vary per step; grouped-GEMM
  efficiency and (multi-GPU) expert-parallel balance both degrade with
  variance. At larger scale this motivates expert parallelism (experts
  sharded across GPUs, tokens all-to-all'ed to their experts and back) —
  out of scope for kllm's single-GPU engine but the natural next system.

## Where this lives in kllm

- Routing: `route_kernel` in `backend/model.cu` — one kernel, both
  families (`router_mode` 0 = §7.2 with renorm, 1 = §7.3 with
  selection-only bias).
- Dataflow: `moe_forward` — router GEMM, device top-k, host-built
  expert-sorted permutation, per-expert segment GEMMs via the quant-aware
  `mm`, weighted `scatter_add_kernel`.
- Validation: `engine/e2emoe` (softmax family vs HF Mixtral),
  `engine/e2emoesig` (sigmoid+bias family vs an independent numpy
  reference — HF has no matching class with plain attention),
  `engine/e2emoew4` (INT4 experts vs the dequantized twin).
- Test models: `tools/make_test_moe.py` (both variants share weights;
  the sigmoid variant adds the selection bias).

## Reading list

1. Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017 — the modern MoE origin; load-balancing losses.
2. Fedus, Zoph & Shazeer, *Switch Transformers*, 2021 — top-1 routing, capacity factors, the scaling case.
3. Jiang et al., *Mixtral of Experts*, 2024 — the softmax-top-2 recipe of §7.2.
4. DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024 — sigmoid routing and the aux-loss-free bias of §7.3–7.4.
5. Lepikhin et al., *GShard*, 2020 — expert parallelism and the all-to-all, for where §7.6 leads.
6. Rajbhandari et al., *DeepSpeed-MoE*, 2022 — the inference-systems view: grouped kernels, expert placement.
