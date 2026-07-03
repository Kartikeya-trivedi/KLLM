# kllm theory — a graduate primer on LLM inference

This is the deep version of [../THEORY.md](../THEORY.md): every concept the
engine uses, treated the way you'd want it treated on day one of a PhD on
inference systems — precise definitions, the actual math, derivations where
they carry insight, complexity and hardware analysis, the trade-off space,
and pointers into both the literature and this codebase.

**How to read it.** The chapters build on each other in order, but each is
self-contained enough to enter directly. Every chapter ends with two
sections: *Where this lives in kllm* (the theory made executable, with file
pointers and measured numbers from [../JOURNAL.md](../JOURNAL.md)) and
*Reading list* (the primary sources, in the order worth reading them).

| # | Chapter | One-line scope |
|---|---------|----------------|
| 1 | [The transformer forward pass](01-transformer-forward-pass.md) | Embeddings, RMSNorm, attention (MHA/GQA), RoPE derivation, gated MLPs, residual stream, architecture variants (Llama vs Gemma 3) |
| 2 | [Autoregressive decoding](02-autoregressive-decoding.md) | The factorization, prefill/decode duality, the KV cache and its correctness argument, sampling, speculative decoding with the acceptance proof |
| 3 | [Memory: the KV cache and paging](03-memory-paged-kv.md) | Cache sizing math, fragmentation, PagedAttention as virtual memory, block-size trade-offs |
| 4 | [Batching and scheduling](04-batching-and-scheduling.md) | Why batching is near-free throughput, static vs continuous batching, dual admission budgets, latency metrics and Little's law |
| 5 | [The GPU performance model](05-gpu-performance-model.md) | Roofline formalism, arithmetic-intensity derivations, coalescing, occupancy, reductions, launch overhead, CUDA graphs, a Turing-vs-Ampere case study |
| 6 | [Weight quantization](06-quantization.md) | Number formats, group-wise int4 math, quantization error analysis, dequant-fused matmul, validation methodology |
| 7 | [Mixture of Experts](07-mixture-of-experts.md) | Conditional computation, both routing families, load balancing, the permute→grouped-GEMM→scatter dataflow |

**Notation.** Throughout: $d$ = hidden (model) dimension, $L$ = number of
layers, $H$ = attention heads, $H_{kv}$ = KV heads, $d_h$ = head dimension,
$V$ = vocabulary size, $T$ = sequence length, $B$ = batch size (sequences),
$f$ = MLP intermediate dimension. Weights are row-major $[\text{out},
\text{in}]$ (the HuggingFace convention), vectors are rows, and $x W^\top$
denotes a projection. All complexity counts are per layer unless stated.

**The one-sentence thesis of the whole field**, which every chapter is a
variation on: *transformer inference is a memory-movement problem wearing a
compute costume* — the interesting engineering is deciding which bytes to
move, when, and how few times.
