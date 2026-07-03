# 3 · Memory: the KV cache and paging

## 3.1 Sizing the problem

Per sequence, the KV cache stores two vectors of width $H_{kv} d_h$ per
layer per token:

$$\text{bytes}(T) \;=\; 2 \cdot L \cdot H_{kv}\, d_h \cdot T \cdot b$$

where $b$ is bytes per element. Worked examples ($b{=}2$, fp16, except
kllm's fp32 lab):

| model | $L$ | $H_{kv} d_h$ | bytes/token | 4K-context sequence |
|---|---:|---:|---:|---:|
| TinyLlama 1.1B | 22 | 256 | 22.5 KB (fp16) | 92 MB |
| Sarvam-1 2B | 28 | 1024 | 114 KB (fp16) | 470 MB |
| Llama-2 70B (GQA) | 80 | 1024 | 327 KB (fp16) | 1.34 GB |
| 30B-class MoE, 48L, kv 1024 | 48 | 1024 | 196 KB (fp16) | 0.8 GB |

Three readings of this table:

1. **Cache rivals weights.** Forty 4K sequences of the 30B model need
   ~32 GB of cache — comparable to the int4 weights themselves. VRAM
   budgeting is weights + cache + activations, and cache is the *elastic*
   term: it alone decides concurrency.
2. **Cache reads tax every token.** Decode at context $T$ reads
   $2 L H_{kv} d_h T b$ cache bytes per token on top of the weights; past
   the crossover $T^* = N b_w / (2 L H_{kv} d_h b)$ tokens, attention —
   not the weights — dominates bandwidth. (TinyLlama fp32: $T^* \approx$
   19K; a 70B fp16: $T^* \approx$ 4K. Long context is an attention-bandwidth
   problem.)
3. **Every KV-shrinking idea attacks a factor**: GQA shrinks $H_{kv}$
   (§1.4), sliding windows cap $T$ per layer (§1.7), KV quantization
   shrinks $b$, and paging (this chapter) attacks not the size but the
   *waste*.

## 3.2 The allocation problem: fragmentation

The naive layout — one contiguous buffer per sequence, sized for the
maximum context $T_{max}$ — reserves worst-case memory for every sequence
regardless of actual length. The waste decomposes exactly like memory-
allocator waste:

- **Internal (reservation) waste:** a sequence at length $t$ wastes
  $(T_{max} - t)/T_{max}$ of its slot. Real request lengths are heavily
  right-skewed; mean utilization in production traces is a small fraction —
  the vLLM paper measured only **20–38% of KV memory holding actual token
  state** under contiguous allocation.
- **External fragmentation:** variable-size contiguous slots leave
  unusable gaps as sequences of different lengths come and go.
- **Unknown lifetimes:** generation length is unknown at admission, so you
  cannot even right-size the reservation up front.

This is *precisely* the problem operating systems solved for process memory
sixty years ago, and the solution transfers wholesale.

## 3.3 Paged KV: virtual memory for attention

**PagedAttention** (Kwon et al., 2023 — the vLLM paper): carve the cache
pool into fixed-size **blocks** of $B_{blk}$ tokens; give each sequence a
**block table** — an ordered list of physical block ids, exactly a page
table; translate on access:

$$\text{token } p \;\mapsto\; \text{block } \mathrm{table}[\lfloor p / B_{blk}\rfloor],\ \text{offset } p \bmod B_{blk}$$

Properties, with their OS analogues:

| paging property | consequence |
|---|---|
| allocation on demand, block granularity | internal waste bounded by $< B_{blk}$ tokens per sequence (last partial block) — not $T_{max} - t$ |
| any free block serves any sequence | external fragmentation eliminated; free list is a stack |
| indirection at access time | attention kernels do one extra integer lookup per position — measured cost ≈ noise, since the table line sits in L2/L1 |
| tables are cheap | one int32 per block: a 4K sequence at $B_{blk}{=}16$ is 256 ints = 1 KB of table for up to hundreds of MB of cache |
| sharing (advanced) | identical prefixes (system prompts; beams) can map to the same physical blocks copy-on-write — the paging structure makes prefix caching *possible*; kllm hasn't built it yet |

**Block size trade-off.** Small $B_{blk}$ → less internal waste, bigger
tables, more translation lookups and (in optimized kernels) less
contiguity per memory transaction; large $B_{blk}$ → the reverse,
degenerating to contiguous allocation as $B_{blk} \to T_{max}$. Production
engines settled on 16–32 tokens; kllm uses 16. The waste bound makes the
sizing insensitive: expected internal waste is $\approx B_{blk}/2$ tokens
per *sequence*, trivial next to contiguous allocation's per-sequence
$T_{max} - t$.

**Capacity arithmetic** an engine should do at startup (kllm's harness
does): pool blocks $\times B_{blk}$ tokens = total token capacity; each
concurrent sequence consumes $\lceil t/B_{blk}\rceil$ blocks. The failure
mode of getting this wrong is instructive — kllm's first Sarvam-1 run
allocated a fixed 8192-block pool, which at Sarvam's $L{=}28$, $kv_{\dim}{=}1024$,
fp32 is $28 \cdot 2 \cdot 8192 \cdot 16 \cdot 1024 \cdot 4 \approx 30$ GB,
OOMing a 40 GB A100 that held only 8.8 GB of weights. Pool size must be a
*derived* quantity: $\min(\text{tokens needed}, \text{VRAM budget}) / (\text{bytes per block})$.

## 3.4 Placement policy vs mechanism

A clean paging design separates:

- **Mechanism** (device side): the physical pool — one large allocation,
  layout $[L][2][\text{blocks}][B_{blk}][kv_{\dim}]$ in kllm — plus kernels
  that read through tables. The device never allocates per sequence.
- **Policy** (host side): the free list, per-sequence tables,
  reserve/commit/release lifecycle, admission decisions when blocks run
  out, and eventually eviction/preemption (swap a victim's blocks to host
  memory or drop-and-recompute — the OS analogy extends to swapping; kllm
  currently fails-fast mid-stream instead, a documented simplification).

The reason to insist on this split is testability and evolution: policy is
pure data-structure code (kllm's is plain Go with unit tests, no GPU), and
policies can change — prefix sharing, priority eviction, chunked prefill —
without touching a kernel.

**Validation note.** The backend validates every table slot it will touch
*before* launching (bounds against the pool), converting caller bugs into
API errors instead of silent cross-sequence corruption — the paging
equivalent of a segfault instead of heap corruption.

## 3.5 GPU memory hierarchy: where these bytes actually live

The cache and weights live in **HBM/GDDR (global memory)** — the top of a
steep hierarchy that Chapter 5 exploits:

| level | size (A100) | bandwidth | latency | managed by |
|---|---|---|---|---|
| registers | 256 KB/SM | ~fabric | 0 cy | compiler |
| shared memory / L1 | 192 KB/SM | ~19 TB/s agg. | ~30 cy | **the kernel author** |
| L2 | 40 MB | ~7 TB/s | ~200 cy | hardware |
| HBM2e | 40 GB | 1.56 TB/s | ~400+ cy | cudaMalloc |
| host DRAM (over PCIe/NVLink) | ~TBs | 32–64 GB/s | µs | driver |

Everything in this chapter is about *not wasting* the 40 GB tier;
everything in Chapter 5 is about *not re-reading* it. The two compose: a
paged pool that fits 3× more sequences (Ch. 3) feeds a batch 3× larger
(Ch. 4), which amortizes each weight read 3× further (Ch. 5).

## Where this lives in kllm

- Policy: `engine/kv/kv.go` — LIFO free list, block tables,
  `Reserve/Commit/Release`, unit-tested (`kv_test.go`) with the OOM and
  reuse cases.
- Mechanism: `backend/model.cu` — single-`cudaMalloc` pool,
  `kv_append_paged_kernel`, `attn_paged_*` reading through tables,
  create-time validation of every used slot.
- ABI shape: `te_forward_batch` takes flattened per-sequence block tables;
  the backend holds **no** per-sequence state.
- The capacity lesson: `tools/modal_lab.py` `bench_model` derives the pool
  from model dims with an 8 GiB cap (journal: the Sarvam-1 30 GB OOM).

## Reading list

1. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023 — the vLLM paper; read §3 for the fragmentation measurements.
2. Any OS textbook's virtual-memory chapter (e.g. Arpaci-Dusseau, *OSTEP*, ch. 18–21) — the source of every idea in §3.3, worth reading *as* systems lineage.
3. NVIDIA A100 whitepaper / CUDA C++ Programming Guide §"Memory Hierarchy" — the numbers in §3.5.
4. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*, 2024 — RadixAttention: prefix sharing carried to its logical end.
