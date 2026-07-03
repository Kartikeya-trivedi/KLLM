# Toy Inference Engine in Go + CUDA Kernels: Build Plan

Architecture: a **Go engine** (HTTP/gRPC server, scheduler, continuous batching, paged-KV bookkeeping, weight loading, sampling and spec-decode orchestration) over a **CUDA C++ backend** (all kernels, GPU memory, the forward step, cuBLAS/CUTLASS, CUDA graphs), bridged by a narrow C-ABI. This is the Ollama pattern: Go orchestrating a native GPU backend. Hard kernels are prototyped in Python/Triton first, then ported to CUDA C++ and wired into Go.

Target models: 30B-class, all MoE — Gemma 4 26B (softmax routing), Qwen 30B-class MoE e.g. Qwen3-30B-A3B (softmax top-k), Sarvam-30B (sigmoid routing + expert-bias normalization). Hardware: 3x A6000 (Ampere sm_86, 48GB, ~768 GB/s, tensor cores, no FP8) as the engine box; GTX 1650 (Turing sm_75, no tensor cores, 4GB) as an isolated CUDA kernel lab.

---

## The layered picture

- Go side: HTTP/gRPC server with SSE token streaming; request queue (channels); scheduler + continuous batching (goroutines); paged-KV block tables + free list; safetensors weight loader; sampling decisions; spec-decode accept/reject loop.
- C-ABI boundary (narrow): a handful of C functions exported from a shared lib built by nvcc.
- CUDA C++ side: CUDA context; GPU memory (weights, KV block pool); all kernels; cuBLAS (dense GEMM) and CUTLASS (custom matmul cores); a `forward_step` that turns a batch descriptor into logits; CUDA graphs to capture and replay the decode step.
- Python `tools/` (offline, NOT in the serving path): reference generation (HF dumps for correctness) and Triton prototypes of the hard kernels.

---

## The two decisions that define the Go build

### 1. Narrow the FFI boundary
Expose a small C-ABI: `init`, `load_weights`, `alloc_kv_blocks` / `free_kv_blocks`, `forward_step(batch_desc) -> logits or token_ids`, `sample`. Keep all big tensors on-device; cross the boundary with small control data only (token ids, positions, block tables, seq lens). Minimize crossings: **one `forward_step` call per decode step for the whole batch, not per layer.** Capture the decode step as a **CUDA graph** and replay it, so per-launch and per-call overhead is paid once. Go stays at orchestration; C++ owns "batch in, logits out."

(Windows adaptation: on the Windows lab box the boundary is crossed via `syscall.LoadDLL` — no cgo, no MinGW-w64 needed. On the Linux A6000 engine box it's plain cgo against the `.so`. Same C-ABI, two thin Go loaders behind one interface — see `engine/backend/`.)

### 2. Move the correctness oracle offline
No in-process HF diffing. Instead: `tools/gen_reference.py` dumps HF's final logits **and per-layer activations** for fixed prompts to disk; Go's correctness test loads them and compares. Per-layer dumps let you binary-search exactly which kernel/layer diverges. Every hard kernel is validated twice: **Triton-vs-torch** in Python, then **CUDA-vs-dumped-reference** in Go. This is the whole reason Phase 0 exists — do not skip it.

---

## Stack + hardware

- Go: engine, server (net/http or gRPC), scheduler, KV bookkeeping, loader, sampling orchestration.
- CUDA C++ (nvcc -> `.dll` on Windows / `.so` on Linux): kernels, GPU memory, forward step, cuBLAS, CUTLASS, CUDA graphs.
- FFI: hand-rolled shim (syscall on Windows, cgo on Linux). Not `gorgonia/cu`.
- Python (offline tools only): reference dumps + Triton kernel prototypes. Never in the serving path.
- Tokenizer: a Go tokenizer lib or bind HF `tokenizers` (Rust). Plumbing — keep it a black box.
- Weight format: safetensors (JSON header + raw bytes; parse in Go, upload via the shim).
- GTX 1650 (this box): isolated CUDA kernel microbenchmarks (SIMT fundamentals, coalescing, shared-mem tiling, occupancy, DP4A int8). No tensor cores, 4GB — cannot run the engine on a 30B model. A scratchpad to get a kernel's algorithm right, then move to the A6000. Small dense models for Phase 0 spine dev DO fit here.
- A6000 box: the engine, the 30B models, all tensor-core / quantization kernels.
- Reference projects to read: Ollama (Go + ggml over cgo — this architecture), llama.cpp server (C++ reference for kernels and the forward loop).

Model progression: small dense (spine) -> 30B dense (Gemma 31B) -> 30B MoE (Gemma 26B / Qwen MoE / Sarvam-30B). Engine code is size-agnostic once written. Don't fight MoE routing while getting the KV cache correct.

---

## Phases (each independently demoable)

### Phase 0 — Correctness spine (bigger in Go; budget for it)
- Goal: Go greedy-decodes a tiny dense model and matches dumped HF tokens.
- Build (Go): safetensors loader; tokenizer; batch descriptor; greedy sampling (copy logits to host for now).
- Build (CUDA C++): CUDA context + shim [DONE]; `forward_step` using cuBLAS for the matmuls and simple/naive kernels for elementwise; contiguous KV.
- Build (Python): `gen_reference.py` dumping logits + per-layer activations [SKELETON DONE].
- Kernel: none custom yet (cuBLAS + naive elementwise). Attention: a straightforward kernel or cuBLAS path; correctness over speed.
- Deliverable: `go run ./cmd/serve --prompt "..."` emits correct text; a Go test diffing logits vs the HF dump over N prompts; per-layer diffs pass.
- Done when: top-1 tokens match HF greedy for ~50 prompts, logit MSE under threshold, and each layer's activations match the dump.
- Note: biggest phase. Loader + tokenizer + CUDA context + cuBLAS forward + the offline oracle before the first token. Start with a 2-layer test config, then a sub-1B model, before anything 30B.

### Phase 1 — First fused kernels + roofline
- Goal: measurable per-token speedup; learn Nsight; wire Go metrics.
- Build (Go): metrics — TTFT / ITL / tok/s.
- Kernel (CUDA C++, written directly): fused RMSNorm + residual, then RoPE, then SiLU/GeGLU. Validate each vs the dumped reference; measure the delta.
- Deliverable: before/after ITL, Nsight traces, a bytes-moved / launch-overhead note.
- Done when: measurable ITL win and you can state your roofline position.

### Phase 2 — Paged KV (clean Go/CUDA split)
- Goal: the engine's memory core.
- Build (Go): block-table allocator, per-sequence tables, free list.
- Build (CUDA): the block memory pool (one big `cudaMalloc`), the paged-attention gather kernel that reads block tables. Go passes the (small) block table to the kernel each step.
- Deliverable: many sequences / long context without the contiguous-KV OOM; VRAM utilization shown.
- Done when: more concurrent sequences fit than contiguous KV allowed, outputs still correct.

### Phase 3 — Scheduler + continuous batching (Go's payoff)
- Goal: static batch -> continuous/in-flight batching.
- Build (Go): a request goroutine per stream; a channel-based queue; one batching goroutine that pulls pending requests, forms/updates the batch, calls `forward_step`, and scatters next tokens back to per-request channels; add/remove sequences each step. HTTP/SSE streaming.
- Kernel: none new.
- Deliverable: throughput-vs-batch-size curve; p99 under a Go load generator with mixed lengths; token streaming over HTTP.
- Done when: tok/s scales with batch, p99 stays sane, correctness holds under mixed batches.

### Phase 4 — W4A16 kernel (Triton-prototype-then-port; the big Ampere win)
- Goal: the main decode tok/s lever.
- Build (Python/Triton): prototype the int4 dequant-fused-matmul; validate vs torch; autotune tiling.
- Build (CUDA C++): port it (CUTLASS core), using the Triton kernel as the tiling spec; validate in Go vs dumped reference.
- Build (Go): int4 weight loading (AWQ / GPTQ / compressed-tensors), per-group scales.
- Deliverable: ~3-4x decode tok/s vs bf16 at matched quality; quality validated vs reference.
- Done when: speedup lands and quality holds within threshold.

### Phase 5 — MoE routing + fused grouped-GEMM (portfolio centerpiece)
- Goal: serve all three 30B MoE models; kernel general across routing families.
- Build (Python/Triton first): quantized grouped-GEMM MoE prototype; then port to CUDA C++.
- Routing: top-k selection (cheap — a small kernel or Go-side), token permutation/gather, grouped-GEMM, un-permute. One interface over softmax (Gemma/Qwen) and sigmoid + expert-bias-norm (Sarvam).
- Deliverable: Gemma / Sarvam / Qwen 30B served + benchmarked; sigmoid-vs-softmax generality demonstrated; INT4 Sarvam-30B with validated Indic quality.
- Done when: correct + fast across the three.

### Phase 6 — Own attention kernel + spec decode (learning + stretch)
- Goal: a FlashAttention-2-style tiled kernel; speculative decoding.
- Build (CUDA C++): tiled attention, online softmax, shared-mem tiling, GQA-aware.
- Build (Go): the spec-decode orchestration — manage draft + target `forward_step` calls and the accept/reject loop; the verify step's kernel is CUDA.
- Deliverable: attention kernel vs FA2 (slower — explain why; that's the lesson); acceptance-rate + speedup for spec decode.
- Done when: correct attention kernel + working spec decode with measured acceptance.

---

## Sequencing

- Weeks 1-2: Phase 0. Bigger than a Python version — don't rush it; it's the CUDA-context + loader + oracle foundation. (Started 2026-07-03.)
- Week 3: Phase 1.
- Weeks 4-5: Phase 2-3 (systems core; Go shines in Phase 3).
- Weeks 6-7: Phase 4-5 (kernel payoff; prototype-then-port).
- Week 8+: Phase 6 (timebox it — highest effort, lowest throughput return).

---

## Rules

- Narrow the FFI boundary: small control data across, big tensors stay on-device; one `forward_step` per step; CUDA graphs to replay.
- Two-stage kernel correctness: Triton-vs-torch, then CUDA-vs-dumped-reference. Per-layer dumps to locate divergence.
- Correctness gate every phase before believing any speedup. One hard thing at a time (don't add MoE while debugging KV).
- cuBLAS for dense GEMM, CUTLASS for custom matmul cores. Don't hand-write dense GEMM; don't expect to beat FA2.
- Profile before optimizing — Nsight, not intuition, picks the next kernel.
- Ampere: int4/int8 quant kernels, FA2-style tiling. No FP8 (no FP8 tensor cores), no FA3 (Hopper-only).
- Keep the tokenizer, and initially attention, as black boxes.
