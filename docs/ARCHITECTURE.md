# Architecture — how kllm is put together

kllm is an LLM inference engine split across two languages by responsibility:

- **Go** does everything that is orchestration and bookkeeping: the HTTP
  server, the request scheduler, continuous batching, paged-KV block
  accounting, weight loading from disk, and sampling.
- **CUDA C++** does everything that touches the GPU: kernels, device memory,
  the forward pass, cuBLAS, and (planned) CUDA graphs.

They meet at a deliberately **narrow C-ABI**. This is the same architecture
Ollama uses (Go orchestrating a native GPU backend), and the split is the
whole design: Go is where concurrency, lifetimes, and I/O are pleasant;
CUDA C++ is where the math lives. Neither language reaches into the other's
job.

```
   ┌──────────────────────── Go engine (this is most of the code) ───────────────────────┐
   │                                                                                      │
   │  cmd/serve ─► server (HTTP + SSE)  ─►  engine.Scheduler  ─►  engine.Engine           │
   │                  ▲  browser UI          (goroutine +            │                    │
   │                  │  at "/"               channels,              │ engine/kv          │
   │              cmd/loadgen                 continuous             │ (block tables,     │
   │                                          batching)              │  free list)        │
   │   loader (safetensors) ─► models (config.json) ─► backend (Go side of the C-ABI)     │
   └────────────────────────────────────────────┬─────────────────────────────────────────┘
                                                 │  small control data only
                        ══════════════ C-ABI (backend/shim.h) ══════════════
                                                 │  (tokens, positions, block tables)
   ┌─────────────────────────────────────────────┴────────────────────────────────────────┐
   │                       CUDA C++ backend  (nvcc ─► .dll / .so)                            │
   │                                                                                        │
   │   shim.cu (C-ABI impl, error plumbing)   model.cu (weights, KV pool, forward_step,     │
   │                                           all kernels, cuBLAS, MoE, W4)                 │
   └────────────────────────────────────────────────────────────────────────────────────────┘
```

## The two design decisions that shape everything

### 1. The C-ABI boundary is narrow

Every crossing of the Go↔CUDA boundary is a function call with a fixed C
signature (see [`backend/shim.h`](../backend/shim.h)). The rule: **small
control data crosses; big tensors never do.** Weights, the KV cache, and all
activations live on the GPU for the model's whole lifetime. What crosses per
step is a few kilobytes at most — token ids, positions, block tables, and
the resulting logits.

The load-bearing consequence is `te_forward_batch`: **one call runs the
entire forward pass for a whole batch of sequences** — all layers, all
kernels, launched from C++ — and returns just the last-token logits. Go
never calls the backend per-layer or per-kernel. This keeps the (relatively
expensive) FFI crossing off the hot path and leaves the door open for CUDA
graphs, which capture the whole decode step as one replayable unit.

The functions, grouped:

| Group | Functions | Purpose |
|---|---|---|
| Context | `te_init`, `te_device_info`, `te_last_error` | bring up CUDA, report errors per-thread |
| Model lifecycle | `te_model_create`, `te_model_load_tensor`, `te_model_load_tensor_w4`, `te_model_finalize` | build the model, upload weights (fp32 or int4), validate + allocate |
| Inference | `te_forward_batch`, `te_set_fusion` | run a batch step; toggle fused kernels |
| Correctness taps | `te_debug_set/count/size/read` | read per-layer activations back to host for the oracle |
| Bench | `te_bench_matmul` | isolated matmul timing (fp32 vs W4) |

### 2. The correctness oracle is offline

Because the reference implementation (HuggingFace Transformers) can't run
in-process next to a Go binary, correctness is checked against **dumps on
disk**. `tools/gen_reference.py` runs HF on fixed prompts and saves, per
prompt: the greedy token ids, the per-step logits, and the **per-layer
hidden states**. Go's test suite loads these and diffs.

Per-layer dumps are the key: if the engine diverges, the failing layer index
tells you exactly which kernel is wrong — you binary-search the forward pass
instead of staring at a wrong final token. The backend's debug taps
(`te_debug_*`) capture the residual stream after each layer on device and
copy it back for the diff. Tolerances: per-layer max-abs-diff `< 1e-4`,
logits `< 1e-3`, tokens exact.

This is why Phase 0 was the biggest phase — the oracle plus the loader plus
the CUDA context all had to exist before the first token.

## Request lifecycle (serving path)

What happens when a request hits the running server:

1. **HTTP in.** `POST /v1/generate` with `{"ids": [...], "max_new_tokens": N}`
   arrives at [`server/server.go`](../server/server.go). The handler calls
   `scheduler.Submit(...)` and gets back a Go channel of token events.
2. **Queued.** `Submit` puts a request on a buffered channel and returns.
   One dedicated **scheduler goroutine** owns the GPU; nothing else calls the
   backend.
3. **Batch formation (every step).** The scheduler wakes, admits queued
   requests into free batch slots, and builds one batch: new requests
   contribute their whole **prompt** (prefill), running requests contribute
   their **one** next token (decode). Admission respects two budgets — the
   max batch size and the backend's per-step token capacity.
4. **KV reservation.** For each sequence, [`engine/kv`](../engine/kv/kv.go)
   reserves enough physical KV blocks to cover its new tokens and hands back
   a block table (a small `[]int32`).
5. **One forward call.** The scheduler flattens the batch into the C-ABI's
   parallel arrays and calls `te_forward_batch` once. The backend runs the
   full network and returns each sequence's last-token logits.
6. **Sample + stream + retire.** The scheduler argmaxes each row, sends the
   token to that request's channel (→ SSE frame to the browser), and either
   keeps the sequence for the next step or, at EOS / max tokens, releases its
   KV blocks back to the pool.

Because step 3 rebuilds the batch from scratch every iteration, a new request
joins the running batch within one step and a finished one leaves it
immediately — this is **continuous (in-flight) batching**, and it is where
the Go design earns its place. Concurrent browser tabs are merged into one
GPU batch automatically.

## Portability: two loaders, one interface

The Go `backend` package defines an `impl` interface; the platform provides
the loader behind it:

- **Windows** ([`backend_windows.go`](../engine/backend/backend_windows.go)):
  `syscall.LoadDLL` + `GetProcAddress`. **No cgo** — the box's MinGW is
  32-bit and can't back cgo on amd64, and since the ABI is narrow, raw
  syscalls cover it with zero extra toolchain.
- **Linux** ([`backend_cgo.go`](../engine/backend/backend_cgo.go), build tag
  `linux`): plain cgo against `libtoyengine.so`.

Same C-ABI, same Go-facing API. The inner development loop runs on the
Windows GTX 1650 (sm_75); the target-architecture and 30B runs happen on
Linux/Ampere (sm_86) via Modal — see [modal_lab.py](../tools/modal_lab.py).

## Repo map

```
cmd/
  serve/      the inference server (engine + scheduler + HTTP/SSE + UI)
  generate/   one-shot greedy decode from raw token ids
  bench/      TTFT / ITL / tok-s microbench, fused-vs-unfused
  loadgen/    concurrent HTTP load generator, throughput-vs-concurrency
  wbench/     matmul microbench: fp32 cuBLAS vs W4 kernel
  smoke/      walking-skeleton: Go → DLL → CUDA kernel → verify
  inspect/    dump the tensors in a safetensors checkpoint

engine/
  engine.go       load a checkpoint, Sequence API, greedy Generate
  scheduler.go    continuous-batching scheduler (goroutine + channels)
  backend/        Go side of the C-ABI (windows syscall / linux cgo loaders)
  kv/             paged-KV block allocator + per-sequence block tables
  loader/         safetensors reader (single-file + sharded index)
  npy/            minimal .npy reader (for the oracle dumps)
  oracle/         shared correctness harness (diff engine vs HF dumps)
  e2e/ e2ew4/ e2emoe/ e2emoesig/ e2emoew4/   per-variant correctness suites

models/           config.json parsing → backend model config
server/           HTTP/SSE handlers + embedded browser UI (ui.html)

backend/          CUDA C++: shim.h/.cu (C-ABI), model.cu (kernels+forward),
                  common.h (error plumbing)

tools/            offline Python (never in the serving path):
                  gen_reference.py, gen_reference_numpy_moe.py,
                  make_test_model.py, make_test_moe.py, quantize_w4.py,
                  modal_lab.py

docs/             PLAN.md (the build plan), JOURNAL.md (build log),
                  ARCHITECTURE.md (this file), THEORY.md (the concepts)
```

## Why these boundaries pay off

- **One hard thing at a time.** KV correctness was nailed before MoE was
  added; W4 was validated against a dequantized oracle before its speed
  mattered. Each phase has a green gate the next phase can't silently break.
- **The engine is size-agnostic.** Nothing in the Go code knows the model is
  tiny. Moving to a 30B MoE is a loader change (dtype conversion) plus more
  VRAM — not new architecture.
- **Kernels are swappable.** Every kernel is naive-but-correct today. Because
  the oracle suites pin the numerics, the planned optimized kernels (W4
  matmul, paged FlashAttention, fused grouped-GEMM, CUDA graphs) can be
  dropped in and instantly regression-checked.
