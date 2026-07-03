# kllm — an LLM inference engine in Go + CUDA, from scratch

kllm serves transformer language models (dense and Mixture-of-Experts) with
its own engine: **Go** does all the orchestration — HTTP server, request
scheduler, continuous batching, paged-KV bookkeeping, weight loading,
sampling — and **CUDA C++** does everything on the GPU — kernels, device
memory, the forward pass, cuBLAS. They meet at a deliberately narrow C-ABI.
It's the Ollama pattern (Go over a native GPU backend), built up phase by
phase with a correctness gate at every step.

Nothing here is a wrapper around a serving framework — the KV cache, the
paged-attention gather, the continuous-batching scheduler, the int4
dequant-matmul, and the MoE router + grouped-GEMM are all implemented
directly.

> **New here? Read [docs/THEORY.md](docs/THEORY.md)** for the concepts
> (KV cache, paged attention, continuous batching, quantization, MoE) and
> **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for how the pieces fit and
> how a request flows through. [docs/JOURNAL.md](docs/JOURNAL.md) is the
> build log with every measured number; [docs/PLAN.md](docs/PLAN.md) is the
> original plan.

## What works today

- **Correct.** The fp32 forward pass matches HuggingFace **exactly** —
  tokens, per-step logits, and per-layer activations — validated by an
  offline oracle on every model variant.
- **Paged KV + continuous batching.** A goroutine scheduler forms a fresh
  GPU batch every decode step, admitting and retiring sequences in flight;
  ~20× throughput from batching on the lab GPU.
- **Serving.** HTTP server with SSE token streaming and a built-in browser
  playground; a load generator for throughput/latency curves.
- **Quantization.** Group-wise int4 weights with a dequant-fused matmul,
  proven to compute exactly `dequant(w)·x` against a dequantized oracle.
- **MoE, both routing families.** Softmax top-k (Mixtral/Qwen, validated vs
  HF) and sigmoid + expert-bias (Sarvam / DeepSeek-V3, validated vs a numpy
  reference), including **INT4 experts** — the Sarvam-30B-INT4 serving path
  in miniature.
- **Portable + real-Ampere gated.** Windows (syscall loader, sm_75) for the
  inner loop; Linux (cgo loader, sm_86) verified on cloud A10G via Modal.

Kernels are currently **naive-but-correct on purpose** — the whole system is
green, and kernel optimization (W4 matmul, paged FlashAttention, fused
grouped-GEMM, CUDA graphs) is the deliberate endgame, each with a measured
baseline and a correctness net that catches any regression.

## Try it in your browser (no GPU needed locally)

The engine + playground is deployed on a Modal A10G (scales to zero when
idle). Open the URL, enter token ids, watch them stream:

```
modal deploy tools/modal_lab.py     # prints your public https URL
```

The playground streams tokens over SSE from `POST /v1/generate`; multiple
tabs generating at once are merged into one GPU batch by the scheduler.

## Run it locally (Windows lab box)

Requirements: CUDA Toolkit 12.x, Visual Studio 2022 with the C++ workload
(nvcc uses cl.exe as host compiler on Windows), Go 1.26+, an NVIDIA GPU.

```powershell
# 1. Build the CUDA backend (locates MSVC via vswhere automatically)
.\scripts\build_backend.ps1 -Arch sm_75     # sm_86 on Ampere

# 2. Generate a tiny test model + its HF oracle (one-time)
python tools\make_test_model.py
python tools\gen_reference.py --model testmodels\tiny-llama `
    --prompts tools\prompts_tiny.txt --out refdumps\tiny-llama `
    --raw-ids --max-new-tokens 16 --dtype float32 --device cpu

# 3. Prove correctness against HuggingFace, then generate
go test ./...                                # all oracle suites green
go run .\cmd\generate --ids "1 17 42 100 7"  # matches HF token-for-token

# 4. Serve it + open the browser UI
go run .\cmd\serve --addr 127.0.0.1:8080
#   → http://127.0.0.1:8080   (playground at /, SSE API at /v1/generate)
```

Commands: `serve` (server + UI), `generate` (one-shot decode), `bench`
(TTFT/ITL), `loadgen` (throughput sweep), `wbench` (matmul fp32-vs-W4),
`inspect` (dump a checkpoint), `smoke` (walking skeleton).

## The build, in phases

Each phase is independently demoable and gated on matching the oracle before
its speedups count. Full detail + numbers in [JOURNAL.md](docs/JOURNAL.md).

| Phase | What it adds | Gate |
|------:|--------------|------|
| **0** | safetensors loader, CUDA context, cuBLAS forward, offline HF oracle | tokens + logits + per-layer activations match HF |
| **1** | fused add+RMSNorm, metrics, roofline analysis | fused ≡ unfused ≡ HF; measurable ITL win |
| **2** | paged KV — Go block allocator, stateless batched forward | mixed-length batch matches solo, more seqs fit |
| **3** | continuous-batching scheduler + HTTP/SSE server | concurrent streams match HF; tok/s scales with batch |
| **4** | W4 group-quant weights + dequant-fused matmul | W4 engine ≡ HF-on-dequantized-weights |
| **5** | MoE: softmax & sigmoid routing, grouped expert GEMM, INT4 experts | both routers match their oracles |
| **next** | kernel optimization: W4 matmul, paged attention, grouped-GEMM, CUDA graphs | oracle suites stay green |

## How correctness works

There's no in-process reference to diff against, so the oracle is offline:
`tools/gen_reference.py` runs HuggingFace on fixed prompts and dumps the
greedy tokens, per-step logits, **and per-layer hidden states**. The Go test
suite ([`engine/oracle`](engine/oracle/oracle.go)) loads them and diffs. The
per-layer dumps mean a divergence points straight at the failing layer — you
binary-search the forward pass instead of guessing. For quantized paths, the
quantizer also emits a dequantized-fp32 twin checkpoint, and HF runs on that,
so the kernel is proven to compute exactly `dequant(w)·x` — separating kernel
bugs from quantization error.

## Layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#repo-map) for the annotated
tree. Top level: `cmd/` (binaries), `engine/` (Go engine + backend loaders +
KV + oracle), `models/` (config parsing), `server/` (HTTP/SSE + UI),
`backend/` (CUDA C++), `tools/` (offline Python: oracles, quantizer, Modal),
`docs/`.

### Note: why there's no cgo on Windows

cgo on windows/amd64 needs a 64-bit MinGW-w64 GCC; this box has only 32-bit
MinGW, which cgo can't use. Because the C-ABI is narrow by design, the
Windows loader uses `syscall.LoadDLL` + `GetProcAddress` — zero cgo, zero
extra toolchain ([`backend_windows.go`](engine/backend/backend_windows.go)).
Linux uses a plain cgo loader against `libtoyengine.so`
([`backend_cgo.go`](engine/backend/backend_cgo.go)) behind the same Go
interface. Same ABI, two thin loaders.
