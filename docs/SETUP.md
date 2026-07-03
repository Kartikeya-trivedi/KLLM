# Setup guide — from bare machine to passing test suite

This walks a fresh machine to a working kllm build: CUDA toolchain, Go,
Python tooling, the cloud GPU lab, and the verification ladder that proves
each layer works before you depend on the next. Every step and version here
is what this repo was actually developed and verified against; the
troubleshooting section at the end is real errors this project hit, with
their fixes.

There are two supported environments:

| | Windows (lab box) | Linux (incl. Modal containers) |
|---|---|---|
| GPU | any CUDA GPU (developed on a GTX 1650, sm_75) | any CUDA GPU (verified on A10 sm_86, A100 sm_80) |
| backend build | nvcc + **MSVC** → `toyengine_backend.dll` | nvcc + gcc → `libtoyengine.so` |
| Go↔CUDA bridge | `syscall.LoadDLL` (**no cgo**) | cgo |

You don't need a big GPU locally: the whole correctness suite runs on a
4 GB card, and real models run on cloud GPUs via Modal (§5).

---

## 1. Windows setup

### 1.1 NVIDIA driver + CUDA Toolkit 12.x

1. Install a current NVIDIA driver (GeForce/RTX driver is fine).
2. Install the **CUDA Toolkit 12.x** (this repo used 12.8) from
   https://developer.nvidia.com/cuda-downloads — the default install adds
   `nvcc` to `PATH`.
3. Verify both halves — the driver and the compiler are separate things:

```powershell
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
nvcc --version
```

Note your GPU's **compute capability** from the first command (e.g.
`7.5` → build arch `sm_75`; `8.6` → `sm_86`). You'll pass it to the build.

### 1.2 Visual Studio 2022 with the C++ workload (required)

On Windows, **nvcc cannot compile without MSVC's `cl.exe`** as host
compiler — MinGW/GCC is not a supported nvcc host on Windows. Install
**Visual Studio 2022 Community** with the workload **"Desktop development
with C++"** (that's the part that installs `cl.exe` and vswhere).

You do *not* need to run anything from a "Developer PowerShell":
[`scripts/build_backend.ps1`](../scripts/build_backend.ps1) locates the
newest MSVC via `vswhere` automatically.

### 1.3 Go 1.26+

Install from https://go.dev/dl/ and verify `go version`.

**The cgo note (important, counterintuitive):** you do **not** need any
GCC/MinGW on Windows. kllm's Windows loader talks to the CUDA DLL through
`syscall.LoadDLL`/`GetProcAddress` — zero cgo — precisely because cgo on
windows/amd64 would require a 64-bit MinGW-w64 toolchain and nvcc would
*still* require MSVC, leaving you maintaining two C toolchains. If you have
an old 32-bit MinGW on PATH (as this lab box did), it is simply unused.

### 1.4 Python 3.11+ for the offline tools

Python is **never in the serving path** — it powers the correctness oracle
and model tooling only.

```powershell
pip install numpy torch transformers safetensors huggingface_hub matplotlib
```

- `numpy` — test-model generators, quantizer, plots (hard requirement).
- `torch` + `transformers` — the HuggingFace oracle (`gen_reference.py`).
  CPU-only torch is sufficient for the tiny test models; a CUDA torch just
  makes reference dumps faster. **Gemma 3 oracles need
  `transformers >= 4.50`** — if yours is older, generate Gemma dumps on
  Modal instead (§5), which is what this repo does.
- `matplotlib` — `tools/plot_kernels.py` only.
- `wandb` (optional) — `tools/wandb_bench.py` logs tok/s per kernel version
  if installed + `wandb login`; without it the script still runs and prints.

### 1.5 Build and verify (the ladder)

Run these in order from the repo root; each rung proves a layer the next
rung depends on.

```powershell
# Rung 1 — toolchain: nvcc + MSVC produce the backend DLL
.\scripts\build_backend.ps1 -Arch sm_75        # your GPU's arch here

# Rung 2 — the boundary: Go -> DLL -> CUDA context -> kernel -> verified
go run .\cmd\smoke                              # expect "PASS"

# Rung 3 — generate the tiny test models + HF oracle dumps (one-time)
python tools\make_test_model.py
python tools\gen_reference.py --model testmodels\tiny-llama `
  --prompts tools\prompts_tiny.txt --out refdumps\tiny-llama `
  --raw-ids --max-new-tokens 16 --dtype float32 --device cpu
python tools\quantize_w4.py --model testmodels\tiny-llama `
  --out testmodels\tiny-llama-w4 --group-size 32
python tools\gen_reference.py --model testmodels\tiny-llama-w4-dq `
  --prompts tools\prompts_tiny.txt --out refdumps\tiny-llama-w4dq `
  --raw-ids --max-new-tokens 16 --dtype float32 --device cpu
python tools\make_test_moe.py
python tools\gen_reference.py --model testmodels\tiny-mixtral `
  --prompts tools\prompts_tiny.txt --out refdumps\tiny-mixtral `
  --raw-ids --max-new-tokens 16 --dtype float32 --device cpu
python tools\gen_reference_numpy_moe.py --model testmodels\tiny-mixtral-sigmoid `
  --prompts tools\prompts_tiny.txt --out refdumps\tiny-mixtral-sigmoid --max-new-tokens 16
python tools\quantize_w4.py --model testmodels\tiny-mixtral `
  --out testmodels\tiny-mixtral-w4 --group-size 32
python tools\gen_reference.py --model testmodels\tiny-mixtral-w4-dq `
  --prompts tools\prompts_tiny.txt --out refdumps\tiny-mixtral-w4dq `
  --raw-ids --max-new-tokens 16 --dtype float32 --device cpu

# Rung 4 — the whole correctness suite vs HuggingFace
go vet ./...
go test -count=1 ./...                          # every e2e* suite must pass
```

Suites whose model/dump directories are missing **skip** rather than fail
(the Gemma suite skips locally unless you generate its artifacts — fine).

### 1.6 Run it

```powershell
go run .\cmd\generate --ids "1 17 42 100 7"     # greedy decode, matches HF
go run .\cmd\serve                               # server + browser UI at http://127.0.0.1:8177
go run .\cmd\bench                               # TTFT / ITL / tok/s
go run .\cmd\wbench                              # W4 kernel attempts vs cuBLAS
go run .\cmd\loadgen --url http://127.0.0.1:8177 # throughput sweep
```

**Rebuild the DLL after any change under `backend/`** — Go does not track
`.cu` files.

---

## 2. Linux setup

Same shape, different loader. Requirements: NVIDIA driver, CUDA Toolkit
12.x, gcc (any recent; it's both nvcc's host compiler and cgo's compiler),
Go 1.26+, Python as in §1.4.

```bash
# backend (pick your arch: sm_80 A100, sm_86 A10/A6000/30-series, sm_89 L4/40-series)
mkdir -p build
nvcc -shared -O2 -lineinfo -arch=sm_86 -Xcompiler -fPIC \
  -o build/libtoyengine.so backend/*.cu -lcublas

# engine: cgo links libtoyengine.so directly (see engine/backend/backend_cgo.go;
# CGO_ENABLED=1 is the default when gcc is present)
go vet ./... && go test -count=1 ./...
go run ./cmd/smoke --backend build/libtoyengine.so
```

The cgo directives bake an rpath to `./build`, so binaries run from the
repo root find the `.so` without `LD_LIBRARY_PATH`. The reference Linux
environment is codified in [`tools/modal_lab.py`](../tools/modal_lab.py)'s
image: `nvidia/cuda:12.8.0-devel-ubuntu22.04` + Go 1.26.2 tarball +
`pip install torch transformers safetensors huggingface_hub accelerate numpy`
— if in doubt, mirror that.

---

## 3. Multi-arch note (which `-Arch` do I build?)

One binary can serve multiple GPU generations — pass several `-gencode`
flags — but the simple path is: build for the machine you're on.

| GPU | arch |
|---|---|
| GTX 16xx (Turing) | `sm_75` |
| A100 | `sm_80` |
| A10 / A6000 / RTX 30xx | `sm_86` |
| L4 / RTX 40xx | `sm_89` |
| H100 | `sm_90` |

Kernel-performance conclusions **do not transfer across archs** — this
repo measured the same W4 kernel losing on Turing and winning on Ampere
(journal, kernel loop) — so re-run `cmd/wbench` after changing arch.

---

## 4. Real models (HF checkpoint → engine format)

```bash
python tools/convert_hf.py --hf TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out testmodels/real
go run ./cmd/serve --model testmodels/real --max-seq 2048
```

Supported archs: plain Llama (Llama 2/3, TinyLlama, Sarvam-1, Mistral
without sliding) and **Gemma 3 text-only** (`gemma-3-1b`). The converter
*refuses* what the kernels don't implement (Qwen2 QKV-bias, multimodal
Gemma, rope scaling) rather than emitting silent garbage. Everything is
cast to fp32 (the current compute dtype), so budget VRAM at
`4 bytes × params + KV pool`: a 1.1B model ≈ 4.4 GB, 2B ≈ 8.8 GB. Use the
W4 pipeline (`tools/quantize_w4.py`) to shrink weights ~8×. Gated repos
(meta-llama, google/gemma) need `huggingface-cli login` or `HF_TOKEN` set;
the ungated mirrors (`unsloth/gemma-3-1b-it`) need nothing.

---

## 5. Modal (cloud GPU lab) setup

Modal supplies the big GPUs: A10 (sm_86) for arch-gates and A100 for real
models. One-time:

```powershell
pip install modal
modal token new          # opens browser auth; writes ~/.modal.toml
```

Then, from the repo root (first run of each builds a container image,
~2–5 min; later runs reuse it):

```powershell
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'    # Windows: required, see §6

modal run tools/modal_lab.py::gpu_smoke          # cheapest sanity: prints the GPU
modal run tools/modal_lab.py::build_and_test     # A10: build sm_86 + FULL suite + wbench
modal run tools/modal_lab.py::validate_gemma     # A10: Gemma3 per-layer oracle gate
modal run tools/modal_lab.py::bench_model        # A100: TinyLlama end-to-end + tok/s
modal run tools/modal_lab.py::bench_model --model-id sarvamai/sarvam-1
modal deploy tools/modal_lab.py                  # public URL serving the browser UI
```

Costs are per-GPU-second; everything above runs minutes, not hours, and
the deployed web endpoint scales to zero when idle (`modal app stop
kllm-lab` tears it down). HF downloads inside `bench_model` are cached in a
Modal Volume, so repeat runs skip the download. For gated models, add a
Modal secret containing `HF_TOKEN` and attach it to the function.

---

## 6. Troubleshooting (every entry happened to this repo)

| Symptom | Cause → fix |
|---|---|
| `nvcc fatal: Cannot find compiler 'cl.exe'` | MSVC missing or not located. Install VS 2022 C++ workload; use `scripts/build_backend.ps1` (finds it via vswhere). |
| `go test` passes but your `.cu` change did nothing | The DLL/.so is stale — Go doesn't track it. Rebuild the backend, rerun. |
| `listen tcp ...: bind: ... forbidden by its access permissions` | Windows reserves port ranges (WinNAT) — 8080 is often inside one, which is why `cmd/serve` defaults to 8177. If your chosen port hits this, pick another; `netsh interface ipv4 show excludedportrange protocol=tcp` lists the reserved ranges. |
| Modal CLI crashes: `'charmap' codec can't encode character '✓'` | Windows console encoding vs Modal's ✓ glyphs. Set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` before `modal ...`. |
| Benchmarks show `0s` / quantized timings on Windows | The monotonic clock ticks ~0.5 ms. Time in aggregate (N iterations ÷ N) as `cmd/bench` does, or use CUDA events (`te_bench_matmul`). |
| `backend error 2: ... out of memory` at serve time on a model that loaded fine | KV pool oversized: pool bytes = `layers × 2 × blocks × block_size × kv_dim × 4`. Derive blocks from the model's dims (see `bench_model` in `tools/modal_lab.py`); don't hardcode. |
| Engine matches HF on sliding-window layers but diverges on full-attention layers | RoPE thetas parsed from a transformers-5 config: top-level `rope_theta` is gone, replaced by nested `rope_parameters` per layer type. Handled in `models/config.go`; if you touch config parsing, keep both paths. |
| Gemma layer types wrong / oracle diverges mid-stack | Never derive sliding/full from `sliding_window_pattern` formulas — versions disagree. Use the checkpoint's `layer_types` verbatim (`convert_hf.py` writes the runtime list; the engine passes explicit flags). |
| `transformers` errors on `torch_dtype=` or can't load `gemma3_text` | transformers ≥5 renamed it `dtype` (handled in `gen_reference.py`); Gemma 3 classes need ≥4.50 — upgrade, or generate those dumps on Modal. |
| Negative backend error prints as `4294967294` | C `int` is 32-bit: truncate `uintptr` to `int32` before sign-reading (already done in `backend_windows.go`; a trap for new syscall code). |
| A kernel is fast on one GPU and slow on another | Expected (see §3 and journal). Keep versions runtime-selectable (`te_set_kernels`) and measure per arch with `cmd/wbench`. |
| Everything builds but generation is subtly wrong | Run the oracle: `go test ./engine/e2e/ -v`. The first diverging layer index localizes the bug (per-layer dumps exist exactly for this). |

---

## 7. What "healthy" looks like

After setup, this is the steady state to expect (numbers from this repo's
journal, your hardware will vary):

- `go test -count=1 ./...` — all suites `ok`, ~15 s on a 4 GB GPU.
- `cmd/smoke` — `max abs error 0`.
- `cmd/generate` on `tiny-llama` — token-for-token identical to the HF dump.
- `cmd/wbench` — W4 v1/v2 beating fp32 cuBLAS (1.7–2.7× depending on arch).
- `modal run ...::build_and_test` — same suite green on sm_86.
