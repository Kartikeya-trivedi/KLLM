# kllm — toy LLM inference engine (Go + CUDA)

A from-scratch inference engine: **Go** does orchestration (server, scheduler,
continuous batching, paged-KV bookkeeping, weight loading, sampling); **CUDA
C++** does everything on the GPU (kernels, memory, the forward step, cuBLAS/
CUTLASS, CUDA graphs). The two meet at a deliberately narrow C-ABI
([backend/shim.h](backend/shim.h)). Full plan: [docs/PLAN.md](docs/PLAN.md).

## Status

- [x] Walking skeleton: Go → shared lib → CUDA context → kernel launch →
      results verified in Go (`cmd/smoke`)
- [x] Safetensors loader (single-file + sharded index) with tests; `cmd/inspect`
- [x] Tiny 2-layer Llama-style test model generator
      (`tools/make_test_model.py`, driven by raw token ids — no tokenizer)
- [ ] Phase 0 remainder: weight upload, cuBLAS forward pass, greedy decode
      matching `gen_reference.py` dumps; then tokenizer + sub-1B model

## Layout

```
cmd/smoke/            walking-skeleton test binary
cmd/serve/            (Phase 0+) the server
engine/backend/       Go side of the C-ABI: syscall loader on Windows,
                      cgo loader on Linux (planned), one shared interface
backend/              CUDA C++: shim.cu (C-ABI impl), kernels/ (coming)
tools/                offline Python: gen_reference.py (HF oracle dumps),
                      triton/ kernel prototypes — never in the serving path
scripts/              build_backend.ps1 (nvcc + MSVC → build/*.dll)
docs/PLAN.md          the build plan
```

## Building (Windows lab box)

Requirements: CUDA toolkit (12.x), Visual Studio 2022 with the C++ workload
(nvcc requires cl.exe as host compiler on Windows), Go 1.26+.

```powershell
.\scripts\build_backend.ps1 -Arch sm_75   # GTX 1650; use sm_86 on the A6000 box
go run .\cmd\smoke                        # expect: device info + PASS
```

### Windows FFI note (why there is no cgo here)

cgo on windows/amd64 requires a 64-bit MinGW-w64 GCC, which this box doesn't
have (only 32-bit MinGW.org 6.3, which cgo can't use). Since the C-ABI is
narrow by design, the Windows loader uses `syscall.LoadDLL` +
`GetProcAddress` instead — zero cgo, zero extra toolchain. The Linux/A6000
engine box will use a plain cgo loader against `libtoyengine.so` behind the
same Go interface (`engine/backend`). If cgo is ever wanted on Windows,
install MinGW-w64 (MSYS2 `mingw-w64-ucrt-x86_64-gcc` or w64devkit).
