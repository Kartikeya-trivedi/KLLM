# kllm — working notes for Claude

Toy LLM inference engine: Go orchestration over a CUDA C++ backend across a
narrow C-ABI. The authoritative plan (phases, rules, hardware) is
docs/PLAN.md — read it before making architectural changes.

## This machine

This is the **GTX 1650 kernel lab box** (sm_75, 4 GiB, no tensor cores),
Windows 11. It runs the walking skeleton, Phase 0 spine dev on tiny/sub-1B
models, and CUDA kernel microbenchmarks. The 30B models and the real engine
run on a separate 3x A6000 Linux box (sm_86).

## Build & test

- Backend: `.\scripts\build_backend.ps1 -Arch sm_75` → `build\toyengine_backend.dll`
  (locates MSVC cl.exe via vswhere; do NOT use MinGW as nvcc host compiler)
- Smoke: `go run .\cmd\smoke` from repo root — must print PASS
- Rebuild the DLL after any backend/*.cu or *.h change; Go does not track it.

## Hard rules (from the plan)

- The C-ABI stays narrow: small control data across, tensors stay on-device,
  one `forward_step` call per decode step for the whole batch.
- No cgo on Windows: `engine/backend/backend_windows.go` uses syscall.LoadDLL
  (the local 32-bit MinGW cannot back cgo). Linux gets a cgo loader behind
  the same `impl` interface — don't merge the two paths.
- Python under tools/ is offline-only (reference dumps, Triton prototypes) —
  never in the serving path.
- Correctness before speed: every kernel validates against
  tools/gen_reference.py dumps before its perf numbers count.
