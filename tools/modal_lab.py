"""Modal GPU lab: run kllm's CUDA backend on real Ampere (A10G = sm_86, the
same arch as the target A6000 engine box) without leaving the laptop.

Two entrypoints:

  modal run tools/modal_lab.py::gpu_smoke
      Cheap sanity check: confirms the account can get a GPU and prints it.

  modal run tools/modal_lab.py::build_and_test
      Ships backend/ + engine/ + cmd/ + go.mod into a CUDA 12.8 devel
      container, builds libtoyengine.so for sm_86, builds the Go engine
      (cgo loader on Linux), and runs the smoke + test suite on the GPU.
      This is the per-phase "real Ampere" correctness gate.

Uses the active profile in ~/.modal.toml.
"""

import subprocess

import modal

app = modal.App("kllm-lab")

REPO_FILES = ["backend", "engine", "cmd", "go.mod", "scripts"]

# Light image for the smoke check.
smoke_image = modal.Image.debian_slim()

# Full toolchain image for building the backend + Go engine on Linux.
cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("wget", "git", "build-essential")
    .run_commands(
        "wget -q https://go.dev/dl/go1.26.2.linux-amd64.tar.gz -O /tmp/go.tgz",
        "tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz",
    )
    .env({"PATH": "/usr/local/go/bin:/usr/local/cuda/bin:$PATH"})
)
for name in REPO_FILES:
    cuda_image = cuda_image.add_local_dir(name, remote_path=f"/repo/{name}") \
        if name not in ("go.mod",) else cuda_image.add_local_file(name, remote_path="/repo/go.mod")


@app.function(gpu="A10G", image=smoke_image, timeout=120)
def gpu_smoke():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
                          "--format=csv"], capture_output=True, text=True)
    print(out.stdout or out.stderr)
    return out.stdout


@app.function(gpu="A10G", image=cuda_image, timeout=900)
def build_and_test():
    def sh(cmd):
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, cwd="/repo", capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            raise RuntimeError(f"failed: {cmd}")
        return r.stdout

    sh("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader")
    # Build the backend for sm_86 (real target arch).
    sh("mkdir -p build && nvcc -shared -O2 -lineinfo -arch=sm_86 -Xcompiler -fPIC "
       "-o build/libtoyengine.so backend/*.cu -lcublas")
    # Build + test the Go engine (Linux uses the cgo loader when present;
    # falls back to purego-style loading otherwise).
    sh("go vet ./... && go test ./...")
    sh("go run ./cmd/smoke --backend build/libtoyengine.so")
    return "ok"
