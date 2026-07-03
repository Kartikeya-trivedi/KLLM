"""Modal GPU lab: run kllm's CUDA backend on real Ampere (A10G = sm_86, the
same arch as the target A6000 engine box) without leaving the laptop.

Entrypoints:

  modal run tools/modal_lab.py::gpu_smoke
      Cheap sanity check: confirms the account can get a GPU and prints it.

  modal run tools/modal_lab.py::build_and_test
      Ships the repo into a CUDA 12.8 devel container, builds
      libtoyengine.so for sm_86, builds the Go engine (cgo loader on Linux),
      and runs the smoke + test suite on the GPU. The per-phase "real
      Ampere" correctness gate.

  modal serve tools/modal_lab.py        (live, ephemeral — dies on Ctrl-C)
  modal deploy tools/modal_lab.py       (persistent public URL)
      Serves the inference engine + browser playground on an A10G. Modal
      prints a public https URL; open it and generate tokens in the browser.
      Scales to zero when idle, so it only bills GPU time while in use.

Uses the active profile in ~/.modal.toml.
"""

import subprocess

import modal

app = modal.App("kllm-lab")

REPO_DIRS = ["backend", "engine", "cmd", "models", "server", "testmodels", "refdumps"]

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
    .env({"PATH": "/usr/local/go/bin:/usr/local/cuda/bin:"
                  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "CGO_ENABLED": "1"})
)
cuda_image = cuda_image.add_local_file("go.mod", remote_path="/repo/go.mod")
for name in REPO_DIRS:
    cuda_image = cuda_image.add_local_dir(name, remote_path=f"/repo/{name}")

# Model image: cuda_image + the HF/torch stack needed to download, convert,
# and oracle-check a real checkpoint. Reused by the A100 model benchmark.
model_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("wget", "git", "build-essential")
    .run_commands(
        "wget -q https://go.dev/dl/go1.26.2.linux-amd64.tar.gz -O /tmp/go.tgz",
        "tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz",
    )
    .pip_install("torch", "transformers", "safetensors", "huggingface_hub",
                 "accelerate", "numpy")
    .env({"PATH": "/usr/local/go/bin:/usr/local/cuda/bin:"
                  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "CGO_ENABLED": "1", "HF_HOME": "/cache/hf"})
    .add_local_file("go.mod", remote_path="/repo/go.mod")
)
for name in ["backend", "engine", "cmd", "models", "server", "tools"]:
    model_image = model_image.add_local_dir(name, remote_path=f"/repo/{name}")

hf_cache = modal.Volume.from_name("kllm-hf-cache", create_if_missing=True)

# Serving image: copy the repo into the image (copy=True so build steps can
# see it), prebuild the .so + Go binary, so container cold-start just runs it.
serving_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("wget", "build-essential")
    .run_commands(
        "wget -q https://go.dev/dl/go1.26.2.linux-amd64.tar.gz -O /tmp/go.tgz",
        "tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz",
    )
    .env({"PATH": "/usr/local/go/bin:/usr/local/cuda/bin:"
                  "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "CGO_ENABLED": "1"})
    .add_local_file("go.mod", remote_path="/repo/go.mod", copy=True)
)
for name in ["backend", "engine", "cmd", "models", "server", "testmodels"]:
    serving_image = serving_image.add_local_dir(name, remote_path=f"/repo/{name}", copy=True)
serving_image = serving_image.run_commands(
    "cd /repo && mkdir -p build && nvcc -shared -O2 -lineinfo -arch=sm_86 "
    "-Xcompiler -fPIC -o build/libtoyengine.so backend/*.cu -lcublas",
    "cd /repo && go build -o /repo/kllm-serve ./cmd/serve",
)


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
    # Build + test the Go engine through the Linux cgo loader.
    sh("go vet ./...")
    sh("go test -v -count=1 ./... 2>&1 | grep -E '^(ok|FAIL|--- (PASS|FAIL|SKIP))' || true")
    sh("go test -count=1 ./...")
    sh("go run ./cmd/smoke --backend build/libtoyengine.so")
    # Decode-shape matmul microbenchmark on real Ampere for the journal.
    sh("go run ./cmd/wbench --backend build/libtoyengine.so "
       "--shapes 4096x4096,11008x4096 --n 1 --iters 50")
    return "ok"


@app.function(gpu="A100", image=model_image, timeout=2400, volumes={"/cache": hf_cache})
def bench_model(model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                prompt_len: int = 64, steps: int = 128):
    """Run kllm on a real Llama-arch model on a single A100: convert →
    build (sm_80) → correctness vs HuggingFace → tok/s (single-stream +
    continuous-batched aggregate).

        modal run tools/modal_lab.py::bench_model
        modal run tools/modal_lab.py::bench_model --model-id <hub-id> --steps 256
    """
    import json
    import subprocess
    import time
    import urllib.request

    def sh(cmd, check=True, **kw):
        print(f"$ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd="/repo", capture_output=True, text=True, **kw)
        if r.stdout:
            print(r.stdout, flush=True)
        if r.returncode != 0:
            print(r.stderr, flush=True)
            if check:
                raise RuntimeError(f"failed: {cmd}")
        return r.stdout

    sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")

    # 1. Download + convert the checkpoint to the engine's fp32 format.
    from huggingface_hub import snapshot_download
    hf_dir = snapshot_download(model_id, ignore_patterns=["*.bin", "*.pth", "*.gguf", "original/*"])
    sh(f"python tools/convert_hf.py --hf {hf_dir} --out /repo/testmodels/real")

    # 2. Build the backend for A100 (sm_80) and the Go binaries once.
    sh("mkdir -p build && nvcc -shared -O2 -lineinfo -arch=sm_80 -Xcompiler -fPIC "
       "-o build/libtoyengine.so backend/*.cu -lcublas")
    for b in ["generate", "bench", "serve", "loadgen"]:
        sh(f"go build -o /repo/bin/{b} ./cmd/{b}")
    BK = "build/libtoyengine.so"
    M = "/repo/testmodels/real"

    # 3. Correctness: HF greedy vs engine greedy on a fixed raw-id prompt.
    prompt = "1 15043 29892 590 1024 338"  # arbitrary valid token ids
    with open("/tmp/p.txt", "w") as f:
        f.write(prompt + "\n")
    # Non-fatal: an oracle hiccup shouldn't block the tok/s benchmark below.
    match, hf_ids, eng_ids = -1, [], []
    ref_ok = sh(f"python tools/gen_reference.py --model {hf_dir} --prompts /tmp/p.txt "
                f"--out /tmp/ref --raw-ids --max-new-tokens 16 --dtype float32 "
                f"--device cuda", check=False) is not None
    try:
        hf_ids = json.load(open("/tmp/ref/prompt_0/tokens.json"))["generated_ids"]
        out = sh(f'/repo/bin/generate --backend {BK} --model {M} --ids "{prompt}" '
                 f"--steps 16 --max-seq 512")
        eng_ids = [int(x) for x in out.split("generated:")[1].strip().strip("[]").split()]
        match = 0
        for a, b in zip(hf_ids, eng_ids):
            if a != b:
                break
            match += 1
        print(f"\nCORRECTNESS: engine matched HF greedy for {match}/{len(hf_ids)} tokens")
        print(f"  HF:     {hf_ids}")
        print(f"  engine: {eng_ids}\n")
    except Exception as e:
        print(f"\nCORRECTNESS: skipped ({e})\n")

    # 4. Single-stream decode tok/s.
    js = sh(f"/repo/bin/bench --backend {BK} --model {M} --prompt {prompt_len} "
            f"--steps {steps} --reps 3 --max-seq 2048 --json")
    single = json.loads([l for l in js.splitlines() if l.strip().startswith("{")][-1])
    print(f"SINGLE-STREAM: {single['decode_tok_s']:.1f} tok/s "
          f"(TTFT {single['ttft_ms']:.1f} ms, ITL {single['itl_ms']:.3f} ms)\n")

    # 5. Aggregate throughput via the continuous-batching server + loadgen.
    srv = subprocess.Popen(
        ["/repo/bin/serve", "--backend", BK, "--model", M, "--addr", "127.0.0.1:8080",
         "--max-batch", "32", "--max-seq", "512", "--kv-blocks", "8192"], cwd="/repo")
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=2).status == 200:
                    break
            except Exception:
                time.sleep(1)
        agg = sh('/repo/bin/loadgen --url http://127.0.0.1:8080 '
                 '--concurrency 1,8,32 --requests 64 --steps 64 '
                 f'--vocab {json.load(open(M + "/config.json"))["vocab_size"]}')
        stats = json.load(urllib.request.urlopen("http://127.0.0.1:8080/stats.json"))
        print(f"AGGREGATE peak EWMA: {stats['tokens_per_second']:.0f} tok/s\n")
    finally:
        srv.terminate()

    return {"model": model_id, "correct_tokens": match,
            "single_stream_tok_s": single["decode_tok_s"]}


@app.function(gpu="A10G", image=serving_image, scaledown_window=300)
@modal.concurrent(max_inputs=100)  # one container fans requests into the batch
@modal.web_server(8080, startup_timeout=120)
def serve():
    # Launch the prebuilt Go server bound to the web port; the browser
    # playground is at "/", the SSE API at "/v1/generate".
    import subprocess
    subprocess.Popen(
        ["/repo/kllm-serve", "--addr", "0.0.0.0:8080",
         "--backend", "build/libtoyengine.so", "--model", "testmodels/tiny-llama",
         "--max-batch", "16", "--max-seq", "256", "--kv-blocks", "2048"],
        cwd="/repo",
    )
