"""Track engine throughput (tok/s) in Weights & Biases.

Two modes, both offline-friendly:

  BENCHMARK SWEEP (default) — run cmd/bench across configs and log a run:
      python tools/wandb_bench.py --tag naive-w4-kernel
      python tools/wandb_bench.py --tag tiled-w4-kernel   # ...compare later
  Use --tag to label the kernel/config version; W&B then charts decode
  tok/s across your kernel-optimization iterations so you can see each
  kernel change move the number.

  LIVE SERVER (--serve-url) — poll a running server's /stats.json and log
  its tokens_per_second as a time series:
      python tools/wandb_bench.py --serve-url http://127.0.0.1:8080 --duration 60

Setup:  pip install wandb && wandb login    (once)
Without wandb installed, or with --dry-run / WANDB_MODE=disabled, it runs
the benchmarks and prints the numbers without logging — so it's always
runnable.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request

# --- Optional W&B ------------------------------------------------------------
try:
    import wandb
    HAVE_WANDB = True
except ImportError:
    HAVE_WANDB = False


def bench_once(backend, model, fused, prompt, steps, reps):
    """Run cmd/bench --json for one config and return the parsed Result."""
    cmd = [
        "go", "run", "./cmd/bench", "--json",
        "--backend", backend, "--model", model,
        f"--fused={'true' if fused else 'false'}",
        "--prompt", str(prompt), "--steps", str(steps), "--reps", str(reps),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"bench failed: {out.stderr.strip()}")
    line = [l for l in out.stdout.splitlines() if l.strip().startswith("{")][-1]
    return json.loads(line)


def sweep(args):
    configs = [
        dict(fused=False, prompt=32, steps=args.steps),
        dict(fused=True,  prompt=32, steps=args.steps),
        dict(fused=True,  prompt=8,  steps=args.steps),
        dict(fused=True,  prompt=128, steps=args.steps),
    ]
    run = None
    if HAVE_WANDB and not args.dry_run:
        run = wandb.init(
            project=args.project,
            name=args.tag,
            config=dict(model=args.model, backend=args.backend, tag=args.tag,
                        reps=args.reps, steps=args.steps),
            tags=[args.tag] if args.tag else None,
        )

    print(f"{'fused':6} {'prompt':7} {'steps':6} {'TTFT ms':10} {'ITL ms':10} {'tok/s':10}")
    best_tps = 0.0
    for c in configs:
        r = bench_once(args.backend, args.model, c["fused"], c["prompt"], c["steps"], args.reps)
        print(f"{str(c['fused']):6} {c['prompt']:<7} {c['steps']:<6} "
              f"{r['ttft_ms']:<10.3f} {r['itl_ms']:<10.4f} {r['decode_tok_s']:<10.1f}")
        best_tps = max(best_tps, r["decode_tok_s"])
        if run:
            # One logged step per config; W&B charts decode_tok_s vs config.
            wandb.log({
                "decode_tok_s": r["decode_tok_s"],
                "ttft_ms": r["ttft_ms"],
                "itl_ms": r["itl_ms"],
                "fused": int(c["fused"]),
                "prompt_len": c["prompt"],
            })

    if run:
        run.summary["best_decode_tok_s"] = best_tps
        run.finish()
        print(f"\nlogged to W&B project '{args.project}'"
              + (f" as run '{args.tag}'" if args.tag else ""))
    else:
        why = "wandb not installed (pip install wandb)" if not HAVE_WANDB else "dry-run"
        print(f"\n[not logged - {why}]  best decode {best_tps:.1f} tok/s")


def live(args):
    if not (HAVE_WANDB and not args.dry_run):
        print("live mode needs wandb; showing /stats.json instead")
    run = None
    if HAVE_WANDB and not args.dry_run:
        run = wandb.init(project=args.project, name=args.tag or "live-serve",
                         tags=["live"])
    end = time.time() + args.duration
    while time.time() < end:
        try:
            with urllib.request.urlopen(args.serve_url.rstrip("/") + "/stats.json", timeout=5) as f:
                s = json.load(f)
        except Exception as e:
            print("poll failed:", e); time.sleep(args.interval); continue
        print(f"tok/s={s['tokens_per_second']:.0f} "
              f"tokens={s['tokens_total']} running={s['running_sequences']:.0f} "
              f"ttft_ms={s['ttft_mean_ms']:.1f}")
        if run:
            wandb.log({k: s[k] for k in
                       ("tokens_per_second", "tokens_total", "running_sequences",
                        "queued_requests", "ttft_mean_ms", "itl_mean_ms",
                        "avg_batch_size", "kv_blocks_used")})
        time.sleep(args.interval)
    if run:
        run.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="kllm")
    p.add_argument("--tag", default="", help="run name / kernel-version label")
    p.add_argument("--backend", default="build/toyengine_backend.dll")
    p.add_argument("--model", default="testmodels/tiny-llama")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="run benches, skip W&B")
    p.add_argument("--serve-url", help="live mode: poll this server's /stats.json")
    p.add_argument("--duration", type=int, default=60, help="live mode seconds")
    p.add_argument("--interval", type=float, default=2.0, help="live mode poll seconds")
    args = p.parse_args()

    if not HAVE_WANDB and not args.dry_run:
        print("note: wandb not installed — running without logging "
              "(pip install wandb && wandb login to enable)\n", file=sys.stderr)
    if args.serve_url:
        live(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
