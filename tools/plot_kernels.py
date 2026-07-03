"""Render the kernel-optimization progress graph from bench/kernel_attempts.json.

Two panels:
  left  — isolated W4 matmul effective bandwidth by attempt (per GPU/shape),
          with the cuBLAS fp32 reference as dashed lines
  right — end-to-end decode tok/s by attempt (per GPU/model)

Usage:  python tools/plot_kernels.py   (writes docs/assets/kernel_progress.png)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "bench" / "kernel_attempts.json").read_text())

attempts = {a["id"]: a["name"] for a in DATA["attempts"]}
ids = sorted(attempts)
labels = [f"{i}\n{attempts[i]}" for i in ids]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
fig.suptitle("kllm kernel-optimization loop — every attempt oracle-validated before it counts",
             fontsize=11)

# --- Panel 1: W4 matmul effective GB/s ---------------------------------------
series = defaultdict(dict)   # (gpu, shape) -> {attempt: gbps}
ref = {}                     # (gpu, shape) -> cuBLAS fp32 gbps
for row in DATA["wbench"]:
    key = (row["gpu"], row["shape"])
    if row["attempt"] is None:
        ref[key] = row["gbps"]
    else:
        series[key][row["attempt"]] = row["gbps"]

colors = plt.cm.tab10.colors
for i, (key, points) in enumerate(sorted(series.items())):
    xs = sorted(points)
    ax1.plot(xs, [points[x] for x in xs], "o-", color=colors[i % 10],
             label=f"{key[0]} · {key[1]}")
    if key in ref:
        ax1.axhline(ref[key], color=colors[i % 10], linestyle="--", alpha=0.4,
                    label=f"{key[0]} · {key[1]} cuBLAS fp32")
ax1.set_title("W4 dequant-matmul, decode shape (n=1)")
ax1.set_xlabel("attempt")
ax1.set_ylabel("effective GB/s (weight bytes moved / time)")
ax1.set_xticks(ids[:3] if len(ids) > 3 else ids)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

# --- Panel 2: end-to-end decode tok/s ----------------------------------------
mseries = defaultdict(dict)
for row in DATA["model_bench"]:
    mseries[(row["gpu"], row["model"])][row["attempt"]] = row["tok_s"]
for i, (key, points) in enumerate(sorted(mseries.items())):
    xs = sorted(points)
    ax2.plot(xs, [points[x] for x in xs], "s-", color=colors[(i + 4) % 10],
             label=f"{key[0]}\n{key[1]}")
ax2.set_title("end-to-end decode throughput")
ax2.set_xlabel("attempt")
ax2.set_ylabel("decode tok/s")
ax2.set_xticks(ids)
ax2.set_xticklabels(labels, fontsize=7)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

out = ROOT / "docs" / "assets" / "kernel_progress.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=140)
print(f"wrote {out}")
