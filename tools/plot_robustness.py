"""Plot the robustness sweep (bench/robustness/results.csv, from cmd/robusteval).

    python tools/plot_robustness.py
    python tools/plot_robustness.py --results bench/robustness/results.csv --out docs/assets

Writes two figures:

  robustness_accuracy.png   exact match against the number of adversarial nodes,
                            one panel per attack, one line per strategy
  robustness_tradeoff.png   what each dispatch policy costs and what it gives up

Colours are the first five slots of a categorical palette checked for
colour-vision-deficiency separation; each strategy also has its own marker,
so identity never rests on colour alone. The numbers behind both figures are
tabulated in bench/robustness/report.md.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import IdentityTransform

SURFACE, GRID = "#fcfcfb", "#e7e6e2"
TEXT, TEXT2 = "#0b0b0b", "#52514e"
NEUTRAL = "#a3a29c"

# strategy -> (label, colour, marker). Fixed order: colour follows the strategy.
STRATEGIES = {
    "majority": ("majority vote", "#2a78d6", "o"),
    "token-vote": ("token-level vote", "#eb6834", "s"),
    "confidence": ("confidence-weighted", "#1baf7a", "^"),
    "reputation": ("reputation-weighted", "#eda100", "D"),
    "adaptive": ("adaptive", "#e87ba4", "p"),
}
ATTACKS = [
    ("collude", "collude: one shared wrong answer"),
    ("stealth", "stealth: honest prefix, then the lie"),
    ("truncate", "truncate: half an answer"),
    ("overconfident", "overconfident: lie + claimed certainty"),
    ("sleeper", "sleeper: honest, then turns"),
    ("fast", "fast: the lie arrives first"),
]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=9)
    ax.set_axisbelow(True)


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    cell = {(r["attack"], int(r["adversaries"]), r["method"]): r for r in rows}
    nodes = int(rows[0]["nodes"])
    max_adv = max(int(r["adversaries"]) for r in rows)
    return cell, nodes, max_adv


def accuracy(cell, nodes, max_adv, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), sharex=True, sharey=True, facecolor=SURFACE)
    xs = list(range(max_adv + 1))

    def series(attack, method):
        ys = []
        for f in xs:
            row = cell.get(("normal", 0, method)) if f == 0 else cell.get((attack, f, method))
            ys.append(100 * float(row["exact_match"]) if row else None)
        return ys

    for ax, (attack, title) in zip(axes.flat, ATTACKS):
        style(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.plot(xs, series(attack, "single"), color=NEUTRAL, linewidth=1.5, zorder=2)
        for method, (_, colour, marker) in STRATEGIES.items():
            ax.plot(xs, series(attack, method), color=colour, linewidth=2, marker=marker, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3, clip_on=False)
        ax.set_title(title, loc="left", fontsize=10.5, color=TEXT, pad=8)
        ax.set_ylim(-5, 106)  # room for markers at 0 and 100 clear of the tick labels
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(xs)
    for ax in axes[-1]:
        ax.set_xlabel(f"adversarial nodes (of {nodes})", color=TEXT2, fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("exact match, %", color=TEXT2, fontsize=9.5)

    # One direct label, on the line the figure is about; the legend names the rest.
    last = max_adv - 1  # the largest adversarial minority
    y = series("stealth", "adaptive")[last]
    axes.flat[1].text(last + 0.12, y, f"adaptive {y:.0f}%", va="center", ha="left", fontsize=9, color=TEXT)

    handles = [plt.Line2D([], [], color=c, linewidth=2, marker=m, markersize=8, markeredgecolor=SURFACE,
                          markeredgewidth=2, label=label) for label, c, m in STRATEGIES.values()]
    handles.append(plt.Line2D([], [], color=NEUTRAL, linewidth=1.5, label="single node (no redundancy)"))
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9.5,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 0.945), handlelength=2.2, columnspacing=1.6)
    fig.suptitle("Accuracy as nodes turn adversarial", x=0.012, y=0.985, ha="left", fontsize=14,
                 color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tradeoff(cell, nodes, out):
    k = nodes // 2 + 1
    methods = [
        ("adaptive", "wait for all"),
        (f"adaptive@first-{k}", f"first {k} answers"),
        ("adaptive@adaptive-3+2", "ask 3, escalate"),
        ("adaptive@adaptive-3+2-trust", "ask 3 most trusted,\nescalate"),
    ]
    panels = [
        ("tokens generated per answer", lambda m: float(cell[("normal", 0, m)]["tokens"]), "{:.0f}"),
        ("p99 latency, ms", lambda m: float(cell[("normal", 0, m)]["latency_p99_ms"]), "{:.0f}"),
        ("exact match, no adversary, %", lambda m: 100 * float(cell[("normal", 0, m)]["exact_match"]), "{:.1f}"),
        ("exact match, 2 answer-first\nadversaries, %", lambda m: 100 * float(cell[("fast", 2, m)]["exact_match"]), "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True, facecolor=SURFACE)
    ys = list(range(len(methods)))[::-1]
    bars = []
    for ax, (title, value, fmt) in zip(axes, panels):
        style(ax)
        ax.spines["bottom"].set_visible(False)
        vals = [value(m) for m, _ in methods]
        top = max(vals) * 1.22
        ax.set_xlim(0, top)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xticks([])
        ax.set_yticks(ys)
        ax.set_yticklabels([label for _, label in methods], fontsize=9.5, color=TEXT)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=8)
        for y, v in zip(ys, vals):
            bars.append((ax, y, v))
            ax.text(v + top * 0.025, y, fmt.format(v), va="center", ha="left", fontsize=9.5, color=TEXT)
    fig.suptitle("What each dispatch policy costs, and what it gives up", x=0.012, y=0.97, ha="left",
                 fontsize=14, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    # Thin bars with a rounded data end and a square baseline, drawn in pixels
    # once the layout is final so the corners stay circular.
    fig.canvas.draw()
    thick, radius = 18.0, 4.0
    scale = fig.dpi / 100.0
    for ax, y, v in bars:
        (x0, yc), (x1, _) = ax.transData.transform([(0, y), (v, y)])
        h, r = thick * scale, radius * scale
        if x1 - x0 > 2 * r:
            fig.patches.append(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, transform=IdentityTransform(),
                                              boxstyle=f"round,pad=0,rounding_size={r}",
                                              facecolor="#2a78d6", edgecolor="none"))
        fig.patches.append(Rectangle((x0, yc - h / 2), max(x1 - x0 - r, 1), h, transform=IdentityTransform(),
                                     facecolor="#2a78d6", edgecolor="none"))
    fig.savefig(out, dpi=fig.dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="bench/robustness/results.csv")
    ap.add_argument("--out", default="docs/assets")
    args = ap.parse_args()

    cell, nodes, max_adv = load(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    accuracy(cell, nodes, max_adv, out / "robustness_accuracy.png")
    plt.rcParams["figure.dpi"] = 160
    tradeoff(cell, nodes, out / "robustness_tradeoff.png")
    print("wrote", out / "robustness_accuracy.png", "and", out / "robustness_tradeoff.png")


if __name__ == "__main__":
    main()
