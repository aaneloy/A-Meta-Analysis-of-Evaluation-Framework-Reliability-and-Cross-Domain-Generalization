#!/usr/bin/env python3
"""Generate clean presentation diagrams matching the deck theme.

Outputs (white background, navy/red/blue palette):
  rag_pipeline_single.png  - RAG flow with faithfulness checkpoint
  rag_pipeline_multi.png   - one input, four evaluators, four different scores
  consensus_donut.png      - 6 / 1.5 / 92.5 cross-cluster consensus
  human_corr_bar.png       - mean human correlation by cluster
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

NAVY = "#002147"
RED = "#E4002B"
BLUE = "#0056b3"
MIDBLUE = "#5b8db8"
LIGHT = "#eef3fb"
INK = "#222222"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

OUT = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, text, fill, edge, tcolor, fs=13, bold=True):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=2, edgecolor=edge, facecolor=fill, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tcolor, fontsize=fs, fontweight="bold" if bold else "normal",
            zorder=3, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=18,
                 linewidth=lw, color=color, zorder=1))


# ----------------------------------------------------------------------
# 1. Single RAG pipeline with faithfulness checkpoint
# ----------------------------------------------------------------------
def rag_single():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.4); ax.axis("off")

    y, h = 1.7, 0.95
    boxes = [
        (0.2, "Query", LIGHT, NAVY, NAVY, 1.7),
        (2.3, "Retriever", LIGHT, NAVY, NAVY, 1.9),
        (4.6, "Retrieved\nContext", "#dce8fb", BLUE, NAVY, 1.9),
        (7.0, "LLM\nGenerator", LIGHT, NAVY, NAVY, 1.9),
        (9.3, "Answer", "#dce8fb", BLUE, NAVY, 1.5),
    ]
    centers = []
    for x, t, fill, edge, tc, w in boxes:
        box(ax, x, y, w, h, t, fill, edge, tc, fs=13)
        centers.append((x, x + w))
    for i in range(len(boxes) - 1):
        arrow(ax, centers[i][1], y + h / 2, centers[i + 1][0], y + h / 2)

    # faithfulness bracket between Context and Answer
    cx0 = boxes[2][0]
    cx1 = boxes[4][0] + boxes[4][5]
    by = 1.25
    ax.plot([cx0, cx0, cx1, cx1], [by, by - 0.18, by - 0.18, by],
            color=RED, lw=2.2, zorder=1)
    ax.text((cx0 + cx1) / 2, 0.55,
            "Faithfulness:  every claim in the answer is supported by the context",
            ha="center", va="center", color=RED, fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(OUT, "rag_pipeline_single.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. One input, four evaluators, four very different scores
# ----------------------------------------------------------------------
def rag_multi():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")

    # input block
    box(ax, 0.2, 1.85, 2.4, 0.95, "Context\n+ Answer", "#dce8fb", BLUE, NAVY, 13)

    evaluators = [
        ("LLM-as-Judge", "0.92", BLUE),
        ("NLI entailment", "0.06", RED),
        ("Embedding", "0.24", NAVY),
        ("Interpretability", "0.28", MIDBLUE),
    ]
    ex = 4.3; ew = 3.4; eh = 0.78
    ys = [3.55, 2.62, 1.69, 0.76]
    for (name, score, col), yy in zip(evaluators, ys):
        box(ax, ex, yy, ew, eh, name, "white", col, col, 13)
        arrow(ax, 2.6, 2.32, ex, yy + eh / 2, color="#888888", lw=1.8)
        # score badge
        bx = ex + ew + 0.35
        b = FancyBboxPatch((bx, yy + 0.04), 1.5, eh - 0.08,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=2, edgecolor=col, facecolor=col, zorder=2)
        ax.add_patch(b)
        ax.text(bx + 0.75, yy + eh / 2, score, ha="center", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=3)

    ax.text(8.85, 4.25, "same input", ha="center", color="#666666", fontsize=11, style="italic")
    ax.text(8.85, 0.32, "four different scores", ha="center", color=RED, fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(OUT, "rag_pipeline_multi.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Cross-cluster consensus donut
# ----------------------------------------------------------------------
def consensus_donut():
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sizes = [92.5, 6.0, 1.5]
    labels = ["Contested\n92.5%", "Unanimous\nfaithful  6.0%", "Unanimous\nunfaithful  1.5%"]
    colors = [NAVY, BLUE, RED]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
    ax.text(0, 0, "200\nsamples", ha="center", va="center",
            fontsize=16, fontweight="bold", color=NAVY)
    ax.legend(wedges, labels, loc="center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, frameon=False, fontsize=10.5, handlelength=1.1)
    ax.set(aspect="equal")
    fig.savefig(os.path.join(OUT, "consensus_donut.png"))
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Human correlation by cluster
# ----------------------------------------------------------------------
def human_bar():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    cats = ["LLM-as-Judge", "Mixed Methods", "Outlier"]
    vals = [0.51, 0.19, 0.05]
    colors = [BLUE, MIDBLUE, RED]
    bars = ax.bar(cats, vals, color=colors, width=0.6, zorder=3, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=NAVY)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Mean correlation with human labels (r)", fontsize=12, color=INK)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="y", color="#dddddd", zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.99, 0.96, "n = 50", transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, color="#666666", style="italic")
    fig.savefig(os.path.join(OUT, "human_corr_bar.png"))
    plt.close(fig)


if __name__ == "__main__":
    rag_single()
    rag_multi()
    consensus_donut()
    human_bar()
    print("Diagrams written to", OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".png"):
            print(" -", f)
