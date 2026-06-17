"""Regenerate LOLO figures from the 5-seed aggregated results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

LOLO_PATH = ACTIVE_ROOT / "artifacts" / "lolo_5seed_revised" / "aggregated_results.json"
FIGS_DIR = REPO_ROOT / "manuscript" / "figures"

NETWORKS = ["net3", "d_town", "l_town"]
NETWORK_LABELS = {"net3": "Net3", "d_town": "D-Town", "l_town": "L-TOWN"}
AUG_ORDER = ["baseline", "noise", "gmm", "smote", "gnn_baseline"]
AUG_LABELS = {
    "baseline": "No aug.",
    "noise": "Noise",
    "gmm": "GMM",
    "smote": "SMOTE",
    "gnn_baseline": "GCN Det.",
}
AUG_COLORS = {
    "baseline": "#2C3E50",
    "noise": "#3498DB",
    "gmm": "#E74C3C",
    "smote": "#27AE60",
    "gnn_baseline": "#9B59B6",
}


def plot_per_network_grouped(data: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_aug = len(AUG_ORDER)
    width = 0.14
    x = np.arange(len(NETWORKS))

    for i, aug in enumerate(AUG_ORDER):
        means = []
        stds = []
        for nid in NETWORKS:
            m = data.get(aug, {}).get(nid, {}).get("mean")
            s = data.get(aug, {}).get(nid, {}).get("std")
            means.append(m if m is not None else 0.0)
            stds.append(s if s is not None else 0.0)
        offset = (i - (n_aug - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=AUG_LABELS[aug], color=AUG_COLORS[aug], alpha=0.9,
               edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([NETWORK_LABELS[n] for n in NETWORKS], fontsize=11)
    ax.set_ylabel("AUPRC", fontsize=12)
    ax.set_title("Per-Network LOLO Transfer AUPRC (5 seeds, revised suite)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9, ncol=5, framealpha=0.9)
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_macro_forest(data: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = []
    means = []
    ci_lows = []
    ci_highs = []
    colors = []

    for aug in AUG_ORDER:
        w = data.get(aug, {}).get("weighted", {})
        m = w.get("mean")
        s = w.get("std")
        if m is None:
            continue
        labels.append(AUG_LABELS[aug])
        means.append(m)
        ci_lows.append(m - s)
        ci_highs.append(m + s)
        colors.append(AUG_COLORS[aug])

    y_pos = np.arange(len(labels))
    errs = [[m - lo for m, lo in zip(means, ci_lows)],
            [hi - m for m, hi in zip(means, ci_highs)]]
    ax.errorbar(means, y_pos, xerr=errs, fmt="o", markersize=10,
                capsize=5, linewidth=2, color="#2C3E50")
    for i, (m, c) in enumerate(zip(means, colors)):
        ax.plot(m, i, "o", markersize=12, color=c, markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)

    ax.axvline(means[0], linestyle="--", alpha=0.4, color="#2C3E50", label="baseline")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Macro AUPRC (weighted, 5 seeds)", fontsize=11)
    ax.set_title("Macro LOLO AUPRC with $\\pm 1$ std", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    with LOLO_PATH.open() as f:
        data = json.load(f)

    plot_per_network_grouped(data, FIGS_DIR / "revision_lolo_per_network_grouped.png")
    plot_macro_forest(data, FIGS_DIR / "revision_lolo_macro_forest.png")


if __name__ == "__main__":
    main()
