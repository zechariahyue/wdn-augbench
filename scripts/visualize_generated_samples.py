"""Visualize real vs. synthetic disturbance samples for the manuscript.

Generates:
  1. Pressure-field comparison (real vs SMOTE vs GMM vs Graph-CVAE)
  2. Feature distribution overlays (pressure, flow, demand)
  3. Per-augmenter plausibility score distributions

These figures address the review request to "show what the generator produces."
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from run_cross_network_eval import (  # noqa: E402
    _build_network_index,
    _simulate_network_group,
    _apply_augmenter,
    FEATURE_COLUMNS,
    TRAIN_NETWORK_IDS,
)
from watergen.evaluation import HydraulicResidualScorer  # noqa: E402

OUTPUT_DIR = ACTIVE_ROOT / "artifacts" / "sample_visualization"

# Features to visualize
VIZ_FEATURES = [
    "pressure_mean", "pressure_std", "pressure_min", "pressure_max",
    "flow_abs_mean", "flow_std", "demand_mean", "demand_sum",
]

AUGMENTER_NAMES = ["baseline", "noise", "gmm", "smote"]
AUGMENTER_COLORS = {
    "baseline": "#2C3E50",
    "noise": "#3498DB",
    "gmm": "#E74C3C",
    "smote": "#27AE60",
}


def _get_positive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Extract positive (anomaly) rows."""
    return df[df["event_label"] == 1].copy()


def plot_feature_distributions(
    real_positive: pd.DataFrame,
    augmented_sets: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Overlay histograms of key features: real positive vs each augmenter."""
    available = [f for f in VIZ_FEATURES if f in real_positive.columns]
    if not available:
        print("[warn] No visualization features found in data")
        return

    import math
    n_features = len(available)
    ncols = 4 if n_features > 4 else n_features
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()

    legend_handles = None
    for i, feat in enumerate(available):
        ax = axes[i]
        real_vals = pd.to_numeric(real_positive[feat], errors="coerce").dropna()
        if real_vals.empty:
            continue

        bins = np.linspace(real_vals.quantile(0.01), real_vals.quantile(0.99), 40)

        ax.hist(real_vals, bins=bins, alpha=0.6, label="Real positive",
                color=AUGMENTER_COLORS["baseline"], density=True, edgecolor="white")

        for aug_name, aug_df in augmented_sets.items():
            if aug_name == "baseline":
                continue
            aug_positive = _get_positive_rows(aug_df)
            if feat not in aug_positive.columns:
                continue
            aug_vals = pd.to_numeric(aug_positive[feat], errors="coerce").dropna()
            if aug_vals.empty:
                continue
            ax.hist(aug_vals, bins=bins, alpha=0.35, label=f"{aug_name} synthetic",
                    color=AUGMENTER_COLORS.get(aug_name, "#999"), density=True,
                    edgecolor="white", linestyle="--")

        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.tick_params(labelsize=7)
        if legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    # hide any unused grid cells
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    # single shared legend instead of one per panel
    if legend_handles and legend_handles[0]:
        fig.legend(*legend_handles, loc="lower center",
                   ncol=len(legend_handles[0]), fontsize=9, frameon=False,
                   bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Feature Distributions: Real Positive vs Synthetic Samples", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_dir / "feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Feature distribution plot saved")


def plot_pressure_scatter(
    real_positive: pd.DataFrame,
    augmented_sets: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """2D scatter: pressure_mean vs pressure_std for real vs synthetic."""
    if "pressure_mean" not in real_positive.columns or "pressure_std" not in real_positive.columns:
        print("[warn] pressure_mean/pressure_std not in data — skipping scatter")
        return

    n_aug = len([a for a in augmented_sets if a != "baseline"])
    fig, axes = plt.subplots(1, max(n_aug, 1), figsize=(5 * max(n_aug, 1), 4.5))
    if n_aug <= 1:
        axes = [axes]

    real_x = pd.to_numeric(real_positive["pressure_mean"], errors="coerce").to_numpy()
    real_y = pd.to_numeric(real_positive["pressure_std"], errors="coerce").to_numpy()

    idx = 0
    for aug_name, aug_df in augmented_sets.items():
        if aug_name == "baseline":
            continue
        ax = axes[idx]
        ax.scatter(real_x, real_y, alpha=0.4, s=15, label="Real", color=AUGMENTER_COLORS["baseline"])

        aug_pos = _get_positive_rows(aug_df)
        if "pressure_mean" in aug_pos.columns and "pressure_std" in aug_pos.columns:
            aug_x = pd.to_numeric(aug_pos["pressure_mean"], errors="coerce").to_numpy()
            aug_y = pd.to_numeric(aug_pos["pressure_std"], errors="coerce").to_numpy()
            ax.scatter(aug_x, aug_y, alpha=0.4, s=15, label=f"{aug_name}",
                       color=AUGMENTER_COLORS.get(aug_name, "#E74C3C"))

        ax.set_xlabel("pressure_mean")
        ax.set_ylabel("pressure_std")
        ax.set_title(f"Real vs {aug_name}")
        ax.legend(fontsize=8)
        idx += 1

    plt.suptitle("Pressure Field: Real vs Synthetic", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "pressure_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Pressure scatter plot saved")


def plot_plausibility_comparison(
    augmented_sets: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Box plot of plausibility scores across augmenters."""
    scorer = HydraulicResidualScorer()

    scores_by_aug: dict[str, np.ndarray] = {}
    for aug_name, aug_df in augmented_sets.items():
        pos = _get_positive_rows(aug_df)
        if pos.empty:
            continue
        try:
            result = scorer.score_rows(pos)
            scores_by_aug[aug_name] = result.plausibility_scores
        except Exception as exc:
            print(f"[warn] Plausibility scoring failed for {aug_name}: {exc}")

    if not scores_by_aug:
        print("[warn] No plausibility scores computed — skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(scores_by_aug.keys())
    data = [scores_by_aug[l] for l in labels]
    colors = [AUGMENTER_COLORS.get(l, "#999") for l in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Plausibility Score")
    ax.set_title("Plausibility Score Distribution by Augmenter (Positive Samples)")
    plt.tight_layout()
    fig.savefig(output_dir / "plausibility_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Plausibility comparison plot saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
    print(f"[info] Building network index from {manifest_path}")
    network_index = _build_network_index(manifest_path)

    # Simulate training data
    print("[info] Simulating training data (Anytown + Hanoi)...")
    train_split = _simulate_network_group(
        TRAIN_NETWORK_IDS,
        network_index,
        split_name="train",
        sensor_coverage_levels=[0.50],
        disturbance_types=["leak", "pipe_closure", "pump_outage"],
        max_candidates=5,
    )
    if train_split.empty:
        print("[ERROR] Empty training split — cannot proceed")
        return

    real_positive = _get_positive_rows(train_split)
    print(f"[info] Training data: {len(train_split)} rows, {len(real_positive)} positive")

    # Generate augmented versions
    print("[info] Generating augmented datasets...")
    augmented_sets: dict[str, pd.DataFrame] = {"baseline": train_split}

    for aug_name in AUGMENTER_NAMES:
        if aug_name == "baseline":
            continue
        try:
            aug_df = _apply_augmenter(train_split, aug_name, random_state=42)
            augmented_sets[aug_name] = aug_df
            n_pos = int(aug_df["event_label"].sum())
            print(f"  {aug_name}: {len(aug_df)} rows ({n_pos} positive)")
        except Exception as exc:
            print(f"  [warn] {aug_name} failed: {exc}")

    # Generate all plots
    print("\n[info] Generating visualizations...")
    plot_feature_distributions(real_positive, augmented_sets, OUTPUT_DIR)
    plot_pressure_scatter(real_positive, augmented_sets, OUTPUT_DIR)
    plot_plausibility_comparison(augmented_sets, OUTPUT_DIR)

    print(f"\n[info] All visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
