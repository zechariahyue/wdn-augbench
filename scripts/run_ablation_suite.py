"""Ablation ladder suite runner for the physics-aware augmentation paper.

Executes levels L0-L5 across 4 sensor coverage regimes and generates:
1. ablation_results_table.md — AUPRC × coverage × level
2. sensing_regime_curves.png — key paper figure (Figure 2)

Usage:
    python scripts/run_ablation_suite.py
    python scripts/run_ablation_suite.py --matrix configs/final_experiment_matrix.yaml
    python scripts/run_ablation_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

# Ablation level metadata — order matters (L0 is baseline, L5 is full proposed)
ABLATION_LEVELS: list[dict[str, Any]] = [
    {"run_id": "ablation_L0_rf_noise", "label": "L0: RF+Noise", "color": "#7f7f7f", "marker": "o"},
    {"run_id": "ablation_L1_rf_gmm", "label": "L1: RF+GMM", "color": "#1f77b4", "marker": "s"},
    {"run_id": "ablation_L2_rf_gmm_filter", "label": "L2: RF+GMM+Filter", "color": "#ff7f0e", "marker": "^"},
    {"run_id": "ablation_L3_rf_smote", "label": "L3: RF+SMOTE", "color": "#2ca02c", "marker": "D"},
    {"run_id": "ablation_L4_rf_graph_cvae", "label": "L4: RF+GraphCVAE", "color": "#9467bd", "marker": "P"},
    {"run_id": "ablation_L5_rf_graph_cvae_physics", "label": "L5: RF+GraphCVAE+Physics", "color": "#d62728", "marker": "*"},
]

COVERAGE_LEVELS = [0.25, 0.50, 0.75, 1.00]
COVERAGE_LABELS = ["0.25", "0.50", "0.75", "1.00"]


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return data


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_one_config(config_path: Path, log_path: Path) -> int:
    """Run a single experiment config via run_experiment.py."""
    command = [
        sys.executable,
        str(ACTIVE_ROOT / "scripts" / "run_experiment.py"),
        "--config",
        str(config_path),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""),
        encoding="utf-8",
    )
    return int(result.returncode)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    import copy

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def extract_auprc_by_coverage(
    run_id: str,
    artifact_dir: Path,
    split: str = "validation",
) -> dict[str, float | None]:
    """Extract per-coverage AUPRC from a smoke experiment summary.

    Returns a dict mapping coverage fraction string → AUPRC float (or None if unavailable).
    """
    smoke_path = artifact_dir / "smoke" / "smoke_experiment_summary.json"
    if not smoke_path.exists():
        return {c: None for c in COVERAGE_LABELS}

    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    except Exception:
        return {c: None for c in COVERAGE_LABELS}

    # Look for random_forest_baseline experiment (the canonical ablation detector)
    experiments = smoke.get("experiments", {})
    target_experiment = experiments.get("random_forest_baseline", {})
    split_data = target_experiment.get("splits", {}).get(split, {})
    by_coverage = split_data.get("by_sensor_coverage", {})

    result: dict[str, float | None] = {}
    for cov_label in COVERAGE_LABELS:
        cov_data = by_coverage.get(cov_label, {})
        auprc = cov_data.get("auprc")
        result[cov_label] = float(auprc) if isinstance(auprc, (int, float)) else None

    return result


def extract_auprc_std_by_coverage(
    all_seed_results: list[dict[str, float | None]],
) -> dict[str, float | None]:
    """Compute per-coverage std across seeds. Returns None when fewer than 2 seeds."""
    import statistics

    stds: dict[str, float | None] = {}
    for cov in COVERAGE_LABELS:
        values = [r[cov] for r in all_seed_results if isinstance(r.get(cov), float)]
        stds[cov] = statistics.stdev(values) if len(values) >= 2 else None
    return stds


def write_ablation_table(
    level_results: list[dict[str, Any]],
    output_path: Path,
    split: str = "validation",
) -> None:
    """Write the ablation results table as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ablation Results Table",
        "",
        f"Metric: validation AUPRC (split={split})",
        "",
        "| Level | Augmenter | " + " | ".join(f"cov={c}" for c in COVERAGE_LABELS) + " |",
        "|---|---|" + "|".join(["---"] * len(COVERAGE_LABELS)) + "|",
    ]

    for entry in level_results:
        meta = entry["meta"]
        auprc_by_cov = entry["auprc_by_coverage"]
        cells = []
        for cov in COVERAGE_LABELS:
            val = auprc_by_cov.get(cov)
            std = entry.get("auprc_std_by_coverage", {}).get(cov)
            if isinstance(val, float):
                cell = f"{val:.3f}"
                if isinstance(std, float):
                    cell += f" ±{std:.3f}"
            else:
                cell = "n/a"
            cells.append(cell)
        level_num = meta["run_id"].split("_")[1]  # e.g. "L0"
        augmenter = meta["label"].split(": ", 1)[1] if ": " in meta["label"] else meta["label"]
        lines.append(f"| {level_num} | {augmenter} | " + " | ".join(cells) + " |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Ablation results table: {output_path}")


def plot_sensing_regime_curves(
    level_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate the sensing regime curves figure (Figure 2 of the paper).

    X-axis: sensor coverage (0.25, 0.50, 0.75, 1.00)
    Y-axis: validation AUPRC
    One line per ablation level (L0-L5) with error bars when multi-seed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    x_positions = COVERAGE_LEVELS

    for entry in level_results:
        meta = entry["meta"]
        auprc_by_cov = entry["auprc_by_coverage"]
        std_by_cov = entry.get("auprc_std_by_coverage", {})

        y_values = [auprc_by_cov.get(c) for c in COVERAGE_LABELS]
        y_errs = [std_by_cov.get(c) for c in COVERAGE_LABELS]

        # Build plot-ready arrays, substituting None with NaN
        import math

        y_plot = [v if isinstance(v, float) else math.nan for v in y_values]
        yerr_plot = [v if isinstance(v, float) else 0.0 for v in y_errs]
        has_errors = any(isinstance(v, float) for v in y_errs)

        if has_errors:
            ax.errorbar(
                x_positions,
                y_plot,
                yerr=yerr_plot,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=8,
                linewidth=2,
                capsize=4,
            )
        else:
            ax.plot(
                x_positions,
                y_plot,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                markersize=8,
                linewidth=2,
            )

    # Publication-quality formatting
    ax.set_xlabel("Sensor Coverage Fraction", fontsize=14, labelpad=8)
    ax.set_ylabel("Validation AUPRC", fontsize=14, labelpad=8)
    ax.set_title(
        "Sensing Regime Curves: AUPRC vs. Sensor Coverage\nby Augmentation Level",
        fontsize=15,
        pad=12,
    )
    ax.set_xlim(0.15, 1.10)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(COVERAGE_LEVELS)
    ax.set_xticklabels([f"{c:.2f}" for c in COVERAGE_LEVELS], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(
        title="Ablation Level",
        title_fontsize=11,
        fontsize=10,
        loc="lower right",
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Sensing regime curves figure saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the L0-L5 ablation ladder and generate sensing regime curves."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ACTIVE_ROOT / "configs" / "final_experiment_matrix.yaml",
        help="Path to the experiment matrix YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ACTIVE_ROOT / "artifacts" / "ablation",
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Which split to report AUPRC from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual experiment execution; only generate plots from existing artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    matrix = load_yaml(args.matrix)
    runs = matrix.get("runs", [])
    if not isinstance(runs, list) or not runs:
        print("[error] Experiment matrix has no runs")
        return 1

    # Index runs by run_id for lookup
    run_index = {str(run["run_id"]): run for run in runs}

    generated_dir = ACTIVE_ROOT / "artifacts" / "suite" / "_generated_configs"
    log_dir = ACTIVE_ROOT / "artifacts" / "suite" / "logs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    level_results: list[dict[str, Any]] = []

    for meta in ABLATION_LEVELS:
        run_id = meta["run_id"]
        run_spec = run_index.get(run_id)

        if run_spec is None:
            print(f"[warn] run_id '{run_id}' not found in matrix — skipping")
            auprc_by_cov = {c: None for c in COVERAGE_LABELS}
        else:
            # Resolve artifact dir from merged config
            base_config_path = resolve_repo_path(run_spec["base_config"])
            base_config = load_yaml(base_config_path)
            overrides = run_spec.get("overrides", {})
            merged_config = deep_merge(base_config, overrides)
            artifact_dir_raw = merged_config.get("outputs", {}).get("artifact_dir", "")
            artifact_dir = resolve_repo_path(artifact_dir_raw) if artifact_dir_raw else (
                ACTIVE_ROOT / "artifacts" / "suite" / run_id
            )

            if not args.dry_run:
                print(f"[info] Running ablation level: {run_id}")
                generated_config_path = generated_dir / f"{run_id}.yaml"
                generated_config_path.write_text(
                    yaml.safe_dump(merged_config, sort_keys=False), encoding="utf-8"
                )
                exit_code = run_one_config(
                    generated_config_path, log_dir / f"{run_id}.log"
                )
                if exit_code != 0:
                    print(f"[warn] {run_id} exited with code {exit_code}")
            else:
                print(f"[dry-run] Skipping execution for: {run_id}")

            auprc_by_cov = extract_auprc_by_coverage(run_id, artifact_dir, split=args.split)

        level_results.append(
            {
                "meta": meta,
                "auprc_by_coverage": auprc_by_cov,
                "auprc_std_by_coverage": {},  # populated if multi-seed data were available
            }
        )

    # Write outputs
    write_ablation_table(
        level_results,
        args.output_dir / "ablation_results_table.md",
        split=args.split,
    )
    plot_sensing_regime_curves(
        level_results,
        args.output_dir / "sensing_regime_curves.png",
    )

    print("[info] Ablation suite complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
