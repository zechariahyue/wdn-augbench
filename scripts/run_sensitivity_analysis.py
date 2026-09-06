"""Sensitivity analysis for WDN-AugBench disturbance parameters.

Sweeps over:
  1. Emitter coefficient magnitude (leak severity)
  2. Number of leak candidate locations per network
  3. Sensor placement strategy (random vs betweenness centrality)

For each sweep, trains RF (no augmentation) on Anytown + Hanoi and evaluates
zero-shot transfer to Net3 and L-TOWN (D-Town excluded due to instability).

Results are saved to artifacts/sensitivity/ as JSON and summary plots.
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
    _evaluate_on_network,
    FEATURE_COLUMNS,
    TRAIN_NETWORK_IDS,
)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import TabularDetectorBaseline, prepare_feature_matrix  # noqa: E402

# Stable test networks only (D-Town excluded)
TEST_NETWORK_IDS = ["net3", "l_town"]
SEEDS = [42, 43, 44, 45, 46]

OUTPUT_DIR = ACTIVE_ROOT / "artifacts" / "sensitivity"


def _size_weighted_auprc(per_network: dict[str, dict[str, Any]]) -> float:
    total_rows = 0
    weighted_sum = 0.0
    for nid, result in per_network.items():
        auprc = result.get("auprc")
        rows = result.get("rows", 0)
        if auprc is not None and rows > 0:
            weighted_sum += auprc * rows
            total_rows += rows
    return weighted_sum / total_rows if total_rows > 0 else 0.0


def _run_single_config(
    network_index: dict,
    *,
    leak_area_values: list[float],
    max_candidates: int,
    sensor_coverage_levels: list[float],
    disturbance_types: list[str],
    seed: int,
) -> dict[str, Any]:
    """Train on Anytown+Hanoi, evaluate on Net3+L-TOWN for one config+seed."""

    train_split = _simulate_network_group(
        TRAIN_NETWORK_IDS,
        network_index,
        split_name="train",
        sensor_coverage_levels=sensor_coverage_levels,
        disturbance_types=disturbance_types,
        max_candidates=max_candidates,
        leak_area_values=leak_area_values,
    )
    if train_split.empty:
        return {"error": "empty train split"}

    test_frames: dict[str, pd.DataFrame] = {}
    for nid in TEST_NETWORK_IDS:
        df = _simulate_network_group(
            [nid],
            network_index,
            split_name="test",
            sensor_coverage_levels=sensor_coverage_levels,
            disturbance_types=disturbance_types,
            max_candidates=max_candidates,
            leak_area_values=leak_area_values,
        )
        if not df.empty:
            test_frames[nid] = df

    if not test_frames:
        return {"error": "no test frames"}

    detector = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    detector.fit_from_frame(train_split, feature_columns=FEATURE_COLUMNS)

    per_network: dict[str, dict[str, Any]] = {}
    for nid, test_df in test_frames.items():
        per_network[nid] = _evaluate_on_network(detector, test_df, nid)

    return {
        "per_network": per_network,
        "weighted_auprc": _size_weighted_auprc(per_network),
        "train_rows": len(train_split),
        "train_positive_rows": int(train_split["event_label"].sum()),
    }


# ---------------------------------------------------------------------------
# Sweep 1: Emitter coefficient magnitude
# ---------------------------------------------------------------------------

def sweep_emitter_magnitude(network_index: dict) -> dict:
    """Test leak detection sensitivity to emitter area magnitude."""
    configs = [
        {"label": "small_only", "values": [0.0001, 0.0003]},
        {"label": "small_medium", "values": [0.0001, 0.0003, 0.0005, 0.001]},
        {"label": "full_range", "values": [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]},
        {"label": "large_only", "values": [0.001, 0.003, 0.005]},
    ]
    results = {}
    for cfg in configs:
        seed_results = []
        for seed in SEEDS:
            r = _run_single_config(
                network_index,
                leak_area_values=cfg["values"],
                max_candidates=5,
                sensor_coverage_levels=[0.25, 0.50],
                disturbance_types=["leak", "pipe_closure", "pump_outage"],
                seed=seed,
            )
            seed_results.append(r)
        auprcs = [r["weighted_auprc"] for r in seed_results if "weighted_auprc" in r]
        results[cfg["label"]] = {
            "emitter_areas": cfg["values"],
            "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
            "std_auprc": float(np.std(auprcs)) if auprcs else None,
            "n_seeds": len(auprcs),
            "per_seed": seed_results,
        }
        print(f"  emitter={cfg['label']}: AUPRC={results[cfg['label']]['mean_auprc']:.3f} "
              f"± {results[cfg['label']]['std_auprc']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Sweep 2: Number of leak locations per network
# ---------------------------------------------------------------------------

def sweep_leak_locations(network_index: dict) -> dict:
    """Test sensitivity to number of leak candidate locations."""
    k_values = [1, 2, 3, 5, 8]
    results = {}
    for k in k_values:
        seed_results = []
        for seed in SEEDS:
            r = _run_single_config(
                network_index,
                leak_area_values=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
                max_candidates=k,
                sensor_coverage_levels=[0.25, 0.50],
                disturbance_types=["leak", "pipe_closure", "pump_outage"],
                seed=seed,
            )
            seed_results.append(r)
        auprcs = [r["weighted_auprc"] for r in seed_results if "weighted_auprc" in r]
        results[f"k={k}"] = {
            "k": k,
            "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
            "std_auprc": float(np.std(auprcs)) if auprcs else None,
            "n_seeds": len(auprcs),
            "per_seed": seed_results,
        }
        print(f"  k={k}: AUPRC={results[f'k={k}']['mean_auprc']:.3f} "
              f"± {results[f'k={k}']['std_auprc']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Sweep 3: Sensor coverage level
# ---------------------------------------------------------------------------

def sweep_sensor_coverage(network_index: dict) -> dict:
    """Test sensitivity to sensor coverage density."""
    coverage_configs = [
        {"label": "sparse_25", "levels": [0.25]},
        {"label": "moderate_50", "levels": [0.50]},
        {"label": "dense_75", "levels": [0.75]},
        {"label": "full_100", "levels": [1.00]},
    ]
    results = {}
    for cfg in coverage_configs:
        seed_results = []
        for seed in SEEDS:
            r = _run_single_config(
                network_index,
                leak_area_values=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
                max_candidates=5,
                sensor_coverage_levels=cfg["levels"],
                disturbance_types=["leak", "pipe_closure", "pump_outage"],
                seed=seed,
            )
            seed_results.append(r)
        auprcs = [r["weighted_auprc"] for r in seed_results if "weighted_auprc" in r]
        results[cfg["label"]] = {
            "coverage_levels": cfg["levels"],
            "mean_auprc": float(np.mean(auprcs)) if auprcs else None,
            "std_auprc": float(np.std(auprcs)) if auprcs else None,
            "n_seeds": len(auprcs),
            "per_seed": seed_results,
        }
        print(f"  coverage={cfg['label']}: AUPRC={results[cfg['label']]['mean_auprc']:.3f} "
              f"± {results[cfg['label']]['std_auprc']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sensitivity_summary(all_results: dict, output_dir: Path) -> None:
    """Generate a 3-panel sensitivity summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Emitter magnitude
    ax = axes[0]
    emitter = all_results.get("emitter_magnitude", {})
    labels = list(emitter.keys())
    means = [emitter[l]["mean_auprc"] or 0 for l in labels]
    stds = [emitter[l]["std_auprc"] or 0 for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color="#4C72B0", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Weighted AUPRC")
    ax.set_title("(A) Emitter Magnitude")

    # Panel 2: Leak locations
    ax = axes[1]
    locations = all_results.get("leak_locations", {})
    labels = list(locations.keys())
    means = [locations[l]["mean_auprc"] or 0 for l in labels]
    stds = [locations[l]["std_auprc"] or 0 for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color="#55A868", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Weighted AUPRC")
    ax.set_title("(B) Leak Locations (k)")

    # Panel 3: Sensor coverage
    ax = axes[2]
    coverage = all_results.get("sensor_coverage", {})
    labels = list(coverage.keys())
    means = [coverage[l]["mean_auprc"] or 0 for l in labels]
    stds = [coverage[l]["std_auprc"] or 0 for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color="#C44E52", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Weighted AUPRC")
    ax.set_title("(C) Sensor Coverage")

    plt.tight_layout()
    fig.savefig(output_dir / "sensitivity_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[info] Sensitivity plot saved: {output_dir / 'sensitivity_analysis.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
    print(f"[info] Building network index from {manifest_path}")
    network_index = _build_network_index(manifest_path)

    # Each sweep is ~5-7 h; checkpoint after every sweep so a killed run
    # resumes instead of restarting (a full run is ~20 h).
    json_path = OUTPUT_DIR / "sensitivity_results.json"
    ckpt_path = OUTPUT_DIR / "sensitivity_results.partial.json"
    all_results: dict[str, Any] = {}
    if ckpt_path.exists():
        all_results = json.loads(ckpt_path.read_text())
        print(f"[info] Resuming from checkpoint with sweeps: {sorted(all_results)}")

    sweeps = [
        ("emitter_magnitude", "Sweep 1: Emitter Magnitude", sweep_emitter_magnitude),
        ("leak_locations", "Sweep 2: Leak Locations", sweep_leak_locations),
        ("sensor_coverage", "Sweep 3: Sensor Coverage", sweep_sensor_coverage),
    ]
    for key, title, fn in sweeps:
        if key in all_results:
            print(f"\n=== {title} === (checkpointed, skipping)")
            continue
        print(f"\n=== {title} ===")
        all_results[key] = fn(network_index)
        ckpt_path.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"[info] Checkpoint saved: {ckpt_path}")

    # Save results
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    ckpt_path.unlink(missing_ok=True)
    print(f"\n[info] Results saved: {json_path}")

    # Plot
    try:
        plot_sensitivity_summary(all_results, OUTPUT_DIR)
    except Exception as exc:
        print(f"[warn] Plotting failed: {exc}", file=sys.stderr)

    print("\n[info] Sensitivity analysis complete.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
