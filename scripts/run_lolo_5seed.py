"""Run the LOLO cross-network evaluation across 5 seeds and aggregate.

Calls run_cross_network_eval for each seed, then aggregates per-network
AUPRC values into mean ± std for manuscript Table 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ACTIVE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT / "scripts"))

from run_cross_network_eval import run_cross_network_eval  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_ROOT = ACTIVE_ROOT / "artifacts" / "lolo_5seed_revised"
COVERAGE = [0.25, 0.50]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]

TEST_NETWORKS = ["net3", "d_town", "l_town"]
AUGMENTERS = ["baseline", "noise", "gmm", "smote"]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_seed_results: list[dict] = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}\n")
        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        result = run_cross_network_eval(
            MANIFEST, seed_dir,
            sensor_coverage_levels=COVERAGE,
            disturbance_types=DISTURBANCES,
            seed=seed,
        )
        all_seed_results.append(result)

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print("  AGGREGATION")
    print(f"{'='*60}\n")

    agg: dict[str, dict[str, list[float]]] = {}
    for aug in AUGMENTERS + ["gnn_baseline"]:
        agg[aug] = {nid: [] for nid in TEST_NETWORKS}
        agg[aug]["weighted"] = []

    for result in all_seed_results:
        per_aug = result.get("per_augmenter", {})
        for aug_name, aug_data in per_aug.items():
            if aug_name not in agg:
                continue
            if "error" in aug_data:
                continue
            pn = aug_data.get("per_network", {})
            for nid in TEST_NETWORKS:
                val = pn.get(nid, {}).get("auprc")
                if val is not None:
                    agg[aug_name][nid].append(float(val))
            w = aug_data.get("weighted_auprc")
            if w is not None:
                agg[aug_name]["weighted"].append(float(w))

    # --- Print Table 3 format ---
    print(f"{'Augmenter':20s} {'Net3':>18s} {'D-Town':>18s} {'L-TOWN':>18s} {'Macro':>12s}")
    print("-" * 90)
    for aug in AUGMENTERS + ["gnn_baseline"]:
        parts = []
        for nid in TEST_NETWORKS:
            vals = agg[aug][nid]
            if vals:
                m, s = np.mean(vals), np.std(vals)
                parts.append(f"{m:.3f} ± {s:.3f}")
            else:
                parts.append("---")
        wvals = agg[aug]["weighted"]
        macro = f"{np.mean(wvals):.3f}" if wvals else "---"
        print(f"{aug:20s} {parts[0]:>18s} {parts[1]:>18s} {parts[2]:>18s} {macro:>12s}")

    # --- Save ---
    summary = {}
    for aug in AUGMENTERS + ["gnn_baseline"]:
        summary[aug] = {}
        for nid in TEST_NETWORKS + ["weighted"]:
            vals = agg[aug][nid]
            summary[aug][nid] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "n": len(vals),
                "values": vals,
            }

    out_path = OUTPUT_ROOT / "aggregated_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] Aggregated results saved: {out_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
