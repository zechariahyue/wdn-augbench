"""Per-disturbance LOLO: run the full augmenter + GNN-baseline suite for each
disturbance type individually, across 5 seeds.

Evaluates {baseline, noise, gmm, smote, gnn_baseline} on leak-only,
pipe-closure-only, and pump-outage-only LOLO configurations, producing
per-disturbance weighted AUPRC tables that substantiate the 'leaks are
hardest' claim with quantitative per-type breakdowns.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

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
OUTPUT_ROOT = ACTIVE_ROOT / "artifacts" / "per_disturbance_lolo"
COVERAGE = [0.25, 0.50]
TEST_NETWORKS = ["net3", "d_town", "l_town"]
AUGMENTERS = ["baseline", "noise", "gmm", "smote", "gnn_baseline"]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for dist in ["leak", "pipe_closure", "pump_outage"]:
        print(f"\n{'='*60}\n  DISTURBANCE: {dist}\n{'='*60}\n")
        dist_results = []
        for seed in SEEDS:
            seed_dir = OUTPUT_ROOT / dist / f"seed_{seed}"
            result = run_cross_network_eval(
                MANIFEST, seed_dir,
                sensor_coverage_levels=COVERAGE,
                disturbance_types=[dist],
                seed=seed,
            )
            dist_results.append(result)

        # Aggregate
        agg: dict = {}
        for aug in AUGMENTERS:
            agg[aug] = {nid: [] for nid in TEST_NETWORKS}
            agg[aug]["weighted"] = []

        for result in dist_results:
            per_aug = result.get("per_augmenter", {})
            for aug_name, aug_data in per_aug.items():
                if aug_name not in agg or "error" in aug_data:
                    continue
                pn = aug_data.get("per_network", {})
                for nid in TEST_NETWORKS:
                    v = pn.get(nid, {}).get("auprc")
                    if isinstance(v, (int, float)):
                        agg[aug_name][nid].append(float(v))
                w = aug_data.get("weighted_auprc")
                if isinstance(w, (int, float)):
                    agg[aug_name]["weighted"].append(float(w))

        summary = {}
        for aug in AUGMENTERS:
            summary[aug] = {}
            for key in TEST_NETWORKS + ["weighted"]:
                vals = agg[aug][key]
                summary[aug][key] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "n": len(vals),
                }
        all_results[dist] = summary

    out_path = OUTPUT_ROOT / "per_disturbance_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[info] Saved: {out_path}\n")

    # Print summary table
    for dist, summary in all_results.items():
        print(f"\n--- {dist} ---")
        print(f"{'Method':15s} {'Net3':>14s} {'D-Town':>14s} {'L-TOWN':>14s} {'Weighted':>14s}")
        print("-" * 75)
        for aug in AUGMENTERS:
            s = summary[aug]
            parts = []
            for nid in TEST_NETWORKS + ["weighted"]:
                m = s[nid]["mean"]
                sd = s[nid]["std"]
                parts.append(f"{m:.3f}±{sd:.3f}" if m is not None else "---")
            print(f"{aug:15s} {parts[0]:>14s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
