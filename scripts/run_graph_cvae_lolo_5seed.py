"""Run Graph-CVAE LOLO across 5 seeds and aggregate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ACTIVE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT / "scripts"))

from run_graph_cvae_lolo import run as run_graph_cvae_lolo  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_ROOT = ACTIVE_ROOT / "artifacts" / "graph_cvae_lolo_5seed_revised"
TEST_NETWORKS = ["net3", "d_town", "l_town"]


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_seed_results = []

    for seed in SEEDS:
        print(f"\n{'='*60}\n  GRAPH-CVAE SEED {seed}\n{'='*60}\n")
        seed_dir = OUTPUT_ROOT / f"seed_{seed}"
        result = run_graph_cvae_lolo(MANIFEST, seed_dir, seed=seed)
        per_seed_results.append(result)

    # --- Aggregate ---
    agg = {nid: [] for nid in TEST_NETWORKS}
    agg["macro"] = []
    for result in per_seed_results:
        pn = result.get("per_network", {})
        for nid in TEST_NETWORKS:
            v = pn.get(nid, {}).get("auprc")
            if isinstance(v, (int, float)):
                agg[nid].append(float(v))
        m = result.get("macro_auprc")
        if isinstance(m, (int, float)):
            agg["macro"].append(float(m))

    print(f"\n{'='*60}\n  GRAPH-CVAE 5-SEED AGGREGATE\n{'='*60}\n")
    print(f"{'Network':15s} {'Mean':>10s} {'Std':>10s} {'n':>4s}")
    print("-" * 45)
    summary = {}
    for nid in TEST_NETWORKS + ["macro"]:
        vals = agg[nid]
        if vals:
            m, s = np.mean(vals), np.std(vals)
            print(f"{nid:15s} {m:>10.3f} {s:>10.3f} {len(vals):>4d}")
            summary[nid] = {"mean": float(m), "std": float(s), "n": len(vals), "values": vals}
        else:
            print(f"{nid:15s} {'---':>10s} {'---':>10s} {0:>4d}")
            summary[nid] = {"mean": None, "std": None, "n": 0, "values": []}

    out_path = OUTPUT_ROOT / "aggregated_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] Aggregated: {out_path}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
