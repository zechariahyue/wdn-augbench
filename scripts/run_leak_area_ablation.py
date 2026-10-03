"""Ablation: expanded 8-network LOLO with the `leak_area` input feature removed.

Why: `leak_area` is the true emitter area of the simulated scenario, written as a
constant on every row of a leak scenario (simulation.py: frame["leak_area"] = ...)
and listed in FEATURE_COLUMNS, so the detector receives ground-truth scenario
metadata that no operator has. In the pooled evaluation, rows from leak scenarios
are ~33% positive and rows from non-leak scenarios are ~80-86% positive, so the
trivial rule "leak_area == 0" already scores row AUPRC 0.51-0.59 (the RF: 0.52-0.63).

This driver reruns run_expanded_lolo's protocol unchanged except for dropping that
one column from the detector/augmenter feature list. Resumable per seed.

    python dev/active/scripts/run_leak_area_ablation.py            # all 5 seeds
    python dev/active/scripts/run_leak_area_ablation.py --seeds 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(ACTIVE_ROOT / "src"))
sys.path.insert(0, str(ACTIVE_ROOT / "scripts"))

import run_cross_network_eval as cne  # noqa: E402

MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUT = ACTIVE_ROOT / "artifacts" / "leak_area_ablation"
REF = ACTIVE_ROOT / "artifacts" / "expanded_lolo"
TRAIN = ["anytown", "hanoi"]
TEST = ["net3", "d_town", "l_town", "ky2", "ky3", "ky5", "ky6", "ky7"]
AUGMENTERS = ["baseline", "noise", "gmm", "smote"]
DROP = ["leak_area"]
# Variant B additionally drops the raw network-size counts, which take values <= 16 on the
# training networks and up to ~400 on the test networks (the forest extrapolates on them).
VARIANTS = {
    "no_leak_area": (["leak_area"], "leak_area_ablation"),
    "no_leak_area_no_counts": (["leak_area", "observed_junction_count", "observed_link_count"], "leak_area_ablation_nocounts"),
}


def run_one(seed: int) -> dict:
    out_json = OUT / f"ablation_seed{seed}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())
    cne.RUN_GNN_BASELINE = False
    cne.TRAIN_NETWORK_IDS = list(TRAIN)
    cne.TEST_NETWORK_IDS = list(TEST)
    cne.FEATURE_COLUMNS = [c for c in cne.FEATURE_COLUMNS if c not in DROP]
    assert not set(DROP) & set(cne.FEATURE_COLUMNS) and len(cne.FEATURE_COLUMNS) == 16 - len(DROP)
    seed_dir = OUT / f"ablation_seed{seed}_run"
    t0 = time.time()
    res = cne.run_cross_network_eval(
        MANIFEST, seed_dir,
        sensor_coverage_levels=[0.25, 0.50],
        disturbance_types=["leak", "pipe_closure", "pump_outage"],
        seed=seed,
    )
    res["_elapsed_sec"] = round(time.time() - t0, 1)
    res["_feature_columns"] = list(cne.FEATURE_COLUMNS)
    out_json.write_text(json.dumps(res, indent=2))
    csv = seed_dir / "cross_network_triage_records.csv"
    if csv.exists():
        csv.unlink()  # per-row records are large and not needed for the summary
    print(f"[done] seed {seed} in {res['_elapsed_sec']}s", flush=True)
    return res


def summarize(seeds: list[int]) -> dict:
    def collect(folder: Path, stem: str) -> dict:
        per = {n: {a: [] for a in AUGMENTERS} for n in TEST}
        for s in seeds:
            p = folder / f"{stem}{s}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            for n in TEST:
                for a in AUGMENTERS:
                    cell = r["per_augmenter"][a]["per_network"][n]
                    per[n][a].append(float(cell["auprc"]))
                    per[n].setdefault("_prior", cell["positive_rows"] / cell["rows"])
        return per

    out = {"seeds_done": [s for s in seeds if (OUT / f"ablation_seed{s}.json").exists()], "per_network": {}, "macro_eqwt": {}}
    with_f, without = collect(REF, "expanded_seed"), collect(OUT, "ablation_seed")
    done = out["seeds_done"]
    for n in TEST:
        out["per_network"][n] = {"prior": round(without[n].get("_prior", float("nan")), 4)}
        for a in AUGMENTERS:
            w = [json.loads((REF / f"expanded_seed{s}.json").read_text())["per_augmenter"][a]["per_network"][n]["auprc"] for s in done]
            out["per_network"][n][a] = {
                "with_leak_area": round(float(np.mean(w)), 4) if w else None,
                "without_leak_area": round(float(np.mean(without[n][a])), 4) if without[n][a] else None,
                "without_std": round(float(np.std(without[n][a])), 4) if without[n][a] else None,
            }
    for a in AUGMENTERS:
        wo = [out["per_network"][n][a]["without_leak_area"] for n in TEST]
        wi = [out["per_network"][n][a]["with_leak_area"] for n in TEST]
        if all(v is not None for v in wo):
            out["macro_eqwt"][a] = {"with_leak_area": round(float(np.mean(wi)), 4), "without_leak_area": round(float(np.mean(wo)), 4)}
    out["macro_prior_eqwt"] = round(float(np.mean([out["per_network"][n]["prior"] for n in TEST])), 4)
    (OUT / "ablation_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--variant", choices=list(VARIANTS), default="no_leak_area")
    args = ap.parse_args()
    DROP, _dir = VARIANTS[args.variant]
    OUT = ACTIVE_ROOT / "artifacts" / _dir
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        run_one(s)
        summ = summarize(args.seeds)
        print(json.dumps(summ["macro_eqwt"]), "prior", summ["macro_prior_eqwt"], flush=True)
