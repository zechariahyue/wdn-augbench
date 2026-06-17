"""E1: intra-scale (large-to-large) LOLO fold.

Removes the small->large scale-gap confound flagged by reviewers: instead of
training on Anytown+Hanoi (19-31 junctions) and testing on 92-782-junction
networks, we leave-one-out among the three large held-out networks, training on
two of {Net3, D-Town, L-TOWN} and testing on the third. Train and test are then
matched in scale, so a negative augmentation finding here cannot be attributed
to topology-scale shift.

Heavy: every fold trains and tests on large networks. Writes per-(fold, seed)
JSON immediately and is resumable (skips completed fold-seed combos), so partial
progress survives interruption. Aggregates whatever has completed.
"""
from __future__ import annotations

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
OUT = ACTIVE_ROOT / "artifacts" / "intrascale_lolo"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
COVERAGE = [0.25, 0.50]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
LARGE = ["net3", "d_town", "l_town"]
# fold name -> (train pair, test network)
FOLDS = {
    "test_net3": (["d_town", "l_town"], "net3"),
    "test_d_town": (["net3", "l_town"], "d_town"),
    "test_l_town": (["net3", "d_town"], "l_town"),
}


def run_one(fold_name, train_nets, test_net, seed):
    out_json = OUT / f"{fold_name}_seed{seed}.json"
    if out_json.exists():
        print(f"[skip] {fold_name} seed {seed} already done", flush=True)
        return json.loads(out_json.read_text())
    cne.TRAIN_NETWORK_IDS = list(train_nets)
    cne.TEST_NETWORK_IDS = [test_net]
    seed_dir = OUT / f"{fold_name}_seed{seed}_run"
    t0 = time.time()
    print(f"[run] {fold_name} (train={train_nets}, test={test_net}) seed {seed}", flush=True)
    res = cne.run_cross_network_eval(
        MANIFEST, seed_dir,
        sensor_coverage_levels=COVERAGE,
        disturbance_types=DISTURBANCES,
        seed=seed,
    )
    res["_elapsed_sec"] = round(time.time() - t0, 1)
    res["_fold"] = fold_name
    res["_train_networks"] = list(train_nets)
    res["_test_network"] = test_net
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[done] {fold_name} seed {seed} in {res['_elapsed_sec']}s", flush=True)
    return res


def aggregate():
    """Aggregate per-augmenter test AUPRC across folds/seeds that have completed."""
    augs = ["baseline", "noise", "gmm", "smote"]
    per_fold = {f: {a: [] for a in augs} for f in FOLDS}
    for fold_name, (_, test_net) in FOLDS.items():
        for seed in SEEDS:
            p = OUT / f"{fold_name}_seed{seed}.json"
            if not p.exists():
                continue
            res = json.loads(p.read_text())
            for a in augs:
                ad = res.get("per_augmenter", {}).get(a, {})
                v = ad.get("per_network", {}).get(test_net, {}).get("auprc")
                if v is not None:
                    per_fold[fold_name][a].append(float(v))
    summary = {"folds": {}, "macro_eqwt": {}}
    for fold_name in FOLDS:
        summary["folds"][fold_name] = {
            a: {"mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v)) if v else None,
                "n_seeds": len(v)}
            for a, v in per_fold[fold_name].items()
        }
    # equal-weight macro across folds (only folds with data for that augmenter)
    for a in augs:
        means = [summary["folds"][f][a]["mean"] for f in FOLDS
                 if summary["folds"][f][a]["mean"] is not None]
        summary["macro_eqwt"][a] = float(np.mean(means)) if means else None
    (OUT / "intrascale_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Intra-scale LOLO summary (equal-weight macro AUPRC across folds) ===")
    for a in augs:
        m = summary["macro_eqwt"][a]
        print(f"  {a:<10}: {m:.4f}" if m is not None else f"  {a:<10}: (no data yet)")
    print("\n=== Per-fold (mean over completed seeds) ===")
    for fold_name in FOLDS:
        row = " ".join(
            f"{a}={summary['folds'][fold_name][a]['mean']:.3f}(n{summary['folds'][fold_name][a]['n_seeds']})"
            if summary['folds'][fold_name][a]['mean'] is not None else f"{a}=NA"
            for a in augs)
        print(f"  {fold_name:<14}: {row}")
    return summary


def main():
    grand_t0 = time.time()
    for fold_name, (train_nets, test_net) in FOLDS.items():
        for seed in SEEDS:
            try:
                run_one(fold_name, train_nets, test_net, seed)
            except Exception as e:  # keep going; partial results survive
                print(f"[error] {fold_name} seed {seed}: {e!r}", flush=True)
            aggregate()  # refresh summary after every completed combo
    print(f"\n[ALL DONE] total {round(time.time()-grand_t0,1)}s")
    aggregate()


if __name__ == "__main__":
    main()
