"""Classifier-generality check for the LOLO augmentation negative finding.

Third reviewer (MAJOR): Table 8 evaluates augmenters only with RandomForest as
the downstream classifier. If augmentation helps a weaker classifier (LogReg)
where it does not help RF, the negative finding is classifier-specific.

This script re-runs the EXACT standard saturated LOLO protocol (include_baseline
=True, leak+pipe_closure+pump_outage, coverage 0.25/0.50) on the held-out
networks, fitting BOTH RandomForest and LogisticRegression on each augmenter's
training set in a single simulation pass. RF reproduces Table 8 per-network
AUPRC (validation); LR answers the classifier-generality question.

Test nets restricted to net3 + d_town for tractability (l_town is the 5-min /
168-h cost driver and is omitted, as elsewhere for the GCN subset); the RF-vs-LR
comparison is internally consistent on this common subset. No change to the
augmenter family or the simulation; only the downstream classifier varies.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import run_cross_network_eval as rce  # noqa: E402
from watergen.models import TabularDetectorBaseline  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
TEST_NETS = ["net3", "d_town"]
COVERAGE = [0.25, 0.50]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
ALGOS = {"rf": "random_forest", "lr": "logreg"}
OUT = HERE.parent / "artifacts" / "lolo_classifier_check"
OUT.mkdir(parents=True, exist_ok=True)


def run_seed(network_index, seed):
    train_df = rce._simulate_network_group(
        rce.TRAIN_NETWORK_IDS, network_index, split_name="train",
        sensor_coverage_levels=COVERAGE, disturbance_types=DISTURBANCES)
    test_frames = {}
    for nid in TEST_NETS:
        tdf = rce._simulate_network_group(
            [nid], network_index, split_name="test",
            sensor_coverage_levels=COVERAGE, disturbance_types=DISTURBANCES)
        test_frames[nid] = tdf
    train_split = train_df.copy()
    train_split["split"] = "train"
    out = {}
    for aug in rce.AUGMENTER_LABELS:
        try:
            aug_train = rce._apply_augmenter(train_split, aug, random_state=seed)
        except Exception as exc:
            print(f"[warn] augmenter {aug} failed: {exc}", file=sys.stderr)
            aug_train = train_split.copy()
        for algo_key, algo_name in ALGOS.items():
            det = TabularDetectorBaseline(algorithm=algo_name, random_state=seed)
            det.fit_from_frame(aug_train, feature_columns=rce.FEATURE_COLUMNS)
            per_net = {}
            for nid, tdf in test_frames.items():
                res = rce._evaluate_on_network(det, tdf, nid)
                per_net[nid] = res
            macro = rce.size_weighted_auprc(per_net)
            out[f"{aug}__{algo_key}"] = {
                "macro_auprc": macro,
                "per_net": {nid: per_net[nid].get("auprc") for nid in TEST_NETS},
            }
            pn = "  ".join(f"{nid}={per_net[nid].get('auprc'):.3f}" for nid in TEST_NETS)
            print(f"[info]   seed{seed} {aug:9s} {algo_key:3s} macro={macro:.3f}  {pn}", flush=True)
    return out


def main():
    manifest = HERE.parent / "configs" / "benchmark_manifest.yaml"
    network_index = rce._build_network_index(manifest)
    allres = {}
    for s in SEEDS:
        print(f"[info] === seed {s} ===", flush=True)
        allres[f"seed_{s}"] = run_seed(network_index, s)
    (OUT / "classifier_check_results.json").write_text(json.dumps(
        {"seeds": SEEDS, "test_nets": TEST_NETS, "per_seed": allres}, indent=2))

    # aggregate
    print("\n=== 5-seed macro AUPRC (net3+d_town, node-weighted) ===")
    print(f"{'augmenter':10s} {'RF':>16s} {'LR':>16s}")
    agg = {}
    for aug in rce.AUGMENTER_LABELS:
        row = {}
        for algo_key in ALGOS:
            vals = [allres[f"seed_{s}"][f"{aug}__{algo_key}"]["macro_auprc"] for s in SEEDS]
            row[algo_key] = (float(np.mean(vals)), float(np.std(vals)))
        agg[aug] = row
        print(f"{aug:10s} {row['rf'][0]:.3f} +/- {row['rf'][1]:.3f}   "
              f"{row['lr'][0]:.3f} +/- {row['lr'][1]:.3f}")
    base = {k: agg['baseline'][k][0] for k in ALGOS}
    print("\n=== augmenter - baseline delta (does augmentation help?) ===")
    print(f"{'augmenter':10s} {'RF delta':>10s} {'LR delta':>10s}")
    for aug in ["noise", "gmm", "smote"]:
        print(f"{aug:10s} {agg[aug]['rf'][0]-base['rf']:+10.3f} {agg[aug]['lr'][0]-base['lr']:+10.3f}")
    (OUT / "classifier_check_summary.json").write_text(json.dumps({
        "description": "5-seed standard LOLO (net3+d_town, node-weighted macro AUPRC). "
                       "RF vs LR downstream classifier x augmenter family. Tests whether "
                       "the augmentation negative finding is classifier-specific.",
        "macro": {aug: {k: {"mean": agg[aug][k][0], "std": agg[aug][k][1]} for k in ALGOS}
                  for aug in rce.AUGMENTER_LABELS},
        "baseline_deltas": {aug: {k: agg[aug][k][0]-base[k] for k in ALGOS}
                            for aug in ["noise", "gmm", "smote"]},
    }, indent=2))
    print(f"\n[info] wrote {OUT/'classifier_check_summary.json'}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
