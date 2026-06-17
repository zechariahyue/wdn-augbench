"""Non-saturated event-level endpoint experiment.

Generates a benchmark variant in which each network carries MANY normal-operation
(baseline) scenarios as scenario-level NEGATIVES, alongside the existing
disturbance scenarios. All scenarios (disturbance AND baseline) receive the same
seeded demand-noise treatment, so the only systematic difference between the
classes is the disturbance itself (no clean-vs-noisy confound).

This de-saturates the scenario-level "disturbance vs normal" label (positive rate
~ n_dist / (n_dist + n_baseline) instead of ~0.97), letting us compute a
quantitative RSI in a regime where scenario-level AUPRC is informative.

Reuses the exact RF detector + augmenter setup from run_cross_network_eval.py.
Writes results to artifacts/nonsat_endpoint/.
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_cross_network_eval as rce  # noqa: E402
from watergen.data import build_leak_scenarios, simulate_many  # noqa: E402
from watergen.data.simulation import LeakScenarioSpec  # noqa: E402
from watergen.models import TabularDetectorBaseline, prepare_feature_matrix  # noqa: E402
from watergen.models.baselines import GNNDetector  # noqa: E402

ACTIVE = HERE.parent
OUT = ACTIVE / "artifacts" / "nonsat_endpoint"
OUT.mkdir(parents=True, exist_ok=True)
LEAK_AREAS = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
COVERAGE = [0.25, 0.5]
FEATS = rce.FEATURE_COLUMNS


def specs_for_network(network_index, nid, split_name, n_baseline, seed_base):
    resolved = rce._build_split_for_networks([nid], network_index, split_name=split_name)
    specs = build_leak_scenarios(
        resolved, smoke_splits=[split_name], leak_area_values=LEAK_AREAS,
        max_candidates_per_network=5, disturbance_types=DISTURBANCES, include_baseline=False,
    )
    specs = [s for s in specs if s.benchmark_id == nid]
    if not specs:
        return []
    bpath, stage = specs[0].benchmark_path, specs[0].stage
    # seed every disturbance spec (uniform noise treatment); spec is frozen
    specs = [dataclasses.replace(s, demand_seed=seed_base + i) for i, s in enumerate(specs)]
    # add N seeded baseline (normal-operation) scenarios as scenario-level negatives
    for j in range(n_baseline):
        specs.append(LeakScenarioSpec(
            scenario_id=f"{nid}__baseline__{j}", benchmark_id=nid, benchmark_path=bpath,
            split=split_name, stage=stage, kind="baseline", demand_seed=seed_base + 100000 + j,
        ))
    return specs


def simulate_group(network_index, nids, split_name, n_baseline, seed_base):
    frames = []
    for nid in nids:
        specs = specs_for_network(network_index, nid, split_name, n_baseline, seed_base)
        if not specs:
            print(f"[warn] no specs for {nid}", file=sys.stderr); continue
        df = simulate_many(specs, sensor_coverage_levels=COVERAGE)
        frames.append(df)
        print(f"[info] {nid}: {len(specs)} scenarios -> {len(df)} rows", flush=True)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()


def scen_metrics(test_df, base_prob):
    """row-level AUPRC and max-pool scenario-level AUPRC (any-positive label)."""
    y = test_df["event_label"].to_numpy(dtype=int)
    row_auprc = float(average_precision_score(y, base_prob)) if 0 < y.sum() < len(y) else float("nan")
    by = defaultdict(list)
    for sid, s, yy in zip(test_df["scenario_id"], base_prob, y):
        by[sid].append((s, yy))
    smax, slab = [], []
    for sid, lst in by.items():
        smax.append(max(v[0] for v in lst))
        slab.append(int(any(v[1] > 0 for v in lst)))
    smax, slab = np.array(smax), np.array(slab)
    npos, n = int(slab.sum()), len(slab)
    scen_auprc = float(average_precision_score(slab, smax)) if 0 < npos < n else float("nan")
    return row_auprc, scen_auprc, n, npos


def run_seed(network_index, seed, n_baseline, test_nets):
    seed_base = seed * 1_000_000
    print(f"[info] === seed {seed} ===", flush=True)
    train_df = simulate_group(network_index, rce.TRAIN_NETWORK_IDS, "train", n_baseline, seed_base + 1)
    test_frames = {nid: simulate_group(network_index, [nid], "test", n_baseline, seed_base + 2000 + i * 7)
                   for i, nid in enumerate(test_nets)}
    train_split = train_df.copy(); train_split["split"] = "train"
    out = {}
    for aug in rce.AUGMENTER_LABELS:
        aug_train = rce._apply_augmenter(train_split, aug, random_state=seed)
        det = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
        det.fit_from_frame(aug_train, feature_columns=FEATS)
        for nid, tdf in test_frames.items():
            feats = [c for c in FEATS if c in tdf.columns]
            x, _, _ = prepare_feature_matrix(tdf, target_column="event_label", feature_columns=feats)
            prob = det.predict_proba(x)[:, 1]
            r, sc, n, npos = scen_metrics(tdf, prob)
            out[f"{aug}__{nid}"] = {"row_auprc": r, "scen_auprc": sc,
                                    "n_scen": n, "n_pos": npos, "scen_pos_rate": npos / n}
            print(f"[info]   {aug:9s} {nid:7s} row={r:.3f} scen={sc:.3f} pos_rate={npos/n:.2f}", flush=True)
    # GCN detector (sample-space k-NN graph), no augmentation
    try:
        gcn = GNNDetector(random_state=seed)
        gcn.fit_from_frame(train_split, feature_columns=FEATS)
        for nid, tdf in test_frames.items():
            feats = [c for c in FEATS if c in tdf.columns]
            x, _, _ = prepare_feature_matrix(tdf, target_column="event_label", feature_columns=feats)
            prob = gcn.predict_proba(x)[:, 1]
            r, sc, n, npos = scen_metrics(tdf, prob)
            out[f"gcn__{nid}"] = {"row_auprc": r, "scen_auprc": sc,
                                  "n_scen": n, "n_pos": npos, "scen_pos_rate": npos / n}
            print(f"[info]   {'gcn':9s} {nid:7s} row={r:.3f} scen={sc:.3f} pos_rate={npos/n:.2f}", flush=True)
    except Exception as e:
        print(f"[warn] gcn failed: {e}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-baseline", type=int, default=40)
    ap.add_argument("--test-nets", nargs="+", default=rce.TEST_NETWORK_IDS)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    manifest = ACTIVE / "configs" / "benchmark_manifest.yaml"
    network_index = rce._build_network_index(manifest)
    allres = {}
    for s in args.seeds:
        allres[f"seed_{s}"] = run_seed(network_index, s, args.n_baseline, args.test_nets)
    path = OUT / f"nonsat_results_{args.tag}.json"
    path.write_text(json.dumps({"n_baseline": args.n_baseline, "test_nets": args.test_nets,
                                "seeds": args.seeds, "per_seed": allres}, indent=2))
    print(f"[info] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
