"""RSI(hydraulic-residual detector, RF) on the balanced non-saturated endpoint.

Answers reviewer FATAL-1 (PEER_REVIEW_S33): the manuscript's only quantitative RSI
demonstration (Table 6) is computed on the saturated LOLO scenario metric (one
negative scenario per network, prior 0.97), where RSI reduces to minus the row gap
by construction. This script re-simulates the balanced endpoint exactly as
run_nonsat_endpoint.py does (same seeds, same demand seeds, 40 seeded baseline
scenarios per network -> scenario prior ~0.47), trains only the no-augmentation RF,
scores every test row with both the RF and the label-free causal residual detector
of compute_early_warning_endpoint.py, and reports row/scenario AUPRC, RSI, and a
scenario-level bootstrap CI (reviewer MAJOR-3). Per-row records are archived.

Outputs: artifacts/nonsat_hydraulic_rsi/{results.json, records_seed{S}.csv.gz}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nonsat_endpoint as ne  # noqa: E402  (reuses its simulation + RF setup)
from compute_early_warning_endpoint import causal_residual_score  # noqa: E402

OUT = ne.ACTIVE / "artifacts" / "nonsat_hydraulic_rsi"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000


def hydraulic_scores(tdf: pd.DataFrame) -> np.ndarray:
    """Causal residual+persistence score, computed per (scenario, coverage) stream."""
    s = np.zeros(len(tdf))
    for _, idx in tdf.groupby(["scenario_id", "sensor_coverage"]).groups.items():
        sub = tdf.loc[idx].sort_values("time_seconds")
        rows = sub[["mass_residual", "energy_residual"]].to_dict("records")
        s[tdf.index.get_indexer(sub.index)] = causal_residual_score(rows)
    return s


def scen_table(tdf, score):
    y = tdf["event_label"].to_numpy(int)
    by = defaultdict(list)
    for sid, sc, yy in zip(tdf["scenario_id"], score, y):
        by[sid].append((sc, yy))
    sids = list(by)
    smax = np.array([max(v[0] for v in by[s]) for s in sids])
    slab = np.array([int(any(v[1] > 0 for v in by[s])) for s in sids])
    return sids, smax, slab


def rsi_with_ci(tdf, s_hyd, s_rf, rng):
    y = tdf["event_label"].to_numpy(int)
    row_h, row_r = average_precision_score(y, s_hyd), average_precision_score(y, s_rf)
    sids, hmax, slab = scen_table(tdf, s_hyd)
    _, rmax, _ = scen_table(tdf, s_rf)
    scen_h, scen_r = average_precision_score(slab, hmax), average_precision_score(slab, rmax)
    rsi = (row_h - row_r) - (scen_h - scen_r)
    # scenario-level bootstrap: resample scenarios with replacement, recompute both units.
    # Row AUPRC under resampling = AP with per-row weights equal to the scenario's draw
    # multiplicity (avoids concatenating ~1e5-row frames 2000 times on L-TOWN).
    sid_index = {s: i for i, s in enumerate(sids)}
    row_sid = np.array([sid_index[s] for s in tdf["scenario_id"]])
    draws = []
    n = len(sids)
    for _ in range(B):
        pick = rng.integers(0, n, n)
        mult = np.bincount(pick, minlength=n)
        w = mult[row_sid].astype(float)
        yw = y[w > 0]
        if 0 < yw.sum() < len(yw) and 0 < slab[pick].sum() < n:
            rh = average_precision_score(y, s_hyd, sample_weight=w)
            rr = average_precision_score(y, s_rf, sample_weight=w)
            sh, sr = average_precision_score(slab[pick], hmax[pick]), average_precision_score(slab[pick], rmax[pick])
            draws.append((rh - rr) - (sh - sr))
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    return dict(row_hyd=row_h, row_rf=row_r, scen_hyd=scen_h, scen_rf=scen_r, rsi=rsi,
                rsi_ci_lo=float(lo), rsi_ci_hi=float(hi), n_scen=n, n_pos=int(slab.sum()),
                scen_prior=float(slab.mean()), row_prior=float(y.mean()), n_boot=len(draws))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-baseline", type=int, default=40)
    ap.add_argument("--test-nets", nargs="+", default=ne.rce.TEST_NETWORK_IDS)
    args = ap.parse_args()
    manifest = ne.ACTIVE / "configs" / "benchmark_manifest.yaml"
    network_index = ne.rce._build_network_index(manifest)
    results = {"n_baseline": args.n_baseline, "test_nets": args.test_nets, "B": B, "per_seed": {}}
    for seed in args.seeds:
        seed_base = seed * 1_000_000  # identical to run_nonsat_endpoint.run_seed
        print(f"[info] === seed {seed} ===", flush=True)
        train_df = ne.simulate_group(network_index, ne.rce.TRAIN_NETWORK_IDS, "train", args.n_baseline, seed_base + 1)
        train_df["split"] = "train"
        det = ne.TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
        det.fit_from_frame(train_df, feature_columns=ne.FEATS)
        # residual columns come from the plausibility scorer fitted on normal training rows,
        # exactly as run_cross_network_eval does before writing the LOLO records
        normal_train = train_df.loc[train_df["event_label"] == 0]
        scorer = ne.rce.HydraulicResidualScorer().fit(normal_train if not normal_train.empty else train_df)
        rng = np.random.default_rng(seed)
        recs = []
        for i, nid in enumerate(args.test_nets):
            tdf = ne.simulate_group(network_index, [nid], "test", args.n_baseline, seed_base + 2000 + i * 7)
            tdf = scorer.score_rows(tdf.reset_index(drop=True)).frame.reset_index(drop=True)
            for c in ("time_seconds", "sensor_coverage", "mass_residual", "energy_residual", "scenario_id", "event_label"):
                assert c in tdf.columns, f"missing column {c}"
            feats = [c for c in ne.FEATS if c in tdf.columns]
            x, _, _ = ne.prepare_feature_matrix(tdf, target_column="event_label", feature_columns=feats)
            s_rf = det.predict_proba(x)[:, 1]
            s_hyd = hydraulic_scores(tdf)
            r = rsi_with_ci(tdf, s_hyd, s_rf, rng)
            results["per_seed"].setdefault(str(seed), {})[nid] = r
            print(f"[info]   {nid:7s} row hyd/rf={r['row_hyd']:.3f}/{r['row_rf']:.3f}  "
                  f"scen hyd/rf={r['scen_hyd']:.3f}/{r['scen_rf']:.3f}  prior={r['scen_prior']:.2f}  "
                  f"RSI={r['rsi']:+.3f} [{r['rsi_ci_lo']:+.3f},{r['rsi_ci_hi']:+.3f}]", flush=True)
            keep = tdf[["scenario_id", "sensor_coverage", "time_seconds", "event_label",
                        "mass_residual", "energy_residual"]].copy()
            keep["network_id"], keep["rf_score"], keep["hyd_score"] = nid, s_rf, s_hyd
            recs.append(keep)
        pd.concat(recs).to_csv(OUT / f"records_seed{seed}.csv.gz", index=False, compression="gzip")
        (OUT / "results.json").write_text(json.dumps(results, indent=2))  # checkpoint per seed
    # summary over seeds
    summ = {}
    for nid in args.test_nets:
        vals = [results["per_seed"][s][nid] for s in results["per_seed"] if nid in results["per_seed"][s]]
        summ[nid] = {k: (float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])))
                     for k in ("row_hyd", "row_rf", "scen_hyd", "scen_rf", "rsi", "scen_prior")}
        summ[nid]["rsi_ci_lo_mean"] = float(np.mean([v["rsi_ci_lo"] for v in vals]))
        summ[nid]["rsi_ci_hi_mean"] = float(np.mean([v["rsi_ci_hi"] for v in vals]))
        summ[nid]["n_seeds"] = len(vals)
    results["summary"] = summ
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== RSI(hydraulic, RF) on the balanced endpoint (mean +- seed std; mean bootstrap CI) ===")
    for nid, s in summ.items():
        print(f"{nid:7s} RSI {s['rsi'][0]:+.3f} +- {s['rsi'][1]:.3f}  CI[{s['rsi_ci_lo_mean']:+.3f},{s['rsi_ci_hi_mean']:+.3f}]  "
              f"row {s['row_hyd'][0]:.3f}/{s['row_rf'][0]:.3f}  scen {s['scen_hyd'][0]:.3f}/{s['scen_rf'][0]:.3f}  prior {s['scen_prior'][0]:.2f}")


if __name__ == "__main__":
    main()
