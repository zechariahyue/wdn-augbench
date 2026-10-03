"""v2 benchmark: simulate + score, writing per-row records (S36 redesign).

Design (docs/superpowers/plans/2026-10-03-benchmark-v2-redesign.md):
  * every network (train and test) carries N normal-operation scenarios with distinct
    demand seeds; disturbance scenarios get the same seeded demand noise (no clean-vs-noisy
    confound), so disturbances are rare at the scenario level (prior = n_dist/(n_dist+N));
  * sensor layout is drawn at random per (run seed, network, coverage); leak onset and
    length are sampled per scenario;
  * detector inputs are per-network-normalised sensor aggregates (FEATURE_COLUMNS_V2);
    no scenario metadata, no network-size counts;
  * arms: RF x {baseline, noise, gmm, smote} on v2 features; RF on raw aggregates
    (ablation); LR baseline; sample-space GCN baseline. The label-free residual
    comparator is computed later from the recorded residual proxies (analyze_v2.py).

Per seed it writes artifacts/v2_benchmark/records_seed{S}.parquet (one row per test
timestep x coverage, with every arm's score) and summary_seed{S}.json. Resumable.

    python dev/active/scripts/run_v2_benchmark.py                      # 5 seeds, 8 nets, N=100
    python dev/active/scripts/run_v2_benchmark.py --seeds 42 --test-nets net3 --n-baseline 10 --tag canary
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parent
sys.path.insert(0, str(ACTIVE / "src"))
sys.path.insert(0, str(HERE))

import run_cross_network_eval as rce  # noqa: E402
from watergen.data import build_leak_scenarios, normalise_by_network_reference, simulate_many  # noqa: E402
from watergen.data.simulation import LeakScenarioSpec  # noqa: E402
from watergen.evaluation import HydraulicResidualScorer  # noqa: E402
from watergen.models import TabularDetectorBaseline, prepare_feature_matrix  # noqa: E402
from watergen.models.baselines import GNNDetector  # noqa: E402

LEAK_AREAS = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
COVERAGE = [0.25, 0.5]
TRAIN = ["anytown", "hanoi"]
TEST = ["net3", "d_town", "l_town", "ky2", "ky3", "ky5", "ky6", "ky7"]
AUGMENTERS = ["baseline", "noise", "gmm", "smote"]
RECORD_COLUMNS = ["seed", "network_id", "scenario_id", "scenario_kind", "sensor_coverage",
                  "time_seconds", "event_label", "mass_residual", "energy_residual"]


def _net_seed(seed: int, nid: str, salt: int) -> int:
    return int((seed * 1_000_003 + zlib.crc32(nid.encode("utf-8")) + salt) % (2**31 - 1))


def specs_for_network(network_index, nid: str, split: str, n_baseline: int, seed: int) -> list[LeakScenarioSpec]:
    resolved = rce._build_split_for_networks([nid], network_index, split_name=split)
    rng = np.random.default_rng([seed, zlib.crc32(nid.encode("utf-8")), 7])
    specs = build_leak_scenarios(
        resolved, smoke_splits=[split], leak_area_values=LEAK_AREAS, max_candidates_per_network=5,
        disturbance_types=DISTURBANCES, include_baseline=False, rng=rng,
    )
    specs = [s for s in specs if s.benchmark_id == nid]
    if not specs:
        return []
    bpath, stage = specs[0].benchmark_path, specs[0].stage
    base = _net_seed(seed, nid, 0)
    specs = [dataclasses.replace(s, demand_seed=base + i) for i, s in enumerate(specs)]
    for j in range(n_baseline):
        specs.append(LeakScenarioSpec(
            scenario_id=f"{nid}__baseline__{j:03d}", benchmark_id=nid, benchmark_path=bpath,
            split=split, stage=stage, kind="baseline", demand_seed=base + 100_000 + j,
        ))
    return specs


def simulate_network(network_index, nid: str, split: str, n_baseline: int, seed: int) -> pd.DataFrame:
    specs = specs_for_network(network_index, nid, split, n_baseline, seed)
    if not specs:
        raise RuntimeError(f"no scenarios for {nid}")
    t0 = time.time()
    df = simulate_many(specs, sensor_coverage_levels=COVERAGE, placement_seed=seed)
    df, _ = normalise_by_network_reference(df)
    print(f"[sim] seed {seed} {nid:7s} {len(specs):4d} scenarios -> {len(df):7d} rows  "
          f"row prior {df.event_label.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return df


def _scores(det, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x, _, _ = prepare_feature_matrix(frame, target_column="event_label", feature_columns=cols)
    return det.predict_proba(x)[:, 1]


def _quick(frame: pd.DataFrame, s: np.ndarray) -> dict:
    y = frame["event_label"].to_numpy(int)
    row = float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    g = pd.DataFrame({"sid": frame["scenario_id"].values, "cov": frame["sensor_coverage"].values, "s": s, "y": y})
    agg = g.groupby(["sid", "cov"]).agg(smax=("s", "max"), ypos=("y", "max"))
    scen = float(average_precision_score(agg.ypos, agg.smax)) if 0 < agg.ypos.sum() < len(agg) else float("nan")
    return {"row_auprc": row, "scen_auprc": scen, "row_prior": float(y.mean()), "scen_prior": float(agg.ypos.mean())}


def run_seed(seed: int, test_nets: list[str], n_baseline: int, out: Path, *, run_gcn: bool) -> dict:
    rec_path, sum_path = out / f"records_seed{seed}.parquet", out / f"summary_seed{seed}.json"
    if rec_path.exists() and sum_path.exists():
        print(f"[skip] seed {seed} done", flush=True)
        return json.loads(sum_path.read_text())
    t_seed = time.time()
    network_index = rce._build_network_index(ACTIVE / "configs" / "benchmark_manifest.yaml")
    rce.FEATURE_COLUMNS = list(rce.FEATURE_COLUMNS_V2)  # augmenters/filter read the module global

    train = pd.concat([simulate_network(network_index, n, "train", n_baseline, seed) for n in TRAIN], ignore_index=True)
    train["split"] = "train"
    tests = {n: simulate_network(network_index, n, "test", n_baseline, seed) for n in test_nets}

    # label-free residual proxies, fitted on training normal-operation rows (raw columns)
    scorer = HydraulicResidualScorer().fit(train[train.event_label == 0])

    arms: dict[str, tuple[object, list[str]]] = {}
    for aug in AUGMENTERS:
        aug_train = rce._apply_augmenter(train, aug, random_state=seed)
        det = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
        det.fit_from_frame(aug_train, feature_columns=rce.FEATURE_COLUMNS_V2)
        arms[f"rf_{aug}"] = (det, rce.FEATURE_COLUMNS_V2)
        print(f"[fit] seed {seed} rf_{aug}: {len(aug_train)} training rows", flush=True)
    raw = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    raw.fit_from_frame(train, feature_columns=rce.FEATURE_COLUMNS_RAW)
    arms["rf_raw_baseline"] = (raw, rce.FEATURE_COLUMNS_RAW)
    lr = TabularDetectorBaseline(algorithm="logreg", random_state=seed)
    lr.fit_from_frame(train, feature_columns=rce.FEATURE_COLUMNS_V2)
    arms["lr_baseline"] = (lr, rce.FEATURE_COLUMNS_V2)
    if run_gcn:
        try:
            gcn = GNNDetector(random_state=seed)
            gcn.fit_from_frame(train, feature_columns=rce.FEATURE_COLUMNS_V2)
            arms["gcn_baseline"] = (gcn, rce.FEATURE_COLUMNS_V2)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] GCN fit failed: {exc}", file=sys.stderr, flush=True)

    summary = {"seed": seed, "n_baseline": n_baseline, "train_networks": TRAIN, "test_networks": test_nets,
               "feature_columns_v2": rce.FEATURE_COLUMNS_V2, "feature_columns_raw": rce.FEATURE_COLUMNS_RAW,
               "train_rows": int(len(train)), "train_row_prior": float(train.event_label.mean()),
               "per_network": {}, "arms": list(arms)}
    frames = []
    for nid, tdf in tests.items():
        res = scorer.score_rows(tdf)
        rec = tdf[["scenario_id", "scenario_kind", "sensor_coverage", "time_seconds", "event_label"]].copy()
        rec.insert(0, "network_id", nid)
        rec.insert(0, "seed", seed)
        rec["mass_residual"] = np.asarray(res.mass_residuals, float)
        rec["energy_residual"] = np.asarray(res.energy_residuals, float)
        summary["per_network"][nid] = {"rows": int(len(tdf)), "scenarios": int(tdf.scenario_id.nunique())}
        for arm, (det, cols) in arms.items():
            t0 = time.time()
            try:
                s = _scores(det, tdf, cols)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {arm} failed on {nid}: {exc}", file=sys.stderr, flush=True)
                s = np.full(len(tdf), np.nan)
            rec[f"score__{arm}"] = s
            q = _quick(tdf, s) if np.isfinite(s).all() else {}
            summary["per_network"][nid][arm] = q
            if q:
                print(f"[eval] seed {seed} {nid:7s} {arm:16s} row {q['row_auprc']:.3f} (prior {q['row_prior']:.3f})  "
                      f"scen {q['scen_auprc']:.3f} (prior {q['scen_prior']:.3f})  {time.time()-t0:.0f}s", flush=True)
        frames.append(rec)
    records = pd.concat(frames, ignore_index=True)
    records.to_parquet(rec_path, index=False)
    summary["elapsed_sec"] = round(time.time() - t_seed, 1)
    sum_path.write_text(json.dumps(summary, indent=2))
    print(f"[done] seed {seed} in {summary['elapsed_sec']/60:.1f} min -> {rec_path.name} ({len(records)} rows)", flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--test-nets", nargs="+", default=TEST)
    ap.add_argument("--n-baseline", type=int, default=100)
    ap.add_argument("--tag", default="v2_benchmark")
    ap.add_argument("--no-gcn", action="store_true")
    args = ap.parse_args()
    out = ACTIVE / "artifacts" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        run_seed(s, args.test_nets, args.n_baseline, out, run_gcn=not args.no_gcn)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
