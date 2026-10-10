"""Split-calibration check for the v2 normalisation (S38; Codex S37 round-2 MAJOR 1).

In the v2 benchmark each test network is z-scored against its own N = 100 normal-operation
scenarios, and those same scenarios are the evaluated negatives. This check separates the
two: the normal scenarios of each (seed, network) are split in half by scenario_id; one half
is the normalisation reference (and the alarm-threshold reference), the other half plus all
disturbance scenarios is the evaluation set. Both assignments of the halves are run.

  simulate  re-simulates the test networks (same seeds as v2_benchmark, so identical rows)
            and saves the RAW aggregate features to artifacts/v2_splitcal/raw_seed{S}_{net}.parquet.
            Resumable, parallel over (seed, network).
  analyze   re-fits the v2 RF (and LR) exactly as run_v2_benchmark does, checks that
            full-reference normalisation reproduces the archived scores, then scores the
            split-reference variant and compares it with the archived full-reference scores
            on the SAME evaluation rows. Writes artifacts/v2_splitcal/splitcal_summary.json.

    python dev/active/scripts/run_v2_splitcal.py simulate --workers 3
    python dev/active/scripts/run_v2_splitcal.py analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parent
sys.path.insert(0, str(ACTIVE / "src"))
sys.path.insert(0, str(HERE))

import analyze_v2 as av  # noqa: E402
import run_cross_network_eval as rce  # noqa: E402
import run_v2_benchmark as rvb  # noqa: E402
from watergen.data import normalise_by_network_reference, simulate_many  # noqa: E402
from watergen.data.normalise import AGGREGATE_FEATURES  # noqa: E402
from watergen.models import TabularDetectorBaseline  # noqa: E402

OUT = ACTIVE / "artifacts" / "v2_splitcal"
ARCHIVE = ACTIVE / "artifacts" / "v2_benchmark"
SEEDS = [42, 43, 44, 45, 46]
KEEP = ["benchmark_id", "scenario_id", "scenario_kind", "sensor_coverage", "time_seconds", "event_label"] + AGGREGATE_FEATURES


def _sim_job(seed: int, nid: str, n_baseline: int) -> str:
    path = OUT / f"raw_seed{seed}_{nid}.parquet"
    if path.exists():
        return f"[skip] {path.name}"
    t0 = time.time()
    network_index = rce._build_network_index(ACTIVE / "configs" / "benchmark_manifest.yaml")
    specs = rvb.specs_for_network(network_index, nid, "test", n_baseline, seed)
    df = simulate_many(specs, sensor_coverage_levels=rvb.COVERAGE, placement_seed=seed)
    df[KEEP].to_parquet(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)
    return f"[sim] seed {seed} {nid} {len(df)} rows ({time.time() - t0:.0f}s)"


def simulate(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    nets = sorted(args.nets, key=lambda n: n != "l_town")  # longest jobs first
    jobs = [(s, n) for n in nets for s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_sim_job, s, n, args.n_baseline) for s, n in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)
    (OUT / "simulate.done").write_text("done")
    return 0


# ------------------------------------------------------------------ analysis
def _fit(seed: int):
    network_index = rce._build_network_index(ACTIVE / "configs" / "benchmark_manifest.yaml")
    rce.FEATURE_COLUMNS = list(rce.FEATURE_COLUMNS_V2)
    train = pd.concat([rvb.simulate_network(network_index, n, "train", 100, seed) for n in rvb.TRAIN], ignore_index=True)
    train["split"] = "train"
    rf = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    rf.fit_from_frame(rce._apply_augmenter(train, "baseline", random_state=seed), feature_columns=rce.FEATURE_COLUMNS_V2)
    lr = TabularDetectorBaseline(algorithm="logreg", random_state=seed)
    lr.fit_from_frame(train, feature_columns=rce.FEATURE_COLUMNS_V2)
    return {"rf_baseline": rf, "lr_baseline": lr}


def _halves(seed: int, nid: str, normal_ids: list[str]) -> tuple[set, set]:
    ids = sorted(normal_ids)
    perm = np.random.default_rng([seed, zlib.crc32(nid.encode("utf-8")), 11]).permutation(len(ids))
    a = {ids[i] for i in perm[: len(ids) // 2]}
    return a, set(ids) - a


def _threshold(g: pd.DataFrame, score: str) -> dict:
    """1 - 1/L quantile of the score over the normal streams in g, per coverage (as analyze_v2)."""
    thr = {}
    for cov, gc in g[g.scenario_kind == "baseline"].groupby("sensor_coverage"):
        lens = gc.groupby("scenario_id").size()
        thr[cov] = float(np.quantile(gc[score].to_numpy(float), max(0.0, 1.0 - 1.0 / lens.mean())))
    return thr


def _ttd(g: pd.DataFrame, score: str, thr: dict) -> dict:
    delays, early, fa = [], [], []
    for (_, cov), gg in g[g.scenario_kind == "leak"].groupby(["scenario_id", "sensor_coverage"], sort=False):
        yy = gg.event_label.to_numpy(int)
        if yy.sum() == 0:
            continue
        d, hit = av.detection_delay(yy, gg[score].to_numpy(float), gg.time_seconds.to_numpy(float) / 3600.0, thr[cov])
        delays.append(d); early.append(int(hit and d < av.FIRST_K_H))
    for (_, cov), gg in g[g.scenario_kind == "baseline"].groupby(["scenario_id", "sensor_coverage"], sort=False):
        fa.append(int((gg[score].to_numpy(float) >= thr[cov]).sum()))
    return {"first3h_leaks": float(np.mean(early)) if early else float("nan"),
            "ttd_median_h_leaks": float(np.median(delays)) if delays else float("nan"),
            "false_alarm_rate_normals": float(np.mean(fa)) if fa else float("nan")}


def _score(models: dict, frame: pd.DataFrame, ref_ids: set | None) -> pd.DataFrame:
    """Normalise against the normals in ref_ids (None = all normals) and score every row."""
    f = frame.copy()
    if ref_ids is not None:
        f.loc[(f.scenario_kind == "baseline") & ~f.scenario_id.isin(ref_ids), "scenario_kind"] = "baseline_eval"
    f, _ = normalise_by_network_reference(f)
    f["scenario_kind"] = frame["scenario_kind"].values
    for arm, det in models.items():
        f[f"score__{arm}"] = rvb._scores(det, f, rce.FEATURE_COLUMNS_V2)
    return f


def _metrics(g: pd.DataFrame, score: str) -> dict:
    m = av.metrics_for(g, score)
    return {k: m[k] for k in ("row_auprc", "row_prior", "scen_auprc_max", "scen_prior", "n_normal_scen",
                              "first3h_leaks", "ttd_median_h_leaks", "false_alarm_rate_normals") if k in m}


def analyze(args) -> int:
    arch_cols = ["network_id", "scenario_id", "sensor_coverage", "time_seconds",
                 "score__rf_baseline", "score__lr_baseline", "score__rf_raw_baseline"]
    per, canary = {}, {}
    for seed in args.seeds:
        t0 = time.time()
        models = _fit(seed)
        arch = pd.read_parquet(ARCHIVE / f"records_seed{seed}.parquet", columns=arch_cols)
        for nid in args.nets:
            raw = pd.read_parquet(OUT / f"raw_seed{seed}_{nid}.parquet")
            a = arch[arch.network_id == nid].reset_index(drop=True)
            keys = ["scenario_id", "sensor_coverage", "time_seconds"]
            if len(a) != len(raw) or not (a[keys].to_numpy() == raw[keys].to_numpy()).all():
                raise RuntimeError(f"seed {seed} {nid}: re-simulated rows do not align with the archive")
            full = _score(models, raw, None)
            canary[f"{seed}/{nid}"] = {arm: float(np.max(np.abs(full[f"score__{arm}"].to_numpy() - a[f"score__{arm}"].to_numpy())))
                                       for arm in models}
            if canary[f"{seed}/{nid}"]["rf_baseline"] > 1e-6 or canary[f"{seed}/{nid}"]["lr_baseline"] > 1e-3:
                raise RuntimeError(f"seed {seed} {nid}: full-reference scores differ from archive {canary[f'{seed}/{nid}']}")
            full["score__rf_raw_baseline"] = a["score__rf_raw_baseline"].to_numpy()
            normals = raw.loc[raw.scenario_kind == "baseline", "scenario_id"].unique().tolist()
            folds = []
            for ref_ids, eval_ids in (lambda h: [(h[0], h[1]), (h[1], h[0])])(_halves(seed, nid, normals)):
                split = _score(models, raw, ref_ids)
                split["score__rf_raw_baseline"] = a["score__rf_raw_baseline"].to_numpy()
                ev = (raw.scenario_kind != "baseline") | raw.scenario_id.isin(eval_ids)
                ref_rows = raw.scenario_kind.eq("baseline") & raw.scenario_id.isin(ref_ids)
                fold = {}
                for arm in ("rf_baseline", "lr_baseline", "rf_raw_baseline"):
                    sc = f"score__{arm}"
                    fold[f"{arm}__full"] = _metrics(full[ev], sc)
                    fold[f"{arm}__split"] = _metrics(split[ev], sc)
                    # alarm threshold from the reference half only, applied to the evaluation half
                    thr = _threshold(split[ref_rows], sc)
                    fold[f"{arm}__split"].update({f"{k}_refthr": v for k, v in _ttd(split[ev], sc, thr).items()})
                folds.append(fold)
            per[f"{seed}/{nid}"] = {k: {m: float(np.nanmean([f[k][m] for f in folds])) for m in folds[0][k]} for k in folds[0]}
            r = per[f"{seed}/{nid}"]
            print(f"[{seed} {nid:6s}] rf row full {r['rf_baseline__full']['row_auprc']:.3f} split {r['rf_baseline__split']['row_auprc']:.3f} "
                  f"raw {r['rf_raw_baseline__full']['row_auprc']:.3f} prior {r['rf_baseline__full']['row_prior']:.3f}", flush=True)
        print(f"[seed {seed}] {time.time() - t0:.0f}s", flush=True)

    # aggregate: per network seed mean/std, then equal-weight macro over networks
    agg = {"per_network": {}, "macro": {}}
    for nid in args.nets:
        agg["per_network"][nid] = {}
        for k in per[f"{args.seeds[0]}/{nid}"]:
            agg["per_network"][nid][k] = {m: {"mean": float(np.nanmean([per[f"{s}/{nid}"][k][m] for s in args.seeds])),
                                              "std": float(np.nanstd([per[f"{s}/{nid}"][k][m] for s in args.seeds]))}
                                          for m in per[f"{args.seeds[0]}/{nid}"][k]}
    for k in agg["per_network"][args.nets[0]]:
        agg["macro"][k] = {m: float(np.nanmean([agg["per_network"][n][k][m]["mean"] for n in args.nets]))
                           for m in agg["per_network"][args.nets[0]][k]}
    # paired network-level difference split - full for the RF, and the lift over raw under each reference
    d = {n: agg["per_network"][n]["rf_baseline__split"]["row_auprc"]["mean"] - agg["per_network"][n]["rf_baseline__full"]["row_auprc"]["mean"]
         for n in args.nets}
    agg["rf_row_split_minus_full"] = {"per_network": d, "macro": float(np.mean(list(d.values()))),
                                      "max_abs": float(np.max(np.abs(list(d.values()))))}
    out = {"design": "normals split in half by scenario_id (seeded); one half = normalisation + threshold reference, "
                     "other half + all disturbances = evaluation; both assignments averaged",
           "seeds": args.seeds, "networks": args.nets, "canary_max_abs_diff": canary, "per_seed_network": per, **agg}
    (OUT / "splitcal_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"macro": agg["macro"], "rf_row_split_minus_full": agg["rf_row_split_minus_full"]}, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
        p.add_argument("--nets", nargs="+", default=rvb.TEST)
        if name == "simulate":
            p.add_argument("--workers", type=int, default=3)
            p.add_argument("--n-baseline", type=int, default=100)
    args = ap.parse_args()
    return simulate(args) if args.cmd == "simulate" else analyze(args)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
