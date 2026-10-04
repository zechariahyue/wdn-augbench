"""Derive every v2 table from the per-row records written by run_v2_benchmark.py.

No simulation here. Inputs: artifacts/<tag>/records_seed*.parquet. Outputs:
artifacts/<tag>/analysis/v2_summary.json and v2_tables.md.

Metrics, per (seed, network, arm):
  row AUPRC (+ row prior); scenario AUPRC under max-pool and mean-pool over
  (scenario, coverage) streams (+ scenario prior); per-disturbance slices (one
  disturbance class against ALL normal scenarios, with the slice prior); WSTL AUPRC on
  leak scenarios (chance = within-scenario positive fraction); time-to-detection inside the
  active window at a threshold set per (seed, network, coverage, detector) on the N normal
  streams so that the expected number of exceedance rows per normal stream is one.
The label-free residual comparator is built here (causal_residual_score) from the
recorded mass/energy residual proxies, standardised by the network's normal-operation
reference (never by whole-scenario statistics).
Aggregates: seed mean/std per network; equal-weight macro; deltas vs rf_baseline with a
paired Wilcoxon over networks, the Kentucky-family check, a network bootstrap CI, and an
MDE from the between-network spread of paired deltas; RSI with a pooled scenario
bootstrap; a scenario-prior sweep by subsampling normal scenarios.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score as _ap

HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parent

WARMUP = 4        # steps before the causal z-score activates
PERSIST = 3       # persistence window (steps)
FIRST_K_H = 3.0   # early-detection horizon (hours after onset)
KY = ["ky2", "ky3", "ky5", "ky6", "ky7"]


def ap(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y) or not np.isfinite(s).all():
        return float("nan")
    return float(_ap(y, s))


# ---------------------------------------------------------------- residual comparator
def causal_residual_score(mass, energy, ref):
    """Causal, label-free residual score for one (scenario, coverage) stream sorted by time.

    Each stream is standardised by the network's NORMAL-OPERATION reference statistics
    (ref = {"mass": (mu, sd), "energy": (mu, sd)}), never by the scenario's own
    statistics; the blend is then turned into a trailing-window z-score (strictly past
    data, WARMUP steps of warm-up) and smoothed by a trailing mean over PERSIST steps.
    The value at step t depends only on steps <= t.
    """
    def z(x, key):
        mu, sd = ref[key]
        return (np.asarray(x, float) - mu) / (sd if sd > 0 else 1.0)
    blend = z(mass, "mass") + z(energy, "energy")
    T = len(blend)
    out = np.zeros(T)
    for t in range(WARMUP, T):
        hist = blend[:t]
        out[t] = (blend[t] - hist.mean()) / (hist.std() + 1e-9)
    sm = np.empty(T)
    for t in range(T):
        sm[t] = out[max(0, t - PERSIST + 1):t + 1].mean()
    return sm


def add_residual_arm(rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.sort_values(["seed", "network_id", "scenario_id", "sensor_coverage", "time_seconds"]).reset_index(drop=True)
    rec["score__resid"] = np.nan
    for (seed, net, cov), g in rec.groupby(["seed", "network_id", "sensor_coverage"], sort=False):
        normal = g[g.scenario_kind == "baseline"]
        ref = {"mass": (normal.mass_residual.mean(), normal.mass_residual.std(ddof=0)),
               "energy": (normal.energy_residual.mean(), normal.energy_residual.std(ddof=0))}
        for sid, gg in g.groupby("scenario_id", sort=False):
            rec.loc[gg.index, "score__resid"] = causal_residual_score(gg.mass_residual.values, gg.energy_residual.values, ref)
    return rec


# ---------------------------------------------------------------- per (seed, network, arm)
STREAM_KEY = ["scenario_id", "sensor_coverage"]
REP_STREAM_KEY = ["rep", "scenario_id", "sensor_coverage"]   # bootstrap replicates: every drawn copy is its own stream


def stream_table(g: pd.DataFrame, score: str, by=STREAM_KEY) -> pd.DataFrame:
    """One row per stream (default: (scenario, coverage)): max/mean score, any-positive label, kind."""
    agg = g.groupby(list(by), sort=False).agg(
        smax=(score, "max"), smean=(score, "mean"), ypos=("event_label", "max"), kind=("scenario_kind", "first"))
    return agg.reset_index()


def replicate_concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate bootstrap-drawn scenario frames, tagging each drawn copy with a replicate
    index `rep` so that a scenario drawn twice yields two streams under REP_STREAM_KEY."""
    return pd.concat(parts, keys=range(len(parts)), names=["rep", "row"]).reset_index(level="rep")


def metrics_for(g: pd.DataFrame, score: str, normal_ids: set | None = None) -> dict:
    """All metrics for one (seed, network) frame and one score column.
    normal_ids restricts the normal scenarios used (prior sweep); None = all."""
    if normal_ids is not None:
        g = g[(g.scenario_kind != "baseline") | g.scenario_id.isin(normal_ids)]
    y, s = g.event_label.to_numpy(int), g[score].to_numpy(float)
    st = stream_table(g, score)
    out = {"row_auprc": ap(y, s), "row_prior": float(y.mean()),
           "scen_auprc_max": ap(st.ypos, st.smax), "scen_auprc_mean": ap(st.ypos, st.smean),
           "scen_prior": float(st.ypos.mean()), "n_streams": int(len(st)), "n_normal_scen": int(g[g.scenario_kind == "baseline"].scenario_id.nunique())}
    for kind in ("leak", "pipe_closure", "pump_outage"):
        gk = g[(g.scenario_kind == kind) | (g.scenario_kind == "baseline")]
        if (g.scenario_kind == kind).any():
            out[f"slice_{kind}_auprc"] = ap(gk.event_label, gk[score])
            out[f"slice_{kind}_prior"] = float(gk.event_label.mean())
            out[f"n_pos_scen_{kind}"] = int(g[g.scenario_kind == kind].scenario_id.nunique())
    # WSTL on leak scenarios (the only kind with an onset inside the horizon)
    w, chance = [], []
    for _, gg in g[g.scenario_kind == "leak"].groupby(["scenario_id", "sensor_coverage"], sort=False):
        yy = gg.event_label.to_numpy(int)
        if 0 < yy.sum() < len(yy):
            w.append(ap(yy, gg[score])); chance.append(yy.mean())
    out["wstl_auprc"] = float(np.nanmean(w)) if w else float("nan")
    out["wstl_chance"] = float(np.mean(chance)) if chance else float("nan")
    # time-to-detection. Threshold: per (seed, network, coverage, detector) the 1 - 1/L quantile
    # of the detector's own scores over the N normal streams (L = mean stream length), i.e.
    # the level at which the EXPECTED number of exceedance rows per normal stream is one.
    # An alarm counts only if it fires INSIDE the active window [onset, last active step];
    # alarms after the window are misses, with the delay censored at the window end.
    normals = {}
    for (_, cov), gg in g[g.scenario_kind == "baseline"].groupby(["scenario_id", "sensor_coverage"], sort=False):
        normals.setdefault(cov, []).append(gg[score].to_numpy(float))
    if normals:
        thr = {}
        for cov, arrs in normals.items():
            flat = np.concatenate(arrs); mean_len = np.mean([len(x) for x in arrs])
            thr[cov] = float(np.quantile(flat, max(0.0, 1.0 - 1.0 / mean_len)))
        delays, early, inwin = [], [], []
        for (_, cov), gg in g[g.scenario_kind == "leak"].groupby(["scenario_id", "sensor_coverage"], sort=False):
            yy = gg.event_label.to_numpy(int)
            if yy.sum() == 0:
                continue
            t_h = gg.time_seconds.to_numpy(float) / 3600.0
            d, hit = detection_delay(yy, gg[score].to_numpy(float), t_h, thr[cov])
            delays.append(d); inwin.append(int(hit)); early.append(int(hit and d < FIRST_K_H))
        out["ttd_median_h_leaks"] = float(np.median(delays)) if delays else float("nan")
        out["first3h_leaks"] = float(np.mean(early)) if early else float("nan")
        out["detected_in_window_leaks"] = float(np.mean(inwin)) if inwin else float("nan")
        out["n_leak_streams"] = int(len(delays))
        out["false_alarm_rate_normals"] = float(np.mean([(x >= thr[cov]).sum() for cov, arrs in normals.items() for x in arrs]))
    return out


def detection_delay(yy, ss, t_h, thr):
    """(delay in hours, detected) for one leak stream: the first exceedance of thr INSIDE the
    active window [onset, last active step]. No exceedance inside the window -> not detected,
    delay censored at the end of the active window (exceedances after it do not count)."""
    act = np.where(np.asarray(yy) > 0)[0]
    onset, last = int(act[0]), int(act[-1])
    fired = np.where(np.asarray(ss, float)[onset:last + 1] >= thr)[0]
    if len(fired):
        return float(t_h[onset + int(fired[0])] - t_h[onset]), True
    return float(t_h[last] - t_h[onset]), False


# ---------------------------------------------------------------- aggregation
def aggregate(per: dict, nets: list[str], arms: list[str], seeds: list[int]) -> dict:
    """per[(seed, net, arm)] -> metric dict. Returns network means/stds and macro + tests."""
    res = {"per_network": {}, "macro_eqwt": {}, "deltas_vs_rf_baseline": {}}
    keys = sorted({k for d in per.values() for k in d})
    for net in nets:
        res["per_network"][net] = {}
        for arm in arms:
            vals = [per[(s, net, arm)] for s in seeds if (s, net, arm) in per]
            if not vals:
                continue
            res["per_network"][net][arm] = {k: (float(np.nanmean([v.get(k, np.nan) for v in vals])),
                                                float(np.nanstd([v.get(k, np.nan) for v in vals]))) for k in keys}
    for arm in arms:
        res["macro_eqwt"][arm] = {k: float(np.nanmean([res["per_network"][n][arm][k][0] for n in nets if arm in res["per_network"][n]]))
                                  for k in ("row_auprc", "scen_auprc_max", "scen_auprc_mean", "wstl_auprc", "first3h_leaks", "detected_in_window_leaks", "row_prior", "scen_prior")}
    base = np.array([res["per_network"][n]["rf_baseline"]["row_auprc"][0] for n in nets])
    prior = np.array([res["per_network"][n]["rf_baseline"]["row_prior"][0] for n in nets])
    res["baseline_vs_prior"] = {"lift_mean": float((base - prior).mean()), "wilcoxon_p": float(wilcoxon(base, prior).pvalue) if len(nets) >= 6 else None,
                                "per_network_lift": {n: float(b - p) for n, b, p in zip(nets, base, prior)}}
    rng = np.random.default_rng(0)
    for arm in arms:
        if arm == "rf_baseline" or arm not in res["macro_eqwt"]:
            continue
        x = np.array([res["per_network"][n][arm]["row_auprc"][0] if arm in res["per_network"][n] else np.nan for n in nets])
        ok = np.isfinite(x)
        if ok.sum() < 3:
            continue
        d = x[ok] - base[ok]; nn = [n for n, o in zip(nets, ok) if o]
        boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(2000)]
        ky_mask = np.array([n in KY for n in nn])
        fam = list(d[~ky_mask]) + ([d[ky_mask].mean()] if ky_mask.any() else [])
        res["deltas_vs_rf_baseline"][arm] = {
            "delta_mean": float(d.mean()), "wins": int((d > 0).sum()), "n": int(len(d)),
            "wilcoxon_p": float(wilcoxon(x[ok], base[ok]).pvalue) if len(d) >= 6 else None,
            "network_bootstrap_95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "family_level_deltas": [float(v) for v in fam],
            "mde_between_network": float(2.5 * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
            "per_network": {n: float(v) for n, v in zip(nn, d)},
        }
    return res


def rsi_with_bootstrap(rec: pd.DataFrame, nets, seeds, pairs, B=500, normal_ids_by_net=None):
    """RSI(A,B) = (rowA-rowB) - (scenA-scenB) per network: seed mean, seed std, and a pooled
    scenario bootstrap (resample scenario ids within the network, recompute every seed,
    average over seeds; percentile interval of that seed-averaged statistic). Each drawn
    copy of a scenario is tagged with a replicate index and counted as its own stream
    (replicate_concat + REP_STREAM_KEY), so the scenario-level term is reweighted exactly
    like the row-level term. One set of B draws is shared by all pairs of a network."""
    rng = np.random.default_rng(1)
    out = {}
    cols = sorted({c for a, b in pairs for c in (a, b) if c in rec.columns})
    for net in nets:
        gnet = rec[rec.network_id == net]
        if normal_ids_by_net is not None:
            gnet = gnet[(gnet.scenario_kind != "baseline") | gnet.scenario_id.isin(normal_ids_by_net[net])]
        sids = sorted(gnet.scenario_id.unique())
        # cache per (seed, scenario): row label/score arrays and the per-coverage stream table
        cache = {}
        for s in seeds:
            for sid, g in gnet[gnet.seed == s].groupby("scenario_id", sort=False):
                st = g.groupby(STREAM_KEY, sort=False).agg(ypos=("event_label", "max"), kind=("scenario_kind", "first"),
                                                          **{c: (c, "max") for c in cols}).reset_index()
                cache[(s, sid)] = (g.event_label.to_numpy(int), {c: g[c].to_numpy(float) for c in cols}, st)

        def gaps(chosen):
            """Per seed: {col: (row AP, scenario AP)} for the drawn (possibly repeated) scenarios."""
            res = []
            for s in seeds:
                parts = [cache[(s, sid)] for sid in chosen if (s, sid) in cache]
                if not parts:
                    continue
                y = np.concatenate([p[0] for p in parts])
                streams = replicate_concat([p[2] for p in parts])          # one stream per drawn copy
                res.append({c: (ap(y, np.concatenate([p[1][c] for p in parts])), ap(streams.ypos, streams[c])) for c in cols})
            return res

        def rsi_from(gp, a, b):
            return [(g[a][0] - g[b][0]) - (g[a][1] - g[b][1]) for g in gp]

        point = gaps(sids)
        boots = [gaps(list(rng.choice(sids, size=len(sids), replace=True))) for _ in range(B)]
        out[net] = {}
        for a, b in pairs:
            if a not in cols or b not in cols:
                continue
            pv = rsi_from(point, a, b)
            bv = [float(np.nanmean(rsi_from(gp, a, b))) for gp in boots if gp]
            out[net][f"RSI({a},{b})"] = {"mean": float(np.nanmean(pv)), "seed_std": float(np.nanstd(pv)),
                                         "bootstrap_95": [float(np.nanpercentile(bv, 2.5)), float(np.nanpercentile(bv, 97.5))] if bv else None}
    return out


def _f(x, spec=".3f"):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def md_tables(summary: dict) -> str:
    L = ["# v2 benchmark analysis", ""]
    nets = summary["networks"]; arms = summary["arms"]
    L += ["## Row AUPRC (seed mean ± std) and row prior", "", "| network | prior | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for n in nets:
        pn = summary["aggregate"]["per_network"][n]
        L.append(f"| {n} | {pn['rf_baseline']['row_prior'][0]:.3f} | " + " | ".join(f"{_f(pn[a]['row_auprc'][0])}±{_f(pn[a]['row_auprc'][1])}" if a in pn else "—" for a in arms) + " |")
    L.append("| **macro** | " + f"{summary['aggregate']['macro_eqwt']['rf_baseline']['row_prior']:.3f} | " + " | ".join(f"**{_f(summary['aggregate']['macro_eqwt'][a]['row_auprc'])}**" if a in summary['aggregate']['macro_eqwt'] else "—" for a in arms) + " |")
    L += ["", "## Scenario AUPRC, max-pool (seed mean) and scenario prior", "", "| network | prior | " + " | ".join(arms) + " |", "|---|---|" + "---|" * len(arms)]
    for n in nets:
        pn = summary["aggregate"]["per_network"][n]
        L.append(f"| {n} | {pn['rf_baseline']['scen_prior'][0]:.3f} | " + " | ".join(_f(pn[a]['scen_auprc_max'][0]) if a in pn else "—" for a in arms) + " |")
    L += ["", "## Deltas vs rf_baseline (row AUPRC, paired over networks)", "", "| arm | Δ mean | wins | Wilcoxon p | network bootstrap 95% | MDE (between-network) |", "|---|---|---|---|---|---|"]
    for a, d in summary["aggregate"]["deltas_vs_rf_baseline"].items():
        L.append(f"| {a} | {_f(d['delta_mean'], '+.3f')} | {d['wins']}/{d['n']} | {_f(d['wilcoxon_p'])} | [{_f(d['network_bootstrap_95'][0], '+.3f')}, {_f(d['network_bootstrap_95'][1], '+.3f')}] | {_f(d['mde_between_network'])} |")
    b = summary["aggregate"]["baseline_vs_prior"]
    L += ["", f"Baseline lift over prior: mean {_f(b['lift_mean'], '+.3f')}, Wilcoxon p = {_f(b['wilcoxon_p'])}; per network: " + ", ".join(f"{n} {v:+.3f}" for n, v in b["per_network_lift"].items()), ""]
    L += ["## Early warning (rf_baseline vs resid): WSTL on leak scenarios, first-3h / in-window detection", "", "| network | WSTL rf | WSTL resid | chance | first-3h rf | first-3h resid | in-window rf | in-window resid | FA/normal rf | FA/normal resid |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in nets:
        pn = summary["aggregate"]["per_network"][n]
        if "resid" in pn:
            L.append(f"| {n} | {_f(pn['rf_baseline']['wstl_auprc'][0])} | {_f(pn['resid']['wstl_auprc'][0])} | {_f(pn['rf_baseline']['wstl_chance'][0])} | {_f(pn['rf_baseline']['first3h_leaks'][0])} | {_f(pn['resid']['first3h_leaks'][0])} | {_f(pn['rf_baseline']['detected_in_window_leaks'][0])} | {_f(pn['resid']['detected_in_window_leaks'][0])} | {_f(pn['rf_baseline']['false_alarm_rate_normals'][0], '.2f')} | {_f(pn['resid']['false_alarm_rate_normals'][0], '.2f')} |")
    L += ["", "## RSI (seed mean ± seed std; pooled scenario bootstrap 95%)", ""]
    for n, d in summary["rsi"].items():
        for k, v in d.items():
            ci = f"[{_f(v['bootstrap_95'][0], '+.3f')}, {_f(v['bootstrap_95'][1], '+.3f')}]" if v["bootstrap_95"] else "—"
            L.append(f"- {n} {k}: {_f(v['mean'], '+.3f')} ± {_f(v['seed_std'])} {ci}")
    L += ["", "## Scenario-prior sweep (first n normal scenarios kept per network): rf_baseline scen AUPRC max-pool and RSI(resid, rf_baseline)", ""]
    for nb, d in summary["prior_sweep"].items():
        L.append(f"- N normals = {nb}: " + "; ".join(f"{n}: prior {_f(v['scen_prior'], '.2f')}, scen {_f(v['scen_auprc_max'])}, RSI {_f(v['rsi_resid_rf'], '+.3f')}" for n, v in d.items()))
    return "\n".join(L) + "\n"


def main() -> int:
    apr = argparse.ArgumentParser()
    apr.add_argument("--tag", default="v2_benchmark")
    apr.add_argument("--bootstrap", type=int, default=500)
    args = apr.parse_args()
    d = ACTIVE / "artifacts" / args.tag
    files = sorted(d.glob("records_seed*.parquet"))
    if not files:
        print("no records", file=sys.stderr); return 1
    rec = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rec = add_residual_arm(rec)
    seeds = sorted(rec.seed.unique().tolist()); nets = [n for n in ["net3", "d_town", "l_town", "ky2", "ky3", "ky5", "ky6", "ky7"] if n in set(rec.network_id)]
    score_cols = [c for c in rec.columns if c.startswith("score__")]
    arms = [c.replace("score__", "") for c in score_cols]
    print(f"[analyze] {len(rec)} rows, seeds {seeds}, nets {nets}, arms {arms}", flush=True)

    per = {}
    for (s, n), g in rec.groupby(["seed", "network_id"], sort=False):
        for arm, col in zip(arms, score_cols):
            if g[col].notna().all():
                per[(s, n, arm)] = metrics_for(g, col)
    agg = aggregate(per, nets, arms, seeds)

    pairs = [("score__resid", "score__rf_baseline")] + [(f"score__rf_{a}", "score__rf_baseline") for a in ("noise", "gmm", "smote")]
    if "score__gcn_baseline" in rec.columns and rec["score__gcn_baseline"].notna().all():
        pairs.append(("score__gcn_baseline", "score__rf_baseline"))
    rsi = rsi_with_bootstrap(rec, nets, seeds, pairs, B=args.bootstrap)
    rsi = {n: {k.replace("score__", ""): v for k, v in d_.items()} for n, d_ in rsi.items()}

    # prior sweep: keep the first n normal scenarios per network
    sweep = {}
    all_normals = {n: sorted(rec[(rec.network_id == n) & (rec.scenario_kind == "baseline")].scenario_id.unique()) for n in nets}
    n_total = min(len(v) for v in all_normals.values())
    for nb in sorted({20, 40, n_total} if n_total >= 40 else {n_total}):
        ids = {n: set(all_normals[n][:nb]) for n in nets}
        sweep[nb] = {}
        for n in nets:
            vals, r = [], []
            for s in seeds:
                g = rec[(rec.seed == s) & (rec.network_id == n)]
                m_rf = metrics_for(g, "score__rf_baseline", ids[n]); m_rs = metrics_for(g, "score__resid", ids[n])
                vals.append(m_rf); r.append((m_rs["row_auprc"] - m_rf["row_auprc"]) - (m_rs["scen_auprc_max"] - m_rf["scen_auprc_max"]))
            sweep[nb][n] = {"scen_prior": float(np.mean([v["scen_prior"] for v in vals])), "scen_auprc_max": float(np.nanmean([v["scen_auprc_max"] for v in vals])),
                            "row_prior": float(np.mean([v["row_prior"] for v in vals])), "rsi_resid_rf": float(np.nanmean(r))}

    summary = {"tag": args.tag, "seeds": seeds, "networks": nets, "arms": arms, "n_rows": int(len(rec)),
               "aggregate": agg, "rsi": rsi, "prior_sweep": sweep,
               "per_seed_network_arm": {f"{s}|{n}|{a}": v for (s, n, a), v in per.items()}}
    out = d / "analysis"; out.mkdir(exist_ok=True)
    (out / "v2_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "v2_tables.md").write_text(md_tables(summary), encoding="utf-8")
    print((out / "v2_tables.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
