"""Early warning at a common false-alarm budget (episodes per day of normal operation).

analyze_v2.py sets one threshold per (seed, network, coverage, detector) so that each normal
stream expects one exceedance row; with 24 h streams on Net3/Kentucky and 168 h streams on
D-Town/L-TOWN that is a different alarm burden per day. Here the threshold is instead the
lowest level at which the normal streams of that (seed, network, coverage) raise at most B
false-alarm episodes (runs of consecutive exceedances) per day, for B in BUDGETS. Per leak
stream we record detection inside the active window, detection with delay < 3 h, and whether
the stream already exceeds the threshold before onset. Inputs: records_seed*.parquet.
Output: artifacts/v2_benchmark/analysis/v2_alarm_budget.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v2 import FIRST_K_H, add_residual_arm, detection_delay  # noqa: E402

ACTIVE = Path(__file__).resolve().parent.parent
TAG = ACTIVE / "artifacts" / "v2_benchmark"
BUDGETS = (0.5, 1.0, 2.0)
ARMS = ("rf_baseline", "resid")
LEVELS = 1.0 - np.geomspace(0.3, 1e-6, 400)


def episodes(x, thr):
    return int(np.sum(np.diff(np.r_[0, (x >= thr).astype(int)]) == 1))


def budget_threshold(normals, days, b):
    """Lowest candidate threshold whose normal streams give <= b episodes per day."""
    flat = np.concatenate(normals)
    for thr in np.unique(np.quantile(flat, LEVELS)):
        if sum(episodes(x, thr) for x in normals) / days <= b:
            return float(thr)
    return float(np.nextafter(flat.max(), np.inf))


def per_frame(g, arm):
    col = f"score__{arm}"
    step_h = float(np.median(np.diff(np.sort(g.time_seconds.unique())))) / 3600.0
    out = {"step_h": step_h, "opportunities_3h": int(round(FIRST_K_H / step_h))}
    for b in BUDGETS:
        early, inwin, pre = [], [], []
        for cov, gc in g.groupby("sensor_coverage"):
            nm = [gg[col].to_numpy(float) for _, gg in gc[gc.scenario_kind == "baseline"].groupby("scenario_id")]
            days = sum(len(x) for x in nm) * step_h / 24.0
            thr = budget_threshold(nm, days, b)
            for _, gg in gc[gc.scenario_kind == "leak"].groupby("scenario_id"):
                yy = gg.event_label.to_numpy(int)
                if yy.sum() == 0:
                    continue
                ss = gg[col].to_numpy(float)
                t_h = gg.time_seconds.to_numpy(float) / 3600.0
                d, hit = detection_delay(yy, ss, t_h, thr)
                onset = int(np.argmax(yy > 0))
                inwin.append(int(hit)); early.append(int(hit and d < FIRST_K_H))
                pre.append(int((ss[:onset] >= thr).any()))
        out[str(b)] = {"first3h_leaks": float(np.mean(early)), "detected_in_window_leaks": float(np.mean(inwin)),
                       "pre_onset_alarm_leaks": float(np.mean(pre)), "n_leak_streams": len(early)}
    return out


def main() -> int:
    per = {}
    for f in sorted(TAG.glob("records_seed*.parquet")):
        rec = add_residual_arm(pd.read_parquet(f))
        for (seed, net), g in rec.groupby(["seed", "network_id"]):
            for arm in ARMS:
                per.setdefault(net, {}).setdefault(arm, []).append(per_frame(g, arm))
        print(f"[budget] {f.name} done", flush=True)
    summary = {"budgets_episodes_per_day": list(BUDGETS), "first_k_h": FIRST_K_H, "per_network": {}}
    for net, arms in per.items():
        summary["per_network"][net] = {}
        for arm, runs in arms.items():
            d = {"step_h": runs[0]["step_h"], "opportunities_3h": runs[0]["opportunities_3h"]}
            for b in map(str, BUDGETS):
                d[b] = {k: [float(np.mean([r[b][k] for r in runs])), float(np.std([r[b][k] for r in runs]))]
                        for k in ("first3h_leaks", "detected_in_window_leaks", "pre_onset_alarm_leaks")}
            summary["per_network"][net][arm] = d
    out = TAG / "analysis" / "v2_alarm_budget.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[budget] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
