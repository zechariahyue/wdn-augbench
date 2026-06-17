"""Early-warning / temporal-localization endpoint for the LOLO benchmark.

Addresses two reviewer MAJOR items at once:
  (1) "Missing early-warning / time-to-detection endpoint."
  (2) "Physical baseline too weak" -> a deployable causal residual+persistence
      hydraulic baseline, evaluated on the same endpoints + row/scenario RSI.

All metrics are computed from existing per-row records
(dev/active/artifacts/lolo_5seed_revised/seed_*/cross_network_triage_records.csv),
which retain a per-hour base_score plus per-hour mass_residual / energy_residual.

Detectors compared:
  - RF base_score (the paper's tabular detector, per-row probability)
  - Hydraulic residual + persistence (CAUSAL: trailing rolling z-score of a
    mass/energy residual blend, smoothed over a k-hour persistence window).
    This is deployable -- it never uses future hours or the onset time.

Metrics (per network, mean +/- std over 5 seeds):
  - WSTL-AUPRC: within-scenario temporal-localization AUPRC. For each positive
    scenario, rank its own hours by detector score and score against the
    active-hour labels; average over positive scenarios. Anchored vs the
    within-scenario positive fraction (chance AUPRC).
  - Row-level AUPRC and max-pool scenario-level AUPRC (for RSI vs RF).
  - TTD: median time-to-detection (hours after onset) at a false-alarm budget
    of <=1 pre-onset alarm per scenario, threshold calibrated per (network,seed)
    on the negative (baseline) scenarios; plus detection rate within first
    k=3 hours of onset.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ART = Path(__file__).resolve().parents[1] / "artifacts" / "lolo_5seed_revised"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "early_warning_endpoint"
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
NETWORKS = ["net3", "d_town", "l_town"]
WARMUP = 4          # hours of trailing history before the causal z-score activates
PERSIST_K = 3       # persistence smoothing window (hours)
FIRST_K = 3         # early-detection horizon (hours after onset)


def causal_residual_score(rows_sorted):
    """Deployable hydraulic score: causal trailing z-score of a mass/energy
    residual blend, smoothed over a PERSIST_K-hour trailing mean.
    Never uses future hours or the onset label."""
    mass = np.array([float(r["mass_residual"]) for r in rows_sorted])
    energy = np.array([float(r["energy_residual"]) for r in rows_sorted])
    # Standardise each residual stream by its own scenario-global scale first
    # (a fixed, label-free transform) then blend.
    def _scale(x):
        s = x.std() + 1e-9
        return (x - x.mean()) / s
    blend = _scale(mass) + _scale(energy)
    T = len(blend)
    z = np.zeros(T)
    for t in range(T):
        if t < WARMUP:
            z[t] = 0.0
            continue
        hist = blend[:t]                      # strictly causal
        mu, sd = hist.mean(), hist.std() + 1e-9
        z[t] = (blend[t] - mu) / sd
    # persistence: trailing mean over PERSIST_K hours
    sm = np.copy(z)
    for t in range(T):
        lo = max(0, t - PERSIST_K + 1)
        sm[t] = z[lo:t + 1].mean()
    return sm


def load_baseline_rows(seed, net):
    csv_path = ART / f"seed_{seed}" / "cross_network_triage_records.csv"
    by_key = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["augmenter"] != "baseline" or r["network_id"] != net:
                continue
            by_key[(r["scenario_id"], r["sensor_coverage"])].append(r)
    for key in by_key:
        by_key[key].sort(key=lambda r: float(r["time_seconds"]))
    return by_key


def wstl_auprc(by_key, score_fn):
    """Within-scenario temporal localization AUPRC over positive scenarios."""
    vals = []
    for key, rows in by_key.items():
        y = np.array([int(r["y_true"]) for r in rows])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = score_fn(rows)
        vals.append(average_precision_score(y, s))
    return float(np.mean(vals)) if vals else float("nan"), len(vals)


def chance_auprc(by_key):
    fracs = []
    for rows in by_key.values():
        y = np.array([int(r["y_true"]) for r in rows])
        if 0 < y.sum() < len(y):
            fracs.append(y.mean())
    return float(np.mean(fracs)) if fracs else float("nan")


def row_and_scen_auprc(by_key, score_fn):
    all_s, all_y, scen_max_s, scen_lab = [], [], [], []
    for key, rows in by_key.items():
        y = np.array([int(r["y_true"]) for r in rows])
        s = score_fn(rows)
        all_s.extend(s.tolist())
        all_y.extend(y.tolist())
        scen_max_s.append(float(s.max()))
        scen_lab.append(int(y.sum() > 0))
    all_y = np.array(all_y); all_s = np.array(all_s)
    scen_lab = np.array(scen_lab); scen_max_s = np.array(scen_max_s)
    row = average_precision_score(all_y, all_s) if 0 < all_y.sum() < len(all_y) else float("nan")
    scen = average_precision_score(scen_lab, scen_max_s) if 0 < scen_lab.sum() < len(scen_lab) else float("nan")
    return float(row), float(scen)


def ttd_at_far(by_key, score_fn, far_budget=1):
    """Calibrate a per-(network,seed) threshold on negative scenarios so that
    <= far_budget pre-onset/negative alarms occur per scenario on average,
    then measure detection delay distribution on positive scenarios."""
    # Collect candidate threshold from negative scenarios: pick the threshold
    # such that mean alarms-per-negative-scenario <= far_budget.
    neg_scores = []
    for key, rows in by_key.items():
        y = np.array([int(r["y_true"]) for r in rows])
        if y.sum() > 0:
            continue
        neg_scores.append(score_fn(rows))
    if not neg_scores:
        return float("nan"), float("nan")
    flat = np.concatenate(neg_scores)
    # mean negative scenario length
    mean_len = np.mean([len(s) for s in neg_scores])
    # choose threshold = quantile so that expected alarms/scenario = far_budget
    q = max(0.0, 1.0 - far_budget / mean_len)
    thr = np.quantile(flat, q)

    delays, detected_first_k = [], []
    for key, rows in by_key.items():
        y = np.array([int(r["y_true"]) for r in rows])
        if y.sum() == 0:
            continue
        t_h = np.array([float(r["time_seconds"]) / 3600.0 for r in rows])
        onset = int(np.argmax(y > 0))
        onset_t = t_h[onset]
        s = score_fn(rows)
        fired = np.where(s[onset:] >= thr)[0]
        if len(fired) > 0:
            delay_h = t_h[onset + int(fired[0])] - onset_t   # elapsed hours after onset
            delays.append(delay_h)
            detected_first_k.append(1 if delay_h < FIRST_K else 0)
        else:
            delays.append(t_h[-1] - onset_t)  # censored at horizon
            detected_first_k.append(0)
    median_delay = float(np.median(delays)) if delays else float("nan")
    first_k_rate = float(np.mean(detected_first_k)) if detected_first_k else float("nan")
    return median_delay, first_k_rate


def rf_score(rows):
    return np.array([float(r["base_score"]) for r in rows])


def main():
    detectors = {"RF_base": rf_score, "HydResidPersist": causal_residual_score}
    results = {net: {d: defaultdict(list) for d in detectors} for net in NETWORKS}
    chance = {net: [] for net in NETWORKS}

    for seed in SEEDS:
        for net in NETWORKS:
            by_key = load_baseline_rows(seed, net)
            chance[net].append(chance_auprc(by_key))
            for dname, fn in detectors.items():
                wstl, n_scen = wstl_auprc(by_key, fn)
                row, scen = row_and_scen_auprc(by_key, fn)
                md, fk = ttd_at_far(by_key, fn)
                results[net][dname]["wstl"].append(wstl)
                results[net][dname]["row"].append(row)
                results[net][dname]["scen"].append(scen)
                results[net][dname]["ttd_median"].append(md)
                results[net][dname]["first_k_rate"].append(fk)
        print(f"seed {seed} done", flush=True)

    summary = {"params": {"warmup": WARMUP, "persist_k": PERSIST_K, "first_k": FIRST_K},
               "chance_auprc": {n: float(np.mean(chance[n])) for n in NETWORKS},
               "per_network": {}}
    for net in NETWORKS:
        summary["per_network"][net] = {}
        for dname in detectors:
            r = results[net][dname]
            summary["per_network"][net][dname] = {
                k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
                for k, v in r.items()
            }

    with open(OUT / "early_warning_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\n=== Within-Scenario Temporal Localization (WSTL) AUPRC, mean+/-std over 5 seeds ===")
    print(f"{'Network':<8} {'chance':>8} {'RF base':>16} {'HydResid+Persist':>20}")
    for net in NETWORKS:
        c = summary["chance_auprc"][net]
        rf = summary["per_network"][net]["RF_base"]["wstl"]
        hy = summary["per_network"][net]["HydResidPersist"]["wstl"]
        print(f"{net:<8} {c:>8.3f} {rf['mean']:>8.3f}+/-{rf['std']:.3f}   {hy['mean']:>8.3f}+/-{hy['std']:.3f}")

    print("\n=== Time-to-detection (median hrs after onset) and first-3h detection rate ===")
    print(f"{'Network':<8} {'RF TTD':>9} {'RF 1st-3h':>11} {'Hyd TTD':>9} {'Hyd 1st-3h':>11}")
    for net in NETWORKS:
        rf = summary["per_network"][net]["RF_base"]
        hy = summary["per_network"][net]["HydResidPersist"]
        print(f"{net:<8} {rf['ttd_median']['mean']:>9.2f} {rf['first_k_rate']['mean']:>11.3f} "
              f"{hy['ttd_median']['mean']:>9.2f} {hy['first_k_rate']['mean']:>11.3f}")

    print("\n=== Row vs max-pool scenario AUPRC for the hydraulic baseline (RSI vs RF) ===")
    print(f"{'Network':<8} {'RF row':>9} {'Hyd row':>9} {'RF scen':>9} {'Hyd scen':>9} {'RSI(Hyd,RF)':>13}")
    for net in NETWORKS:
        rf = summary["per_network"][net]["RF_base"]
        hy = summary["per_network"][net]["HydResidPersist"]
        rsi = (hy["row"]["mean"] - rf["row"]["mean"]) - (hy["scen"]["mean"] - rf["scen"]["mean"])
        print(f"{net:<8} {rf['row']['mean']:>9.3f} {hy['row']['mean']:>9.3f} "
              f"{rf['scen']['mean']:>9.3f} {hy['scen']['mean']:>9.3f} {rsi:>+13.3f}")


if __name__ == "__main__":
    main()
