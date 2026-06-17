"""Compute mean-pool vs. max-pool scenario AUPRC and RSI for the 5-seed LOLO benchmark.

Inputs: dev/active/artifacts/lolo_5seed_revised/seed_*/cross_network_triage_records.csv
Output: dev/active/artifacts/lolo_meanpool_rsi/meanpool_rsi_results.json

For each seed x augmenter x network:
  - row-level AUPRC over all rows (base_score, y_true)
  - max-pool scenario AUPRC: per scenario_id, take max(base_score) and OR(y_true)
  - mean-pool scenario AUPRC: per scenario_id, take mean(base_score), OR(y_true)
Then compute RSI(method, baseline) under both aggregations.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ART = Path(r"c:/Users/Zachy/OneDrive/Desktop/LLM water/dev/active/artifacts/lolo_5seed_revised")
OUT = Path(r"c:/Users/Zachy/OneDrive/Desktop/LLM water/dev/active/artifacts/lolo_meanpool_rsi")
OUT.mkdir(exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
AUGMENTERS = ["baseline", "noise", "gmm", "smote"]
NETWORKS = ["net3", "d_town", "l_town"]


def per_seed_metrics(seed: int) -> dict:
    csv_path = ART / f"seed_{seed}" / "cross_network_triage_records.csv"
    # buckets[(aug, network)] = list of (scenario_id, score, y_true)
    rows = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            aug = r["augmenter"]
            net = r["network_id"]
            sid = r["scenario_id"]
            try:
                s = float(r["base_score"])
                y = int(r["y_true"])
            except ValueError:
                continue
            rows[(aug, net)].append((sid, s, y))

    out = {}
    for (aug, net), records in rows.items():
        # Row-level
        scores = np.array([s for _, s, _ in records])
        ys = np.array([y for _, _, y in records])
        if ys.sum() == 0 or ys.sum() == len(ys):
            row_auprc = float("nan")
        else:
            row_auprc = float(average_precision_score(ys, scores))

        # Group by scenario_id
        by_sid = defaultdict(list)
        for sid, s, y in records:
            by_sid[sid].append((s, y))

        scen_max_scores, scen_mean_scores, scen_labels = [], [], []
        for sid, lst in by_sid.items():
            ss = [x[0] for x in lst]
            ys_ = [x[1] for x in lst]
            scen_max_scores.append(max(ss))
            scen_mean_scores.append(sum(ss) / len(ss))
            scen_labels.append(int(any(y > 0 for y in ys_)))

        scen_max_scores = np.array(scen_max_scores)
        scen_mean_scores = np.array(scen_mean_scores)
        scen_labels = np.array(scen_labels)
        n_scen = len(scen_labels)
        n_pos = int(scen_labels.sum())

        if n_pos == 0 or n_pos == n_scen:
            scen_max_auprc = float("nan")
            scen_mean_auprc = float("nan")
        else:
            scen_max_auprc = float(average_precision_score(scen_labels, scen_max_scores))
            scen_mean_auprc = float(average_precision_score(scen_labels, scen_mean_scores))

        out[f"{aug}__{net}"] = {
            "augmenter": aug,
            "network": net,
            "row_auprc": row_auprc,
            "scen_max_auprc": scen_max_auprc,
            "scen_mean_auprc": scen_mean_auprc,
            "n_rows": int(len(records)),
            "n_scenarios": n_scen,
            "n_scenarios_positive": n_pos,
            "scen_pos_rate_max": n_pos / n_scen,
        }
    return out


def main():
    all_results = {}
    for seed in SEEDS:
        print(f"Processing seed {seed}...", flush=True)
        all_results[f"seed_{seed}"] = per_seed_metrics(seed)
        print(f"  done.", flush=True)

    # Aggregate across seeds: per (aug, net) compute mean +/- std of each metric
    agg = {}
    for aug in AUGMENTERS:
        for net in NETWORKS:
            key = f"{aug}__{net}"
            row_vals = [all_results[f"seed_{s}"].get(key, {}).get("row_auprc", float("nan")) for s in SEEDS]
            sm_vals = [all_results[f"seed_{s}"].get(key, {}).get("scen_max_auprc", float("nan")) for s in SEEDS]
            sn_vals = [all_results[f"seed_{s}"].get(key, {}).get("scen_mean_auprc", float("nan")) for s in SEEDS]
            agg[key] = {
                "augmenter": aug,
                "network": net,
                "row_auprc_mean": float(np.nanmean(row_vals)),
                "row_auprc_std": float(np.nanstd(row_vals)),
                "scen_max_auprc_mean": float(np.nanmean(sm_vals)),
                "scen_max_auprc_std": float(np.nanstd(sm_vals)),
                "scen_mean_auprc_mean": float(np.nanmean(sn_vals)),
                "scen_mean_auprc_std": float(np.nanstd(sn_vals)),
                "per_seed_row": row_vals,
                "per_seed_scen_max": sm_vals,
                "per_seed_scen_mean": sn_vals,
            }

    # Compute RSI(aug, baseline) per network and aggregated
    rsi = {}
    for aug in AUGMENTERS:
        if aug == "baseline":
            continue
        for net in NETWORKS:
            base_key = f"baseline__{net}"
            aug_key = f"{aug}__{net}"
            # Per-seed RSI under max-pool and mean-pool
            row_diffs = [all_results[f"seed_{s}"][aug_key]["row_auprc"]
                         - all_results[f"seed_{s}"][base_key]["row_auprc"] for s in SEEDS]
            scen_max_diffs = [all_results[f"seed_{s}"][aug_key]["scen_max_auprc"]
                              - all_results[f"seed_{s}"][base_key]["scen_max_auprc"] for s in SEEDS]
            scen_mean_diffs = [all_results[f"seed_{s}"][aug_key]["scen_mean_auprc"]
                               - all_results[f"seed_{s}"][base_key]["scen_mean_auprc"] for s in SEEDS]
            rsi_max = [r - sm for r, sm in zip(row_diffs, scen_max_diffs)]
            rsi_mean = [r - smn for r, smn in zip(row_diffs, scen_mean_diffs)]
            rsi[f"{aug}_vs_baseline__{net}"] = {
                "row_diff_mean": float(np.mean(row_diffs)),
                "row_diff_std": float(np.std(row_diffs)),
                "scen_max_diff_mean": float(np.mean(scen_max_diffs)),
                "scen_mean_diff_mean": float(np.mean(scen_mean_diffs)),
                "rsi_under_max_pool_mean": float(np.mean(rsi_max)),
                "rsi_under_max_pool_std": float(np.std(rsi_max)),
                "rsi_under_mean_pool_mean": float(np.mean(rsi_mean)),
                "rsi_under_mean_pool_std": float(np.std(rsi_mean)),
            }

    # Macro across networks (equal-weight)
    for aug in AUGMENTERS:
        if aug == "baseline":
            continue
        per_net_rsi_max = [rsi[f"{aug}_vs_baseline__{n}"]["rsi_under_max_pool_mean"] for n in NETWORKS]
        per_net_rsi_mean = [rsi[f"{aug}_vs_baseline__{n}"]["rsi_under_mean_pool_mean"] for n in NETWORKS]
        rsi[f"{aug}_vs_baseline__macro_eqwt"] = {
            "rsi_under_max_pool_macro": float(np.mean(per_net_rsi_max)),
            "rsi_under_mean_pool_macro": float(np.mean(per_net_rsi_mean)),
        }

    out_path = OUT / "meanpool_rsi_results.json"
    with open(out_path, "w") as f:
        json.dump({"per_seed": all_results, "aggregated": agg, "rsi": rsi}, f, indent=2)
    print(f"Wrote {out_path}")

    # Print summary table
    print("\n=== Per-network mean+/-std (across 5 seeds) ===")
    print(f"{'Aug':<10} {'Net':<8} {'row AUPRC':>14} {'max-pool AUPRC':>18} {'mean-pool AUPRC':>20} {'pos rate(max)':>16}")
    for aug in AUGMENTERS:
        for net in NETWORKS:
            r = agg[f"{aug}__{net}"]
            # also pull mean pos rate
            prs = [all_results[f"seed_{s}"][f"{aug}__{net}"]["scen_pos_rate_max"] for s in SEEDS]
            print(f"{aug:<10} {net:<8} {r['row_auprc_mean']:>7.4f}+/-{r['row_auprc_std']:.3f} "
                  f"{r['scen_max_auprc_mean']:>7.4f}+/-{r['scen_max_auprc_std']:.3f}    "
                  f"{r['scen_mean_auprc_mean']:>7.4f}+/-{r['scen_mean_auprc_std']:.3f}    "
                  f"{np.mean(prs):>7.4f}")

    print("\n=== RSI summaries (eq-weight macro across 3 networks) ===")
    print(f"{'Comparison':<28} {'RSI under max-pool':>22} {'RSI under mean-pool':>24}")
    for aug in AUGMENTERS:
        if aug == "baseline":
            continue
        m = rsi[f"{aug}_vs_baseline__macro_eqwt"]
        print(f"{aug+' vs baseline':<28} {m['rsi_under_max_pool_macro']:>+9.4f}             {m['rsi_under_mean_pool_macro']:>+9.4f}")


if __name__ == "__main__":
    main()
