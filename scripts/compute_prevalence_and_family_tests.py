"""Reviewer FATAL-2 / MAJOR-2 / MAJOR-3 numbers from archived artifacts (no simulation).

1. Row-level class prevalence per network and per disturbance slice (baseline arm)
   from the LOLO and per-disturbance per-row records.
2. Family-level robustness of the 8-network GMM/noise/SMOTE paired tests:
   families = {Net3, D-Town, L-TOWN, Kentucky(mean of ky2/3/5/6/7)}; leave-one-KY-out.
3. Network-level bootstrap CI on the equal-weight macro AUPRC and on the
   augmenter-minus-baseline macro delta (resample the 8 held-out networks, B=2000).

Output: artifacts/prevalence_family_tests/results.json (+ printed summary).
"""
from __future__ import annotations

import csv
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ACTIVE = Path(__file__).resolve().parents[1]
ART = ACTIVE / "artifacts"
OUT = ART / "prevalence_family_tests"
OUT.mkdir(exist_ok=True)


def prevalence(csv_path, arm="baseline"):
    tot, pos = defaultdict(int), defaultdict(int)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("augmenter", arm) != arm:
                continue
            net = r["network_id"]
            tot[net] += 1; pos[net] += int(float(r["event_label"]) > 0)
            tot["ALL"] += 1; pos["ALL"] += int(float(r["event_label"]) > 0)
    return {k: round(pos[k] / tot[k], 4) for k in tot}, dict(tot)


def main():
    res = {}
    # ---- 1. prevalence -------------------------------------------------------
    prev = {}
    prev["all_disturbances"], n_all = prevalence(ART / "lolo_5seed_revised" / "seed_42" / "cross_network_triage_records.csv")
    for slc in ("leak", "pipe_closure", "pump_outage"):
        p = ART / "per_disturbance_lolo" / slc / "seed_42" / "cross_network_triage_records.csv"
        if p.exists():
            prev[slc], _ = prevalence(p)
    res["row_prevalence_seed42_baseline_arm"] = prev
    res["row_counts_all_disturbances"] = n_all
    print("== row-level positive rate (seed 42, baseline arm) ==")
    for k, v in prev.items():
        print(f"  {k:17s} {v}")

    # ---- 2. family-level tests ----------------------------------------------
    e = json.load(open(ART / "expanded_lolo" / "expanded_summary.json"))
    nets = list(e["per_network"])
    augs = [a for a in e["per_network"][nets[0]] if a != "baseline"]
    val = {n: {a: e["per_network"][n][a]["mean"] for a in e["per_network"][n]} for n in nets}
    ky = [n for n in nets if n.startswith("ky")]
    core = [n for n in nets if not n.startswith("ky")]
    fam = {}
    for a in augs:
        deltas = {n: val[n][a] - val[n]["baseline"] for n in nets}
        d8 = [deltas[n] for n in nets]
        p8 = wilcoxon(d8).pvalue if any(d8) else 1.0
        # 4 families: 3 core + KY mean
        d4 = [deltas[n] for n in core] + [float(np.mean([deltas[n] for n in ky]))]
        p4 = wilcoxon(d4).pvalue
        # leave-one-KY-out (n=7)
        loo = {}
        for drop in ky:
            d7 = [deltas[n] for n in nets if n != drop]
            loo[drop] = round(wilcoxon(d7).pvalue, 4)
        fam[a] = dict(
            delta_per_network={n: round(deltas[n], 4) for n in nets},
            wins_of_8=int(sum(d > 0 for d in d8)), p_n8=round(p8, 4),
            wins_core_of_3=int(sum(deltas[n] > 0 for n in core)),
            core_deltas={n: round(deltas[n], 4) for n in core},
            ky_mean_delta=round(d4[-1], 4), wins_of_4_families=int(sum(d > 0 for d in d4)), p_n4=round(p4, 4),
            sign_floor_n4=round(2 / 2 ** 4, 4), leave_one_ky_out_p=loo,
        )
        print(f"\n== {a}: wins {fam[a]['wins_of_8']}/8, p(n=8)={p8:.4f}; core deltas {fam[a]['core_deltas']}; "
              f"KY mean {d4[-1]:+.3f}; families {fam[a]['wins_of_4_families']}/4, p(n=4)={p4:.3f} (floor {2/16:.3f}); "
              f"LOO-KY p={loo}")
    res["family_tests"] = fam

    # ---- 3. network-level bootstrap on macro AUPRC and macro delta ------------
    rng = np.random.default_rng(0)
    boot = {}
    for a in ["baseline"] + augs:
        v = np.array([val[n][a] for n in nets]); d = np.array([val[n][a] - val[n]["baseline"] for n in nets])
        mac, dl = [], []
        for _ in range(2000):
            pick = rng.integers(0, len(nets), len(nets))
            mac.append(v[pick].mean()); dl.append(d[pick].mean())
        boot[a] = dict(macro_eqwt=round(float(v.mean()), 4),
                       macro_ci=[round(float(x), 4) for x in np.percentile(mac, [2.5, 97.5])],
                       delta_vs_baseline=round(float(d.mean()), 4),
                       delta_ci=[round(float(x), 4) for x in np.percentile(dl, [2.5, 97.5])])
        print(f"boot {a:9s} macro {boot[a]['macro_eqwt']:.3f} CI{boot[a]['macro_ci']}  delta {boot[a]['delta_vs_baseline']:+.3f} CI{boot[a]['delta_ci']}")
    res["network_bootstrap_B2000"] = boot
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\n[info] wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
