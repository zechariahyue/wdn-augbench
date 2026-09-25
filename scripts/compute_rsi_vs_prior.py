"""RSI as a function of scenario-level positive rate (prevalence).

Reviewer (2nd external, MAJOR): the quantitative RSI is shown at ~0.47 prior,
but the paper motivates rare-event (1:25) monitoring. This script reads the
existing n_baseline sweep (nb20/nb40/nb80 -> priors ~0.64/0.47/0.31) and reports
whether the GCN RSI sign/ordering persists as prevalence drops toward the
rare-event regime. Common net subset is net3+d_town (gcn OOMs on l_town).
No new simulation: pure re-read of archived aggregate AUPRC.
"""
import json, glob, os
import numpy as np
from pathlib import Path

import os
_REPO = Path(__file__).resolve().parents[1]
# Override with WDN_AUGBENCH_ARTIFACTS / WDN_AUGBENCH_FIGS to point at a full regenerated run.
_ART = Path(os.environ.get("WDN_AUGBENCH_ARTIFACTS", _REPO / "artifacts"))
_FIGS = Path(os.environ.get("WDN_AUGBENCH_FIGS", _REPO / "figures"))
ART = _ART / "nonsat_endpoint"
FILES = sorted(glob.glob(str(ART / "nonsat_results_*.json")))
COMMON_NETS = ["net3", "d_town"]  # present across all nb levels


def load(f):
    d = json.loads(Path(f).read_text())
    if "per_seed" not in d:
        return None
    return d


def cell(ps, seeds, aug, net, key):
    out = []
    for s in seeds:
        k = f"{aug}__{net}"
        rec = ps[f"seed_{s}"].get(k)
        if rec is None or rec.get(key) is None:
            return None
        v = rec[key]
        if isinstance(v, float) and np.isnan(v):
            return None
        out.append(v)
    return out


rows = []
for f in FILES:
    d = load(f)
    if d is None:
        continue
    nb = d.get("n_baseline")
    if nb is None:
        continue
    seeds = d["seeds"]
    ps = d["per_seed"]
    nets = [n for n in COMMON_NETS if n in d["test_nets"]]
    # mean positive rate over common nets / augs / seeds
    prs = []
    for net in nets:
        pr = cell(ps, seeds, "baseline", net, "scen_pos_rate")
        if pr:
            prs.extend(pr)
    posrate = float(np.mean(prs)) if prs else float("nan")
    # macro RSI(aug, baseline) eq-weight over common nets
    for aug in ["noise", "gmm", "smote", "gcn"]:
        rsis = []
        for net in nets:
            ra = cell(ps, seeds, aug, net, "row_auprc")
            rb = cell(ps, seeds, "baseline", net, "row_auprc")
            sa = cell(ps, seeds, aug, net, "scen_auprc")
            sb = cell(ps, seeds, "baseline", net, "scen_auprc")
            if None in (ra, rb, sa, sb):
                continue
            rg = np.array(ra) - np.array(rb)
            sg = np.array(sa) - np.array(sb)
            rsi = rg - sg
            rsis.append(rsi)
        if not rsis:
            continue
        rsi_per_seed = np.mean(np.vstack(rsis), axis=0)  # eq-weight over nets, per seed
        rows.append({
            "n_baseline": nb, "posrate": posrate, "aug": aug,
            "rsi_mean": float(np.mean(rsi_per_seed)),
            "rsi_std": float(np.std(rsi_per_seed)),
            "nets": ",".join(nets),
        })

rows.sort(key=lambda r: (-r["posrate"], r["aug"]))
print(f"{'pos_rate':>8s} {'n_base':>6s} {'aug':>7s} {'macroRSI':>10s} {'std':>6s}  nets")
for r in rows:
    print(f"{r['posrate']:8.3f} {r['n_baseline']:6d} {r['aug']:>7s} "
          f"{r['rsi_mean']:+10.3f} {r['rsi_std']:6.3f}  {r['nets']}")

print("\n=== GCN RSI sign vs prevalence (the headline question) ===")
gcn = [r for r in rows if r["aug"] == "gcn"]
gcn.sort(key=lambda r: -r["posrate"])
for r in gcn:
    sign = "POS" if r["rsi_mean"] > 0 else ("NEG" if r["rsi_mean"] < 0 else "ZERO")
    print(f"  posrate {r['posrate']:.3f}: GCN macro RSI {r['rsi_mean']:+.3f} +/- {r['rsi_std']:.3f}  [{sign}]")

print("\n=== tabular augmenters RSI ~ 0 check vs prevalence ===")
for aug in ["noise", "gmm", "smote"]:
    a = [r for r in rows if r["aug"] == aug]
    a.sort(key=lambda r: -r["posrate"])
    s = "  ".join(f"{r['posrate']:.2f}:{r['rsi_mean']:+.3f}" for r in a)
    print(f"  {aug:6s}  {s}")
