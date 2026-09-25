"""Compute RSI on the non-saturated endpoint from nonsat_results_full.json."""
import json, numpy as np
from pathlib import Path
import os
_REPO = Path(__file__).resolve().parents[1]
# Override with WDN_AUGBENCH_ARTIFACTS / WDN_AUGBENCH_FIGS to point at a full regenerated run.
_ART = Path(os.environ.get("WDN_AUGBENCH_ARTIFACTS", _REPO / "artifacts"))
_FIGS = Path(os.environ.get("WDN_AUGBENCH_FIGS", _REPO / "figures"))
ART = _ART / "nonsat_endpoint"
d = json.loads((ART/"nonsat_results_full.json").read_text())
seeds = d["seeds"]; nets = d["test_nets"]
_first = d["per_seed"][f"seed_{seeds[0]}"]
augs = [a for a in ["baseline","noise","gmm","smote","gcn"]
        if any(k.startswith(a+"__") for k in _first)]
ps = d["per_seed"]

def have(aug, net):
    return all(f"{aug}__{net}" in ps[f"seed_{s}"] for s in seeds)

def vals(aug, net, key):
    return [ps[f"seed_{s}"][f"{aug}__{net}"][key] for s in seeds]

print("=== Per (aug,net): mean+/-std over seeds ===")
print(f"{'aug':9s} {'net':7s} {'row':>14s} {'scen':>14s} {'pos_rate':>9s}")
for aug in augs:
    for net in nets:
        if not have(aug, net):
            print(f"{aug:9s} {net:7s} {'N/A (OOM/missing)':>30s}"); continue
        r=vals(aug,net,"row_auprc"); sc=vals(aug,net,"scen_auprc"); pr=vals(aug,net,"scen_pos_rate")
        print(f"{aug:9s} {net:7s} {np.mean(r):6.3f}+/-{np.std(r):.3f} {np.mean(sc):6.3f}+/-{np.std(sc):.3f} {np.mean(pr):8.3f}")

print("\n=== RSI(aug, baseline) per net: mean+/-std over seeds, with row/scen gaps ===")
print(f"{'aug':7s} {'net':7s} {'row_gap':>9s} {'scen_gap':>9s} {'RSI':>14s}")
rsi_macro = {a:[] for a in augs if a!='baseline'}
for aug in augs:
    if aug=='baseline': continue
    for net in nets:
        if not (have(aug,net) and have("baseline",net)):
            print(f"{aug:7s} {net:7s} {'N/A':>30s}"); continue
        rg=[a-b for a,b in zip(vals(aug,net,"row_auprc"), vals("baseline",net,"row_auprc"))]
        sg=[a-b for a,b in zip(vals(aug,net,"scen_auprc"), vals("baseline",net,"scen_auprc"))]
        rsi=[r-s for r,s in zip(rg,sg)]
        rsi_macro[aug].append(np.mean(rsi))
        print(f"{aug:7s} {net:7s} {np.mean(rg):+8.3f} {np.mean(sg):+8.3f} {np.mean(rsi):+8.3f}+/-{np.std(rsi):.3f}")

print("\n=== eq-weight macro RSI(aug, baseline) across nets ===")
for aug,vv in rsi_macro.items():
    print(f"{aug:7s} macro RSI = {np.mean(vv):+.3f}")

print("\n=== sanity: scen pos-rate range and saturation check ===")
cells=[(s,a,n) for s in seeds for a in augs for n in nets if f"{a}__{n}" in ps[f"seed_{s}"]]
allpr=[ps[f"seed_{s}"][f"{a}__{n}"]["scen_pos_rate"] for s,a,n in cells]
print(f"scen pos_rate min/max: {min(allpr):.3f}/{max(allpr):.3f}  (target: well below 0.97)")
allsc=[ps[f"seed_{s}"][f"{a}__{n}"]["scen_auprc"] for s,a,n in cells]
print(f"scen AUPRC min/max: {min(allsc):.3f}/{max(allsc):.3f}  (spread => informative, not pinned at prior)")
print("note: gcn OOMs on l_town (k-NN graph over ~3e5 rows); reported on net3+d_town only.")
