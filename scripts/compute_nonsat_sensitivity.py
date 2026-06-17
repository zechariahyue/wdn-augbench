"""RSI sensitivity to the injected baseline-scenario count.

Reads nonsat_results_{tag}.json for several tags (different --n-baseline
values) and reports macro RSI(aug, baseline) on the common Net3+D-Town
subset for each, so we can check whether the headline GCN RSI is stable
across the scenario-level prior.
"""
import json, numpy as np
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "artifacts" / "nonsat_endpoint"
# (tag, label) — n_baseline is read from the file itself
TAGS = [("nb20", None), ("full", None), ("nb80", None)]
NETS = ["net3", "d_town"]  # common subset where GCN is tractable
AUGS = ["noise", "gmm", "smote", "gcn"]


def load(tag):
    p = ART / f"nonsat_results_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def macro_rsi(d, aug):
    seeds = d["seeds"]
    ps = d["per_seed"]
    def have(a, n):
        return all(f"{a}__{n}" in ps[f"seed_{s}"] for s in seeds)
    def v(a, n, k):
        return [ps[f"seed_{s}"][f"{a}__{n}"][k] for s in seeds]
    per_net = []
    for n in NETS:
        if not (have(aug, n) and have("baseline", n)):
            return None, None
        rg = [a - b for a, b in zip(v(aug, n, "row_auprc"), v("baseline", n, "row_auprc"))]
        sg = [a - b for a, b in zip(v(aug, n, "scen_auprc"), v("baseline", n, "scen_auprc"))]
        per_net.append(np.mean([r - s for r, s in zip(rg, sg)]))
    return float(np.mean(per_net)), per_net


def pos_rate(d):
    seeds = d["seeds"]; ps = d["per_seed"]
    prs = [ps[f"seed_{s}"][f"baseline__{n}"]["scen_pos_rate"]
           for s in seeds for n in NETS if f"baseline__{n}" in ps[f"seed_{s}"]]
    return np.mean(prs)


print(f"{'tag':6s} {'n_base':>6s} {'pos_rate':>8s} " + " ".join(f"{a:>10s}" for a in AUGS))
for tag, _ in TAGS:
    d = load(tag)
    if d is None:
        print(f"{tag:6s} {'(missing)':>6s}")
        continue
    nb = d.get("n_baseline", "?")
    row = f"{tag:6s} {str(nb):>6s} {pos_rate(d):8.3f} "
    for a in AUGS:
        m, _ = macro_rsi(d, a)
        row += f"{('%+.3f' % m) if m is not None else 'N/A':>10s} "
    print(row)
print("\nNets:", NETS, "(common subset; GCN OOMs on l_town)")
