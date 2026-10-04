"""Emit LaTeX table bodies for the manuscript from analyze_v2's v2_summary.json.

Every manuscript table that reports a v2 number is \\input from manuscript/tables_v2/, so
no number is transcribed by hand. Captions/labels live in the .tex; this script writes
only the tabular environments.

    python dev/active/scripts/make_v2_tables.py [--tag v2_benchmark]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE = HERE.parent
OUT = ACTIVE.parents[1] / "manuscript" / "tables_v2"
NET_LABEL = {"net3": "Net3", "d_town": "D-Town", "l_town": "L-TOWN", "ky2": "ky2", "ky3": "ky3", "ky5": "ky5", "ky6": "ky6", "ky7": "ky7"}
ARM_LABEL = {"rf_baseline": "RF, no aug.", "rf_noise": "RF + noise", "rf_gmm": "RF + GMM", "rf_smote": "RF + SMOTE",
             "rf_raw_baseline": "RF, raw features", "lr_baseline": "Logistic regr.", "gcn_baseline": "GCN (sample-space)", "resid": "Residual (label-free)"}
# short column heads for the two wide (10-column) per-network tables; the caption spells them out
ARM_SHORT = {"rf_baseline": "RF", "rf_noise": "RF+noise", "rf_gmm": "RF+GMM", "rf_smote": "RF+SMOTE",
             "rf_raw_baseline": "RF raw", "lr_baseline": "LR", "gcn_baseline": "GCN", "resid": "Resid."}
MAIN_ARMS = ["rf_baseline", "rf_noise", "rf_gmm", "rf_smote", "rf_raw_baseline", "lr_baseline", "gcn_baseline", "resid"]


def f(x, d=3, plus=False):
    if x is None or (isinstance(x, float) and x != x):
        return "---"
    return f"{x:+.{d}f}" if plus else f"{x:.{d}f}"


def wrap(cols: str, header: str, rows: list[str], size: str = r"\footnotesize", sep: str = "3pt", star: bool = True) -> str:
    env = r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + cols + "}" if star else r"\begin{tabular}{" + cols + "}"
    end = r"\end{tabular*}" if star else r"\end{tabular}"
    return "\n".join([size, rf"\setlength{{\tabcolsep}}{{{sep}}}", env, r"\toprule", header + r" \\", r"\midrule", *[r + r" \\" for r in rows], r"\bottomrule", end, ""])


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="v2_benchmark"); a = ap.parse_args()
    S = json.loads((ACTIVE / "artifacts" / a.tag / "analysis" / "v2_summary.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    nets = S["networks"]; agg = S["aggregate"]; pn = agg["per_network"]; macro = agg["macro_eqwt"]
    arms = [x for x in MAIN_ARMS if x in S["arms"]]

    # T2: row AUPRC per network (mean ± std over seeds) + prior; macro row
    hdr = "Network & Prior & " + " & ".join(ARM_SHORT[x] for x in arms)
    rows = []
    for n in nets:
        cells = [f"${f(pn[n][x]['row_auprc'][0])} \\pm {f(pn[n][x]['row_auprc'][1], 2)}$" if x in pn[n] else "---" for x in arms]
        rows.append(f"{NET_LABEL[n]} & ${f(pn[n]['rf_baseline']['row_prior'][0])}$ & " + " & ".join(cells))
    rows.append(r"\midrule Macro (eq.) & $" + f(macro["rf_baseline"]["row_prior"]) + "$ & " + " & ".join(f"$\\mathbf{{{f(macro[x]['row_auprc'])}}}$" for x in arms))
    (OUT / "tab_row_auprc.tex").write_text(wrap("l" + "c" * (len(arms) + 1), hdr, rows, size=r"\scriptsize", sep="2pt"), encoding="utf-8")

    # T3: deltas vs rf_baseline
    d = agg["deltas_vs_rf_baseline"]
    rows = [f"{ARM_LABEL[x]} & ${f(v['delta_mean'], plus=True)}$ & {v['wins']}/{v['n']} & ${f(v['wilcoxon_p'])}$ & "
            f"$[{f(v['network_bootstrap_95'][0], plus=True)},\\,{f(v['network_bootstrap_95'][1], plus=True)}]$ & ${f(v['mde_between_network'])}$"
            for x, v in d.items() if x in ARM_LABEL]
    (OUT / "tab_deltas.tex").write_text(wrap("lccccc", r"Arm & $\Delta$ row AUPRC & Wins & Wilcoxon $p$ & Network bootstrap 95\% & MDE", rows), encoding="utf-8")

    # T4: scenario AUPRC (max-pool) + prior, selected arms
    sel = [x for x in ("rf_baseline", "rf_raw_baseline", "rf_noise", "rf_gmm", "rf_smote", "lr_baseline", "gcn_baseline", "resid") if x in arms]
    rows = [f"{NET_LABEL[n]} & ${f(pn[n]['rf_baseline']['scen_prior'][0])}$ & " + " & ".join(f"${f(pn[n][x]['scen_auprc_max'][0])}$" if x in pn[n] else "---" for x in sel) for n in nets]
    rows.append(r"\midrule Macro (eq.) & $" + f(macro["rf_baseline"]["scen_prior"]) + "$ & " + " & ".join(f"${f(macro[x]['scen_auprc_max'])}$" for x in sel))
    (OUT / "tab_scen_auprc.tex").write_text(wrap("l" + "c" * (len(sel) + 1), "Network & Prior & " + " & ".join(ARM_SHORT[x] for x in sel), rows, size=r"\scriptsize", sep="2pt"), encoding="utf-8")

    # T5: early warning rf_baseline vs resid
    rows = []
    for n in nets:
        r, s = pn[n]["rf_baseline"], pn[n].get("resid")
        rows.append(f"{NET_LABEL[n]} & ${f(r['wstl_chance'][0])}$ & ${f(r['wstl_auprc'][0])}$ & ${f(s['wstl_auprc'][0]) if s else '---'}$ & "
                    f"${f(r['first3h_leaks'][0])}$ & ${f(s['first3h_leaks'][0]) if s else '---'}$ & ${f(r['ttd_median_h_leaks'][0], 1)}$ & ${f(s['ttd_median_h_leaks'][0], 1) if s else '---'}$")
    (OUT / "tab_early_warning.tex").write_text(wrap("lccccccc", r"Network & Chance & WSTL RF & WSTL res. & 3-h RF & 3-h res. & TTD RF (h) & TTD res. (h)", rows), encoding="utf-8")

    # S: per-disturbance slices (rf_baseline) with priors
    rows = []
    for n in nets:
        r = pn[n]["rf_baseline"]
        cells = []
        for k in ("leak", "pipe_closure", "pump_outage"):
            if f"slice_{k}_auprc" in r:
                cells.append(f"${f(r[f'slice_{k}_auprc'][0])}$ / ${f(r[f'slice_{k}_prior'][0])}$")
            else:
                cells.append("---")
        rows.append(f"{NET_LABEL[n]} & " + " & ".join(cells))
    (OUT / "tab_slices.tex").write_text(wrap("lccc", "Network & Leak (AUPRC / prior) & Pipe closure (AUPRC / prior) & Pump outage (AUPRC / prior)", rows), encoding="utf-8")

    # S: max-pool vs mean-pool (rf_baseline, resid)
    rows = [f"{NET_LABEL[n]} & ${f(pn[n]['rf_baseline']['scen_auprc_max'][0])}$ & ${f(pn[n]['rf_baseline']['scen_auprc_mean'][0])}$ & "
            f"${f(pn[n]['resid']['scen_auprc_max'][0]) if 'resid' in pn[n] else '---'}$ & ${f(pn[n]['resid']['scen_auprc_mean'][0]) if 'resid' in pn[n] else '---'}$" for n in nets]
    (OUT / "tab_pooling.tex").write_text(wrap("lcccc", "Network & RF max-pool & RF mean-pool & Resid.\\ max-pool & Resid.\\ mean-pool", rows), encoding="utf-8")

    # S: RSI list
    rows = []
    for n in nets:
        for k, v in S["rsi"][n].items():
            ci = f"$[{f(v['bootstrap_95'][0], plus=True)},\\,{f(v['bootstrap_95'][1], plus=True)}]$" if v["bootstrap_95"] else "---"
            pair = k.replace("_", "\\_")
            rows.append(f"{NET_LABEL[n]} & {pair} & ${f(v['mean'], plus=True)} \\pm {f(v['seed_std'], 2)}$ & {ci}")
    (OUT / "tab_rsi.tex").write_text(wrap("llcc", "Network & Pair & RSI (seed mean $\\pm$ std) & Pooled scenario bootstrap 95\\%", rows), encoding="utf-8")

    # S: prior sweep
    rows = []
    for nb, dd in S["prior_sweep"].items():
        for n, v in dd.items():
            rows.append(f"{nb} & {NET_LABEL[n]} & ${f(v['row_prior'])}$ & ${f(v['scen_prior'])}$ & ${f(v['scen_auprc_max'])}$ & ${f(v['rsi_resid_rf'], plus=True)}$")
    (OUT / "tab_prior_sweep.tex").write_text(wrap("llcccc", "Normals per network & Network & Row prior & Scenario prior & RF scen.\\ AUPRC & RSI(resid, RF)", rows), encoding="utf-8")

    # S: seed-level row AUPRC for the normalised RF baseline and the raw-feature RF
    ps = S["per_seed_network_arm"]; seeds = S["seeds"]
    rows = []
    for arm, lab in (("rf_baseline", "RF, no aug."), ("rf_raw_baseline", "RF, raw features")):
        for n in nets:
            vals = " & ".join(f"${f(ps[f'{sd}|{n}|{arm}']['row_auprc'])}$" for sd in seeds)
            rows.append(f"{lab} & {NET_LABEL[n]} & {vals}")
    hdr = "Arm & Network & " + " & ".join(f"seed {sd}" for sd in seeds)
    (OUT / "tab_seed_level.tex").write_text(wrap("ll" + "c" * len(seeds), hdr, rows), encoding="utf-8")

    # key numbers as LaTeX macros for the prose
    m = [r"\newcommand{\vTwoMacroPrior}{" + f(macro["rf_baseline"]["row_prior"]) + "}",
         r"\newcommand{\vTwoMacroRF}{" + f(macro["rf_baseline"]["row_auprc"]) + "}",
         r"\newcommand{\vTwoMacroRaw}{" + f(macro["rf_raw_baseline"]["row_auprc"]) + "}",
         r"\newcommand{\vTwoMacroResid}{" + f(macro["resid"]["row_auprc"]) + "}",
         r"\newcommand{\vTwoLiftP}{" + f(agg["baseline_vs_prior"]["wilcoxon_p"]) + "}",
         r"\newcommand{\vTwoSeeds}{" + str(len(S["seeds"])) + "}"]
    for x in ("rf_noise", "rf_gmm", "rf_smote", "rf_raw_baseline", "lr_baseline", "gcn_baseline", "resid"):
        if x in d:
            key = x.replace("rf_", "").replace("_baseline", "").title().replace("_", "")
            m += [rf"\newcommand{{\vTwoDelta{key}}}{{{f(d[x]['delta_mean'], plus=True)}}}", rf"\newcommand{{\vTwoWins{key}}}{{{d[x]['wins']}/{d[x]['n']}}}", rf"\newcommand{{\vTwoP{key}}}{{{f(d[x]['wilcoxon_p'])}}}"]
    (OUT / "macros_v2.tex").write_text("\n".join(m) + "\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
