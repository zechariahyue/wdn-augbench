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


def best(vals):
    v = [x for x in vals if x == x]
    return max(v) if v else None


def cell(x, bold=False, d=3):
    return f"$\\mathbf{{{f(x, d)}}}$" if bold else f"${f(x, d)}$"


def pm(ms, d=3, bold=False):
    """'mean +- std' cell from a (mean, std) pair; both at the same precision."""
    body = f"{f(ms[0], d)} \\pm {f(ms[1], d)}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


PAIR_LABEL = {"RSI(resid,rf_baseline)": "RSI(Resid., RF)", "RSI(rf_noise,rf_baseline)": "RSI(RF+noise, RF)",
              "RSI(rf_gmm,rf_baseline)": "RSI(RF+GMM, RF)", "RSI(rf_smote,rf_baseline)": "RSI(RF+SMOTE, RF)",
              "RSI(gcn_baseline,rf_baseline)": "RSI(GCN, RF)"}


def wrap(cols: str, header: str, rows: list[str], size: str = r"\footnotesize", sep: str = "3pt", star: bool = True) -> str:
    env = r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}" + cols + "}" if star else r"\begin{tabular}{" + cols + "}"
    end = r"\end{tabular*}" if star else r"\end{tabular}"
    return "\n".join([size, rf"\setlength{{\tabcolsep}}{{{sep}}}", env, r"\toprule", header + r" \\", r"\midrule", *[r + r" \\" for r in rows], r"\bottomrule", end, ""])


def forest_plot(d: dict, nets: list[str]) -> None:
    """Single manuscript figure: per-network paired delta vs rf_baseline for every arm, with the
    8-network mean and its network-bootstrap 95% interval. Writes manuscript/figures/v2_forest_deltas.png."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # figure is optional; tables carry the result
        print("matplotlib not available; skipping forest plot")
        return
    arms = [x for x in MAIN_ARMS if x in d]
    # Okabe-Ito colour-blind-safe palette, one colour per network
    pal = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, x in enumerate(arms):
        y = len(arms) - 1 - i
        for j, n in enumerate(nets):
            ax.plot(d[x]["per_network"][n], y + 0.18, marker="o", ms=4, color=pal[j], ls="none",
                    label=NET_LABEL[n] if i == 0 else None, alpha=0.9)
        lo, hi = d[x]["network_bootstrap_95"]
        ax.plot([lo, hi], [y - 0.18, y - 0.18], color="0.3", lw=1.2)
        ax.plot(d[x]["delta_mean"], y - 0.18, marker="D", ms=5, color="0.1", ls="none")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(range(len(arms))); ax.set_yticklabels([ARM_LABEL[x] for x in reversed(arms)], fontsize=8)
    ax.set_xlabel(r"$\Delta$ row AUPRC vs normalised RF baseline (seed mean per network)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False,
              handletextpad=0.2, columnspacing=0.9)
    ax.set_ylim(-0.5, len(arms) - 0.5)
    fig.tight_layout()
    out = OUT.parent / "figures" / "v2_forest_deltas.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out.name)


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
        vals = [pn[n][x]["row_auprc"][0] if x in pn[n] else float("nan") for x in arms]
        cells = [pm(pn[n][x]["row_auprc"], bold=(v == best(vals))) if x in pn[n] else "---" for x, v in zip(arms, vals)]
        rows.append(f"{NET_LABEL[n]} & ${f(pn[n]['rf_baseline']['row_prior'][0])}$ & " + " & ".join(cells))
    vals = [macro[x]["row_auprc"] for x in arms]
    rows.append(r"\midrule Macro (eq.) & $" + f(macro["rf_baseline"]["row_prior"]) + "$ & " + " & ".join(cell(v, bold=(v == best(vals))) for v in vals))
    (OUT / "tab_row_auprc.tex").write_text(wrap("l" + "c" * (len(arms) + 1), hdr, rows, size=r"\scriptsize", sep="1.5pt"), encoding="utf-8")

    # T3: deltas vs rf_baseline
    d = agg["deltas_vs_rf_baseline"]
    rows = [f"{ARM_LABEL[x]} & ${f(v['delta_mean'], plus=True)}$ & {v['wins']}/{v['n']} & ${f(v['wilcoxon_p'])}$ & "
            f"$[{f(v['network_bootstrap_95'][0], plus=True)},\\,{f(v['network_bootstrap_95'][1], plus=True)}]$ & ${f(v['mde_between_network'])}$"
            for x, v in d.items() if x in ARM_LABEL]
    # last column: 2.5 x the between-network standard error of the paired delta (JSON key
    # mde_between_network) -- a descriptive scale for detectable effects, not a powered MDE
    (OUT / "tab_deltas.tex").write_text(wrap("lccccc", r"Arm & $\Delta$ row AUPRC & Wins & Wilcoxon $p$ & Network bootstrap 95\% & $2.5\,\mathrm{SE}_\Delta$", rows), encoding="utf-8")

    # S: family-level deltas vs rf_baseline: [Net3, D-Town, L-TOWN, mean of the five Kentucky networks]
    rows = [f"{ARM_LABEL[x]} & " + " & ".join(f"${f(v, plus=True)}$" for v in d[x]["family_level_deltas"]) + f" & ${f(d[x]['delta_mean'], plus=True)}$"
            for x in d if x in ARM_LABEL]
    (OUT / "tab_family.tex").write_text(wrap("lccccc", r"Arm & Net3 & D-Town & L-TOWN & Kentucky (mean of 5) & All 8 (mean)", rows), encoding="utf-8")
    forest_plot(d, nets)

    # T4: scenario AUPRC (max-pool) + prior, selected arms
    sel = [x for x in ("rf_baseline", "rf_raw_baseline", "rf_noise", "rf_gmm", "rf_smote", "lr_baseline", "gcn_baseline", "resid") if x in arms]
    rows = []
    for n in nets:
        vals = [pn[n][x]["scen_auprc_max"][0] if x in pn[n] else float("nan") for x in sel]
        rows.append(f"{NET_LABEL[n]} & ${f(pn[n]['rf_baseline']['scen_prior'][0])}$ & "
                    + " & ".join(pm(pn[n][x]["scen_auprc_max"], bold=(v == best(vals))) if x in pn[n] else "---" for x, v in zip(sel, vals)))
    vals = [macro[x]["scen_auprc_max"] for x in sel]
    rows.append(r"\midrule Macro (eq.) & $" + f(macro["rf_baseline"]["scen_prior"]) + "$ & " + " & ".join(cell(v, bold=(v == best(vals))) for v in vals))
    (OUT / "tab_scen_auprc.tex").write_text(wrap("l" + "c" * (len(sel) + 1), "Network & Prior & " + " & ".join(ARM_SHORT[x] for x in sel), rows, size=r"\scriptsize", sep="1.5pt"), encoding="utf-8")

    # T5: early warning rf_baseline vs resid
    rows = []
    for n in nets:
        r, s = pn[n]["rf_baseline"], pn[n]["resid"]
        rows.append(f"{NET_LABEL[n]} & {int(round(r['n_leak_streams'][0]))} & ${f(r['wstl_chance'][0])}$ & {pm(r['wstl_auprc'])} & {pm(s['wstl_auprc'])} & "
                    f"{pm(r['first3h_leaks'])} & {pm(s['first3h_leaks'])} & ${f(r['detected_in_window_leaks'][0])}$ & ${f(s['detected_in_window_leaks'][0])}$ & "
                    f"{pm(r['ttd_median_h_leaks'], 1)} & {pm(s['ttd_median_h_leaks'], 1)} & ${f(r['false_alarm_rate_normals'][0], 2)}$ & ${f(s['false_alarm_rate_normals'][0], 2)}$")
    (OUT / "tab_early_warning.tex").write_text(wrap("lcccccccccccc", r"Network & $n$ & Chance & WSTL RF & WSTL res. & 3-h RF & 3-h res. & In-win.\ RF & In-win.\ res. & TTD RF (h) & TTD res. (h) & FA RF & FA res.", rows, size=r"\footnotesize", sep="3pt"), encoding="utf-8")

    # S: per-disturbance slices (rf_baseline) with priors
    rows = []
    KIND = {"leak": "Leak", "pipe_closure": "Pipe closure", "pump_outage": "Pump outage"}
    for n in nets:
        r, s = pn[n]["rf_baseline"], pn[n]["resid"]
        for k in ("leak", "pipe_closure", "pump_outage"):
            if f"slice_{k}_auprc" in r:
                rows.append(f"{NET_LABEL[n]} & {KIND[k]} & {int(round(r[f'n_pos_scen_{k}'][0]))} & ${f(r[f'slice_{k}_prior'][0])}$ & {pm(r[f'slice_{k}_auprc'])} & {pm(s[f'slice_{k}_auprc'])}")
    (OUT / "tab_slices.tex").write_text(wrap("llcccc", r"Network & Disturbance & $n$ scen. & Prior & RF, no aug. & Resid.", rows), encoding="utf-8")

    # S: max-pool vs mean-pool (rf_baseline, resid)
    rows = [f"{NET_LABEL[n]} & {pm(pn[n]['rf_baseline']['scen_auprc_max'])} & {pm(pn[n]['rf_baseline']['scen_auprc_mean'])} & "
            f"{pm(pn[n]['resid']['scen_auprc_max'])} & {pm(pn[n]['resid']['scen_auprc_mean'])}" for n in nets]
    (OUT / "tab_pooling.tex").write_text(wrap("lcccc", "Network & RF max-pool & RF mean-pool & Resid.\\ max-pool & Resid.\\ mean-pool", rows), encoding="utf-8")

    # S: RSI list
    rows = []
    for n in nets:
        for k, v in S["rsi"][n].items():
            ci = f"$[{f(v['bootstrap_95'][0], plus=True)},\\,{f(v['bootstrap_95'][1], plus=True)}]$" if v["bootstrap_95"] else "---"
            pair = PAIR_LABEL.get(k, k.replace("_", "\\_"))
            rows.append(f"{NET_LABEL[n]} & {pair} & ${f(v['mean'], plus=True)} \\pm {f(v['seed_std'])}$ & {ci}")
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
