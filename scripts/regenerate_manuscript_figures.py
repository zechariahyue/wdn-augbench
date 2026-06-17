from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"c:/Users/Zachy/OneDrive/Desktop/LLM water")
MANUSCRIPT_FIGS = ROOT / "manuscript" / "figures"
VALIDATION_SUMMARY = ROOT / "dev/active/artifacts/validation_package_full/summary.json"
LOLO_RESULTS = ROOT / "dev/active/artifacts/cross_network_lolo3/aggregated_lolo_results.json"
GRAPH_CVAE_RESULTS = ROOT / "dev/active/artifacts/graph_cvae_lolo3/aggregated_graph_cvae_lolo_results.json"

COLORBLIND = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "gray": "#4D4D4D",
    "purple": "#CC79A7",
    "teal": "#56B4E9",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _style(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#222222")


def make_scenario_bar():
    summary = _load_json(VALIDATION_SUMMARY)
    s24 = summary["scenario_summary"]["24"]

    labels = ["RF scenario", "LSTM scenario", "Triage scenario"]
    values = [
        s24["rf_auprc"]["mean"],
        s24["lstm_auprc"]["mean"],
        s24["triage_auprc"]["mean"],
    ]
    errs = [
        s24["rf_auprc"]["std"],
        s24["lstm_auprc"]["std"],
        s24["triage_auprc"]["std"],
    ]
    colors = [COLORBLIND["blue"], COLORBLIND["orange"], COLORBLIND["green"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.62, yerr=errs,
           error_kw={"elinewidth": 1.2, "capsize": 5, "capthick": 1.2, "ecolor": COLORBLIND["gray"]},
           edgecolor="none")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Scenario AUPRC")
    ax.set_xlabel("Model / score")
    ax.set_ylim(0.0, 1.0)
    _style(ax)
    fig.tight_layout()
    fig.savefig(MANUSCRIPT_FIGS / "scenario_level_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_lolo_bar():
    lolo = _load_json(LOLO_RESULTS)
    gcvae = _load_json(GRAPH_CVAE_RESULTS)

    # Current manuscript methods/order
    labels = ["Baseline", "Noise", "SMOTE", "GMM", "Graph CVAE"]
    values = [
        lolo["per_augmenter"]["baseline"]["macro_auprc_mean"],
        lolo["per_augmenter"]["noise"]["macro_auprc_mean"],
        lolo["per_augmenter"]["smote"]["macro_auprc_mean"],
        lolo["per_augmenter"]["gmm"]["macro_auprc_mean"],
        gcvae["macro_auprc"],
    ]
    colors = [
        COLORBLIND["gray"],
        COLORBLIND["teal"],
        COLORBLIND["green"],
        COLORBLIND["orange"],
        COLORBLIND["purple"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.68, edgecolor="none")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Macro AUPRC")
    ax.set_xlabel("Augmentation strategy")
    ax.set_ylim(0.0, max(values) * 1.18)
    _style(ax)
    fig.tight_layout()
    fig.savefig(MANUSCRIPT_FIGS / "domain_rand_transfer_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    MANUSCRIPT_FIGS.mkdir(parents=True, exist_ok=True)
    make_scenario_bar()
    make_lolo_bar()
    print("[info] Wrote scenario_level_bar.png")
    print("[info] Wrote domain_rand_transfer_bar.png")


if __name__ == "__main__":
    main()
