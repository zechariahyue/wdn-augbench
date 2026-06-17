from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
OUTPUT_DIR = ACTIVE_ROOT / "artifacts" / "figures_v2"

SCENARIO_RESULTS = ACTIVE_ROOT / "artifacts" / "scenario_eval" / "scenario_eval_results.json"
DOMAIN_RAND_RESULTS = ACTIVE_ROOT / "artifacts" / "domain_rand" / "domain_rand_results.json"
LEAK_ONLY_RESULTS = ACTIVE_ROOT / "artifacts" / "leak_only_lolo" / "leak_only_results.json"
BLOCK2_RESULTS = ACTIVE_ROOT / "artifacts" / "block2_graph_vs_physics" / "block2_results.json"

COLORBLIND_SAFE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "teal": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#4D4D4D",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_publication_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "savefig.dpi": 300,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#222222")
    ax.set_axisbelow(True)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_level_bar(payload: dict[str, Any], output_path: Path) -> None:
    labels = ["RF scenario", "LSTM scenario"]
    keys = ["rf_scenario", "lstm_scenario"]
    values = [payload[key]["auprc"] for key in keys]
    lower_errors = [payload[key]["auprc"] - payload[key]["ci_lower"] for key in keys]
    upper_errors = [payload[key]["ci_upper"] - payload[key]["auprc"] for key in keys]
    colors = [COLORBLIND_SAFE["blue"], COLORBLIND_SAFE["orange"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    x = range(len(labels))
    ax.bar(
        x,
        values,
        color=colors,
        width=0.62,
        yerr=[lower_errors, upper_errors],
        error_kw={"elinewidth": 1.2, "capsize": 5, "capthick": 1.2, "ecolor": COLORBLIND_SAFE["gray"]},
        edgecolor="none",
    )
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Scenario AUPRC")
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.0)
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_domain_rand_transfer_bar(payload: dict[str, Any], output_path: Path) -> None:
    systems = payload["per_strategy"]
    ordered_keys = ["baseline", "noise", "smote", "domain_rand", "graph_cvae"]
    labels = ["Baseline", "Noise", "SMOTE", "Domain rand", "Graph CVAE"]
    values = [systems[key]["weighted_auprc"] for key in ordered_keys]
    colors = [
        COLORBLIND_SAFE["gray"],
        COLORBLIND_SAFE["teal"],
        COLORBLIND_SAFE["green"],
        COLORBLIND_SAFE["orange"],
        COLORBLIND_SAFE["purple"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.68, edgecolor="none")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Weighted AUPRC")
    ax.set_xlabel("Augmentation strategy")
    ax.set_ylim(0.0, max(values) * 1.12)
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_leak_only_coverage_curves(payload: dict[str, Any], output_path: Path) -> None:
    ordered_keys = ["baseline", "noise+RF", "smote+RF", "graph_cvae+RF"]
    labels = {
        "baseline": "Baseline",
        "noise+RF": "Noise+RF",
        "smote+RF": "SMOTE+RF",
        "graph_cvae+RF": "Graph CVAE+RF",
    }
    colors = {
        "baseline": COLORBLIND_SAFE["gray"],
        "noise+RF": COLORBLIND_SAFE["teal"],
        "smote+RF": COLORBLIND_SAFE["green"],
        "graph_cvae+RF": COLORBLIND_SAFE["purple"],
    }
    markers = {
        "baseline": "o",
        "noise+RF": "s",
        "smote+RF": "^",
        "graph_cvae+RF": "D",
    }

    coverage_levels = payload["sensor_coverage_levels"]
    x_values = [level * 100.0 for level in coverage_levels]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key in ordered_keys:
        system = payload["systems"][key]
        y_values = [system["per_coverage"][str(level)]["weighted"] for level in coverage_levels]
        ax.plot(
            x_values,
            y_values,
            label=labels[key],
            color=colors[key],
            marker=markers[key],
            linewidth=2.0,
            markersize=6,
        )

    ax.set_xlabel("Sensor coverage (%)")
    ax.set_ylabel("Weighted AUPRC")
    ax.set_xticks(x_values, [f"{int(value)}" for value in x_values])
    ax.legend(title="Method", frameon=False)
    ax.set_ylim(0.28, 0.37)
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_physics_ablation_bar(payload: dict[str, Any], output_path: Path) -> None:
    ordered_keys = [
        "tabular_cvae",
        "graph_no_physics",
        "graph_with_physics",
        "graph_shuffled_adjacency",
    ]
    labels = ["Tabular CVAE", "Graph no physics", "Graph with physics", "Graph shuffled adjacency"]
    values = [payload["variants"][key]["auprc"] for key in ordered_keys]
    colors = [
        COLORBLIND_SAFE["blue"],
        COLORBLIND_SAFE["orange"],
        COLORBLIND_SAFE["green"],
        COLORBLIND_SAFE["red"],
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.68, edgecolor="none")
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylabel("Validation AUPRC")
    ax.set_xlabel("Ablation variant")
    ax.set_ylim(0.0, max(values) * 1.12)
    _style_axes(ax)
    _save_figure(fig, output_path)


def main() -> int:
    _apply_publication_style()

    scenario_payload = _load_json(SCENARIO_RESULTS)
    domain_payload = _load_json(DOMAIN_RAND_RESULTS)
    leak_payload = _load_json(LEAK_ONLY_RESULTS)
    block2_payload = _load_json(BLOCK2_RESULTS)

    outputs = {
        "scenario_level_bar.png": lambda: plot_scenario_level_bar(
            scenario_payload, OUTPUT_DIR / "scenario_level_bar.png"
        ),
        "domain_rand_transfer_bar.png": lambda: plot_domain_rand_transfer_bar(
            domain_payload, OUTPUT_DIR / "domain_rand_transfer_bar.png"
        ),
        "leak_only_coverage_curves.png": lambda: plot_leak_only_coverage_curves(
            leak_payload, OUTPUT_DIR / "leak_only_coverage_curves.png"
        ),
        "physics_ablation_bar.png": lambda: plot_physics_ablation_bar(
            block2_payload, OUTPUT_DIR / "physics_ablation_bar.png"
        ),
    }

    for render in outputs.values():
        render()

    for filename in outputs:
        output_path = OUTPUT_DIR / filename
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output was not created: {output_path}")
        print(f"[info] Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
