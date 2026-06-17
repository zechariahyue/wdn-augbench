"""Aggregate multi-seed LOLO cross-network results into a publication-ready summary.

Reads seed_0..seed_4 results from artifacts/cross_network_lolo3/ and produces:
  - aggregated_lolo_results.json  (per-network mean±std, macro-average)
  - aggregated_lolo_table.md      (publication-ready markdown table)

W1 fix: reports mean±std AUPRC across seeds.
W2 fix: reports per-network breakdown across 3 held-out test networks.
W3 fix: relabels 'ctgan' augmenter as 'gmm_fallback' (CTGAN was unavailable;
         CTGANAugmenter fell back to GaussianMixtureAugmenter silently).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
LOLO3_DIR = ACTIVE_ROOT / "artifacts" / "cross_network_lolo3"
OUTPUT_JSON = LOLO3_DIR / "aggregated_lolo_results.json"
OUTPUT_MD = LOLO3_DIR / "aggregated_lolo_table.md"

# W3 fix: honest augmenter relabeling
AUGMENTER_LABEL_MAP = {
    "ctgan": "gmm_fallback",
    "baseline": "baseline",
    "noise": "noise",
    "gmm": "gmm",
    "smote": "smote",
}

# W4 note: physics filter is a training diagnostic, not a transfer-improvement claim
AUGMENTER_DISPLAY = {
    "baseline": "RF (no aug)",
    "noise": "RF + Noise",
    "gmm": "RF + GMM",
    "gmm_fallback": "RF + GMM (CTGAN fallback)",
    "smote": "RF + SMOTE",
}


def load_seed_results(seed_dir: Path) -> dict[str, Any]:
    result_path = seed_dir / "cross_network_results.json"
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def remap_augmenter(name: str) -> str:
    return AUGMENTER_LABEL_MAP.get(name, name)


def aggregate() -> dict[str, Any]:
    seed_dirs = sorted(LOLO3_DIR.glob("seed_*"))
    if not seed_dirs:
        print(f"[error] No seed directories found in {LOLO3_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Found {len(seed_dirs)} seed runs: {[d.name for d in seed_dirs]}")

    # Collect per-augmenter, per-network AUPRC across seeds
    # Structure: {augmenter -> {network -> [auprc_s0, auprc_s1, ...]}}
    collected: dict[str, dict[str, list[float]]] = {}
    test_networks: list[str] | None = None
    train_networks: list[str] | None = None

    for seed_dir in seed_dirs:
        try:
            data = load_seed_results(seed_dir)
        except FileNotFoundError:
            print(f"[warn] Missing results in {seed_dir}, skipping", file=sys.stderr)
            continue

        if test_networks is None:
            test_networks = data.get("test_networks", [])
        if train_networks is None:
            train_networks = data.get("train_networks", [])

        per_aug = data.get("per_augmenter", {})
        for raw_aug, aug_data in per_aug.items():
            aug = remap_augmenter(raw_aug)
            if aug not in collected:
                collected[aug] = {}
            per_net = aug_data.get("per_network", {})
            for net_id, net_metrics in per_net.items():
                auprc = net_metrics.get("auprc")
                if auprc is not None:
                    collected[aug].setdefault(net_id, []).append(float(auprc))

    if not collected or test_networks is None:
        print("[error] No data collected", file=sys.stderr)
        sys.exit(1)

    # Build aggregated output
    aggregated: dict[str, Any] = {}
    for aug, net_data in collected.items():
        aug_summary: dict[str, Any] = {}
        macro_auprc_means: list[float] = []

        for net_id in sorted(net_data):
            values = net_data[net_id]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aug_summary[net_id] = {
                "auprc_mean": round(mean_val, 4),
                "auprc_std": round(std_val, 4),
                "n_seeds": len(values),
                "auprc_values": [round(v, 4) for v in values],
            }
            macro_auprc_means.append(mean_val)

        macro_mean = float(np.mean(macro_auprc_means)) if macro_auprc_means else 0.0
        aug_summary["macro_auprc_mean"] = round(macro_mean, 4)
        aggregated[aug] = aug_summary

    # Bootstrap CI over networks (network-level resampling)
    rng = np.random.default_rng(0)
    for aug, aug_summary in aggregated.items():
        net_means = [
            aug_summary[n]["auprc_mean"]
            for n in aug_summary
            if isinstance(aug_summary[n], dict) and "auprc_mean" in aug_summary[n]
        ]
        if len(net_means) < 2:
            aug_summary["macro_auprc_bootstrap_ci95"] = None
            continue
        boot_means = [
            float(np.mean(rng.choice(net_means, size=len(net_means), replace=True)))
            for _ in range(2000)
        ]
        lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
        aug_summary["macro_auprc_bootstrap_ci95"] = [round(lo, 4), round(hi, 4)]

    return {
        "protocol": "LOLO_3net",
        "train_networks": train_networks,
        "test_networks": sorted(test_networks),
        "n_seeds": len(seed_dirs),
        "augmenter_relabeling_note": (
            "'ctgan' relabeled as 'gmm_fallback': CTGANAugmenter fell back to "
            "GaussianMixtureAugmenter because sdv/ctgan was not installed. "
            "Rows were mislabeled 'synthetic_ctgan_positive' in prior artifacts."
        ),
        "physics_filter_note": (
            "Plausibility filter (quantile-based) used as a training-time quality gate "
            "on synthetic data. Transfer performance claims are not attributed to physics "
            "regularization — the filter is a diagnostic mechanism only."
        ),
        "per_augmenter": aggregated,
    }


def write_markdown_table(result: dict[str, Any], output_path: Path) -> None:
    test_nets = result["test_networks"]
    per_aug = result["per_augmenter"]
    augmenters = sorted(per_aug.keys())

    net_headers = " | ".join(n.replace("_", "-") for n in test_nets)
    header = f"| Method | {net_headers} | Macro AUPRC | 95% CI |"
    sep = "| --- " * (len(test_nets) + 3) + "|"

    lines = [
        "# Cross-Network LOLO Transfer Results (3 held-out networks, 5 seeds)",
        "",
        f"**Train:** {', '.join(result['train_networks'])}  ",
        f"**Test:** {', '.join(test_nets)}  ",
        f"**Seeds:** {result['n_seeds']}  ",
        "",
        "> Note: 'gmm-fallback' = CTGANAugmenter ran without sdv installed, "
        "falling back to GaussianMixtureAugmenter. Labeled honestly here.",
        "> Note: Plausibility filter is a training-time quality gate only — "
        "not claimed as physics transfer regularization.",
        "",
        header,
        sep,
    ]

    for aug in augmenters:
        display = AUGMENTER_DISPLAY.get(aug, aug)
        aug_data = per_aug[aug]
        cells = []
        net_means = []
        for net in test_nets:
            net_data = aug_data.get(net, {})
            mean_v = net_data.get("auprc_mean", None)
            std_v = net_data.get("auprc_std", None)
            if mean_v is not None:
                cells.append(f"{mean_v:.3f} ±{std_v:.3f}")
                net_means.append(mean_v)
            else:
                cells.append("n/a")
        macro = aug_data.get("macro_auprc_mean", None)
        ci = aug_data.get("macro_auprc_bootstrap_ci95", None)
        macro_str = f"{macro:.3f}" if macro is not None else "n/a"
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
        row = f"| {display} | {' | '.join(cells)} | {macro_str} | {ci_str} |"
        lines.append(row)

    lines += [
        "",
        "AUPRC reported as mean ± std across seeds. "
        "Macro AUPRC = equal-weight average across test networks. "
        "95% CI from network-level bootstrap (2000 resamples).",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Wrote markdown table: {output_path}")


def main() -> int:
    result = aggregate()

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[info] Wrote aggregated results: {OUTPUT_JSON}")

    write_markdown_table(result, OUTPUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
