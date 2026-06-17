"""Aggregate validation package artifacts into summary JSON/CSV/Markdown.

Reads experiment_manifest.json from run_triage_validation.py and summarizes:
- cross-network weighted AUPRC for base/triage/heuristic
- scenario-level AUPRC for RF/LSTM/Triage
- mean/std across seeds
- simple markdown summary for manuscript drafting
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate validation package artifacts")
    parser.add_argument("--input-dir", type=Path, default=Path("dev/active/artifacts/validation_package"))
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        input_dir = (repo_root / input_dir).resolve()

    manifest_path = input_dir / "experiment_manifest.json"
    manifest = _read_json(manifest_path)

    cross_rows = []
    scenario_rows = []

    for run in manifest.get("runs", []):
        out_dir = Path(run["output_dir"])
        if run.get("return_code") != 0:
            continue

        if run.get("kind") == "cross_network":
            result_path = out_dir / "cross_network_results.json"
            if result_path.exists():
                data = _read_json(result_path)
                for aug, info in data.get("per_augmenter", {}).items():
                    cross_rows.append({
                        "seed": run.get("seed"),
                        "augmenter": aug,
                        "base_weighted_auprc": info.get("weighted_auprc"),
                        "triage_weighted_auprc": info.get("triage_weighted_auprc"),
                        "heuristic_weighted_auprc": info.get("heuristic_weighted_auprc"),
                    })

        if run.get("kind") == "scenario_eval":
            result_path = out_dir / "scenario_eval_results.json"
            if result_path.exists():
                data = _read_json(result_path)
                scenario_rows.append({
                    "seed": run.get("seed"),
                    "min_timesteps": run.get("min_timesteps"),
                    "rf_auprc": data.get("rf_scenario", {}).get("auprc"),
                    "lstm_auprc": data.get("lstm_scenario", {}).get("auprc"),
                    "triage_auprc": data.get("triage_scenario", {}).get("auprc"),
                    "n_test_examples": data.get("n_test_examples"),
                })

    # Aggregate cross-network by augmenter
    cross_summary = {}
    augmenters = sorted({r["augmenter"] for r in cross_rows})
    for aug in augmenters:
        subset = [r for r in cross_rows if r["augmenter"] == aug]
        for key in ["base_weighted_auprc", "triage_weighted_auprc", "heuristic_weighted_auprc"]:
            vals = [float(r[key]) for r in subset if r.get(key) is not None]
            m, s = _safe_stats(vals)
            cross_summary.setdefault(aug, {})[key] = {"mean": m, "std": s, "n": len(vals)}

    # Aggregate scenario rows by min_timesteps
    scenario_summary = {}
    for mint in sorted({r["min_timesteps"] for r in scenario_rows}):
        subset = [r for r in scenario_rows if r["min_timesteps"] == mint]
        scenario_summary[mint] = {}
        for key in ["rf_auprc", "lstm_auprc", "triage_auprc"]:
            vals = [float(r[key]) for r in subset if r.get(key) is not None]
            m, s = _safe_stats(vals)
            scenario_summary[mint][key] = {"mean": m, "std": s, "n": len(vals)}

    summary = {
        "experiment_name": manifest.get("experiment_name"),
        "cross_network_summary": cross_summary,
        "scenario_summary": scenario_summary,
        "n_cross_runs": len(cross_rows),
        "n_scenario_runs": len(scenario_rows),
    }

    # Write JSON
    (input_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # Write CSV
    csv_path = input_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "name", "metric", "mean", "std", "n"])
        for aug, metrics in cross_summary.items():
            for metric, vals in metrics.items():
                writer.writerow(["cross_network", aug, metric, vals["mean"], vals["std"], vals["n"]])
        for mint, metrics in scenario_summary.items():
            for metric, vals in metrics.items():
                writer.writerow(["scenario", f"min_timesteps={mint}", metric, vals["mean"], vals["std"], vals["n"]])

    # Write markdown
    lines = [
        "# Validation Package Summary",
        "",
        "## Cross-network weighted AUPRC",
        "",
        "| Augmenter | Base | Learned triage | Heuristic triage | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for aug, metrics in cross_summary.items():
        b = metrics.get("base_weighted_auprc", {})
        t = metrics.get("triage_weighted_auprc", {})
        h = metrics.get("heuristic_weighted_auprc", {})
        lines.append(
            f"| {aug} | {b.get('mean')} ± {b.get('std')} | {t.get('mean')} ± {t.get('std')} | {h.get('mean')} ± {h.get('std')} | {b.get('n')} |"
        )
    lines.extend(["", "## Scenario-level AUPRC by min_timesteps", ""])
    lines.extend([
        "| min_timesteps | RF | LSTM-AE | Triage | n |",
        "| --- | --- | --- | --- | --- |",
    ])
    for mint, metrics in scenario_summary.items():
        r = metrics.get("rf_auprc", {})
        l = metrics.get("lstm_auprc", {})
        t = metrics.get("triage_auprc", {})
        lines.append(
            f"| {mint} | {r.get('mean')} ± {r.get('std')} | {l.get('mean')} ± {l.get('std')} | {t.get('mean')} ± {t.get('std')} | {r.get('n')} |"
        )
    (input_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[info] Summary JSON -> {input_dir / 'summary.json'}")
    print(f"[info] Summary CSV  -> {csv_path}")
    print(f"[info] Summary MD   -> {input_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
