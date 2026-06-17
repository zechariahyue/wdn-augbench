"""Run a frozen multi-run experiment suite and aggregate the outputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]

# Make watergen importable for multi-seed aggregation
SRC_ROOT = ACTIVE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def run_one_config(config_path: Path, log_path: Path) -> int:
    command = [sys.executable, str(ACTIVE_ROOT / "scripts" / "run_experiment.py"), "--config", str(config_path)]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    return int(result.returncode)


def extract_run_summary(run_id: str, artifact_dir: Path) -> dict[str, Any]:
    smoke_summary_path = artifact_dir / "smoke" / "smoke_experiment_summary.json"
    acwa_path = artifact_dir / "acwa" / "acwa_case_study.json"

    summary: dict[str, Any] = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
    }

    if smoke_summary_path.exists():
        smoke = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
        summary["smoke"] = smoke
    if acwa_path.exists():
        acwa = json.loads(acwa_path.read_text(encoding="utf-8"))
        summary["acwa"] = acwa
    return summary


def flatten_for_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    smoke = summary.get("smoke", {})
    for experiment_name, metrics in smoke.get("experiments", {}).items():
        split_map = metrics.get("splits", {})
        row = {
            "run_id": summary["run_id"],
            "experiment": experiment_name,
            "train_f1": split_map.get("train", {}).get("f1"),
            "validation_f1": split_map.get("validation", {}).get("f1"),
            "test_f1": split_map.get("test", {}).get("f1"),
            "train_auprc": split_map.get("train", {}).get("auprc"),
            "validation_auprc": split_map.get("validation", {}).get("auprc"),
            "test_auprc": split_map.get("test", {}).get("auprc"),
        }
        rows.append(row)
    return rows


def _format_auprc(value: Any, summary: "MultiSeedSummary | None" = None) -> str:  # type: ignore[name-defined]
    """Format an AUPRC cell as 'value' or 'mean ± std' when multi-seed."""
    if summary is not None:
        return summary.formatted()
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def write_suite_tables(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    multi_seed_summaries: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "suite_summary.json"
    payload: dict[str, Any] = {"summaries": summaries}
    if multi_seed_summaries:
        payload["multi_seed_summaries"] = multi_seed_summaries
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        rows.extend(flatten_for_table(summary))

    # Determine CSV fieldnames — add mean/std columns when multi-seed
    base_fields = [
        "run_id",
        "experiment",
        "train_f1",
        "validation_f1",
        "test_f1",
        "train_auprc",
        "validation_auprc",
        "test_auprc",
    ]
    extra_fields: list[str] = []
    if multi_seed_summaries:
        extra_fields = [
            "validation_auprc_mean_std",
            "test_auprc_mean_std",
        ]

    csv_path = output_dir / "suite_results_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        for row in rows:
            out_row: dict[str, Any] = {f: row.get(f) for f in base_fields}
            if multi_seed_summaries:
                key = f"{row['run_id']}::{row['experiment']}"
                ms = multi_seed_summaries.get(key, {})
                out_row["validation_auprc_mean_std"] = ms.get("validation_auprc", {}).get("formatted", "")
                out_row["test_auprc_mean_std"] = ms.get("test_auprc", {}).get("formatted", "")
            writer.writerow(out_row)

    lines = ["# Final Suite Results", ""]
    for row in rows:
        val_auprc = row["validation_auprc"]
        test_auprc = row["test_auprc"]
        if multi_seed_summaries:
            key = f"{row['run_id']}::{row['experiment']}"
            ms = multi_seed_summaries.get(key, {})
            val_auprc_str = ms.get("validation_auprc", {}).get("formatted", str(val_auprc))
            test_auprc_str = ms.get("test_auprc", {}).get("formatted", str(test_auprc))
        else:
            val_auprc_str = str(val_auprc)
            test_auprc_str = str(test_auprc)

        lines.append(
            f"- {row['run_id']} | {row['experiment']} | "
            f"train_f1={row['train_f1']} | validation_f1={row['validation_f1']} | test_f1={row['test_f1']} | "
            f"train_auprc={row['train_auprc']} | validation_auprc={val_auprc_str} | test_auprc={test_auprc_str}"
        )
    (output_dir / "suite_results_table.md").write_text("\n".join(lines), encoding="utf-8")


def _aggregate_seed_results(
    all_seed_summaries: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Aggregate per-seed result lists into multi-seed summary dicts.

    Args:
        all_seed_summaries: Outer list indexed by seed; each inner list is
            the flattened rows for that seed (from flatten_for_table).

    Returns:
        Dict keyed by ``"{run_id}::{experiment}"`` → summary dicts.
    """
    try:
        from watergen.evaluation.statistics import aggregate_multi_seed_results
    except ImportError:
        return {}

    # Collect per-key results across seeds
    per_key: dict[str, list[dict[str, Any]]] = {}
    for seed_rows in all_seed_summaries:
        for row in seed_rows:
            key = f"{row['run_id']}::{row['experiment']}"
            per_key.setdefault(key, []).append(row)

    output: dict[str, Any] = {}
    for key, seed_rows in per_key.items():
        key_summary: dict[str, Any] = {}
        for split in ("validation", "test"):
            for metric in ("auprc", "f1"):
                col = f"{split}_{metric}"
                values = [r[col] for r in seed_rows if isinstance(r.get(col), (int, float))]
                if not values:
                    continue
                # Wrap as results_per_seed format aggregate_multi_seed_results expects
                wrapped = [{metric: v} for v in values]
                ms = aggregate_multi_seed_results(wrapped, metric_key=metric, split=split)
                key_summary[col] = {
                    "mean": ms.mean,
                    "std": ms.std,
                    "ci_lower": ms.ci_lower,
                    "ci_upper": ms.ci_upper,
                    "n_seeds": ms.n_seeds,
                    "values": ms.values,
                    "formatted": ms.formatted(),
                }
        output[key] = key_summary

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final experiment suite.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ACTIVE_ROOT / "configs" / "final_experiment_matrix.yaml",
        help="Path to the frozen experiment matrix YAML.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for multi-seed evaluation. Default: [42]",
    )
    args = parser.parse_args()

    matrix = load_yaml(args.matrix)
    runs = matrix.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Experiment matrix must define a non-empty 'runs' list")

    generated_dir = ACTIVE_ROOT / "artifacts" / "suite" / "_generated_configs"
    log_dir = ACTIVE_ROOT / "artifacts" / "suite" / "logs"
    summary_dir = ACTIVE_ROOT / "artifacts" / "suite" / "summary"
    generated_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    multi_seed = len(seeds) > 1

    # Outer: per seed → list of summaries for that seed
    all_seed_flat_rows: list[list[dict[str, Any]]] = []

    # Keep the last seed's summaries as the canonical single-seed result list
    final_summaries: list[dict[str, Any]] = []
    exit_codes: list[tuple[str, int]] = []

    for seed in seeds:
        seed_summaries: list[dict[str, Any]] = []
        seed_flat_rows: list[dict[str, Any]] = []

        for run in runs:
            run_id = str(run["run_id"])
            # Append seed suffix only when running multiple seeds
            effective_run_id = f"{run_id}_seed{seed}" if multi_seed else run_id

            base_config = load_yaml(resolve_repo_path(run["base_config"]))
            overrides = run.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"Overrides for run '{run_id}' must be a mapping")

            merged_config = deep_merge(base_config, overrides)

            # Inject seed into config when multi-seed
            if multi_seed:
                merged_config.setdefault("evaluation", {})["master_seed"] = seed
                # Update artifact_dir to include seed so runs don't collide
                orig_dir = merged_config.get("outputs", {}).get("artifact_dir", "")
                if orig_dir:
                    merged_config.setdefault("outputs", {})["artifact_dir"] = (
                        orig_dir.rstrip("/").rstrip("\\") + f"_seed{seed}"
                    )

            generated_config_path = generated_dir / f"{effective_run_id}.yaml"
            generated_config_path.write_text(yaml.safe_dump(merged_config, sort_keys=False), encoding="utf-8")

            exit_code = run_one_config(generated_config_path, log_dir / f"{effective_run_id}.log")
            exit_codes.append((effective_run_id, exit_code))

            artifact_dir = resolve_repo_path(merged_config["outputs"]["artifact_dir"])
            run_summary = {
                "run_id": effective_run_id,
                "exit_code": exit_code,
                **extract_run_summary(effective_run_id, artifact_dir),
            }
            seed_summaries.append(run_summary)
            seed_flat_rows.extend(flatten_for_table(run_summary))

        all_seed_flat_rows.append(seed_flat_rows)
        final_summaries = seed_summaries

    # Aggregate multi-seed results
    multi_seed_summaries: dict[str, Any] | None = None
    if multi_seed:
        multi_seed_summaries = _aggregate_seed_results(all_seed_flat_rows)

    write_suite_tables(final_summaries, summary_dir, multi_seed_summaries=multi_seed_summaries)

    failed = [item for item in exit_codes if item[1] != 0]
    if failed:
        print("[warn] Some suite runs failed:")
        for run_id, code in failed:
            print(f"  - {run_id}: exit_code={code}")
        return 1

    print(f"[info] Suite summary written to: {summary_dir}")
    if multi_seed:
        print(f"[info] Multi-seed evaluation: {len(seeds)} seeds — {seeds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
