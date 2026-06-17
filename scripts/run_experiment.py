"""Lightweight experiment runner scaffold.

This runner validates configs, resolves benchmark and split manifests, generates
WNTR leak scenarios, and executes a first smoke experiment with a lightweight
tabular detector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.ensemble import IsolationForest
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.data import (  # noqa: E402
    build_acwa_case_study_frame,
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_path,
    resolve_split_manifest,
    scenario_specs_to_records,
    simulate_many,
    summarize_many,
)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (  # noqa: E402
    CTGANAugmenter,
    GaussianMixtureAugmenter,
    GraphCVAEAugmenter,
    PositiveClassAugmenter,
    QuantilePlausibilityFilter,
    SMOTEAugmenter,
    TabularDetectorBaseline,
    WNTRResidualDetector,
    prepare_feature_matrix,
)


def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of YAML file: {path}")
    return data


def select_entries(entries: list[tuple[str, dict]], selected_ids: list[str] | None) -> list[tuple[str, dict]]:
    """Filter manifest entries by benchmark id when a subset is requested."""
    if not selected_ids:
        return entries
    allowed = {item.strip() for item in selected_ids}
    return [entry for entry in entries if str(entry[1].get("id", "")).strip() in allowed]


def write_json(data: object, path: Path) -> None:
    """Write a JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_smoke_dataset(config: dict, resolved_split: dict, artifact_dir: Path) -> Path:
    """Generate scenario specs, run WNTR simulations, and persist the smoke dataset."""
    simulation_cfg = config.get("simulation", {})
    if not isinstance(simulation_cfg, dict):
        simulation_cfg = {}

    smoke_splits = simulation_cfg.get("smoke_splits", ["train", "validation"])
    leak_area_values = simulation_cfg.get("leak_area_values", [0.0001, 0.0005])
    max_candidates = int(simulation_cfg.get("max_candidates_per_network", 2))
    include_baseline = bool(simulation_cfg.get("include_baseline", True))
    sensor_coverage_levels = config.get("data", {}).get("sensor_coverage_levels", [1.0])
    disturbance_types = simulation_cfg.get("disturbance_types", ["leak"])

    scenario_specs = build_leak_scenarios(
        resolved_split,
        smoke_splits=smoke_splits,
        leak_area_values=leak_area_values,
        max_candidates_per_network=max_candidates,
        disturbance_types=disturbance_types,
        include_baseline=include_baseline,
    )

    scenario_artifact = artifact_dir / "scenario_specs.json"
    write_json({"scenarios": scenario_specs_to_records(scenario_specs)}, scenario_artifact)

    dataset = simulate_many(scenario_specs, sensor_coverage_levels=sensor_coverage_levels)
    dataset_path = artifact_dir / "smoke_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    return dataset_path


def evaluate_detector(
    frame: pd.DataFrame,
    *,
    algorithm: str,
    artifact_dir: Path,
    run_name: str,
) -> dict[str, object]:
    """Train and evaluate a tabular detector baseline."""
    feature_columns = [
        "pressure_mean",
        "pressure_min",
        "pressure_max",
        "pressure_std",
        "demand_mean",
        "demand_sum",
        "demand_std",
        "flow_mean",
        "flow_abs_mean",
        "flow_std",
        "tank_head_mean",
        "tank_head_std",
        "sensor_coverage",
        "observed_junction_count",
        "observed_link_count",
        "leak_area",
    ]

    train_df = frame[frame["split"] == "train"].copy()
    if train_df.empty:
        return {"error": f"No train rows available for detector run '{run_name}'"}

    detector = TabularDetectorBaseline(algorithm=algorithm, random_state=42)
    detector.fit_from_frame(train_df, feature_columns=feature_columns)

    results: dict[str, object] = {
        "algorithm": algorithm,
        "feature_columns": feature_columns,
        "rows": int(len(frame)),
        "splits": {},
    }

    for split_name in ["train", "validation", "test"]:
        split_df = frame[frame["split"] == split_name].copy()
        if split_df.empty:
            continue

        x, y, _ = prepare_feature_matrix(
            split_df,
            target_column="event_label",
            feature_columns=feature_columns,
        )
        probabilities = detector.predict_proba(x)[:, 1]
        predictions = detector.predict(x)
        report = binary_report(y, predictions, probabilities)

        by_coverage: dict[str, dict[str, object]] = {}
        if "sensor_coverage" in split_df.columns:
            for coverage_value, coverage_df in split_df.groupby("sensor_coverage"):
                cx, cy, _ = prepare_feature_matrix(
                    coverage_df,
                    target_column="event_label",
                    feature_columns=feature_columns,
                )
                cprobs = detector.predict_proba(cx)[:, 1]
                cpreds = detector.predict(cx)
                by_coverage[str(coverage_value)] = report_as_dict(binary_report(cy, cpreds, cprobs))
                by_coverage[str(coverage_value)]["rows"] = int(len(coverage_df))

        results["splits"][split_name] = {
            **report_as_dict(report),
            "rows": int(len(split_df)),
            "positive_rows": int(split_df["event_label"].sum()),
            "by_sensor_coverage": by_coverage,
        }

    # Record the role of each split so JSON reports are self-describing.
    results["split_roles"] = {
        "train": "training",
        "validation": "hyperparameter_selection",
        "test": "final_holdout",
    }

    metrics_path = artifact_dir / f"{run_name}_metrics.json"
    write_json(results, metrics_path)
    return results


def run_smoke_experiments(dataset_path: Path, artifact_dir: Path) -> dict[str, object]:
    """Run baseline, stronger baseline, and augmented baseline smoke experiments."""
    frame = pd.read_csv(dataset_path)

    results: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "rows": int(len(frame)),
        "experiments": {},
    }

    results["experiments"]["logreg_baseline"] = evaluate_detector(
        frame,
        algorithm="logreg",
        artifact_dir=artifact_dir,
        run_name="logreg_baseline",
    )
    results["experiments"]["random_forest_baseline"] = evaluate_detector(
        frame,
        algorithm="random_forest",
        artifact_dir=artifact_dir,
        run_name="random_forest_baseline",
    )

    feature_columns = results["experiments"]["logreg_baseline"].get("feature_columns", [])
    train_df = frame[frame["split"] == "train"].copy()
    augmenter = PositiveClassAugmenter(random_state=42, noise_scale=0.05)
    plausibility_filter = QuantilePlausibilityFilter().fit(train_df, feature_columns)
    synthetic_positive = augmenter.generate(
        train_df,
        feature_columns=feature_columns,
        target_column="event_label",
        multiplier=1.0,
        preserve_columns=["sensor_coverage", "observed_junction_count", "observed_link_count", "leak_area"],
    )
    filtered_positive = plausibility_filter.filter(synthetic_positive, feature_columns)
    filter_mode = "quantile_filter"
    if filtered_positive.empty and not synthetic_positive.empty:
        filtered_positive = synthetic_positive.copy()
        filter_mode = "fallback_keep_all"
    augmented_train = pd.concat(
        [train_df.assign(is_synthetic=0), filtered_positive],
        axis=0,
        ignore_index=True,
    )
    augmented_frame = pd.concat(
        [augmented_train, frame[frame["split"] != "train"].copy()],
        axis=0,
        ignore_index=True,
    )
    results["augmentation"] = {
        "synthetic_generated_rows": int(len(synthetic_positive)),
        "synthetic_retained_rows": int(len(filtered_positive)),
        "synthetic_rejected_rows": int(len(synthetic_positive) - len(filtered_positive)),
        "filter_mode": filter_mode,
    }
    results["experiments"]["logreg_augmented"] = evaluate_detector(
        augmented_frame,
        algorithm="logreg",
        artifact_dir=artifact_dir,
        run_name="logreg_augmented",
    )

    try:
        gmm_augmenter = GaussianMixtureAugmenter(random_state=42, n_components=2)
        gmm_positive = gmm_augmenter.generate(
            train_df,
            feature_columns=feature_columns,
            target_column="event_label",
            preserve_template_columns=["sensor_coverage", "observed_junction_count", "observed_link_count", "leak_area"],
        )
        gmm_positive = plausibility_filter.filter(gmm_positive, feature_columns)
        if gmm_positive.empty:
            raise ValueError("GMM-generated rows were all filtered out")
        gmm_train = pd.concat([train_df.assign(is_synthetic=0), gmm_positive], axis=0, ignore_index=True)
        gmm_frame = pd.concat([gmm_train, frame[frame["split"] != "train"].copy()], axis=0, ignore_index=True)
        results["augmentation"]["gmm_generated_rows"] = int(len(gmm_positive))
        results["experiments"]["logreg_gmm_augmented"] = evaluate_detector(
            gmm_frame,
            algorithm="logreg",
            artifact_dir=artifact_dir,
            run_name="logreg_gmm_augmented",
        )
    except Exception as exc:
        results["augmentation"]["gmm_status"] = f"skipped: {exc}"

    try:
        smote_augmenter = SMOTEAugmenter(random_state=42)
        smote_positive = smote_augmenter.generate(
            train_df,
            feature_columns=feature_columns,
            target_column="event_label",
        )
        smote_positive = plausibility_filter.filter(smote_positive, feature_columns)
        if smote_positive.empty:
            raise ValueError("SMOTE-generated rows were all filtered out")
        smote_train = pd.concat([train_df.assign(is_synthetic=0), smote_positive], axis=0, ignore_index=True)
        smote_frame = pd.concat([smote_train, frame[frame["split"] != "train"].copy()], axis=0, ignore_index=True)
        results["augmentation"]["smote_generated_rows"] = int(len(smote_positive))
        results["experiments"]["smote_logreg"] = evaluate_detector(
            smote_frame,
            algorithm="logreg",
            artifact_dir=artifact_dir,
            run_name="smote_logreg",
        )
        results["experiments"]["smote_random_forest"] = evaluate_detector(
            smote_frame,
            algorithm="random_forest",
            artifact_dir=artifact_dir,
            run_name="smote_random_forest",
        )
    except Exception as exc:
        results["augmentation"]["smote_status"] = f"skipped: {exc}"

    try:
        ctgan_augmenter = CTGANAugmenter(random_state=42, epochs=50)
        ctgan_positive = ctgan_augmenter.generate(
            train_df,
            feature_columns=feature_columns,
            target_column="event_label",
        )
        ctgan_positive = plausibility_filter.filter(ctgan_positive, feature_columns)
        if ctgan_positive.empty:
            raise ValueError("CTGAN-generated rows were all filtered out")
        ctgan_train = pd.concat([train_df.assign(is_synthetic=0), ctgan_positive], axis=0, ignore_index=True)
        ctgan_frame = pd.concat([ctgan_train, frame[frame["split"] != "train"].copy()], axis=0, ignore_index=True)
        results["augmentation"]["ctgan_generated_rows"] = int(len(ctgan_positive))
        results["experiments"]["ctgan_logreg"] = evaluate_detector(
            ctgan_frame,
            algorithm="logreg",
            artifact_dir=artifact_dir,
            run_name="ctgan_logreg",
        )
        results["experiments"]["ctgan_random_forest"] = evaluate_detector(
            ctgan_frame,
            algorithm="random_forest",
            artifact_dir=artifact_dir,
            run_name="ctgan_random_forest",
        )
    except Exception as exc:
        results["augmentation"]["ctgan_status"] = f"skipped: {exc}"

    # ---- P12/P13: Graph-Conditioned CVAE augmenter (novel contribution) ----
    try:
        cvae_augmenter = GraphCVAEAugmenter(
            random_state=42,
            epochs=30,           # Smoke run — fast; use train_graph_cvae.py for full training
            lambda_physics=0.5,
            warmup_epochs=10,
        )
        cvae_augmenter.fit(train_df, feature_columns=feature_columns, target_column="event_label")
        cvae_positive = cvae_augmenter.generate(
            train_df,
            feature_columns=feature_columns,
            target_column="event_label",
            n_samples=len(train_df[train_df["event_label"] == 1]),
        )
        cvae_positive = plausibility_filter.filter(cvae_positive, feature_columns)
        if cvae_positive.empty:
            raise ValueError("Graph-CVAE-generated rows were all filtered out")
        cvae_train = pd.concat(
            [train_df.assign(is_synthetic=0), cvae_positive], axis=0, ignore_index=True
        )
        cvae_frame = pd.concat(
            [cvae_train, frame[frame["split"] != "train"].copy()], axis=0, ignore_index=True
        )
        results["augmentation"]["cvae_generated_rows"] = int(len(cvae_positive))
        results["experiments"]["graph_cvae_rf"] = evaluate_detector(
            cvae_frame,
            algorithm="random_forest",
            artifact_dir=artifact_dir,
            run_name="graph_cvae_rf",
        )
        results["experiments"]["graph_cvae_logreg"] = evaluate_detector(
            cvae_frame,
            algorithm="logreg",
            artifact_dir=artifact_dir,
            run_name="graph_cvae_logreg",
        )
    except Exception as exc:
        results["experiments"]["graph_cvae_rf"] = {"error": str(exc)}
        results["augmentation"]["cvae_status"] = f"skipped: {exc}"

    # ---- P8: WNTR pressure-residual domain baseline ----
    try:
        train_df_for_wntr = frame[frame["split"] == "train"].copy()
        wntr_detector = WNTRResidualDetector(z_score_threshold=3.0)
        wntr_detector.fit(train_df_for_wntr)

        wntr_result: dict[str, object] = {"splits": {}}
        for split_name in ["train", "validation", "test"]:
            split_df = frame[frame["split"] == split_name].copy()
            if split_df.empty:
                continue
            predictions = wntr_detector.predict(split_df)
            probabilities = wntr_detector.predict_proba(split_df)[:, 1]
            y = split_df["event_label"].to_numpy(dtype=int)
            report = binary_report(y, predictions, probabilities)
            wntr_result["splits"][split_name] = {
                **report_as_dict(report),
                "rows": int(len(split_df)),
                "positive_rows": int(split_df["event_label"].sum()),
            }
        results["experiments"]["wntr_residual"] = wntr_result
        write_json(wntr_result, artifact_dir / "wntr_residual_metrics.json")
    except Exception as exc:
        results["experiments"]["wntr_residual"] = {"error": str(exc)}

    summary_path = artifact_dir / "smoke_experiment_summary.json"
    write_json(results, summary_path)
    return results


def run_acwa_case_study(secondary_case_study: list[str], artifact_dir: Path) -> dict[str, object]:
    """Run a lightweight ACWA anomaly/OOD case study."""
    xlsx_paths = [Path(item) for item in secondary_case_study if str(item).lower().endswith(".xlsx")]
    if len(xlsx_paths) < 2:
        return {"status": "skipped", "reason": "Not enough ACWA workbook paths configured"}

    frame = build_acwa_case_study_frame(xlsx_paths[0], xlsx_paths[1])
    if frame.empty:
        return {"status": "skipped", "reason": "ACWA case-study frame is empty"}

    reports: dict[str, object] = {}
    for sheet_name, sheet_df in frame.groupby("sheet_name"):
        numeric_columns = [
            name
            for name in sheet_df.columns
            if name not in {"source_label", "sheet_name", "row_origin", "Time"} and pd.api.types.is_numeric_dtype(sheet_df[name])
        ]
        if len(numeric_columns) < 3:
            continue

        normal_df = sheet_df[sheet_df["source_label"] == 0].copy()
        attack_df = sheet_df[sheet_df["source_label"] == 1].copy()
        if normal_df.empty or attack_df.empty:
            continue

        sort_column = "UTCmsec" if "UTCmsec" in normal_df.columns else numeric_columns[0]
        normal_df = normal_df.sort_values(by=sort_column)
        split_index = max(1, int(len(normal_df) * 0.7))
        train_df = normal_df.iloc[:split_index].copy()
        holdout_df = normal_df.iloc[split_index:].copy()
        evaluation_df = pd.concat([holdout_df, attack_df], axis=0, ignore_index=True)
        if evaluation_df.empty:
            continue

        contamination = min(0.25, max(0.05, len(attack_df) / max(len(evaluation_df), 1)))
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
        )
        x_train = train_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x_eval = evaluation_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        model.fit(x_train)
        decision_scores = -model.decision_function(x_eval)
        predictions = (model.predict(x_eval) == -1).astype(int)
        report = binary_report(evaluation_df["source_label"], predictions, decision_scores)
        reports[str(sheet_name)] = {
            **report_as_dict(report),
            "rows": int(len(evaluation_df)),
            "attack_rows": int(evaluation_df["source_label"].sum()),
            "normal_rows": int(len(evaluation_df) - evaluation_df["source_label"].sum()),
            "feature_columns": numeric_columns,
        }

    result = {
        "status": "completed" if reports else "skipped",
        "sheet_reports": reports,
        "source_files": [str(path) for path in xlsx_paths[:2]],
    }

    # Sim-to-real framing: evaluate whether augmented EPANET model transfers to ACWA
    # Compare: (a) model trained only on ACWA normal data, (b) model trained on simulated data + fine-tuned
    result["sim_to_real_framing"] = {
        "description": "ACWA is treated as a sim-to-real transfer/OOD validation, not a primary benchmark.",
        "interpretation": (
            "Physics-grounded augmentation trained on EPANET simulations should transfer "
            "to the ACWA real-world testbed because hydraulic signatures are governed by "
            "the same conservation laws, regardless of network scale or sensor technology."
        ),
        "note": "Full sim-to-real evaluation requires training the primary model on EPANET data; see run_cross_network_eval.py",
    }

    write_json(result, artifact_dir / "acwa_case_study.json")
    return result


def summarize_metrics_line(split_name: str, metrics: dict[str, object]) -> str:
    """Format a single markdown line with both F1 and AUPRC for a given split.

    Parameters
    ----------
    split_name:
        The data-split label (e.g. ``"train"``, ``"validation"``, ``"test"``).
    metrics:
        The metrics dictionary for the split, as produced by ``report_as_dict``.

    Returns
    -------
    str
        A formatted string ready to be appended as a markdown list item.
    """
    f1 = metrics.get("f1", float("nan"))
    precision = metrics.get("precision", float("nan"))
    recall = metrics.get("recall", float("nan"))
    auprc = metrics.get("auprc", "n/a")
    rows = metrics.get("rows", "n/a")

    auprc_str = f"{auprc:.3f}" if isinstance(auprc, float) else str(auprc)
    return (
        f"- {split_name}: F1={f1:.3f}, AUPRC={auprc_str}, "
        f"precision={precision:.3f}, recall={recall:.3f}, rows={rows}"
    )


def write_experiment_report(
    *,
    artifact_dir: Path,
    experiment_name: str,
    smoke_results: dict[str, object] | None,
    acwa_results: dict[str, object] | None,
) -> None:
    """Write a compact markdown report and summary figures for F1 and AUPRC."""
    lines = [f"# Experiment Report: {experiment_name}", ""]
    plot_labels: list[str] = []
    plot_f1_values: list[float] = []
    plot_auprc_values: list[float] = []

    if smoke_results:
        lines.extend(["## Smoke Experiments", ""])
        for run_name, run_metrics in smoke_results.get("experiments", {}).items():
            lines.append(f"### {run_name}")
            splits = run_metrics.get("splits", {})
            for split_name in ["train", "validation", "test"]:
                metrics = splits.get(split_name)
                if not isinstance(metrics, dict):
                    continue
                lines.append(summarize_metrics_line(split_name, metrics))
                if split_name in {"validation", "test"}:
                    plot_labels.append(f"{run_name}:{split_name}")
                    plot_f1_values.append(float(metrics["f1"]))
                    auprc = metrics.get("auprc")
                    plot_auprc_values.append(float(auprc) if isinstance(auprc, (int, float)) else 0.0)
            lines.append("")

        augmentation = smoke_results.get("augmentation", {})
        if isinstance(augmentation, dict):
            lines.append("## Augmentation Summary")
            lines.append(
                f"- generated={augmentation.get('synthetic_generated_rows', 0)}, "
                f"retained={augmentation.get('synthetic_retained_rows', 0)}, "
                f"rejected={augmentation.get('synthetic_rejected_rows', 0)}, "
                f"filter_mode={augmentation.get('filter_mode', 'n/a')}"
            )
            if "gmm_generated_rows" in augmentation:
                lines.append(f"- gmm_generated_rows={augmentation.get('gmm_generated_rows')}")
            if "gmm_status" in augmentation:
                lines.append(f"- gmm_status={augmentation.get('gmm_status')}")
            lines.append("")

    if acwa_results:
        lines.extend(["## ACWA Case Study", ""])
        lines.append(f"- status: {acwa_results.get('status', 'unknown')}")
        for sheet_name, metrics in acwa_results.get("sheet_reports", {}).items():
            lines.append(
                f"- {sheet_name}: F1={metrics['f1']:.3f}, precision={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, rows={metrics['rows']}"
            )

        # Sim-to-real framing note
        sim_to_real = acwa_results.get("sim_to_real_framing")
        if isinstance(sim_to_real, dict):
            lines.append("")
            lines.append("### Sim-to-Real Transfer Framing")
            lines.append("")
            lines.append(f"**Role**: {sim_to_real.get('description', '')}")
            lines.append("")
            lines.append(f"**Interpretation**: {sim_to_real.get('interpretation', '')}")
            lines.append("")
            lines.append(f"**Note**: {sim_to_real.get('note', '')}")

        lines.append("")

    report_path = artifact_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    if plot_labels and plot_f1_values:
        # F1 summary figure (kept for backward compatibility)
        plt.figure(figsize=(10, 4))
        plt.bar(plot_labels, plot_f1_values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("F1")
        plt.title(f"{experiment_name} smoke experiment validation/test F1")
        plt.tight_layout()
        plt.savefig(artifact_dir / "smoke_f1_summary.png", dpi=150)
        plt.close()

    if plot_labels and any(v > 0.0 for v in plot_auprc_values):
        # AUPRC summary figure — primary reported metric
        plt.figure(figsize=(10, 4))
        plt.bar(plot_labels, plot_auprc_values)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("AUPRC")
        plt.title(f"{experiment_name} smoke experiment validation/test AUPRC")
        plt.tight_layout()
        plt.savefig(artifact_dir / "smoke_auprc_summary.png", dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scaffolded water experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a config file, e.g. dev/active/configs/plan_a.yaml",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional benchmark manifest override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"[error] Config file not found: {args.config}")
        return 1

    config = load_yaml(args.config)
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        print(f"[error] Expected 'data' section in config: {args.config}")
        return 1

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_value = data_cfg.get("benchmark_manifest", "dev/active/configs/benchmark_manifest.yaml")
        manifest_path = resolve_path(manifest_value, root=REPO_ROOT)
    else:
        manifest_path = resolve_path(manifest_path, root=REPO_ROOT)

    if not manifest_path.exists():
        print(f"[error] Benchmark manifest not found: {manifest_path}")
        return 1

    manifest = load_yaml(manifest_path)
    raw_entries = [(entry["stage"], entry) for entry in collect_benchmark_entries(manifest)]
    requested_ids = data_cfg.get("benchmark_ids")
    if isinstance(requested_ids, list):
        entries = select_entries(raw_entries, [str(item) for item in requested_ids])
    else:
        entries = raw_entries

    if not entries:
        print(f"[error] No staged benchmark entries found in manifest: {manifest_path}")
        return 1

    resolved_paths: list[Path] = []
    staged_rows: list[tuple[str, str, str, Path]] = []
    missing_paths: list[Path] = []

    for stage_name, item in entries:
        item_id = str(item.get("id", "unknown"))
        role = str(item.get("role", "unspecified"))
        raw_path = item.get("path")
        if not raw_path:
            continue
        resolved = resolve_path(str(raw_path), root=REPO_ROOT)
        if resolved.exists():
            resolved_paths.append(resolved)
            staged_rows.append((stage_name, item_id, role, resolved))
        else:
            missing_paths.append(resolved)

    if missing_paths:
        print("[error] Some benchmark paths from the manifest do not exist:")
        for missing in missing_paths:
            print(f"  - {missing}")
        return 1

    artifact_dir_value = config.get("outputs", {}).get("artifact_dir", "dev/active/artifacts")
    artifact_dir = resolve_path(str(artifact_dir_value), root=REPO_ROOT)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summaries = {str(summary.path.resolve()): summary for summary in summarize_many(resolved_paths)}

    split_manifest_value = data_cfg.get("split_manifest")
    resolved_split = None
    if split_manifest_value:
        split_manifest_path = resolve_path(str(split_manifest_value), root=REPO_ROOT)
        resolved_split = resolve_split_manifest(split_manifest_path, root=REPO_ROOT)
        generated_split_path = artifact_dir / "splits" / "resolved_split_manifest.json"
        write_json(resolved_split, generated_split_path)

    print("[info] Experiment scaffold runner")
    print(f"[info] Using config: {args.config}")
    print(f"[info] Experiment: {config.get('experiment_name', 'unnamed_experiment')}")
    print(f"[info] Primary dataset: {data_cfg.get('primary_dataset', 'unspecified')}")
    print(f"[info] Benchmark manifest: {manifest_path}")
    print(f"[info] Resolved benchmark networks: {len(resolved_paths)}")
    print(f"[info] Artifact directory: {artifact_dir}")
    if resolved_split is not None:
        print(f"[info] Split manifest: {data_cfg.get('split_manifest')}")
        print(f"[info] Resolved split artifact: {artifact_dir / 'splits' / 'resolved_split_manifest.json'}")

    print("[info] Staged benchmark summary:")
    for stage_name, item_id, role, resolved in staged_rows:
        summary = summaries[str(resolved.resolve())]
        print(
            "  - "
            f"{stage_name:<6} | "
            f"{item_id:<16} | "
            f"j={summary.junctions:<4} "
            f"r={summary.reservoirs:<2} "
            f"t={summary.tanks:<2} "
            f"p={summary.pipes:<4} "
            f"pu={summary.pumps:<2} "
            f"v={summary.valves:<2} | "
            f"units={summary.units or 'n/a':<6} "
            f"headloss={summary.headloss or 'n/a':<4} | "
            f"{role}"
        )

    secondary_case_study = data_cfg.get("secondary_case_study", [])
    if isinstance(secondary_case_study, list) and secondary_case_study:
        print(f"[info] Secondary case-study files configured: {len(secondary_case_study)}")

    if resolved_split is not None:
        smoke_artifact_dir = artifact_dir / "smoke"
        smoke_artifact_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = build_smoke_dataset(config, resolved_split, smoke_artifact_dir)
        smoke_results = run_smoke_experiments(dataset_path, smoke_artifact_dir)
        print(f"[info] Smoke dataset: {dataset_path}")
        if "error" in smoke_results:
            print(f"[warn] Smoke baseline did not complete: {smoke_results['error']}")
        else:
            print(f"[info] Smoke experiment summary: {smoke_artifact_dir / 'smoke_experiment_summary.json'}")
            for run_name, run_metrics in smoke_results.get("experiments", {}).items():
                train_metrics = run_metrics.get("splits", {}).get("train")
                val_metrics = run_metrics.get("splits", {}).get("validation")
                test_metrics = run_metrics.get("splits", {}).get("test")
                parts = [f"{run_name}"]
                if isinstance(train_metrics, dict):
                    train_auprc = train_metrics.get("auprc", "n/a")
                    train_auprc_str = f"{train_auprc:.3f}" if isinstance(train_auprc, float) else str(train_auprc)
                    parts.append(f"train_f1={train_metrics['f1']:.3f} train_auprc={train_auprc_str}")
                if isinstance(val_metrics, dict):
                    val_auprc = val_metrics.get("auprc", "n/a")
                    val_auprc_str = f"{val_auprc:.3f}" if isinstance(val_auprc, float) else str(val_auprc)
                    parts.append(f"val_f1={val_metrics['f1']:.3f} val_auprc={val_auprc_str}")
                if isinstance(test_metrics, dict):
                    test_auprc = test_metrics.get("auprc", "n/a")
                    test_auprc_str = f"{test_auprc:.3f}" if isinstance(test_auprc, float) else str(test_auprc)
                    parts.append(f"test_f1={test_metrics['f1']:.3f} test_auprc={test_auprc_str}")
                print("  - " + " | ".join(parts))
            augmentation = smoke_results.get("augmentation", {})
            if augmentation:
                print(
                    "  - augmentation "
                    f"generated={augmentation.get('synthetic_generated_rows', 0)} "
                    f"retained={augmentation.get('synthetic_retained_rows', 0)} "
                    f"rejected={augmentation.get('synthetic_rejected_rows', 0)}"
                )
    else:
        smoke_results = None

    acwa_results = None
    if isinstance(secondary_case_study, list) and secondary_case_study:
        acwa_artifact_dir = artifact_dir / "acwa"
        acwa_artifact_dir.mkdir(parents=True, exist_ok=True)
        acwa_results = run_acwa_case_study(secondary_case_study, acwa_artifact_dir)
        print(f"[info] ACWA case-study artifact: {acwa_artifact_dir / 'acwa_case_study.json'}")
        if acwa_results.get("status") == "completed":
            for sheet_name, metrics in acwa_results.get("sheet_reports", {}).items():
                print(
                    "  - acwa "
                    f"{sheet_name}: "
                    f"f1={metrics['f1']:.3f} "
                    f"precision={metrics['precision']:.3f} "
                    f"recall={metrics['recall']:.3f}"
                )

    write_experiment_report(
        artifact_dir=artifact_dir,
        experiment_name=str(config.get("experiment_name", "unnamed_experiment")),
        smoke_results=smoke_results,
        acwa_results=acwa_results,
    )

    print("[todo] Extend smoke features to sparse-sensing subsets")
    print("[todo] Add richer leak and disturbance injection beyond single-junction leaks")
    print("[todo] Replace lightweight augmentation with a stronger learned generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
