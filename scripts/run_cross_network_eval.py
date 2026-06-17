"""Cross-network transfer evaluation for WDN anomaly detection.

Implements the March 31 revised held-out-network protocol:
- Train on: Anytown, Hanoi
- Test on:  Net3, D-Town, L-TOWN

This expands cross-network evaluation beyond a 2-network test fold while
avoiding known WNTR incompatibilities with BWSN_Network_1.

Network IDs are taken directly from benchmark_manifest.yaml staged_benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Path bootstrap — make watergen importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.data import (  # noqa: E402
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_path,
    simulate_many,
    summarize_many,
)
from watergen.evaluation import (  # noqa: E402
    HydraulicResidualScorer,
    PhysicallyInformedTriageModel,
    binary_report,
    build_triage_frame,
    heuristic_triage_score,
    report_as_dict,
)
from watergen.models import (  # noqa: E402
    GaussianMixtureAugmenter,
    PositiveClassAugmenter,
    QuantilePlausibilityFilter,
    TabularDetectorBaseline,
    prepare_feature_matrix,
)
from watergen.models.baselines import GNNDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Network taxonomy — IDs match benchmark_manifest.yaml exactly
# ---------------------------------------------------------------------------
TRAIN_NETWORK_IDS = ["anytown", "hanoi"]
TEST_NETWORK_IDS = ["net3", "d_town", "l_town"]

# Approximate node counts used for size-weighted AUPRC aggregation.
# Values sourced from standard EPANET benchmark literature.
NETWORK_NODE_COUNTS: dict[str, int] = {
    "net1": 11,
    "anytown": 19,
    "hanoi": 32,
    "net3": 97,
    "bwsn_network_1": 126,
    "d_town": 399,
    "l_town": 785,
}

# Feature columns — matches run_experiment.py exactly
FEATURE_COLUMNS = [
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

# Augmenter labels used in output tables/figures
AUGMENTER_LABELS = ["baseline", "noise", "gmm", "smote"]


# ---------------------------------------------------------------------------
# YAML / JSON helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top level: {path}")
    return data


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _build_network_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Return a mapping of benchmark_id -> entry dict (with resolved path)."""
    manifest = _load_yaml(manifest_path)
    entries = collect_benchmark_entries(manifest)
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        nid = str(entry.get("id", "")).strip()
        if not nid:
            continue
        raw_path = entry.get("path", "")
        resolved = resolve_path(str(raw_path), root=REPO_ROOT)
        index[nid] = {**entry, "resolved_path": resolved}
    return index


def _build_split_for_networks(
    network_ids: list[str],
    network_index: dict[str, dict[str, Any]],
    *,
    split_name: str,
) -> dict[str, Any]:
    """Construct a minimal resolved_split_manifest dict for a list of networks."""
    entries = []
    for nid in network_ids:
        if nid not in network_index:
            print(f"[warn] Network id '{nid}' not found in manifest — skipping", file=sys.stderr)
            continue
        entry = network_index[nid]
        resolved = entry["resolved_path"]
        if not resolved.exists():
            print(f"[warn] Network path does not exist: {resolved} — skipping", file=sys.stderr)
            continue
        entries.append(
            {
                "id": nid,
                "stage": entry.get("stage", "unknown"),
                "path": str(resolved),
                "role": entry.get("role", ""),
                "notes": entry.get("notes", ""),
                "split": split_name,
                "split_role": split_name,
            }
        )
    return {"partitions": {split_name: entries}}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_network_group(
    network_ids: list[str],
    network_index: dict[str, dict[str, Any]],
    *,
    split_name: str,
    sensor_coverage_levels: list[float],
    disturbance_types: list[str],
    max_candidates: int = 5,
    leak_area_values: list[float] | None = None,
) -> pd.DataFrame:
    """Simulate all scenarios for a group of networks and return a combined frame."""
    if leak_area_values is None:
        leak_area_values = [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005]

    resolved_split = _build_split_for_networks(
        network_ids, network_index, split_name=split_name
    )

    frames: list[pd.DataFrame] = []
    for entry in resolved_split["partitions"].get(split_name, []):
        nid = entry["id"]
        try:
            specs = build_leak_scenarios(
                resolved_split,
                smoke_splits=[split_name],
                leak_area_values=leak_area_values,
                max_candidates_per_network=max_candidates,
                disturbance_types=disturbance_types,
                include_baseline=True,
            )
            # Filter to only this network's specs to avoid re-simulating
            network_specs = [s for s in specs if s.benchmark_id == nid]
            if not network_specs:
                print(f"[warn] No scenarios generated for '{nid}' — skipping", file=sys.stderr)
                continue
            df = simulate_many(network_specs, sensor_coverage_levels=sensor_coverage_levels)
            frames.append(df)
            print(f"[info] Simulated {len(network_specs)} scenarios for '{nid}' -> {len(df)} rows")
        except Exception as exc:
            print(f"[warn] Simulation failed for '{nid}': {exc} — skipping", file=sys.stderr)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _apply_augmenter(
    train_df: pd.DataFrame,
    augmenter_name: str | None,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Apply the named augmenter to training data, returning the augmented frame.

    augmenter_name: None/'baseline' → no augmentation
                    'noise'         → PositiveClassAugmenter (Gaussian noise)
                    'gmm'           → GaussianMixtureAugmenter
                    'smote'         → SMOTEAugmenter (with noise fallback)
    """
    if augmenter_name in (None, "baseline"):
        return train_df.copy()

    plausibility_filter = QuantilePlausibilityFilter().fit(train_df, FEATURE_COLUMNS)

    if augmenter_name == "noise":
        augmenter = PositiveClassAugmenter(random_state=random_state, noise_scale=0.05)
        synthetic = augmenter.generate(
            train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column="event_label",
            multiplier=1.0,
            preserve_columns=["sensor_coverage", "observed_junction_count", "observed_link_count", "leak_area"],
        )

    elif augmenter_name == "gmm":
        try:
            gmm = GaussianMixtureAugmenter(random_state=random_state, n_components=2)
            synthetic = gmm.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
                preserve_template_columns=["sensor_coverage", "observed_junction_count", "observed_link_count", "leak_area"],
            )
        except Exception as exc:
            print(f"[warn] GMM augmenter failed ({exc}); falling back to noise", file=sys.stderr)
            fallback = PositiveClassAugmenter(random_state=random_state, noise_scale=0.05)
            synthetic = fallback.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
                multiplier=1.0,
            )

    elif augmenter_name == "smote":
        try:
            from watergen.models.augmentation import SMOTEAugmenter
            smote = SMOTEAugmenter(random_state=random_state)
            synthetic = smote.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
            )
        except Exception as exc:
            print(f"[warn] SMOTE augmenter failed ({exc}); falling back to noise", file=sys.stderr)
            fallback = PositiveClassAugmenter(random_state=random_state, noise_scale=0.05)
            synthetic = fallback.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
                multiplier=1.0,
            )
    else:
        raise ValueError(f"Unknown augmenter: {augmenter_name!r}")

    if synthetic.empty:
        print(f"[warn] Augmenter '{augmenter_name}' produced no rows — returning unaugmented train", file=sys.stderr)
        return train_df.copy()

    filtered = plausibility_filter.filter(synthetic, FEATURE_COLUMNS)
    if filtered.empty:
        print(f"[warn] All synthetic rows filtered by plausibility — using unfiltered", file=sys.stderr)
        filtered = synthetic.copy()

    augmented = pd.concat(
        [train_df.assign(is_synthetic=0), filtered],
        axis=0,
        ignore_index=True,
    )
    return augmented


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_on_network(
    detector: TabularDetectorBaseline,
    test_df: pd.DataFrame,
    network_id: str,
) -> dict[str, Any]:
    """Evaluate a fitted detector on a single test network frame."""
    if test_df.empty:
        return {"network_id": network_id, "error": "empty test frame", "auprc": None}

    available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    if not available_features:
        return {"network_id": network_id, "error": "no feature columns present", "auprc": None}

    try:
        x, y, _ = prepare_feature_matrix(
            test_df,
            target_column="event_label",
            feature_columns=available_features,
        )
        probabilities = detector.predict_proba(x)[:, 1]
        predictions = detector.predict(x)
        report = binary_report(y, predictions, probabilities)
        result = report_as_dict(report)
        result["network_id"] = network_id
        result["rows"] = int(len(test_df))
        result["positive_rows"] = int(test_df["event_label"].sum())
        return result
    except Exception as exc:
        return {"network_id": network_id, "error": str(exc), "auprc": None}


def _evaluate_triage_on_network(
    detector: TabularDetectorBaseline,
    triage_model: PhysicallyInformedTriageModel,
    scorer: HydraulicResidualScorer,
    test_df: pd.DataFrame,
    network_id: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate learned triage reranking on a single test network."""
    if test_df.empty:
        return ({"network_id": network_id, "error": "empty test frame", "auprc": None}, pd.DataFrame())

    available_features = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    if not available_features:
        return ({"network_id": network_id, "error": "no feature columns present", "auprc": None}, pd.DataFrame())

    x, y, _ = prepare_feature_matrix(
        test_df,
        target_column="event_label",
        feature_columns=available_features,
    )
    base_prob = detector.predict_proba(x)[:, 1]
    base_pred = detector.predict(x)

    triage = build_triage_frame(test_df, base_scores=base_prob, scorer=scorer)
    triage_prob = triage_model.predict_proba(triage.frame)[:, 1]
    triage_pred = (triage_prob >= 0.5).astype(int)
    heuristic_score = heuristic_triage_score(triage.frame)

    report = binary_report(y, triage_pred, triage_prob)
    result = report_as_dict(report)
    result["network_id"] = network_id
    result["rows"] = int(len(test_df))
    result["positive_rows"] = int(test_df["event_label"].sum())

    records = triage.frame.copy()
    records["network_id"] = network_id
    records["y_true"] = y
    records["base_score"] = base_prob
    records["base_pred"] = base_pred
    records["triage_score"] = triage_prob
    records["triage_pred"] = triage_pred
    records["heuristic_triage_score"] = heuristic_score
    return result, records


# ---------------------------------------------------------------------------
# Size-weighted AUPRC
# ---------------------------------------------------------------------------

def size_weighted_auprc(network_results: dict[str, dict[str, Any]]) -> float:
    """Compute size-weighted macro-average AUPRC.

    Weights by number of nodes so large networks dominate the aggregate,
    reflecting their greater real-world deployment relevance.

    Args:
        network_results: Mapping of network_id -> result dict with 'auprc' key.

    Returns:
        Weighted average AUPRC, or 0.0 if no valid results.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for nid, result in network_results.items():
        auprc = result.get("auprc")
        if auprc is None or not isinstance(auprc, (int, float)):
            continue
        weight = float(NETWORK_NODE_COUNTS.get(nid, 1))
        weighted_sum += float(auprc) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def plot_cross_network_heatmap(results: dict[str, Any], output_path: Path) -> None:
    """Generate augmenter × network heatmap colored by AUPRC.

    Rows: augmenter strategies (baseline, noise, GMM, SMOTE)
    Columns: test networks (d_town, l_town)
    Values: AUPRC (color scale 0-1, green=high, red=low)
    """
    augmenters = AUGMENTER_LABELS
    test_networks = TEST_NETWORK_IDS

    # Build 2D matrix: rows=augmenters, cols=test networks
    matrix = []
    for aug in augmenters:
        row = []
        for nid in test_networks:
            auprc = (
                results
                .get("per_augmenter", {})
                .get(aug, {})
                .get("per_network", {})
                .get(nid, {})
                .get("auprc")
            )
            row.append(float(auprc) if isinstance(auprc, (int, float)) else float("nan"))
        matrix.append(row)

    import numpy as np
    data = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(max(4, len(test_networks) * 2), max(3, len(augmenters) * 1.2)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(test_networks)))
    ax.set_xticklabels(test_networks, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(augmenters)))
    ax.set_yticklabels(augmenters, fontsize=10)

    ax.set_xlabel("Test Network", fontsize=11)
    ax.set_ylabel("Augmenter", fontsize=11)
    ax.set_title("Cross-Network AUPRC (LOLO Evaluation)", fontsize=12)

    # Annotate cells
    for i in range(len(augmenters)):
        for j in range(len(test_networks)):
            val = data[i, j]
            text = f"{val:.3f}" if not (val != val) else "n/a"  # nan check
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black" if 0.3 <= val <= 0.7 else "white" if val < 0.3 else "black")

    plt.colorbar(im, ax=ax, label="AUPRC")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[info] Heatmap saved: {output_path}")


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def _write_markdown_table(results: dict[str, Any], output_path: Path) -> None:
    """Write a markdown table: augmenter × test_network AUPRC + weighted avg."""
    test_networks = TEST_NETWORK_IDS
    augmenters = AUGMENTER_LABELS

    header = "| Augmenter | " + " | ".join(test_networks) + " | Weighted AUPRC |"
    separator = "|-----------|" + "|".join(["----------"] * len(test_networks)) + "|----------------|"

    lines = [
        "# Cross-Network LOLO Evaluation Results",
        "",
        header,
        separator,
    ]
    for aug in augmenters:
        aug_data = results.get("per_augmenter", {}).get(aug, {})
        per_net = aug_data.get("per_network", {})
        weighted = aug_data.get("weighted_auprc", "n/a")
        weighted_str = f"{weighted:.3f}" if isinstance(weighted, (int, float)) else str(weighted)

        cells = []
        for nid in test_networks:
            auprc = per_net.get(nid, {}).get("auprc")
            cells.append(f"{auprc:.3f}" if isinstance(auprc, (int, float)) else "n/a")

        row = f"| {aug} | " + " | ".join(cells) + f" | {weighted_str} |"
        lines.append(row)

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Markdown table saved: {output_path}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_cross_network_eval(
    manifest_path: Path,
    output_dir: Path,
    *,
    sensor_coverage_levels: list[float],
    disturbance_types: list[str],
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full LOLO cross-network evaluation.

    Returns the complete results dict (also saved to output_dir).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[info] Building network index from manifest...")
    network_index = _build_network_index(manifest_path)

    available_train = [nid for nid in TRAIN_NETWORK_IDS if nid in network_index]
    available_test = [nid for nid in TEST_NETWORK_IDS if nid in network_index]

    print(f"[info] Train networks ({len(available_train)}): {available_train}")
    print(f"[info] Test networks  ({len(available_test)}):  {available_test}")

    # ---- Simulate training data ----
    print("[info] Simulating TRAIN data...")
    train_df = _simulate_network_group(
        available_train,
        network_index,
        split_name="train",
        sensor_coverage_levels=sensor_coverage_levels,
        disturbance_types=disturbance_types,
    )
    print(f"[info] Train dataset: {len(train_df)} rows")

    if train_df.empty:
        print("[error] Training simulation produced no data — aborting", file=sys.stderr)
        return {"error": "empty training data"}

    # ---- Simulate test data per network ----
    print("[info] Simulating TEST data per network...")
    test_frames: dict[str, pd.DataFrame] = {}
    for nid in available_test:
        print(f"[info]   Simulating test network '{nid}'...")
        df = _simulate_network_group(
            [nid],
            network_index,
            split_name="test",
            sensor_coverage_levels=sensor_coverage_levels,
            disturbance_types=disturbance_types,
        )
        test_frames[nid] = df
        print(f"[info]   '{nid}': {len(df)} rows")

    # ---- Evaluate each augmenter ----
    results: dict[str, Any] = {
        "protocol": "LOLO",
        "train_networks": available_train,
        "test_networks": available_test,
        "sensor_coverage_levels": sensor_coverage_levels,
        "disturbance_types": disturbance_types,
        "seed": seed,
        "train_rows": int(len(train_df)),
        "per_augmenter": {},
    }
    triage_records_all: list[pd.DataFrame] = []

    train_split = train_df[train_df["split"] == "train"].copy()
    if train_split.empty:
        train_split = train_df.copy()
        train_split["split"] = "train"

    for aug_name in AUGMENTER_LABELS:
        print(f"[info] Evaluating augmenter: {aug_name!r}")

        try:
            augmented_train = _apply_augmenter(train_split, aug_name, random_state=seed)
        except Exception as exc:
            print(f"[warn] Augmenter '{aug_name}' failed: {exc} — using baseline", file=sys.stderr)
            augmented_train = train_split.copy()

        try:
            detector = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
            detector.fit_from_frame(augmented_train, feature_columns=FEATURE_COLUMNS)
        except Exception as exc:
            print(f"[warn] Detector training failed for '{aug_name}': {exc}", file=sys.stderr)
            results["per_augmenter"][aug_name] = {"error": str(exc)}
            continue

        # Fit plausibility scorer + triage reranker
        scorer = None
        triage_model = None
        try:
            normal_train = augmented_train.loc[augmented_train["event_label"] == 0].copy()
            scorer = HydraulicResidualScorer().fit(normal_train if not normal_train.empty else augmented_train)
            available_train_features = [c for c in FEATURE_COLUMNS if c in augmented_train.columns]
            x_train, _, _ = prepare_feature_matrix(
                augmented_train,
                target_column="event_label",
                feature_columns=available_train_features,
            )
            train_base_prob = detector.predict_proba(x_train)[:, 1]
            triage_train = build_triage_frame(augmented_train, base_scores=train_base_prob, scorer=scorer)
            triage_model = PhysicallyInformedTriageModel(random_state=seed).fit(
                triage_train.frame,
                target_column="event_label",
                feature_columns=triage_train.feature_columns,
            )
        except Exception as exc:
            print(f"[warn] Triage fit failed for '{aug_name}': {exc} — triage disabled", file=sys.stderr)

        per_network: dict[str, dict[str, Any]] = {}
        triage_per_network: dict[str, dict[str, Any]] = {}
        heuristic_per_network: dict[str, dict[str, Any]] = {}

        for nid, test_df in test_frames.items():
            network_result = _evaluate_on_network(detector, test_df, nid)
            per_network[nid] = network_result
            auprc = network_result.get("auprc")
            auprc_str = f"{auprc:.3f}" if isinstance(auprc, (int, float)) else str(auprc)
            print(f"[info]   base '{nid}' AUPRC={auprc_str}")

            if triage_model is not None and scorer is not None:
                triage_result, triage_records = _evaluate_triage_on_network(detector, triage_model, scorer, test_df, nid)
                triage_per_network[nid] = triage_result
                triage_auprc = triage_result.get("auprc")
                triage_auprc_str = f"{triage_auprc:.3f}" if isinstance(triage_auprc, (int, float)) else str(triage_auprc)
                print(f"[info]   triage '{nid}' AUPRC={triage_auprc_str}")

                if not triage_records.empty:
                    triage_records["augmenter"] = aug_name
                    triage_records["seed"] = seed
                    triage_records_all.append(triage_records)

                    h_score = triage_records["heuristic_triage_score"].to_numpy(dtype=float)
                    h_pred = (h_score >= 0.5).astype(int)
                    h_report = binary_report(triage_records["y_true"].to_numpy(dtype=int), h_pred, h_score)
                    h_result = report_as_dict(h_report)
                    h_result["network_id"] = nid
                    h_result["rows"] = int(len(triage_records))
                    h_result["positive_rows"] = int(triage_records["y_true"].sum())
                    heuristic_per_network[nid] = h_result

        weighted = size_weighted_auprc(per_network)
        triage_weighted = size_weighted_auprc(triage_per_network) if triage_per_network else None
        heuristic_weighted = size_weighted_auprc(heuristic_per_network) if heuristic_per_network else None
        results["per_augmenter"][aug_name] = {
            "augmenter": aug_name,
            "augmented_train_rows": int(len(augmented_train)),
            "per_network": per_network,
            "weighted_auprc": weighted,
            "triage_per_network": triage_per_network,
            "triage_weighted_auprc": triage_weighted,
            "heuristic_per_network": heuristic_per_network,
            "heuristic_weighted_auprc": heuristic_weighted,
        }
        print(f"[info]   Weighted AUPRC={weighted:.3f}")
        if triage_weighted is not None:
            print(f"[info]   Triage weighted AUPRC={triage_weighted:.3f}")
        if heuristic_weighted is not None:
            print(f"[info]   Heuristic triage weighted AUPRC={heuristic_weighted:.3f}")

    # ----- GNN detector baseline (no augmentation) -----
    print("[info] Evaluating GNN detector baseline (no augmentation)")
    try:
        gnn_detector = GNNDetector(random_state=seed)
        gnn_detector.fit_from_frame(train_split, feature_columns=FEATURE_COLUMNS)
        gnn_per_network: dict[str, dict[str, Any]] = {}
        for nid, test_df in test_frames.items():
            gnn_result = _evaluate_on_network(gnn_detector, test_df, nid)
            gnn_per_network[nid] = gnn_result
            auprc = gnn_result.get("auprc")
            auprc_str = f"{auprc:.3f}" if isinstance(auprc, (int, float)) else str(auprc)
            print(f"[info]   GNN '{nid}' AUPRC={auprc_str}")
        gnn_weighted = size_weighted_auprc(gnn_per_network)
        results["per_augmenter"]["gnn_baseline"] = {
            "augmenter": "gnn_baseline",
            "detector": "GNNDetector",
            "augmented_train_rows": int(len(train_split)),
            "per_network": gnn_per_network,
            "weighted_auprc": gnn_weighted,
        }
        print(f"[info]   GNN Weighted AUPRC={gnn_weighted:.3f}")
    except Exception as exc:
        print(f"[warn] GNN detector failed: {exc}", file=sys.stderr)
        results["per_augmenter"]["gnn_baseline"] = {"error": str(exc)}

    json_path = output_dir / "cross_network_results.json"
    _write_json(results, json_path)
    print(f"[info] Results saved: {json_path}")

    if triage_records_all:
        triage_records_df = pd.concat(triage_records_all, axis=0, ignore_index=True)
        triage_records_path = output_dir / "cross_network_triage_records.csv"
        triage_records_df.to_csv(triage_records_path, index=False)
        print(f"[info] Triage records saved: {triage_records_path}")

    _write_markdown_table(results, output_dir / "cross_network_table.md")

    try:
        plot_cross_network_heatmap(results, output_dir / "cross_network_heatmap.png")
    except Exception as exc:
        print(f"[warn] Heatmap generation failed: {exc}", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-network LOLO transfer evaluation for WDN anomaly detection."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml",
        help="Path to benchmark_manifest.yaml (default: dev/active/configs/benchmark_manifest.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ACTIVE_ROOT / "artifacts" / "cross_network",
        help="Directory for output artifacts (default: dev/active/artifacts/cross_network)",
    )
    parser.add_argument(
        "--sensor-coverage",
        type=float,
        nargs="+",
        default=[0.25, 0.50],
        metavar="LEVEL",
        help="Sensor coverage levels to simulate (default: 0.25 0.50)",
    )
    parser.add_argument(
        "--disturbance-types",
        type=str,
        nargs="+",
        default=["leak", "pipe_closure", "pump_outage"],
        metavar="TYPE",
        help="Disturbance types to simulate (default: leak pipe_closure pump_outage)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    # Force stdout/stderr to UTF-8 on Windows to handle WNTR Unicode warnings
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    args = _parse_args()

    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    if not manifest_path.exists():
        print(f"[error] Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    print(f"[info] Manifest:          {manifest_path}")
    print(f"[info] Output directory:  {output_dir}")
    print(f"[info] Sensor coverage:   {args.sensor_coverage}")
    print(f"[info] Disturbance types: {args.disturbance_types}")
    print(f"[info] Seed:              {args.seed}")

    results = run_cross_network_eval(
        manifest_path,
        output_dir,
        sensor_coverage_levels=args.sensor_coverage,
        disturbance_types=args.disturbance_types,
        seed=args.seed,
    )

    if "error" in results:
        print(f"[error] Evaluation failed: {results['error']}", file=sys.stderr)
        return 1

    print("[info] Cross-network evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
