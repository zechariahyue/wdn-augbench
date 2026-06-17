"""Leak-only LOLO evaluation for WDN anomaly detection.

Evaluates four systems on leak-only scenarios using the March 31 revised
held-out-network protocol:
  Train: anytown, hanoi
  Test:  net3, d_town, l_town

Systems evaluated:
  1. baseline    - RandomForest, no augmentation
  2. noise+RF    - PositiveClassAugmenter (Gaussian noise) + RF
  3. smote+RF    - SMOTEAugmenter + RF
  4. graph_cvae  - GraphCVAEAugmenter + RF

Reports AUPRC per coverage level and overall.
Saves results to artifacts/leak_only_lolo/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap — make watergen importable regardless of cwd
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Import only the simulation/index helpers from run_cross_network_eval.
# _evaluate_on_network and _apply_augmenter are NOT imported because they
# close over that module's 16-column FEATURE_COLUMNS; we define local
# versions below that use the 19-column list required by this experiment.
from run_cross_network_eval import (  # noqa: E402
    _build_network_index,
    _simulate_network_group,
)
from watergen.data import build_leak_scenarios, simulate_many  # noqa: E402  (unused here but keep path consistent)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (  # noqa: E402
    GaussianMixtureAugmenter,
    GraphCVAEAugmenter,
    PositiveClassAugmenter,
    QuantilePlausibilityFilter,
    TabularDetectorBaseline,
    prepare_feature_matrix,
)
try:
    from watergen.models.augmentation import SMOTEAugmenter as _SMOTEAugmenter
    _HAS_SMOTE = True
except Exception:
    _HAS_SMOTE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_NETWORK_IDS = ["anytown", "hanoi"]
TEST_NETWORK_IDS = ["net3", "d_town", "l_town"]

NETWORK_NODE_COUNTS: dict[str, int] = {
    "d_town": 399,
    "l_town": 785,
}

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
    "pressure_trend",
    "pressure_trend_cumsum",
    "demand_trend",
]

SENSOR_COVERAGE_LEVELS = [0.25, 0.5, 0.75, 1.0]

SYSTEMS = ["baseline", "noise+RF", "smote+RF", "graph_cvae+RF"]

OUTPUT_DIR = ACTIVE_ROOT / "artifacts" / "leak_only_lolo"
MANIFEST_PATH = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _size_weighted_auprc(per_network: dict[str, dict[str, Any]]) -> float:
    """Compute size-weighted AUPRC across test networks."""
    total_weight = 0.0
    weighted_sum = 0.0
    for nid, result in per_network.items():
        auprc = result.get("auprc")
        if auprc is None or not isinstance(auprc, (int, float)):
            continue
        weight = float(NETWORK_NODE_COUNTS.get(nid, 1))
        weighted_sum += float(auprc) * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return weighted_sum / total_weight


def _filter_by_coverage(df: pd.DataFrame, coverage: float) -> pd.DataFrame:
    """Filter dataframe to rows matching a specific coverage level."""
    return df[df["sensor_coverage"].round(2) == round(coverage, 2)].copy()


def _apply_augmenter(train_df: pd.DataFrame, mode: str, *, random_state: int = 42) -> pd.DataFrame:
    """Local augmenter wrapper using this script's 19-column feature list."""
    if mode == "baseline":
        return train_df.copy()

    plausibility_filter = QuantilePlausibilityFilter().fit(train_df, FEATURE_COLUMNS)

    if mode == "noise":
        aug = PositiveClassAugmenter(random_state=random_state, noise_scale=0.05)
        synth = aug.generate(
            train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column="event_label",
            multiplier=1.0,
        )
    elif mode == "smote":
        if _HAS_SMOTE:
            aug = _SMOTEAugmenter(random_state=random_state)
            synth = aug.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
            )
        else:
            aug = PositiveClassAugmenter(random_state=random_state, noise_scale=0.05)
            synth = aug.generate(
                train_df,
                feature_columns=FEATURE_COLUMNS,
                target_column="event_label",
                multiplier=1.0,
            )
    elif mode == "gmm":
        aug = GaussianMixtureAugmenter(random_state=random_state, n_components=2)
        synth = aug.generate(
            train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column="event_label",
        )
    else:
        return train_df.copy()

    if synth.empty:
        return train_df.copy()

    filtered = plausibility_filter.filter(synth, FEATURE_COLUMNS)
    if filtered.empty:
        filtered = synth.copy()

    return pd.concat([train_df.assign(is_synthetic=0), filtered], ignore_index=True)


def _evaluate_on_network(detector: TabularDetectorBaseline, test_df: pd.DataFrame, network_id: str) -> dict[str, Any]:
    """Evaluate detector on one network using this script's feature list."""
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


# ---------------------------------------------------------------------------
# Augmented training data builders
# ---------------------------------------------------------------------------

def _build_augmented_train_baseline(train_df: pd.DataFrame) -> pd.DataFrame:
    return _apply_augmenter(train_df, "baseline")


def _build_augmented_train_noise(train_df: pd.DataFrame) -> pd.DataFrame:
    return _apply_augmenter(train_df, "noise")


def _build_augmented_train_smote(train_df: pd.DataFrame) -> pd.DataFrame:
    return _apply_augmenter(train_df, "smote")


def _build_augmented_train_graph_cvae(train_df: pd.DataFrame) -> pd.DataFrame:
    """Fit GraphCVAEAugmenter and return augmented training frame."""
    aug = GraphCVAEAugmenter(
        random_state=42,
        epochs=30,
        lambda_physics=0.5,
        warmup_epochs=10,
    )
    aug.fit(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    synth = aug.generate(
        train_df,
        feature_columns=FEATURE_COLUMNS,
        target_column="event_label",
    )

    if synth.empty:
        print("[warn] GraphCVAE generated no synthetic rows — using unaugmented train", file=sys.stderr)
        return train_df.copy()

    plaus = QuantilePlausibilityFilter().fit(train_df, FEATURE_COLUMNS)
    filtered = plaus.filter(synth, FEATURE_COLUMNS)

    if filtered.empty:
        print("[warn] All GraphCVAE synthetic rows filtered by plausibility — using unfiltered", file=sys.stderr)
        filtered = synth.copy()

    aug_train = pd.concat(
        [train_df.assign(is_synthetic=0), filtered],
        ignore_index=True,
    )
    return aug_train


_AUGMENT_FN = {
    "baseline": _build_augmented_train_baseline,
    "noise+RF": _build_augmented_train_noise,
    "smote+RF": _build_augmented_train_smote,
    "graph_cvae+RF": _build_augmented_train_graph_cvae,
}


# ---------------------------------------------------------------------------
# Single-system, single-coverage evaluation
# ---------------------------------------------------------------------------

def _evaluate_system_at_coverage(
    system: str,
    train_cov: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit detector on augmented train_cov; evaluate on each test network.

    Returns a dict keyed by network_id -> result dict, plus 'weighted' key.
    """
    if train_cov.empty:
        print(f"[warn] Empty training data for system='{system}' — skipping", file=sys.stderr)
        return {nid: {"auprc": None, "error": "empty train"} for nid in TEST_NETWORK_IDS}

    # Build augmented training frame
    try:
        aug_fn = _AUGMENT_FN[system]
        aug_train = aug_fn(train_cov)
    except Exception as exc:
        print(f"[warn] Augmentation failed for '{system}': {exc} — using baseline", file=sys.stderr)
        aug_train = train_cov.copy()

    # Fit detector
    try:
        det = TabularDetectorBaseline(algorithm="random_forest", random_state=42)
        det.fit_from_frame(aug_train, feature_columns=FEATURE_COLUMNS)
    except Exception as exc:
        print(f"[warn] Detector fit failed for '{system}': {exc}", file=sys.stderr)
        return {nid: {"auprc": None, "error": str(exc)} for nid in TEST_NETWORK_IDS}

    # Evaluate per test network
    per_network: dict[str, Any] = {}
    for nid in TEST_NETWORK_IDS:
        ndf = test_df[test_df["benchmark_id"] == nid].copy()
        result = _evaluate_on_network(det, ndf, nid)
        per_network[nid] = result
        auprc = result.get("auprc")
        auprc_str = f"{auprc:.3f}" if isinstance(auprc, (int, float)) else str(auprc)
        print(f"[info]     {nid}: AUPRC={auprc_str}")

    per_network["weighted"] = _size_weighted_auprc(per_network)
    return per_network


# ---------------------------------------------------------------------------
# Markdown table writer
# ---------------------------------------------------------------------------

def _write_markdown_table(results: dict[str, Any], path: Path) -> None:
    """Write a markdown table: system × coverage × network AUPRC."""
    lines = [
        "# Leak-Only LOLO Evaluation Results",
        "",
        "## Per-Coverage AUPRC",
        "",
        "| System | Coverage | d_town | l_town | Weighted |",
        "|--------|----------|--------|--------|----------|",
    ]

    for system in SYSTEMS:
        sys_data = results["systems"].get(system, {})
        per_cov = sys_data.get("per_coverage", {})
        for cov in SENSOR_COVERAGE_LEVELS:
            cov_key = str(round(cov, 2))
            cov_data = per_cov.get(cov_key, {})
            d_town_auprc = cov_data.get("d_town", {}).get("auprc")
            l_town_auprc = cov_data.get("l_town", {}).get("auprc")
            weighted = cov_data.get("weighted")
            d_str = f"{d_town_auprc:.3f}" if isinstance(d_town_auprc, (int, float)) else "n/a"
            l_str = f"{l_town_auprc:.3f}" if isinstance(l_town_auprc, (int, float)) else "n/a"
            w_str = f"{weighted:.3f}" if isinstance(weighted, (int, float)) else "n/a"
            lines.append(f"| {system} | {cov:.2f} | {d_str} | {l_str} | {w_str} |")

    lines += [
        "",
        "## Overall AUPRC (all coverages combined)",
        "",
        "| System | d_town | l_town | Weighted |",
        "|--------|--------|--------|----------|",
    ]
    for system in SYSTEMS:
        overall = results["systems"].get(system, {}).get("overall", {})
        d_town_auprc = overall.get("d_town", {}).get("auprc")
        l_town_auprc = overall.get("l_town", {}).get("auprc")
        weighted = overall.get("weighted")
        d_str = f"{d_town_auprc:.3f}" if isinstance(d_town_auprc, (int, float)) else "n/a"
        l_str = f"{l_town_auprc:.3f}" if isinstance(l_town_auprc, (int, float)) else "n/a"
        w_str = f"{weighted:.3f}" if isinstance(weighted, (int, float)) else "n/a"
        lines.append(f"| {system} | {d_str} | {l_str} | {w_str} |")

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Markdown table saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Force UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    warnings.filterwarnings("ignore")

    print(f"[info] Manifest:          {MANIFEST_PATH}")
    print(f"[info] Output directory:  {OUTPUT_DIR}")
    print(f"[info] Train networks:    {TRAIN_NETWORK_IDS}")
    print(f"[info] Test networks:     {TEST_NETWORK_IDS}")
    print(f"[info] Coverage levels:   {SENSOR_COVERAGE_LEVELS}")
    print(f"[info] Disturbance types: ['leak']")

    # ---- Build network index ----
    print("\n[info] Building network index from manifest...")
    network_index = _build_network_index(MANIFEST_PATH)

    available_train = [nid for nid in TRAIN_NETWORK_IDS if nid in network_index]
    available_test = [nid for nid in TEST_NETWORK_IDS if nid in network_index]
    print(f"[info] Available train: {available_train}")
    print(f"[info] Available test:  {available_test}")

    # ---- Simulate ----
    print("\n[info] Simulating TRAIN data (leak-only)...")
    train_df = _simulate_network_group(
        available_train,
        network_index,
        split_name="train",
        sensor_coverage_levels=SENSOR_COVERAGE_LEVELS,
        disturbance_types=["leak"],
    )
    print(f"[info] Train dataset: {len(train_df)} rows, "
          f"{int(train_df['event_label'].sum()) if not train_df.empty else 0} positives")

    if train_df.empty:
        print("[error] Training simulation produced no data — aborting", file=sys.stderr)
        return 1

    print("\n[info] Simulating TEST data (leak-only)...")
    test_df = _simulate_network_group(
        available_test,
        network_index,
        split_name="test",
        sensor_coverage_levels=SENSOR_COVERAGE_LEVELS,
        disturbance_types=["leak"],
    )
    print(f"[info] Test dataset:  {len(test_df)} rows, "
          f"{int(test_df['event_label'].sum()) if not test_df.empty else 0} positives")

    if test_df.empty:
        print("[error] Test simulation produced no data — aborting", file=sys.stderr)
        return 1

    # ---- Evaluate each system ----
    results: dict[str, Any] = {
        "protocol": "LOLO_leak_only",
        "train_networks": available_train,
        "test_networks": available_test,
        "sensor_coverage_levels": SENSOR_COVERAGE_LEVELS,
        "disturbance_types": ["leak"],
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "systems": {},
    }

    for system in SYSTEMS:
        print(f"\n[info] === System: {system} ===")
        system_result: dict[str, Any] = {"per_coverage": {}, "overall": {}}

        # -- Per-coverage evaluation --
        for coverage in SENSOR_COVERAGE_LEVELS:
            cov_key = str(round(coverage, 2))
            print(f"[info]   Coverage {coverage:.2f}:")

            train_cov = _filter_by_coverage(train_df, coverage)
            test_cov = _filter_by_coverage(test_df, coverage)

            print(f"[info]     train rows={len(train_cov)}, test rows={len(test_cov)}")

            per_net = _evaluate_system_at_coverage(system, train_cov, test_cov)

            # Store per-network AUPRC (not the full result dicts — keep it clean)
            cov_entry: dict[str, Any] = {}
            for nid in TEST_NETWORK_IDS:
                net_res = per_net.get(nid, {})
                cov_entry[nid] = {"auprc": net_res.get("auprc")}
            cov_entry["weighted"] = per_net.get("weighted")
            system_result["per_coverage"][cov_key] = cov_entry

            w = cov_entry["weighted"]
            w_str = f"{w:.3f}" if isinstance(w, (int, float)) else str(w)
            print(f"[info]     weighted AUPRC={w_str}")

        # -- Overall evaluation (all coverages combined) --
        print(f"[info]   Overall (all coverages):")
        overall_per_net = _evaluate_system_at_coverage(system, train_df, test_df)
        overall_entry: dict[str, Any] = {}
        for nid in TEST_NETWORK_IDS:
            net_res = overall_per_net.get(nid, {})
            overall_entry[nid] = {"auprc": net_res.get("auprc")}
        overall_entry["weighted"] = overall_per_net.get("weighted")
        system_result["overall"] = overall_entry

        w = overall_entry["weighted"]
        w_str = f"{w:.3f}" if isinstance(w, (int, float)) else str(w)
        print(f"[info]   Overall weighted AUPRC={w_str}")

        results["systems"][system] = system_result

    # ---- Save outputs ----
    json_path = OUTPUT_DIR / "leak_only_results.json"
    _write_json(results, json_path)
    print(f"\n[info] Results saved: {json_path}")

    _write_markdown_table(results, OUTPUT_DIR / "leak_only_table.md")

    # ---- Print summary table ----
    print("\n=== SUMMARY: Overall Weighted AUPRC (leak-only) ===")
    print(f"{'System':<20} {'d_town':>8} {'l_town':>8} {'Weighted':>10}")
    print("-" * 50)
    for system in SYSTEMS:
        overall = results["systems"].get(system, {}).get("overall", {})
        d = overall.get("d_town", {}).get("auprc")
        l = overall.get("l_town", {}).get("auprc")
        w = overall.get("weighted")
        d_str = f"{d:.3f}" if isinstance(d, (int, float)) else "n/a"
        l_str = f"{l:.3f}" if isinstance(l, (int, float)) else "n/a"
        w_str = f"{w:.3f}" if isinstance(w, (int, float)) else "n/a"
        print(f"{system:<20} {d_str:>8} {l_str:>8} {w_str:>10}")

    print("\n[info] Leak-only LOLO evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
