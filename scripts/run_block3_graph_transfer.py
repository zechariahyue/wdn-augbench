"""Block 3 runner: Graph-CVAE in LOLO transfer.

Extends the existing LOLO cross-network evaluation to include Graph-CVAE.
Outputs a dedicated comparison table for baseline / SMOTE / GMM / Graph-CVAE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_cross_network_eval import (  # type: ignore  # noqa: E402
    _apply_augmenter,
    _build_network_index,
    _evaluate_on_network,
    _simulate_network_group,
    TEST_NETWORK_IDS,
    TRAIN_NETWORK_IDS,
    FEATURE_COLUMNS,
    NETWORK_NODE_COUNTS,
)
from watergen.models import GraphCVAEAugmenter, QuantilePlausibilityFilter, TabularDetectorBaseline  # noqa: E402

import pandas as pd
import yaml

MANIFEST = PROJECT_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "block3_graph_transfer"
OUTPUT_JSON = OUTPUT_DIR / "block3_results.json"
OUTPUT_MD = OUTPUT_DIR / "block3_results.md"


def size_weighted_auprc(network_results: dict[str, dict[str, Any]]) -> float | None:
    weights = []
    values = []
    for nid, res in network_results.items():
        auprc = res.get("auprc")
        if isinstance(auprc, (int, float)):
            w = NETWORK_NODE_COUNTS.get(nid, 1)
            weights.append(w)
            values.append(float(auprc))
    if not values:
        return None
    total = sum(weights)
    return sum(v * w for v, w in zip(values, weights)) / total


def apply_graph_cvae(train_df: pd.DataFrame) -> pd.DataFrame:
    aug = GraphCVAEAugmenter(random_state=42, epochs=30, lambda_physics=0.5, warmup_epochs=10)
    aug.fit(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    synth = aug.generate(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    plaus = QuantilePlausibilityFilter().fit(train_df, FEATURE_COLUMNS)
    synth = plaus.filter(synth, FEATURE_COLUMNS)
    if synth.empty:
        return train_df.copy()
    return pd.concat([train_df.assign(is_synthetic=0), synth], ignore_index=True)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    network_index = _build_network_index(MANIFEST)

    train_df = _simulate_network_group(
        TRAIN_NETWORK_IDS,
        network_index,
        split_name="train",
        sensor_coverage_levels=[0.25, 0.5],
        disturbance_types=["leak", "pipe_closure", "pump_outage"],
    )
    test_df = _simulate_network_group(
        TEST_NETWORK_IDS,
        network_index,
        split_name="test",
        sensor_coverage_levels=[0.25, 0.5],
        disturbance_types=["leak", "pipe_closure", "pump_outage"],
    )

    results: dict[str, Any] = {"per_augmenter": {}}

    systems = ["baseline", "noise", "gmm", "smote", "graph_cvae"]
    for system in systems:
        print(f"[info] Running Block 3 system: {system}")
        if system == "graph_cvae":
            aug_train = apply_graph_cvae(train_df)
        else:
            aug_train = _apply_augmenter(train_df, system, random_state=42)

        det = TabularDetectorBaseline(algorithm="random_forest", random_state=42)
        det.fit_from_frame(aug_train, feature_columns=FEATURE_COLUMNS)

        per_network: dict[str, dict[str, Any]] = {}
        for nid in TEST_NETWORK_IDS:
            split_df = test_df[test_df["benchmark_id"] == nid].copy()
            if split_df.empty:
                per_network[nid] = {"auprc": None, "rows": 0}
                continue
            per_network[nid] = _evaluate_on_network(det, split_df, nid)

        results["per_augmenter"][system] = {
            "per_network": per_network,
            "weighted_auprc": size_weighted_auprc(per_network),
        }

    OUTPUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Block 3: Graph-CVAE in LOLO Transfer", "", "| Augmenter | d_town | l_town | Weighted AUPRC |", "|---|---:|---:|---:|"]
    for system, payload in results["per_augmenter"].items():
        d = payload["per_network"].get("d_town", {}).get("auprc")
        l = payload["per_network"].get("l_town", {}).get("auprc")
        w = payload.get("weighted_auprc")
        lines.append(f"| {system} | {d} | {l} | {w} |")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Block 3 JSON -> {OUTPUT_JSON}")
    print(f"[info] Block 3 Markdown -> {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
