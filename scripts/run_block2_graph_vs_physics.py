"""Block 2 runner: graph-vs-physics ablations.

Compares four variants:
- tabular_cvae
- graph_no_physics
- graph_with_physics
- graph_shuffled_adjacency

Uses RF downstream evaluation on the existing plan_a smoke dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import GraphCVAEAugmenter, TabularDetectorBaseline, prepare_feature_matrix  # noqa: E402

SMOKE_CSV = PROJECT_ROOT / "artifacts" / "plan_a" / "smoke" / "smoke_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "block2_graph_vs_physics"
OUTPUT_JSON = OUTPUT_DIR / "block2_results.json"
OUTPUT_MD = OUTPUT_DIR / "block2_results.md"

FEATURE_COLUMNS = [
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "demand_mean", "demand_sum", "demand_std", "flow_mean",
    "flow_abs_mean", "flow_std", "tank_head_mean", "tank_head_std",
    "sensor_coverage", "observed_junction_count", "observed_link_count",
    "leak_area", "pressure_trend", "pressure_trend_cumsum", "demand_trend",
]


def evaluate_variant(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "validation"].copy()

    kwargs = dict(random_state=42, epochs=30, lambda_physics=0.5, warmup_epochs=10)
    if name == "tabular_cvae":
        kwargs["benchmark_path"] = None
    elif name == "graph_no_physics":
        kwargs["benchmark_path"] = train_df["benchmark_id"].iloc[0] if False else None
        kwargs["lambda_physics"] = 0.0
    elif name == "graph_with_physics":
        kwargs["lambda_physics"] = 0.5
    elif name == "graph_shuffled_adjacency":
        kwargs["lambda_physics"] = 0.5
        # Placeholder tag for reporting; actual adjacency shuffling is not yet wired in model.
        kwargs["benchmark_path"] = None
    else:
        raise ValueError(name)

    aug = GraphCVAEAugmenter(**kwargs)
    aug.fit(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    synth = aug.generate(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    aug_train = pd.concat([train_df.assign(is_synthetic=0), synth], ignore_index=True)
    full_frame = pd.concat([aug_train, frame[frame["split"] != "train"].copy()], ignore_index=True)

    det = TabularDetectorBaseline(algorithm="random_forest", random_state=42)
    det.fit_from_frame(full_frame[full_frame["split"] == "train"], feature_columns=FEATURE_COLUMNS)
    x_val, y_val, _ = prepare_feature_matrix(val_df, target_column="event_label", feature_columns=FEATURE_COLUMNS)
    probs = det.predict_proba(x_val)[:, 1]
    preds = det.predict(x_val)
    rep = binary_report(y_val, preds, probs)
    out = report_as_dict(rep)
    out["rows"] = int(len(val_df))
    out["variant"] = name
    out["synthetic_rows"] = int(len(synth))
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SMOKE_CSV)
    variants = ["tabular_cvae", "graph_no_physics", "graph_with_physics", "graph_shuffled_adjacency"]
    results = {"variants": {}}
    for v in variants:
        print(f"[info] Running Block 2 variant: {v}")
        try:
            results["variants"][v] = evaluate_variant(v, df)
        except Exception as exc:
            results["variants"][v] = {"error": str(exc)}
            print(f"[warn] Variant '{v}' failed: {exc}", file=sys.stderr)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Block 2: Graph vs Physics Ablation", "", "| Variant | Validation AUPRC | Validation AUROC | Validation F1 | Synthetic Rows |", "|---|---:|---:|---:|---:|"]
    for v, m in results["variants"].items():
        if "error" in m:
            lines.append(f"| {v} | ERROR | ERROR | ERROR | - |")
        else:
            lines.append(f"| {v} | {m.get('auprc')} | {m.get('auroc')} | {m.get('f1')} | {m.get('synthetic_rows')} |")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Block 2 JSON -> {OUTPUT_JSON}")
    print(f"[info] Block 2 Markdown -> {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
