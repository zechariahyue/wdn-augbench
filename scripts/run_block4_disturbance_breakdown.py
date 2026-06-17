"""Block 4 runner: per-disturbance-type breakdown.

Computes AUPRC by disturbance type for a compact set of systems:
- RF baseline
- LSTM-AE
- Graph-CVAE + RF
- WNTR residual
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
from watergen.models import GraphCVAEAugmenter, LSTMAEDetector, TabularDetectorBaseline, WNTRResidualDetector, prepare_feature_matrix  # noqa: E402

SMOKE_CSV = PROJECT_ROOT / "artifacts" / "plan_a" / "smoke" / "smoke_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "block4_disturbance_breakdown"
OUTPUT_JSON = OUTPUT_DIR / "block4_results.json"
OUTPUT_MD = OUTPUT_DIR / "block4_results.md"
FEATURE_COLUMNS = [
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "demand_mean", "demand_sum", "demand_std", "flow_mean",
    "flow_abs_mean", "flow_std", "tank_head_mean", "tank_head_std",
    "sensor_coverage", "observed_junction_count", "observed_link_count",
    "leak_area", "pressure_trend", "pressure_trend_cumsum", "demand_trend",
]


def eval_tabular(detector, df: pd.DataFrame) -> dict[str, Any]:
    x, y, _ = prepare_feature_matrix(df, target_column="event_label", feature_columns=FEATURE_COLUMNS)
    probs = detector.predict_proba(x)[:, 1]
    preds = detector.predict(x)
    rep = binary_report(y, preds, probs)
    out = report_as_dict(rep)
    out["rows"] = int(len(df))
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SMOKE_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    # Fit systems
    rf = TabularDetectorBaseline(algorithm="random_forest", random_state=42).fit_from_frame(train_df, feature_columns=FEATURE_COLUMNS)
    wntr = WNTRResidualDetector(z_score_threshold=3.0).fit(train_df)

    gcvae = GraphCVAEAugmenter(random_state=42, epochs=30, lambda_physics=0.5, warmup_epochs=10)
    gcvae.fit(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    gcvae_synth = gcvae.generate(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
    gcvae_train = pd.concat([train_df.assign(is_synthetic=0), gcvae_synth], ignore_index=True)
    gcvae_rf = TabularDetectorBaseline(algorithm="random_forest", random_state=42).fit_from_frame(gcvae_train, feature_columns=FEATURE_COLUMNS)

    # LSTM-AE using existing helper pattern
    lstm = LSTMAEDetector(hidden_size=64, num_layers=2, sequence_length=24, epochs=20, lr=1e-3, batch_size=16, random_state=42, device="cpu")
    lstm.fit_from_frame(train_df, scenario_id_col="scenario_id", target_col="event_label", feature_columns=FEATURE_COLUMNS)

    def predict_lstm(df_sub: pd.DataFrame) -> dict[str, Any]:
        import numpy as np
        y_true = []
        y_score = []
        for _, grp in df_sub.groupby("scenario_id"):
            grp = grp.sort_values("time_seconds")
            arr = grp[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            seq = lstm._pad_or_truncate(arr)[None, ...]
            proba = lstm.predict_proba(seq)
            score = float(proba[0, 1])
            label = int(grp["event_label"].max())
            y_true.extend([label] * len(grp))
            y_score.extend([score] * len(grp))
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]
        rep = binary_report(y_true, y_pred, y_score)
        out = report_as_dict(rep)
        out["rows"] = len(y_true)
        return out

    systems = {
        "rf_baseline": lambda d: eval_tabular(rf, d),
        "graph_cvae_rf": lambda d: eval_tabular(gcvae_rf, d),
        "wntr_residual": lambda d: {
            **report_as_dict(binary_report(d["event_label"].to_numpy(dtype=int), wntr.predict(d), wntr.predict_proba(d)[:,1])),
            "rows": int(len(d)),
        },
        "lstm_ae": predict_lstm,
    }

    disturbance_order = [
        "baseline",
        "leak",
        "pipe_closure",
        "pump_outage",
        "demand_surge",
        "partial_valve_closure",
    ]

    results: dict[str, Any] = {"systems": {}}
    for sys_name, fn in systems.items():
        results["systems"][sys_name] = {}
        for kind in disturbance_order:
            sub = val_df[val_df["scenario_kind"] == kind].copy()
            if sub.empty:
                continue
            results["systems"][sys_name][kind] = fn(sub)

    OUTPUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Block 4: Per-Disturbance-Type Breakdown", ""]
    for sys_name, per_kind in results["systems"].items():
        lines.append(f"## {sys_name}")
        lines.append("| Disturbance | AUPRC | F1 | Rows |")
        lines.append("|---|---:|---:|---:|")
        for kind, m in per_kind.items():
            lines.append(f"| {kind} | {m.get('auprc')} | {m.get('f1')} | {m.get('rows')} |")
        lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Block 4 JSON -> {OUTPUT_JSON}")
    print(f"[info] Block 4 Markdown -> {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
