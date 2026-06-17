"""Block 1 runner: sequence detector + augmentation interaction.

Evaluates whether augmentation still helps when the detector is a strong
sequence model (LSTM-AE) rather than a tabular classifier.

Runs the following systems on the existing plan_a smoke dataset:
- LSTM-AE (no augmentation)
- LSTM-AE + SMOTE
- LSTM-AE + GMM
- LSTM-AE + Graph-CVAE

Outputs:
- dev/active/artifacts/block1_sequence_aug/block1_results.json
- dev/active/artifacts/block1_sequence_aug/block1_results.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # dev/active
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (  # noqa: E402
    GaussianMixtureAugmenter,
    GraphCVAEAugmenter,
    LSTMAEDetector,
    QuantilePlausibilityFilter,
    SMOTEAugmenter,
)

SMOKE_CSV = PROJECT_ROOT / "artifacts" / "plan_a" / "smoke" / "smoke_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "block1_sequence_aug"
OUTPUT_JSON = OUTPUT_DIR / "block1_results.json"
OUTPUT_MD = OUTPUT_DIR / "block1_results.md"

META_COLS = {
    "event_label",
    "scenario_id",
    "time_seconds",
    "benchmark_id",
    "scenario_kind",
    "split",
    "stage",
    "leak_node",
    "disturbance_target",
    "leak_area",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def predict_proba_from_frame(
    detector: LSTMAEDetector,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[list[int], list[int], list[float]]:
    """Return row-aligned y_true, y_pred, y_score arrays.

    LSTM-AE is a sequence-level detector (one score per scenario_id). We
    broadcast each sequence prediction back to all rows in the scenario so
    that we can evaluate against row-level event labels consistently.
    """
    import numpy as np

    has_time = "time_seconds" in frame.columns
    n = len(frame)
    y_true = np.zeros(n, dtype=int)
    y_score = np.zeros(n, dtype=float)

    original_index = frame.index.tolist()
    idx_map = {idx: pos for pos, idx in enumerate(original_index)}

    for _, grp in frame.groupby("scenario_id"):
        if has_time:
            grp = grp.sort_values("time_seconds")
        arr = (
            grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        seq = detector._pad_or_truncate(arr)[None, ...]
        proba = detector.predict_proba(seq)
        score = float(proba[0, 1])
        grp_label = int(grp["event_label"].max())
        for row_idx in grp.index:
            pos = idx_map[row_idx]
            y_true[pos] = grp_label
            y_score[pos] = score

    y_pred = (y_score >= 0.5).astype(int)
    return y_true.tolist(), y_pred.tolist(), y_score.tolist()



def _augment_train(
    name: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    if name == "none":
        return train_df.copy()

    plaus = QuantilePlausibilityFilter().fit(train_df, feature_cols)

    if name == "smote":
        aug = SMOTEAugmenter(random_state=42)
        synth = aug.generate(train_df, feature_columns=feature_cols, target_column="event_label")
    elif name == "gmm":
        aug = GaussianMixtureAugmenter(random_state=42, n_components=2)
        synth = aug.generate(
            train_df,
            feature_columns=feature_cols,
            target_column="event_label",
            preserve_template_columns=[
                "sensor_coverage",
                "observed_junction_count",
                "observed_link_count",
                "leak_area",
                "time_seconds",
                "benchmark_id",
                "split",
                "stage",
            ],
        )
    elif name == "graph_cvae":
        aug = GraphCVAEAugmenter(random_state=42, epochs=30, lambda_physics=0.5, warmup_epochs=10)
        aug.fit(train_df, feature_columns=feature_cols, target_column="event_label")
        synth = aug.generate(train_df, feature_columns=feature_cols, target_column="event_label")
    else:
        raise ValueError(f"Unknown augmenter: {name}")

    if synth.empty:
        return train_df.copy()

    synth = plaus.filter(synth, feature_cols)
    if synth.empty:
        return train_df.copy()

    # Keep sequence-level grouping valid: scenario_id already assigned by augmenters.
    out = pd.concat([train_df.assign(is_synthetic=0), synth], ignore_index=True)
    return out



def evaluate_system(name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    train_used = _augment_train(name, train_df, feature_cols)

    detector = LSTMAEDetector(
        hidden_size=64,
        num_layers=2,
        sequence_length=24,
        epochs=20,
        lr=1e-3,
        batch_size=16,
        random_state=42,
        device="cpu",
    )
    detector.fit_from_frame(
        train_used,
        scenario_id_col="scenario_id",
        target_col="event_label",
        feature_columns=feature_cols,
    )

    y_true, y_pred, y_score = predict_proba_from_frame(detector, val_df, feature_cols)
    rep = binary_report(y_true, y_pred, y_score)
    out = report_as_dict(rep)
    out["rows"] = len(y_true)
    out["positive_rows"] = int(sum(y_true))
    out["augmenter"] = name
    out["train_rows"] = int(len(train_used))
    return out



def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SMOKE_CSV.exists():
        print(f"[error] Missing smoke dataset: {SMOKE_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(SMOKE_CSV)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    feature_cols = get_feature_columns(df)

    systems = ["none", "smote", "gmm", "graph_cvae"]
    results: dict[str, Any] = {"systems": {}}

    for system in systems:
        print(f"[info] Running Block 1 system: LSTM-AE + {system}")
        try:
            results["systems"][system] = evaluate_system(system, train_df, val_df, feature_cols)
        except Exception as exc:
            results["systems"][system] = {"error": str(exc)}
            print(f"[warn] Block 1 system '{system}' failed: {exc}", file=sys.stderr)

    OUTPUT_JSON.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Block 1: Sequence + Augmentation Interaction", ""]
    lines.append("| System | Validation AUPRC | Validation AUROC | Validation F1 | Rows | Positive Rows |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for system, metrics in results["systems"].items():
        if "error" in metrics:
            lines.append(f"| LSTM-AE + {system} | ERROR | ERROR | ERROR | - | - |")
        else:
            lines.append(
                f"| LSTM-AE + {system} | {metrics.get('auprc')} | {metrics.get('auroc')} | {metrics.get('f1')} | {metrics.get('rows')} | {metrics.get('positive_rows')} |"
            )
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[info] Block 1 JSON -> {OUTPUT_JSON}")
    print(f"[info] Block 1 Markdown -> {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
