"""Run scenario-level hierarchical evaluation on the sequence dataset.

Initial smoke implementation: scenario-level benchmark of simple detectors using
scenario labels and score aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = ACTIVE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.evaluation import (  # noqa: E402
    HydraulicResidualScorer,
    PhysicallyInformedTriageModel,
    build_triage_frame,
    hierarchical_bootstrap_auprc,
    scenario_level_report,
)
from watergen.models import LSTMAEDetector, TabularDetectorBaseline  # noqa: E402


def _load_index(dataset_dir: Path) -> pd.DataFrame:
    index_path = dataset_dir / "index.csv"
    return pd.read_csv(index_path)


def _load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _flatten_mean_features(x: np.ndarray, feature_names: np.ndarray | list[str] | None = None) -> np.ndarray:
    """Return a fixed-width scenario summary independent of sensor count.

    Uses grouped mean/std summaries for pressure, demand, and flow channels.
    Falls back to global mean/std when feature names are unavailable.
    """
    if feature_names is None:
        return np.array([float(x.mean()), float(x.std())], dtype=np.float32)

    names = [str(n) for n in list(feature_names)]
    groups = {
        "pressure": [i for i, n in enumerate(names) if n.startswith("pressure__")],
        "demand": [i for i, n in enumerate(names) if n.startswith("demand__")],
        "flow": [i for i, n in enumerate(names) if n.startswith("flow__")],
    }
    feats: list[float] = []
    for _, idxs in groups.items():
        if idxs:
            sub = x[:, idxs]
            feats.extend([
                float(sub.mean()),
                float(sub.std()),
                float(np.abs(sub).mean()),
                float(np.max(sub)),
                float(np.min(sub)),
            ])
        else:
            feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    feats.extend([float(x.shape[0]), float(x.shape[1])])
    return np.array(feats, dtype=np.float32)


def _scenario_summary_frame(index_rows: pd.DataFrame, blobs: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a scenario-level dataframe with aggregate physical features."""
    records: list[dict[str, Any]] = []
    for (_, row), blob in zip(index_rows.iterrows(), blobs):
        x = blob["x"].astype(np.float32)
        feat = _flatten_mean_features(x, blob.get("feature_names"))
        rec = {
            "scenario_id": str(row["scenario_id"]),
            "network_id": str(row["network_id"]),
            "disturbance": str(row["disturbance"]),
            "coverage": float(row["coverage"]),
            "event_label": int(row["scenario_label"]),
            # map grouped summary vector into triage-compatible names
            "pressure_mean": float(feat[0]),
            "pressure_std": float(feat[1]),
            "flow_abs_mean": float(feat[7]) if len(feat) > 7 else 0.0,
            "flow_std": float(feat[6]) if len(feat) > 6 else 0.0,
            "demand_sum": float(feat[5]) if len(feat) > 5 else 0.0,
            "pressure_max": float(feat[3]) if len(feat) > 3 else 0.0,
            "pressure_min": float(feat[4]) if len(feat) > 4 else 0.0,
            "tank_head_mean": float(feat[0]),  # proxy; no separate tank channels here
            "sensor_coverage": float(row["coverage"]),
            "observed_junction_count": float(max(1.0, feat[-1] / 3.0)) if len(feat) >= 2 else 1.0,
            "observed_link_count": float(max(1.0, feat[-1] / 6.0)) if len(feat) >= 2 else 1.0,
            "window_length": float(feat[-2]) if len(feat) >= 2 else float(x.shape[0]),
            "channel_count": float(feat[-1]) if len(feat) >= 1 else float(x.shape[1]),
        }
        records.append(rec)
    return pd.DataFrame(records)


def _positive_class_score_from_proba(proba: np.ndarray) -> float:
    """Return positive-class score robustly even for single-class estimators."""
    if proba.ndim != 2:
        raise ValueError("predict_proba output must be 2D")
    if proba.shape[1] == 1:
        # Single-class fit; sklearn returns only that class probability.
        # Treat it as p(class_present) if the value is in [0,1].
        return float(proba[0, 0])
    return float(proba[0, 1])


def run_eval(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    coverages: list[float] | None = None,
    disturbances: list[str] | None = None,
    min_timesteps: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(dataset_dir)

    if coverages is not None:
        index = index[index["coverage"].isin(coverages)].copy()
    if disturbances is not None:
        allowed = set(disturbances) | {"baseline", "normal"}
        index = index[index["disturbance"].isin(allowed)].copy()
    if min_timesteps is not None:
        index = index[index["timesteps"] >= min_timesteps].copy()

    train = index[index["split"] == "train"].copy()
    test = index[index["split"] == "test"].copy()

    if train.empty or test.empty:
        out = {
            "seed": seed,
            "coverages": coverages,
            "disturbances": disturbances,
            "min_timesteps": min_timesteps,
            "status": "empty_split_after_filtering",
            "n_train_examples": int(len(train)),
            "n_test_examples": int(len(test)),
        }
        (output_dir / "scenario_eval_results.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
        pd.DataFrame().to_csv(output_dir / "scenario_eval_records.csv", index=False)
        print(f"[warn] Empty train/test split after filtering -> {output_dir}")
        return

    # Load arrays
    train_rows = []
    test_examples = []
    train_blobs = []
    test_blobs = []
    for _, row in train.iterrows():
        blob = _load_npz(dataset_dir / row["file"])
        train_blobs.append(blob)
        feat = _flatten_mean_features(blob["x"], blob.get("feature_names"))
        train_rows.append((row["scenario_id"], row["network_id"], int(row["scenario_label"]), feat))
    for _, row in test.iterrows():
        blob = _load_npz(dataset_dir / row["file"])
        test_blobs.append(blob)
        test_examples.append((row, blob))

    x_train = np.stack([r[3] for r in train_rows], axis=0)
    y_train = np.array([r[2] for r in train_rows], dtype=int)

    # Simple tabular baseline over scenario summaries
    rf = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    rf.fit(x_train, y_train)

    # Build scenario-level triage dataframes
    train_scenario_df = _scenario_summary_frame(train, train_blobs)
    test_scenario_df = _scenario_summary_frame(test, test_blobs)

    triage_model = None
    try:
        scorer = HydraulicResidualScorer().fit(train_scenario_df[train_scenario_df["event_label"] == 0])
        train_rf_proba = rf.predict_proba(x_train)
        if train_rf_proba.shape[1] == 1:
            train_rf_score = np.repeat(float(train_rf_proba[0, 0]), len(train_scenario_df))
        else:
            train_rf_score = train_rf_proba[:, 1]
        triage_train = build_triage_frame(train_scenario_df, base_scores=train_rf_score, scorer=scorer)
        if triage_train.frame["event_label"].nunique() >= 2:
            triage_model = PhysicallyInformedTriageModel(random_state=seed).fit(
                triage_train.frame,
                target_column="event_label",
                feature_columns=triage_train.feature_columns,
            )
        else:
            print("[warn] Scenario-level triage disabled: single-class filtered training slice", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] Scenario-level triage disabled: {exc}", file=sys.stderr)
        scorer = None

    # LSTM-AE over normal sequences only when channel count is consistent
    lstm = LSTMAEDetector(hidden_size=64, num_layers=2, sequence_length=24, epochs=20, lr=1e-3, batch_size=16, random_state=seed, device="cpu")
    normal_sequences = []
    normal_widths = set()
    for _, blob in [(r, _load_npz(dataset_dir / r["file"])) for _, r in train.iterrows() if int(r["scenario_label"]) == 0]:
        x = blob["x"].astype(np.float32)
        normal_widths.add(int(x.shape[1]))
        normal_sequences.append(lstm._pad_or_truncate(x))
    lstm_enabled = len(normal_sequences) > 0 and len(normal_widths) == 1
    if lstm_enabled:
        lstm.fit(np.stack(normal_sequences, axis=0))

    records = []
    y_true = []
    y_score_rf = []
    y_score_lstm = []
    y_score_triage = []
    scenario_ids = []
    network_ids = []

    for idx, (row, blob) in enumerate(test_examples):
        x_seq = blob["x"].astype(np.float32)
        feat = _flatten_mean_features(x_seq, blob.get("feature_names"))[None, :]
        rf_score = _positive_class_score_from_proba(rf.predict_proba(feat))
        seq = lstm._pad_or_truncate(x_seq)[None, ...]
        lstm_score = _positive_class_score_from_proba(lstm.predict_proba(seq)) if lstm_enabled else 0.0
        label = int(row["scenario_label"])

        triage_score = rf_score
        if triage_model is not None and scorer is not None:
            triage_row = build_triage_frame(test_scenario_df.iloc[[idx]].copy(), base_scores=[rf_score], scorer=scorer)
            triage_score = _positive_class_score_from_proba(triage_model.predict_proba(triage_row.frame))

        records.append(
            {
                "scenario_id": str(row["scenario_id"]),
                "network_id": str(row["network_id"]),
                "disturbance": str(row["disturbance"]),
                "coverage": float(row["coverage"]),
                "scenario_label": label,
                "rf_score": rf_score,
                "lstm_score": lstm_score,
                "triage_score": triage_score,
            }
        )
        y_true.append(label)
        y_score_rf.append(rf_score)
        y_score_lstm.append(lstm_score)
        y_score_triage.append(triage_score)
        scenario_ids.append(str(row["scenario_id"]))
        network_ids.append(str(row["network_id"]))

    rf_report = scenario_level_report(y_true, y_score_rf)
    lstm_report = scenario_level_report(y_true, y_score_lstm)
    triage_report = scenario_level_report(y_true, y_score_triage)
    _, rf_lo, rf_hi = hierarchical_bootstrap_auprc(scenario_ids, network_ids, y_true, y_score_rf, n_bootstrap=500)
    _, lstm_lo, lstm_hi = hierarchical_bootstrap_auprc(scenario_ids, network_ids, y_true, y_score_lstm, n_bootstrap=500)
    _, triage_lo, triage_hi = hierarchical_bootstrap_auprc(scenario_ids, network_ids, y_true, y_score_triage, n_bootstrap=500)
    rf_report.ci_lower, rf_report.ci_upper = rf_lo, rf_hi
    lstm_report.ci_lower, lstm_report.ci_upper = lstm_lo, lstm_hi
    triage_report.ci_lower, triage_report.ci_upper = triage_lo, triage_hi

    out = {
        "seed": seed,
        "coverages": coverages,
        "disturbances": disturbances,
        "min_timesteps": min_timesteps,
        "rf_scenario": rf_report.to_dict(),
        "lstm_scenario": lstm_report.to_dict(),
        "triage_scenario": triage_report.to_dict(),
        "n_test_examples": len(records),
    }
    (output_dir / "scenario_eval_results.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(records).to_csv(output_dir / "scenario_eval_records.csv", index=False)
    print(f"[info] Scenario eval JSON -> {output_dir / 'scenario_eval_results.json'}")
    print(f"[info] Scenario eval CSV -> {output_dir / 'scenario_eval_records.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scenario-level hierarchical evaluation")
    parser.add_argument("--dataset-dir", type=Path, default=ACTIVE_ROOT / "artifacts" / "datasets" / "sequence")
    parser.add_argument("--output-dir", type=Path, default=ACTIVE_ROOT / "artifacts" / "scenario_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverages", type=float, nargs="*", default=None)
    parser.add_argument("--disturbances", nargs="*", default=None)
    parser.add_argument("--min-timesteps", type=int, default=None)
    args = parser.parse_args()
    run_eval(
        args.dataset_dir,
        args.output_dir,
        seed=args.seed,
        coverages=args.coverages,
        disturbances=args.disturbances,
        min_timesteps=args.min_timesteps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
