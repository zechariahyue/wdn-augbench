"""Standalone evaluation script for the LSTM-AE detector on the smoke dataset.

Run from the project root:
    python dev/active/scripts/eval_lstm_ae.py

Results are written to dev/active/artifacts/lstm_ae_results.json and printed
in a format suitable for inclusion in Table 2 of the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allows running from any working directory as long as the
# project root (containing dev/) is the cwd.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # dev/active
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

SMOKE_CSV = PROJECT_ROOT / "artifacts" / "plan_a" / "smoke" / "smoke_dataset.csv"
RESULTS_JSON = PROJECT_ROOT / "artifacts" / "lstm_ae_results.json"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

try:
    from watergen.models import LSTMAEDetector, select_numeric_feature_columns
    from watergen.evaluation import binary_report, report_as_dict
    _HAS_WATERGEN = True
except ImportError as exc:
    print(f"[ERROR] Could not import watergen: {exc}", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Metadata columns excluded from features (mirrors baselines.py logic)
# ---------------------------------------------------------------------------
_META_COLS = {
    "event_label", "scenario_id", "time_seconds", "benchmark_id",
    "scenario_kind", "split", "stage", "leak_node",
    "disturbance_target", "leak_area",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def predict_proba_from_frame(
    detector: LSTMAEDetector,
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true_row, y_score_row) arrays aligned with frame rows.

    The LSTM-AE produces one prediction per *scenario* (sequence-level).  We
    broadcast that prediction to every row in the scenario so the evaluation
    arrays are row-aligned (matching frame length) and comparable with the
    row-level ground-truth labels.
    """
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
            grp[feature_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        seq = detector._pad_or_truncate(arr)[np.newaxis, ...]  # (1, T, F)
        proba = detector.predict_proba(seq)  # (1, 2)
        score = float(proba[0, 1])           # anomaly probability

        # Ground-truth: max label in the group (1 if any anomaly timestep)
        grp_label = int(grp["event_label"].max())

        for row_idx in grp.index:
            pos = idx_map[row_idx]
            y_true[pos] = grp_label
            y_score[pos] = score

    return y_true, y_score


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]


def _run_single_seed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict:
    """Fit and evaluate LSTM-AE for a single seed, returning result dict."""
    detector = LSTMAEDetector(
        hidden_size=64,
        num_layers=2,
        sequence_length=24,
        epochs=20,
        lr=1e-3,
        batch_size=16,
        random_state=seed,
        device="cpu",
    )
    detector.fit_from_frame(
        train_df,
        scenario_id_col="scenario_id",
        target_col="event_label",
        feature_columns=feature_cols,
    )

    y_true_val, y_score_val = predict_proba_from_frame(detector, val_df, feature_cols)
    y_pred_val = (y_score_val >= 0.5).astype(int)
    val_report = binary_report(
        y_true_val.tolist(), y_pred_val.tolist(), y_score=y_score_val.tolist(),
    )
    val_dict = report_as_dict(val_report)

    y_true_tr, y_score_tr = predict_proba_from_frame(detector, train_df, feature_cols)
    y_pred_tr = (y_score_tr >= 0.5).astype(int)
    train_report = binary_report(
        y_true_tr.tolist(), y_pred_tr.tolist(), y_score=y_score_tr.tolist(),
    )
    train_dict = report_as_dict(train_report)

    pos_rows_val = int(y_true_val.sum())
    return {
        "seed": seed,
        "validation": {
            "auprc": val_dict["auprc"],
            "auroc": val_dict["auroc"],
            "f1": val_dict["f1"],
            "precision": val_dict["precision"],
            "recall": val_dict["recall"],
            "rows": int(len(y_true_val)),
            "positive_rows": pos_rows_val,
        },
        "train": {
            "auprc": train_dict["auprc"],
            "auroc": train_dict["auroc"],
            "f1": train_dict["f1"],
        },
    }


def main() -> None:
    print(f"Loading smoke dataset from: {SMOKE_CSV}")
    if not SMOKE_CSV.exists():
        print(f"[ERROR] Smoke CSV not found: {SMOKE_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SMOKE_CSV)
    print(f"  Total rows: {len(df)}  |  columns: {len(df.columns)}")

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "validation"].copy()
    print(f"  Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")
    print(f"  Train pos rate: {train_df['event_label'].mean():.3f}  "
          f"|  Val pos rate: {val_df['event_label'].mean():.3f}")

    feature_cols = get_feature_columns(df)
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # -----------------------------------------------------------------------
    # Multi-seed evaluation
    # -----------------------------------------------------------------------
    all_seed_results = []
    val_auprcs = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        seed_result = _run_single_seed(train_df, val_df, feature_cols, seed)
        all_seed_results.append(seed_result)
        auprc = seed_result["validation"]["auprc"]
        if auprc is not None:
            val_auprcs.append(auprc)
        print(f"  Val AUPRC = {auprc:.4f}" if auprc is not None else "  Val AUPRC = n/a")

    # -----------------------------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------------------------
    mean_auprc = float(np.mean(val_auprcs)) if val_auprcs else None
    std_auprc = float(np.std(val_auprcs)) if val_auprcs else None

    results = {
        "seeds": SEEDS,
        "per_seed": all_seed_results,
        "aggregate_validation": {
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
            "n_seeds": len(val_auprcs),
        },
    }

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {RESULTS_JSON}")

    # -----------------------------------------------------------------------
    # Paper-ready summary (Table 2)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LSTM-AE  —  Smoke Dataset  (Table 2 results, 5-seed)")
    print("=" * 60)
    if mean_auprc is not None:
        print(f"  Val AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}  (n={len(val_auprcs)} seeds)")
    else:
        print("  Val AUPRC: n/a")
    for sr in all_seed_results:
        a = sr["validation"]["auprc"]
        print(f"    seed {sr['seed']}: {a:.4f}" if a is not None else f"    seed {sr['seed']}: n/a")
    print("=" * 60)

    print(f"\n[PAPER TABLE 2]  LSTM-AE  AUPRC = "
          f"{'N/A' if mean_auprc is None else f'{mean_auprc:.4f} ± {std_auprc:.4f}'}")


if __name__ == "__main__":
    main()
