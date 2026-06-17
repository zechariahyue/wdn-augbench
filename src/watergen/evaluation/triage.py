"""Physically informed anomaly triage / reranking utilities.

This module builds second-stage triage features from:
- base detector scores
- hydraulic plausibility residuals
- coverage/context metadata
- optional domain residual scores

The goal is not to replace the detector, but to rerank candidate anomalies by
physical credibility and operational review priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

from .plausibility import HydraulicResidualScorer


TRIAGE_FEATURE_COLUMNS = [
    "base_score",
    "pressure_range",
    "pressure_head_gap",
    "demand_flow_ratio",
    "pressure_flow_ratio",
    "mass_residual",
    "energy_residual",
    "pressure_bound_flag",
    "plausibility_score",
    "coverage_adjusted_mass_residual",
    "coverage_adjusted_energy_residual",
    "base_x_plausibility",
    "base_x_coverage",
    "wntr_residual_score",
]


@dataclass
class TriageResult:
    frame: pd.DataFrame
    feature_columns: list[str]


class PhysicallyInformedTriageModel:
    """Second-stage reranker built on simple, interpretable features.

    Default model: logistic regression over base detector score + physics-aware
    plausibility features.
    """

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model_ = None
        self.feature_columns_: list[str] = []

    def fit(
        self,
        triage_frame: pd.DataFrame,
        *,
        target_column: str = "event_label",
        feature_columns: Sequence[str] | None = None,
    ) -> "PhysicallyInformedTriageModel":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for PhysicallyInformedTriageModel")

        cols = list(feature_columns) if feature_columns is not None else [
            c for c in TRIAGE_FEATURE_COLUMNS if c in triage_frame.columns
        ]
        if not cols:
            raise ValueError("No triage features available to fit the reranker")

        x = triage_frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = triage_frame[target_column].to_numpy(dtype=int)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state)
        )
        model.fit(x, y)
        self.model_ = model
        self.feature_columns_ = cols
        return self

    def predict_proba(self, triage_frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Triage model must be fitted before predict_proba")
        x = triage_frame.loc[:, self.feature_columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return self.model_.predict_proba(x)

    def predict(self, triage_frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Triage model must be fitted before predict")
        x = triage_frame.loc[:, self.feature_columns_].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return self.model_.predict(x)


def build_triage_frame(
    frame: pd.DataFrame,
    *,
    base_scores: Sequence[float],
    scorer: HydraulicResidualScorer,
    wntr_residual_scores: Sequence[float] | None = None,
) -> TriageResult:
    """Build a triage feature frame from base detector scores + physics signals."""
    if len(frame) != len(base_scores):
        raise ValueError("frame and base_scores must have the same length")

    rowwise = scorer.score_rows(frame)
    triage = rowwise.frame.copy()
    triage["base_score"] = np.asarray(base_scores, dtype=float)

    # Derived features
    pressure_max = _col(triage, "pressure_max")
    pressure_min = _col(triage, "pressure_min")
    pressure_mean = _col(triage, "pressure_mean")
    tank_head_mean = _col(triage, "tank_head_mean")
    demand_sum = _col(triage, "demand_sum")
    flow_abs_mean = _col(triage, "flow_abs_mean")
    obs_junction = np.maximum(_col(triage, "observed_junction_count"), 1.0)
    pressure_std = _col(triage, "pressure_std")
    flow_std = _col(triage, "flow_std")
    coverage = np.maximum(_col(triage, "sensor_coverage"), 1e-6)

    triage["pressure_range"] = pressure_max - pressure_min
    triage["pressure_head_gap"] = tank_head_mean - pressure_mean
    triage["demand_flow_ratio"] = demand_sum / (flow_abs_mean * obs_junction + 1e-9)
    triage["pressure_flow_ratio"] = pressure_std / (flow_std + 1e-9)
    triage["coverage_adjusted_mass_residual"] = triage["mass_residual"] / coverage
    triage["coverage_adjusted_energy_residual"] = triage["energy_residual"] / coverage
    triage["base_x_plausibility"] = triage["base_score"] * triage["plausibility_score"]
    triage["base_x_coverage"] = triage["base_score"] * coverage

    if wntr_residual_scores is None:
        triage["wntr_residual_score"] = 0.0
    else:
        if len(wntr_residual_scores) != len(frame):
            raise ValueError("wntr_residual_scores must match frame length")
        triage["wntr_residual_score"] = np.asarray(wntr_residual_scores, dtype=float)

    triage["triage_bucket"] = triage.apply(_assign_bucket, axis=1)

    feature_columns = [c for c in TRIAGE_FEATURE_COLUMNS if c in triage.columns]
    return TriageResult(frame=triage, feature_columns=feature_columns)


def heuristic_triage_score(frame: pd.DataFrame) -> np.ndarray:
    """Deterministic weighted-score fallback for transparent triage."""
    base = _col(frame, "base_score")
    plaus = _col(frame, "plausibility_score")
    mass = _col(frame, "mass_residual")
    energy = _col(frame, "energy_residual")
    coverage = _col(frame, "sensor_coverage")

    score = (
        0.45 * base
        + 0.25 * plaus
        - 0.15 * mass
        - 0.10 * energy
        + 0.05 * coverage
    )
    return score.astype(float)


def _assign_bucket(row: pd.Series) -> str:
    base = float(row.get("base_score", 0.0))
    plaus = float(row.get("plausibility_score", 0.0))
    if base >= 0.5 and plaus >= 0.5:
        return "likely_physical"
    if base >= 0.5 and plaus < 0.5:
        return "suspicious_implausible"
    return "likely_false_alarm"


def _col(frame: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col in frame.columns:
        return frame[col].fillna(default).to_numpy(dtype=float)
    return np.full(len(frame), default, dtype=float)
