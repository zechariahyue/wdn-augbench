"""Scenario-level evaluation utilities for sequence-first WDN benchmarks.

This module makes the scenario (not the timestep row) the primary statistical
unit. It supports:
- scenario-level labels and scores
- network-stratified hierarchical bootstrap for AUPRC/AUROC
- disturbance-level summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


@dataclass
class ScenarioExample:
    """One scenario-level example for sequence benchmarking."""

    scenario_id: str
    network_id: str
    disturbance: str
    coverage: float
    x: np.ndarray
    y_t: np.ndarray | None = None
    y: int | None = None


@dataclass
class ScenarioMetricReport:
    """Scenario-level metric summary."""

    auprc: float | None
    auroc: float | None
    n_examples: int
    n_positive: int
    ci_lower: float | None = None
    ci_upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "auroc": self.auroc,
            "n_examples": self.n_examples,
            "n_positive": self.n_positive,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


def scenario_level_report(y_true: Iterable[int], y_score: Iterable[float]) -> ScenarioMetricReport:
    """Compute scenario-level AUPRC/AUROC."""
    true = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)
    if true.shape != score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if len(true) == 0:
        return ScenarioMetricReport(None, None, 0, 0)

    auprc: float | None = None
    auroc: float | None = None
    if _HAS_SKLEARN and len(np.unique(true)) == 2:
        auprc = float(average_precision_score(true, score))
        auroc = float(roc_auc_score(true, score))

    return ScenarioMetricReport(
        auprc=auprc,
        auroc=auroc,
        n_examples=int(len(true)),
        n_positive=int(true.sum()),
    )


def hierarchical_bootstrap_auprc(
    scenario_ids: Iterable[str],
    network_ids: Iterable[str],
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float | None, float | None, float | None]:
    """Bootstrap AUPRC by resampling scenarios within networks.

    This preserves within-network structure and treats scenario as the atomic
    statistical unit.
    """
    if not _HAS_SKLEARN:
        return None, None, None

    sids = np.asarray(list(scenario_ids), dtype=object)
    nids = np.asarray(list(network_ids), dtype=object)
    true = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)

    if not (len(sids) == len(nids) == len(true) == len(score)):
        raise ValueError("All inputs must have the same length")
    if len(true) == 0 or len(np.unique(true)) < 2:
        return None, None, None

    rng = np.random.default_rng(random_state)

    # Build scenario-level table
    rows: list[tuple[str, str, int, float]] = []
    seen: set[str] = set()
    for sid, nid, yt, ys in zip(sids, nids, true, score):
        if sid in seen:
            continue
        seen.add(str(sid))
        rows.append((str(sid), str(nid), int(yt), float(ys)))

    if not rows:
        return None, None, None

    by_network: dict[str, list[tuple[str, str, int, float]]] = {}
    for row in rows:
        by_network.setdefault(row[1], []).append(row)

    boot_scores: list[float] = []
    network_keys = list(by_network.keys())
    for _ in range(n_bootstrap):
        sampled_rows: list[tuple[str, str, int, float]] = []
        for nid in network_keys:
            group = by_network[nid]
            idx = rng.integers(0, len(group), size=len(group))
            sampled_rows.extend(group[i] for i in idx)
        yb = np.array([r[2] for r in sampled_rows], dtype=int)
        sb = np.array([r[3] for r in sampled_rows], dtype=float)
        if len(np.unique(yb)) < 2:
            continue
        boot_scores.append(float(average_precision_score(yb, sb)))

    point = float(average_precision_score(true, score))
    if not boot_scores:
        return point, point, point

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    return point, lower, upper


def summarize_by_disturbance(
    examples: Iterable[ScenarioExample],
    scores: dict[str, float],
) -> dict[str, ScenarioMetricReport]:
    """Return scenario-level reports grouped by disturbance type."""
    grouped: dict[str, list[ScenarioExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.disturbance, []).append(ex)

    out: dict[str, ScenarioMetricReport] = {}
    for disturbance, group in grouped.items():
        y_true = [int(ex.y or 0) for ex in group]
        y_score = [float(scores[ex.scenario_id]) for ex in group if ex.scenario_id in scores]
        if len(y_score) != len(group):
            continue
        out[disturbance] = scenario_level_report(y_true, y_score)
    return out
