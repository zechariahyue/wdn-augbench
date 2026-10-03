"""Per-network normal-operation normalisation of the aggregate sensor features (v2).

Raw spatial aggregates (mean pressure, total demand, ...) live on scales that differ by
one to two orders of magnitude between networks, so a detector fitted on two small
networks cannot read them on a large one. v2 expresses every aggregate feature as a
z-score against the *same network's own normal-operation scenarios at the same sensor
coverage*:

    z__f = (f - mean_normal(f)) / std_normal(f)

The reference uses only normal-operation data, which a utility has for its own system,
and never uses disturbance rows or labels. A feature whose normal-operation spread is
zero (e.g. tank head on a tankless network) maps to z = 0 everywhere.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

AGGREGATE_FEATURES: list[str] = [
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "demand_mean", "demand_sum", "demand_std",
    "flow_mean", "flow_abs_mean", "flow_std",
    "tank_head_mean", "tank_head_std",
    "pressure_trend", "pressure_trend_cumsum", "demand_trend",
]
# Columns that describe the simulated scenario rather than what a sensor reads. They
# must never be detector inputs (v1 shipped `leak_area` as one: S36 C1).
SCENARIO_METADATA_COLUMNS: frozenset[str] = frozenset({
    "leak_area", "leak_node", "disturbance_target", "scenario_id", "scenario_kind",
    "benchmark_id", "split", "stage", "event_label", "observed_junction_count",
    "observed_link_count", "demand_seed",
})


def z_columns(features: Iterable[str] = AGGREGATE_FEATURES) -> list[str]:
    return [f"z__{c}" for c in features]


def normalise_by_network_reference(
    frame: pd.DataFrame,
    features: Sequence[str] = AGGREGATE_FEATURES,
    *,
    group_columns: Sequence[str] = ("benchmark_id", "sensor_coverage"),
    kind_column: str = "scenario_kind",
    reference_kind: str = "baseline",
    min_reference_rows: int = 10,
) -> tuple[pd.DataFrame, dict[tuple, dict[str, tuple[float, float]]]]:
    """Add ``z__<feature>`` columns normalised against each group's normal-operation rows.

    Returns the augmented frame (copy) and the reference statistics used, keyed by the
    group tuple. Raises if a group has fewer than ``min_reference_rows`` normal rows,
    because silently falling back to disturbance rows would leak label information.
    """
    feats = [c for c in features if c in frame.columns]
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise KeyError(f"normalise_by_network_reference: missing feature columns {missing}")
    out = frame.copy()
    for c in feats:
        out[f"z__{c}"] = 0.0
    stats: dict[tuple, dict[str, tuple[float, float]]] = {}
    for key, idx in out.groupby(list(group_columns), sort=False).groups.items():
        key_t = key if isinstance(key, tuple) else (key,)
        sub = out.loc[idx]
        ref = sub[sub[kind_column] == reference_kind]
        if len(ref) < min_reference_rows:
            raise ValueError(
                f"normalise_by_network_reference: group {key_t} has {len(ref)} normal-operation rows "
                f"(< {min_reference_rows}); v2 requires normal-operation scenarios on every network"
            )
        mu = ref[feats].astype(float).mean(axis=0)
        sd = ref[feats].astype(float).std(axis=0, ddof=0)
        stats[key_t] = {c: (float(mu[c]), float(sd[c])) for c in feats}
        for c in feats:
            s = float(sd[c])
            if not np.isfinite(s) or s <= 0.0:
                out.loc[idx, f"z__{c}"] = 0.0
            else:
                out.loc[idx, f"z__{c}"] = (sub[c].astype(float) - float(mu[c])) / s
    return out, stats


def assert_no_scenario_metadata(feature_columns: Iterable[str]) -> None:
    """Fail loudly if a detector feature list contains scenario metadata."""
    bad = sorted(set(feature_columns) & SCENARIO_METADATA_COLUMNS)
    if bad:
        raise ValueError(f"detector feature list contains scenario metadata (not sensor data): {bad}")
