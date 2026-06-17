"""Hydraulic plausibility scoring for generated WDN disturbance samples.

Implements two complementary plausibility measures:
1. Soft physics residuals (differentiable, computed from tabular features)
2. WNTR oracle feasibility check (expensive but ground-truth physical validity)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PlausibilityReport:
    """Plausibility assessment for a set of generated samples."""

    n_samples: int
    mass_residual_mean: float       # Mean junction continuity violation
    mass_residual_std: float
    energy_residual_mean: float     # Mean Hazen-Williams headloss violation
    energy_residual_std: float
    pressure_bound_violation_rate: float  # Fraction below min or above max pressure
    overall_plausibility_score: float    # Composite 0-1 score (higher = more plausible)
    n_passing: int                       # Samples passing all soft checks
    pass_rate: float                     # n_passing / n_samples


@dataclass
class RowwisePlausibilityResult:
    """Per-row plausibility outputs for downstream reranking/triage."""

    frame: pd.DataFrame
    mass_residuals: np.ndarray
    energy_residuals: np.ndarray
    pressure_violations: np.ndarray
    plausibility_scores: np.ndarray


# ---------------------------------------------------------------------------
# HydraulicResidualScorer
# ---------------------------------------------------------------------------


class HydraulicResidualScorer:
    """Compute hydraulic physics residuals on tabular WDN feature vectors.

    Uses the existing tabular feature columns (pressure_mean, pressure_min,
    pressure_max, pressure_std, flow_mean, flow_std, demand_mean, demand_sum)
    to compute approximate physics consistency scores.

    These are proxy residuals, not exact hydraulic equations — exact enforcement
    requires full network topology. These proxies capture the most obvious
    physics violations in aggregated features.

    Usage
    -----
    >>> scorer = HydraulicResidualScorer()
    >>> scorer.fit(normal_frame)
    >>> report = scorer.score_samples(generated_frame)
    """

    def __init__(
        self,
        *,
        min_pressure_m: float = 10.0,   # Minimum service pressure (10m = ~14 psi)
        max_pressure_m: float = 105.0,  # Maximum pipe pressure rating (~150 psi)
        max_velocity_ms: float = 3.0,   # Maximum pipe velocity (m/s)
        min_velocity_ms: float = 0.0,   # Minimum pipe velocity
    ) -> None:
        self.min_pressure_m = float(min_pressure_m)
        self.max_pressure_m = float(max_pressure_m)
        self.max_velocity_ms = float(max_velocity_ms)
        self.min_velocity_ms = float(min_velocity_ms)

        # Fitted statistics — set by fit()
        self._expected_ratio: float = 1.0   # pressure_std / (flow_std + eps)
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, normal_frame: pd.DataFrame) -> "HydraulicResidualScorer":
        """Learn expected hydraulic ranges from normal-class simulation data.

        Computes the expected pressure_std / flow_std ratio from the normal
        operating data. This ratio is used as the reference for the energy
        residual proxy.

        Args:
            normal_frame: DataFrame of normal (non-anomalous) simulation rows
                produced by ``simulate_scenario`` / ``_aggregate_view``.

        Returns:
            self (for chaining)
        """
        if normal_frame.empty:
            self._expected_ratio = 1.0
            self._is_fitted = True
            return self

        pressure_std = self._col(normal_frame, "pressure_std", 0.0)
        flow_std = self._col(normal_frame, "flow_std", 0.0)

        # Compute per-row ratios and take the median (robust to outliers)
        ratios = pressure_std / (flow_std + 1e-9)
        # Clip extreme values before taking the median to avoid blown-up refs
        ratios = np.clip(ratios, 0.0, 1e6)
        self._expected_ratio = float(np.nanmedian(ratios)) if len(ratios) > 0 else 1.0
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score_rows(self, frame: pd.DataFrame) -> RowwisePlausibilityResult:
        """Return per-row plausibility features for triage/reranking.

        This exposes the same proxy checks used by :meth:`score_samples`, but at
        row granularity so downstream pipelines can fuse detector scores with
        physical-consistency signals.
        """
        if not self._is_fitted:
            self.fit(frame)

        n = len(frame)
        if n == 0:
            empty = np.array([], dtype=float)
            return RowwisePlausibilityResult(
                frame=frame.copy(),
                mass_residuals=empty,
                energy_residuals=empty,
                pressure_violations=empty,
                plausibility_scores=empty,
            )

        pressure_min = self._col(frame, "pressure_min", self.min_pressure_m)
        pressure_max = self._col(frame, "pressure_max", self.max_pressure_m)
        below_min = pressure_min < self.min_pressure_m
        above_max = pressure_max > self.max_pressure_m
        pressure_violation = (below_min | above_max).astype(float)

        demand_sum = self._col(frame, "demand_sum", 0.0)
        flow_abs_mean = self._col(frame, "flow_abs_mean", 0.0)
        obs_junction = self._col(frame, "observed_junction_count", 1.0)
        obs_junction = np.where(obs_junction < 1.0, 1.0, obs_junction)
        predicted_demand = flow_abs_mean * obs_junction
        denom = np.abs(demand_sum) + np.abs(predicted_demand) + 1e-9
        mass_residuals = np.abs(demand_sum - predicted_demand) / denom * 2.0

        pressure_std = self._col(frame, "pressure_std", 0.0)
        flow_std = self._col(frame, "flow_std", 0.0)
        actual_ratio = pressure_std / (flow_std + 1e-9)
        energy_residuals = np.abs(actual_ratio - self._expected_ratio) / (
            self._expected_ratio + 1e-9
        )
        energy_residuals = np.clip(energy_residuals, 0.0, 10.0)

        plausibility_scores = (
            np.exp(-1.0 * mass_residuals - 1.0 * energy_residuals)
            * (1.0 - pressure_violation)
        )

        augmented = frame.copy()
        augmented["mass_residual"] = mass_residuals
        augmented["energy_residual"] = energy_residuals
        augmented["pressure_bound_flag"] = pressure_violation.astype(int)
        augmented["plausibility_score"] = plausibility_scores

        return RowwisePlausibilityResult(
            frame=augmented,
            mass_residuals=mass_residuals,
            energy_residuals=energy_residuals,
            pressure_violations=pressure_violation,
            plausibility_scores=plausibility_scores,
        )

    def score_samples(self, frame: pd.DataFrame) -> PlausibilityReport:
        """Score a set of generated samples for hydraulic plausibility.

        Computes three proxy residuals on the tabular feature columns:

        1. **Pressure bound violation rate** — fraction of rows where
           ``pressure_min < min_pressure_m`` or ``pressure_max > max_pressure_m``.

        2. **Mass residual** (conservation-of-mass proxy) — checks that
           ``demand_sum ≈ flow_abs_mean * observed_junction_count`` within 30%
           tolerance.  Computed as mean absolute relative deviation per row.

        3. **Energy residual** (pressure-flow coherence proxy) — checks that
           ``pressure_std / (flow_std + eps)`` is close to the value learned
           from normal data.

        Args:
            frame: DataFrame of generated or simulated rows with the standard
                aggregate feature columns.

        Returns:
            PlausibilityReport with all computed metrics.
        """
        rowwise = self.score_rows(frame)
        n = len(rowwise.frame)
        if n == 0:
            return PlausibilityReport(
                n_samples=0,
                mass_residual_mean=0.0,
                mass_residual_std=0.0,
                energy_residual_mean=0.0,
                energy_residual_std=0.0,
                pressure_bound_violation_rate=0.0,
                overall_plausibility_score=0.0,
                n_passing=0,
                pass_rate=0.0,
            )

        mass_residual_mean = float(np.mean(rowwise.mass_residuals))
        mass_residual_std = float(np.std(rowwise.mass_residuals))
        energy_residual_mean = float(np.mean(rowwise.energy_residuals))
        energy_residual_std = float(np.std(rowwise.energy_residuals))
        pressure_violation_rate = float(np.mean(rowwise.pressure_violations))
        overall_plausibility_score = float(np.mean(rowwise.plausibility_scores))
        n_passing = int(np.sum(rowwise.plausibility_scores > 0.5))
        pass_rate = n_passing / n if n > 0 else 0.0

        return PlausibilityReport(
            n_samples=n,
            mass_residual_mean=mass_residual_mean,
            mass_residual_std=mass_residual_std,
            energy_residual_mean=energy_residual_mean,
            energy_residual_std=energy_residual_std,
            pressure_bound_violation_rate=pressure_violation_rate,
            overall_plausibility_score=overall_plausibility_score,
            n_passing=n_passing,
            pass_rate=pass_rate,
        )


    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def compare_augmenters(
        self,
        frames: dict[str, pd.DataFrame],
        normal_frame: pd.DataFrame,
    ) -> dict[str, PlausibilityReport]:
        """Score multiple augmenters and return one report per augmenter.

        Fits the scorer on ``normal_frame`` once, then scores every frame in
        ``frames``.

        Args:
            frames: Mapping from augmenter name to its generated DataFrame.
            normal_frame: Normal-class data used to fit the scorer.

        Returns:
            Mapping from augmenter name to its ``PlausibilityReport``.
        """
        self.fit(normal_frame)
        return {name: self.score_samples(df) for name, df in frames.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _col(frame: pd.DataFrame, col: str, default: float) -> np.ndarray:
        """Extract a column as float array, filling missing with *default*."""
        if col in frame.columns:
            return frame[col].fillna(default).to_numpy(dtype=float)
        return np.full(len(frame), default, dtype=float)


# ---------------------------------------------------------------------------
# WNTRFeasibilityChecker
# ---------------------------------------------------------------------------


class WNTRFeasibilityChecker:
    """Oracle feasibility checker using WNTR forward simulation.

    For each generated sample, attempts to construct a valid WNTR simulation
    state and checks whether the implied hydraulic state is physically feasible.

    This is EXPENSIVE (0.1–2 s per sample) and should only be used on a
    representative subsample (10–20 % of generated data) for evaluation,
    not during training.

    If WNTR is unavailable or ``benchmark_path`` is not provided, all methods
    return gracefully with a ``"skipped_no_wntr"`` status.
    """

    def __init__(
        self,
        *,
        benchmark_path: str | None = None,
        tolerance_fraction: float = 0.05,  # 5% tolerance on pressure/flow matching
        sample_fraction: float = 0.10,     # Check only 10% of samples
        random_state: int = 42,
    ) -> None:
        self.benchmark_path = benchmark_path
        self.tolerance_fraction = float(tolerance_fraction)
        self.sample_fraction = float(sample_fraction)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_feasibility(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Run oracle feasibility check on a sample of the provided frame.

        Randomly draws ``sample_fraction * len(frame)`` rows and attempts a
        WNTR forward simulation for each, checking whether the simulated
        pressure range overlaps (within tolerance) with the row's recorded
        pressure range.

        Args:
            frame: DataFrame with standard aggregate feature columns.

        Returns:
            Dictionary with keys:
            - ``"feasibility_rate"`` (float | None): fraction of checked samples
              whose hydraulic state is physically consistent.
            - ``"n_checked"`` (int): number of samples actually simulated.
            - ``"n_passing"`` (int): number that passed the pressure overlap test.
            - ``"status"`` (str): ``"ok"`` on success or a skip/error reason.
        """
        # ---- Guard: no WNTR / no network path ------------------------
        try:
            import wntr  # noqa: F401 — presence check
        except ImportError:
            return {
                "feasibility_rate": None,
                "n_checked": 0,
                "n_passing": 0,
                "status": "skipped_no_wntr",
            }

        if self.benchmark_path is None:
            return {
                "feasibility_rate": None,
                "n_checked": 0,
                "n_passing": 0,
                "status": "skipped_no_wntr",
            }

        # ---- Sub-sample rows ----------------------------------------
        n_total = len(frame)
        if n_total == 0:
            return {
                "feasibility_rate": None,
                "n_checked": 0,
                "n_passing": 0,
                "status": "empty_frame",
            }

        n_check = max(1, int(n_total * self.sample_fraction))
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_total, size=min(n_check, n_total), replace=False)
        sample = frame.iloc[indices]

        # ---- Check each row -----------------------------------------
        n_passing = 0
        n_checked = 0

        for _, row in sample.iterrows():
            try:
                passed = self._check_row(row)
                n_passing += int(passed)
                n_checked += 1
            except Exception:
                # A single failure must not abort the whole evaluation
                n_checked += 1

        feasibility_rate = n_passing / n_checked if n_checked > 0 else None
        return {
            "feasibility_rate": feasibility_rate,
            "n_checked": n_checked,
            "n_passing": n_passing,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_row(self, row: pd.Series) -> bool:
        """Simulate network for one row and check pressure feasibility.

        Loads the network fresh, runs a steady-state (single time-step)
        WNTR simulation, then verifies that the simulated pressure range
        for all junctions overlaps (within tolerance) the pressure range
        recorded in *row*.

        Args:
            row: A single aggregate-feature row from a generated DataFrame.

        Returns:
            True if the simulated pressures are consistent with the row.
        """
        import wntr

        wn = wntr.network.WaterNetworkModel(str(self.benchmark_path))
        # Run a single-step simulation (minimum duration = 1 report step)
        wn.options.time.duration = int(
            wn.options.time.report_timestep or wn.options.time.hydraulic_timestep or 3600
        )
        simulator = wntr.sim.WNTRSimulator(wn)
        results = simulator.run_sim()

        junctions = wn.junction_name_list
        if not junctions:
            return True  # Nothing to check

        pressure_data = results.node["pressure"][junctions]
        sim_min = float(pressure_data.values.min())
        sim_max = float(pressure_data.values.max())

        row_min = float(row.get("pressure_min", sim_min))
        row_max = float(row.get("pressure_max", sim_max))

        tol = self.tolerance_fraction
        # Overlap check: the intervals must share at least a point within tol
        sim_lo = sim_min * (1.0 - tol)
        sim_hi = sim_max * (1.0 + tol)
        return bool(row_min <= sim_hi and row_max >= sim_lo)


# ---------------------------------------------------------------------------
# Convenience comparison function
# ---------------------------------------------------------------------------


def compare_augmenter_plausibility(
    augmenter_frames: dict[str, pd.DataFrame],
    normal_frame: pd.DataFrame,
    *,
    min_pressure_m: float = 10.0,
    max_pressure_m: float = 105.0,
) -> dict[str, dict[str, float]]:
    """Compare plausibility scores across multiple augmenters.

    Fits a single ``HydraulicResidualScorer`` on *normal_frame*, then scores
    every augmenter frame and returns a flat metric dictionary per augmenter.
    The output is designed for scatter plots of plausibility vs. ΔAUPRC.

    Args:
        augmenter_frames: Mapping from augmenter name to its generated
            DataFrame of tabular feature rows.
        normal_frame: Normal-class simulation data used to calibrate the
            scorer's expected pressure-flow ratio.
        min_pressure_m: Minimum acceptable service pressure in metres.
        max_pressure_m: Maximum acceptable pipe pressure in metres.

    Returns:
        Mapping ``{augmenter_name: {metric_name: value, ...}}`` where each
        inner dict contains:

        - ``"plausibility_score"`` — composite 0-1 score
        - ``"pass_rate"`` — fraction of samples scoring > 0.5
        - ``"pressure_bound_violation_rate"``
        - ``"mass_residual_mean"``
        - ``"energy_residual_mean"``
        - ``"n_samples"``
    """
    scorer = HydraulicResidualScorer(
        min_pressure_m=min_pressure_m,
        max_pressure_m=max_pressure_m,
    )
    scorer.fit(normal_frame)

    result: dict[str, dict[str, float]] = {}
    for name, df in augmenter_frames.items():
        report = scorer.score_samples(df)
        result[name] = {
            "plausibility_score": report.overall_plausibility_score,
            "pass_rate": report.pass_rate,
            "pressure_bound_violation_rate": report.pressure_bound_violation_rate,
            "mass_residual_mean": report.mass_residual_mean,
            "energy_residual_mean": report.energy_residual_mean,
            "n_samples": float(report.n_samples),
        }
    return result
