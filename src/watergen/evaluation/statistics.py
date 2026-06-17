"""Statistical analysis utilities for multi-seed experiment results.

Implements:
- Bootstrap confidence intervals for AUPRC
- Wilcoxon signed-rank test for augmentation improvement claim
- Holm-Bonferroni correction for multiple comparisons
- Multi-seed experiment aggregation
- Spearman correlation between plausibility score and ΔAUPRC
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

@dataclass
class _SeedBundle:
    """All per-run seeds derived from one child of the master SeedSequence."""

    model_seed: int
    placement_seed: int
    split_seed: int
    simulation_seed: int
    augmenter_seed: int


class SeedManager:
    """Deterministic seed generator for reproducible multi-seed experiments.

    All child seeds are derived from a master seed via numpy SeedSequence,
    ensuring reproducible results across runs.

    Example::

        seeds = SeedManager(master_seed=42, n_seeds=5)
        for bundle in seeds:
            run_experiment(
                random_state=bundle.model_seed,
                placement_seed=bundle.placement_seed,
                split_seed=bundle.split_seed,
            )
    """

    # Number of sub-seeds to generate per child (one per named field in _SeedBundle)
    _N_SUB_SEEDS = 5

    def __init__(self, master_seed: int = 42, n_seeds: int = 5) -> None:
        self.master_seed = master_seed
        self.n_seeds = n_seeds
        self._bundles: list[_SeedBundle] = []
        self._generate_seeds()

    # ------------------------------------------------------------------
    # Internal generation
    # ------------------------------------------------------------------

    def _generate_seeds(self) -> None:
        """Generate all child seeds from master via SeedSequence."""
        try:
            self._generate_via_seed_sequence()
        except (AttributeError, TypeError):
            # Fallback: numpy version predates SeedSequence — use linear offsets
            self._generate_via_linear_offsets()

    def _generate_via_seed_sequence(self) -> None:
        """Use numpy SeedSequence for high-quality independent streams."""
        ss = np.random.SeedSequence(self.master_seed)
        children = ss.spawn(self.n_seeds)
        self._bundles = []
        for child in children:
            # Draw 5 independent 32-bit integers from each child stream
            ints = child.generate_state(self._N_SUB_SEEDS, dtype=np.uint32)
            # Convert to Python int so they are always JSON-serialisable
            bundle = _SeedBundle(
                model_seed=int(ints[0]),
                placement_seed=int(ints[1]),
                split_seed=int(ints[2]),
                simulation_seed=int(ints[3]),
                augmenter_seed=int(ints[4]),
            )
            self._bundles.append(bundle)

    def _generate_via_linear_offsets(self) -> None:
        """Fallback: derive seeds via simple integer arithmetic."""
        self._bundles = []
        for i in range(self.n_seeds):
            base = self.master_seed + i * 1000
            bundle = _SeedBundle(
                model_seed=base,
                placement_seed=base + 1,
                split_seed=base + 2,
                simulation_seed=base + 3,
                augmenter_seed=base + 4,
            )
            self._bundles.append(bundle)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __iter__(self):  # type: ignore[override]
        return iter(self._bundles)

    def __len__(self) -> int:
        return len(self._bundles)

    def __getitem__(self, index: int) -> _SeedBundle:
        return self._bundles[index]

    def to_dict(self) -> dict[str, Any]:
        """Serialise all bundles to a plain dict for JSON output."""
        return {
            "master_seed": self.master_seed,
            "n_seeds": self.n_seeds,
            "bundles": [
                {
                    "model_seed": b.model_seed,
                    "placement_seed": b.placement_seed,
                    "split_seed": b.split_seed,
                    "simulation_seed": b.simulation_seed,
                    "augmenter_seed": b.augmenter_seed,
                }
                for b in self._bundles
            ],
        }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_auprc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for AUPRC.

    Bootstraps at the sample level (resample rows with replacement).
    For time-series data, caller should bootstrap at the scenario level
    before passing y_true/y_score arrays.

    Args:
        y_true: Binary ground-truth labels (0/1), shape (n_samples,).
        y_score: Positive-class probability scores, shape (n_samples,).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Width of the confidence interval (default 0.95 → 95% CI).
        random_state: Seed for numpy RNG.

    Returns:
        ``(auprc_mean, ci_lower, ci_upper)``

    Raises:
        ImportError: When scikit-learn is unavailable.
        ValueError: When y_true contains only one class.
    """
    try:
        from sklearn.metrics import average_precision_score
    except ImportError as exc:
        raise ImportError("scikit-learn is required for bootstrap_auprc_ci") from exc

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot_scores: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        s_b = y_score[idx]
        # Skip samples where only one class is present
        if len(np.unique(y_b)) < 2:
            continue
        boot_scores.append(float(average_precision_score(y_b, s_b)))

    if not boot_scores:
        # All bootstrap samples were single-class — return point estimate
        if len(np.unique(y_true)) >= 2:
            point = float(average_precision_score(y_true, y_score))
        else:
            point = 0.0
        return point, point, point

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_scores, 100 * alpha / 2))
    upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
    mean = float(np.mean(boot_scores))
    return mean, lower, upper


# ---------------------------------------------------------------------------
# Multi-seed result aggregation
# ---------------------------------------------------------------------------

@dataclass
class MultiSeedSummary:
    """Summary statistics across multiple random seed runs."""

    metric: str
    mean: float
    std: float
    ci_lower: float   # 95% bootstrap CI lower bound
    ci_upper: float   # 95% bootstrap CI upper bound
    n_seeds: int
    values: list[float] = field(default_factory=list)  # raw values per seed

    def formatted(self, precision: int = 3) -> str:
        """Return a human-readable ``mean ± std (CI)`` string."""
        fmt = f".{precision}f"
        return (
            f"{self.mean:{fmt}} ± {self.std:{fmt}} "
            f"[{self.ci_lower:{fmt}}, {self.ci_upper:{fmt}}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "n_seeds": self.n_seeds,
            "values": self.values,
        }


def aggregate_multi_seed_results(
    results_per_seed: list[dict[str, Any]],
    *,
    metric_key: str = "auprc",
    split: str = "validation",
) -> MultiSeedSummary:
    """Aggregate results across multiple seed runs into summary statistics.

    Args:
        results_per_seed: List of result dicts, one per seed run.  Each dict
            is expected to have a ``"splits"`` key whose value is a mapping of
            split name → metric dict.  Alternatively, a flat dict where
            ``metric_key`` lives at the top level is also accepted.
        metric_key: Which metric to aggregate (``auprc``, ``f1``, ``auroc``).
        split: Which split to read from (``train``/``validation``/``test``).

    Returns:
        A :class:`MultiSeedSummary` with mean, std, and 95% CI.
    """
    values: list[float] = []

    for result in results_per_seed:
        if not isinstance(result, dict):
            continue

        # Try nested path: result["splits"][split][metric_key]
        splits_map = result.get("splits", {})
        if isinstance(splits_map, dict) and split in splits_map:
            split_data = splits_map[split]
            if isinstance(split_data, dict) and metric_key in split_data:
                val = split_data[metric_key]
                if isinstance(val, (int, float)) and val == val:  # nan check
                    values.append(float(val))
                continue

        # Fallback: try flat top-level key
        if metric_key in result:
            val = result[metric_key]
            if isinstance(val, (int, float)) and val == val:
                values.append(float(val))

    if not values:
        return MultiSeedSummary(
            metric=metric_key,
            mean=0.0,
            std=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            n_seeds=0,
            values=[],
        )

    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    # Bootstrap CI over the seed values themselves
    rng = np.random.default_rng(42)
    boot_means: list[float] = []
    for _ in range(2000):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(float(np.mean(sample)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return MultiSeedSummary(
        metric=metric_key,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_seeds=len(values),
        values=list(values),
    )


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_augmentation_test(
    baseline_scores: Sequence[float],
    augmented_scores: Sequence[float],
    *,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test: does augmentation improve AUPRC?

    Tests H0: augmented AUPRC <= baseline AUPRC
    Tests H1: augmented AUPRC > baseline AUPRC (one-sided by default)

    Args:
        baseline_scores: AUPRC values without augmentation (one per
            network/seed pair).
        augmented_scores: AUPRC values with augmentation (paired with
            baseline, same ordering).
        alternative: One of ``"greater"``, ``"less"``, or ``"two-sided"``.
            Passed directly to ``scipy.stats.wilcoxon``.

    Returns:
        Dict with keys:
        - ``statistic`` – Wilcoxon test statistic
        - ``p_value`` – p-value for the chosen alternative
        - ``effect_size_r`` – rank-biserial correlation (effect size)
        - ``significant_at_05`` – bool, True when p_value < 0.05
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy not available", "p_value": None,
                "statistic": None, "effect_size_r": None, "significant_at_05": None}

    base = np.asarray(list(baseline_scores), dtype=float)
    aug = np.asarray(list(augmented_scores), dtype=float)

    if base.shape != aug.shape:
        return {"error": "baseline_scores and augmented_scores must have the same length",
                "p_value": None, "statistic": None, "effect_size_r": None, "significant_at_05": None}

    if len(base) < 2:
        return {"error": "at least 2 paired observations required",
                "p_value": None, "statistic": None, "effect_size_r": None, "significant_at_05": None}

    differences = aug - base
    if np.all(differences == 0):
        return {"statistic": 0.0, "p_value": 1.0, "effect_size_r": 0.0, "significant_at_05": False}

    try:
        stat_result = wilcoxon(aug, base, alternative=alternative)
        statistic = float(stat_result.statistic)
        p_value = float(stat_result.pvalue)
    except Exception as exc:
        return {"error": str(exc), "p_value": None, "statistic": None,
                "effect_size_r": None, "significant_at_05": None}

    # Rank-biserial correlation as effect size: r = 1 - 2W / (n*(n+1)/2)
    n = len(base)
    denom = n * (n + 1) / 2.0
    effect_size_r = float(1.0 - 2.0 * statistic / denom) if denom > 0 else 0.0

    return {
        "statistic": statistic,
        "p_value": p_value,
        "effect_size_r": effect_size_r,
        "significant_at_05": bool(p_value < 0.05),
    }


def holm_bonferroni_correction(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down correction for multiple comparisons.

    Given a family of m hypothesis tests with raw p-values, the Holm procedure
    rejects hypotheses in order of ascending p-value, stopping at the first
    non-rejected hypothesis.

    Args:
        p_values: Raw (uncorrected) p-values, one per comparison.
        alpha: Family-wise error rate (default 0.05).
        labels: Optional human-readable labels for each comparison.

    Returns:
        List of dicts (one per comparison, sorted by raw p-value) with keys:
        - ``label`` – comparison label (or index)
        - ``raw_p`` – original p-value
        - ``adjusted_p`` – Holm-adjusted p-value (capped at 1.0)
        - ``rejected`` – True if the hypothesis is rejected after correction
    """
    m = len(p_values)
    if m == 0:
        return []

    raw = np.asarray(list(p_values), dtype=float)
    if labels is None:
        labels = [f"test_{i}" for i in range(m)]

    # Sort by ascending p-value
    order = np.argsort(raw)
    sorted_p = raw[order]
    sorted_labels = [labels[i] for i in order]

    # Holm-adjusted p-values: p_adj[k] = max(p_adj[k-1], (m - k) * p_raw[k])
    adjusted = np.zeros(m)
    for k in range(m):
        adjusted[k] = sorted_p[k] * (m - k)
    # Enforce monotonicity
    for k in range(1, m):
        adjusted[k] = max(adjusted[k], adjusted[k - 1])
    adjusted = np.minimum(adjusted, 1.0)

    results = []
    for k in range(m):
        results.append({
            "label": sorted_labels[k],
            "raw_p": float(sorted_p[k]),
            "adjusted_p": float(adjusted[k]),
            "rejected": bool(adjusted[k] < alpha),
        })
    return results


def spearman_plausibility_utility(
    plausibility_scores: Sequence[float],
    delta_auprc: Sequence[float],
) -> dict[str, Any]:
    """Spearman correlation between plausibility score and ΔAUPRC.

    Tests the core paper hypothesis: higher plausibility → better detection.

    Args:
        plausibility_scores: Per-experiment or per-augmenter plausibility
            scores (higher = more plausible synthetic samples).
        delta_auprc: Corresponding AUPRC improvements over baseline
            (augmented_auprc - baseline_auprc).

    Returns:
        Dict with keys:
        - ``rho``     – Spearman rank correlation coefficient
        - ``p_value`` – two-tailed p-value
        - ``n``       – number of paired observations
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {"error": "scipy not available", "rho": None, "p_value": None, "n": None}

    plaus = np.asarray(list(plausibility_scores), dtype=float)
    delta = np.asarray(list(delta_auprc), dtype=float)

    if plaus.shape != delta.shape:
        return {"error": "plausibility_scores and delta_auprc must have the same length",
                "rho": None, "p_value": None, "n": None}

    n = int(len(plaus))
    if n < 3:
        return {"error": "at least 3 observations required for Spearman correlation",
                "rho": None, "p_value": None, "n": n}

    try:
        result = spearmanr(plaus, delta)
        rho = float(result.statistic if hasattr(result, "statistic") else result.correlation)
        p_value = float(result.pvalue)
    except Exception as exc:
        return {"error": str(exc), "rho": None, "p_value": None, "n": n}

    return {"rho": rho, "p_value": p_value, "n": n}
