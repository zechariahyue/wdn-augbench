"""Lightweight tabular augmentation and plausibility filtering utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture

    _HAS_GMM = True
except ImportError:  # pragma: no cover - optional dependency path
    _HAS_GMM = False

try:
    from imblearn.over_sampling import SMOTE as _SMOTE  # type: ignore[import-untyped]
    from imblearn.over_sampling import ADASYN as _ADASYN  # type: ignore[import-untyped]

    _HAS_IMBLEARN = True
except ImportError:  # pragma: no cover - optional dependency path
    _HAS_IMBLEARN = False

try:
    from sdv.single_table import CTGANSynthesizer as _CTGANSynthesizer  # type: ignore[import-untyped]
    from sdv.single_table import TVAESynthesizer as _TVAESynthesizer  # type: ignore[import-untyped]

    _HAS_SDV = True
    _SDV_CTGAN = _CTGANSynthesizer
    _SDV_TVAE = _TVAESynthesizer
except ImportError:  # pragma: no cover - optional dependency path
    _HAS_SDV = False
    try:
        from ctgan import CTGANSynthesizer as _CTGANSynthesizer  # type: ignore[import-untyped]

        _HAS_CTGAN_PKG = True
        _SDV_CTGAN = _CTGANSynthesizer  # type: ignore[assignment]
    except ImportError:  # pragma: no cover - optional dependency path
        _HAS_CTGAN_PKG = False
    _SDV_TVAE = None  # type: ignore[assignment]


@dataclass
class QuantilePlausibilityFilter:
    """Simple feature-wise plausibility filter based on train quantile bounds."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    margin_scale: float = 0.10

    def __post_init__(self) -> None:
        self.bounds_: dict[str, tuple[float, float]] = {}

    def fit(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> "QuantilePlausibilityFilter":
        bounds: dict[str, tuple[float, float]] = {}
        for name in feature_columns:
            series = pd.to_numeric(frame[name], errors="coerce").dropna()
            if series.empty:
                continue
            lower = float(series.quantile(self.lower_quantile))
            upper = float(series.quantile(self.upper_quantile))
            spread = max(upper - lower, 1e-9)
            margin = spread * self.margin_scale
            bounds[name] = (lower - margin, upper + margin)
        self.bounds_ = bounds
        return self

    def filter(self, frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
        if not self.bounds_:
            raise RuntimeError("Plausibility filter must be fitted before use")

        mask = pd.Series(True, index=frame.index)
        for name in feature_columns:
            if name not in self.bounds_:
                continue
            lower, upper = self.bounds_[name]
            values = pd.to_numeric(frame[name], errors="coerce")
            mask &= values.between(lower, upper, inclusive="both")
        return frame.loc[mask].copy()


class PositiveClassAugmenter:
    """Create lightweight synthetic positive samples from tabular training data."""

    def __init__(self, *, random_state: int = 42, noise_scale: float = 0.05) -> None:
        self.random_state = random_state
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(random_state)

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        multiplier: float = 1.0,
        preserve_columns: Sequence[str] = (),
    ) -> pd.DataFrame:
        positive = train_frame.loc[train_frame[target_column] == 1].copy()
        if positive.empty:
            return positive

        n_samples = max(1, int(round(len(positive) * multiplier)))
        sampled = positive.sample(n=n_samples, replace=True, random_state=self.random_state).reset_index(drop=True)
        preserved = set(preserve_columns)

        for name in feature_columns:
            if name in preserved:
                continue
            series = pd.to_numeric(positive[name], errors="coerce")
            std = float(series.std()) if series.notna().any() else 0.0
            noise = self._rng.normal(loc=0.0, scale=max(std * self.noise_scale, 1e-9), size=n_samples)
            sampled[name] = pd.to_numeric(sampled[name], errors="coerce").fillna(0.0) + noise

        sampled[target_column] = 1
        sampled["scenario_kind"] = sampled.get("scenario_kind", "synthetic_positive")
        sampled["scenario_id"] = [f"synthetic_positive_{idx}" for idx in range(n_samples)]
        sampled["is_synthetic"] = 1
        return sampled


class GaussianMixtureAugmenter:
    """Learn a simple positive-class tabular generator with Gaussian mixtures."""

    def __init__(self, *, random_state: int = 42, n_components: int = 2) -> None:
        self.random_state = random_state
        self.n_components = n_components
        self.model_: GaussianMixture | None = None

    def fit(self, train_frame: pd.DataFrame, *, feature_columns: Sequence[str], target_column: str = "event_label") -> "GaussianMixtureAugmenter":
        if not _HAS_GMM:
            raise ImportError("scikit-learn GaussianMixture is required for GaussianMixtureAugmenter")

        positive = train_frame.loc[train_frame[target_column] == 1].copy()
        if len(positive) < 2:
            raise ValueError("Need at least two positive rows to fit a Gaussian mixture augmenter")

        x = positive.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        components = max(1, min(self.n_components, len(positive)))
        self.model_ = GaussianMixture(
            n_components=components,
            covariance_type="full",
            random_state=self.random_state,
        )
        self.model_.fit(x)
        return self

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
        preserve_template_columns: Sequence[str] = (),
    ) -> pd.DataFrame:
        if self.model_ is None:
            self.fit(train_frame, feature_columns=feature_columns, target_column=target_column)

        positive = train_frame.loc[train_frame[target_column] == 1].copy()
        if positive.empty:
            return positive

        sample_count = n_samples if n_samples is not None else len(positive)
        generated, _ = self.model_.sample(sample_count)
        generated_df = pd.DataFrame(generated, columns=list(feature_columns))

        template = positive.sample(n=sample_count, replace=True, random_state=self.random_state).reset_index(drop=True)
        output = template.copy()
        for name in feature_columns:
            output[name] = generated_df[name]

        output[target_column] = 1
        output["scenario_kind"] = "synthetic_gmm_positive"
        output["scenario_id"] = [f"synthetic_gmm_positive_{idx}" for idx in range(sample_count)]
        output["is_synthetic"] = 1

        for name in preserve_template_columns:
            if name in template.columns:
                output[name] = template[name].values

        return output


class SMOTEAugmenter:
    """SMOTE-based positive-class augmenter using imbalanced-learn.

    Falls back to PositiveClassAugmenter noise if imbalanced-learn is unavailable.
    """

    def __init__(self, *, random_state: int = 42, k_neighbors: int = 5) -> None:
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self._fitted = False

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Generate SMOTE synthetic samples for the positive class.

        Returns a DataFrame of synthetic positive-class rows only,
        with is_synthetic=1 and scenario_kind='synthetic_smote_positive'.
        """
        if not _HAS_IMBLEARN:
            import sys

            print(
                "[warn] imbalanced-learn not available; SMOTEAugmenter falling back to PositiveClassAugmenter",
                file=sys.stderr,
            )
            positive = train_frame.loc[train_frame[target_column] == 1]
            multiplier = 1.0 if n_samples is None else n_samples / max(len(positive), 1)
            fallback = PositiveClassAugmenter(random_state=self.random_state)
            result = fallback.generate(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                multiplier=multiplier,
            )
            result["scenario_kind"] = "synthetic_smote_positive"
            result["scenario_id"] = [f"synthetic_smote_positive_{i}" for i in range(len(result))]
            return result

        positive = train_frame.loc[train_frame[target_column] == 1]
        negative = train_frame.loc[train_frame[target_column] == 0]

        n_positive = len(positive)
        n_negative = len(negative)

        if n_positive < 2:
            return positive.iloc[:0].copy()

        # Determine effective k_neighbors — must be < n_positive
        effective_k = min(self.k_neighbors, n_positive - 1)

        # Determine sampling_strategy: how many positive samples to end up with
        if n_samples is None:
            target_positive = n_negative  # balance
        else:
            target_positive = n_positive + n_samples

        # sampling_strategy is the ratio of minority to majority after resampling,
        # or a dict {class_label: desired_count}
        sampling_strategy = {1: max(target_positive, n_positive)}

        x = train_frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = train_frame[target_column].to_numpy()

        smote = _SMOTE(  # type: ignore[possibly-undefined]
            sampling_strategy=sampling_strategy,
            k_neighbors=effective_k,
            random_state=self.random_state,
        )
        x_res, y_res = smote.fit_resample(x, y)

        # Rows beyond the original length are newly generated
        original_len = len(train_frame)
        x_new = x_res[original_len:]
        y_new = y_res[original_len:]

        if len(x_new) == 0:
            return positive.iloc[:0].copy()

        generated_df = pd.DataFrame(x_new, columns=list(feature_columns))
        generated_df[target_column] = y_new
        generated_df["scenario_kind"] = "synthetic_smote_positive"
        generated_df["scenario_id"] = [f"synthetic_smote_positive_{i}" for i in range(len(generated_df))]
        generated_df["is_synthetic"] = 1
        return generated_df


class ADASYNAugmenter:
    """ADASYN-based positive-class augmenter using imbalanced-learn.

    Falls back to PositiveClassAugmenter noise if imbalanced-learn is unavailable.
    """

    def __init__(self, *, random_state: int = 42, n_neighbors: int = 5) -> None:
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self._fitted = False

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Generate ADASYN synthetic samples for the positive class.

        Returns a DataFrame of synthetic positive-class rows only,
        with is_synthetic=1 and scenario_kind='synthetic_adasyn_positive'.
        """
        if not _HAS_IMBLEARN:
            import sys

            print(
                "[warn] imbalanced-learn not available; ADASYNAugmenter falling back to PositiveClassAugmenter",
                file=sys.stderr,
            )
            positive = train_frame.loc[train_frame[target_column] == 1]
            multiplier = 1.0 if n_samples is None else n_samples / max(len(positive), 1)
            fallback = PositiveClassAugmenter(random_state=self.random_state)
            result = fallback.generate(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                multiplier=multiplier,
            )
            result["scenario_kind"] = "synthetic_adasyn_positive"
            result["scenario_id"] = [f"synthetic_adasyn_positive_{i}" for i in range(len(result))]
            return result

        positive = train_frame.loc[train_frame[target_column] == 1]
        negative = train_frame.loc[train_frame[target_column] == 0]

        n_positive = len(positive)
        n_negative = len(negative)

        if n_positive < 2:
            return positive.iloc[:0].copy()

        effective_n = min(self.n_neighbors, n_positive - 1)

        if n_samples is None:
            sampling_strategy = "auto"  # balance classes
        else:
            target_positive = n_positive + n_samples
            sampling_strategy = {1: max(target_positive, n_positive)}  # type: ignore[assignment]

        x = train_frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = train_frame[target_column].to_numpy()

        adasyn = _ADASYN(  # type: ignore[possibly-undefined]
            sampling_strategy=sampling_strategy,
            n_neighbors=effective_n,
            random_state=self.random_state,
        )
        x_res, y_res = adasyn.fit_resample(x, y)

        original_len = len(train_frame)
        x_new = x_res[original_len:]
        y_new = y_res[original_len:]

        if len(x_new) == 0:
            return positive.iloc[:0].copy()

        generated_df = pd.DataFrame(x_new, columns=list(feature_columns))
        generated_df[target_column] = y_new
        generated_df["scenario_kind"] = "synthetic_adasyn_positive"
        generated_df["scenario_id"] = [f"synthetic_adasyn_positive_{i}" for i in range(len(generated_df))]
        generated_df["is_synthetic"] = 1
        return generated_df


class CTGANAugmenter:
    """CTGAN-based deep tabular generative model augmenter.

    Uses the SDV (Synthetic Data Vault) library.
    Falls back to GaussianMixtureAugmenter if sdv is unavailable.
    """

    def __init__(self, *, random_state: int = 42, epochs: int = 100, batch_size: int = 500) -> None:
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self._feature_columns: Sequence[str] = []

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
    ) -> "CTGANAugmenter":
        """Fit CTGAN on positive-class rows only."""
        _has_ctgan_pkg = globals().get("_HAS_CTGAN_PKG", False)
        if not _HAS_SDV and not _has_ctgan_pkg:
            raise ImportError("sdv or ctgan package is required for CTGANAugmenter")

        positive = train_frame.loc[train_frame[target_column] == 1, list(feature_columns)].copy()
        positive = positive.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        self._feature_columns = feature_columns

        if _HAS_SDV:
            import sdv.metadata as _meta

            metadata = _meta.SingleTableMetadata()
            metadata.detect_from_dataframe(positive)
            self.model_ = _SDV_CTGAN(  # type: ignore[possibly-undefined]
                metadata=metadata,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            self.model_.fit(positive)
        else:
            # Older ctgan package API
            self.model_ = _SDV_CTGAN(epochs=self.epochs, batch_size=self.batch_size)  # type: ignore[possibly-undefined]
            self.model_.fit(positive)

        return self

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Generate CTGAN synthetic positive samples."""
        if not _HAS_SDV:
            _has_ctgan = False
            try:
                import ctgan as _ctgan_pkg  # type: ignore[import-untyped]  # noqa: F401

                _has_ctgan = True
            except ImportError:
                pass

            if not _has_ctgan:
                import sys

                print(
                    "[warn] sdv/ctgan not available; CTGANAugmenter falling back to GaussianMixtureAugmenter",
                    file=sys.stderr,
                )
                fallback = GaussianMixtureAugmenter(random_state=self.random_state)
                result = fallback.generate(
                    train_frame,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    n_samples=n_samples,
                )
                result["scenario_kind"] = "synthetic_ctgan_positive"
                result["scenario_id"] = [f"synthetic_ctgan_positive_{i}" for i in range(len(result))]
                return result

        if self.model_ is None:
            self.fit(train_frame, feature_columns=feature_columns, target_column=target_column)

        positive = train_frame.loc[train_frame[target_column] == 1]
        sample_count = n_samples if n_samples is not None else len(positive)

        if _HAS_SDV:
            generated = self.model_.sample(num_rows=sample_count)
        else:
            generated = self.model_.sample(sample_count)

        # Keep only feature columns that exist in the generated output
        available_cols = [c for c in feature_columns if c in generated.columns]
        generated_df = generated[available_cols].copy()

        # Clip numeric values to [0, inf] — no negative pressures/flows
        for col in available_cols:
            generated_df[col] = pd.to_numeric(generated_df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

        generated_df[target_column] = 1
        generated_df["scenario_kind"] = "synthetic_ctgan_positive"
        generated_df["scenario_id"] = [f"synthetic_ctgan_positive_{i}" for i in range(len(generated_df))]
        generated_df["is_synthetic"] = 1
        return generated_df


class TVAEAugmenter:
    """TVAE-based deep tabular generative model augmenter.

    Uses the SDV (Synthetic Data Vault) library.
    Falls back to GaussianMixtureAugmenter if sdv is unavailable.
    """

    def __init__(self, *, random_state: int = 42, epochs: int = 100, batch_size: int = 500) -> None:
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self._feature_columns: Sequence[str] = []

    def fit(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
    ) -> "TVAEAugmenter":
        """Fit TVAE on positive-class rows only."""
        if not _HAS_SDV:
            raise ImportError("sdv package is required for TVAEAugmenter")

        positive = train_frame.loc[train_frame[target_column] == 1, list(feature_columns)].copy()
        positive = positive.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        self._feature_columns = feature_columns

        import sdv.metadata as _meta

        metadata = _meta.SingleTableMetadata()
        metadata.detect_from_dataframe(positive)
        self.model_ = _SDV_TVAE(  # type: ignore[possibly-undefined]
            metadata=metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        self.model_.fit(positive)
        return self

    def generate(
        self,
        train_frame: pd.DataFrame,
        *,
        feature_columns: Sequence[str],
        target_column: str = "event_label",
        n_samples: int | None = None,
    ) -> pd.DataFrame:
        """Generate TVAE synthetic positive samples."""
        if not _HAS_SDV:
            import sys

            print(
                "[warn] sdv not available; TVAEAugmenter falling back to GaussianMixtureAugmenter",
                file=sys.stderr,
            )
            fallback = GaussianMixtureAugmenter(random_state=self.random_state)
            result = fallback.generate(
                train_frame,
                feature_columns=feature_columns,
                target_column=target_column,
                n_samples=n_samples,
            )
            result["scenario_kind"] = "synthetic_tvae_positive"
            result["scenario_id"] = [f"synthetic_tvae_positive_{i}" for i in range(len(result))]
            return result

        if self.model_ is None:
            self.fit(train_frame, feature_columns=feature_columns, target_column=target_column)

        positive = train_frame.loc[train_frame[target_column] == 1]
        sample_count = n_samples if n_samples is not None else len(positive)

        generated = self.model_.sample(num_rows=sample_count)

        available_cols = [c for c in feature_columns if c in generated.columns]
        generated_df = generated[available_cols].copy()

        # Clip numeric values to [0, inf] — no negative pressures/flows
        for col in available_cols:
            generated_df[col] = pd.to_numeric(generated_df[col], errors="coerce").fillna(0.0).clip(lower=0.0)

        generated_df[target_column] = 1
        generated_df["scenario_kind"] = "synthetic_tvae_positive"
        generated_df["scenario_id"] = [f"synthetic_tvae_positive_{i}" for i in range(len(generated_df))]
        generated_df["is_synthetic"] = 1
        return generated_df
