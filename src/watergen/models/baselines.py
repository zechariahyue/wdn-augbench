"""Baseline model interfaces for Plan A / Plan B.

This module now includes:
- dataframe -> feature-matrix helpers
- a simple detector baseline wrapper with sklearn and fallback paths
- LSTMAEDetector: LSTM-Autoencoder sequence anomaly detector (P5)
- WNTRResidualDetector: domain-baseline pressure-residual threshold detector (P8)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is expected in local env
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is expected in local env
    pd = None  # type: ignore[assignment]

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - fallback is supported
    _HAS_SKLEARN = False


@dataclass
class BaselineSpec:
    """Simple descriptor for a model baseline."""

    name: str
    category: str  # detector or generator
    notes: str


BASELINE_REGISTRY = [
    BaselineSpec(
        name="lstm_baseline",
        category="detector",
        notes="Plan B default detector baseline.",
    ),
    BaselineSpec(
        name="tcn_baseline",
        category="detector",
        notes="Plan A recommended detector baseline.",
    ),
    BaselineSpec(
        name="conditional_cvae",
        category="generator",
        notes="Plan A recommended initial generator.",
    ),
    BaselineSpec(
        name="timegan_or_cvae",
        category="generator",
        notes="Plan B simplified generator choice.",
    ),
]


def _ensure_dataframe_support() -> None:
    """Validate required dataframe dependencies."""
    if pd is None or np is None:
        raise ImportError("pandas and numpy are required for baseline dataframe helpers")


def select_numeric_feature_columns(
    frame: "pd.DataFrame",
    *,
    target_column: str = "event_label",
    exclude_columns: Iterable[str] = (),
) -> list[str]:
    """Select numeric feature columns from a dataframe.

    Args:
        frame: Source dataframe.
        target_column: Label column excluded from features.
        exclude_columns: Additional columns to exclude.
    """
    _ensure_dataframe_support()
    excluded = {target_column, *exclude_columns}
    numeric_columns = [
        name
        for name in frame.columns
        if name not in excluded and pd.api.types.is_numeric_dtype(frame[name])
    ]
    if not numeric_columns:
        raise ValueError("No numeric feature columns were found in the dataframe")
    return numeric_columns


def prepare_feature_matrix(
    frame: "pd.DataFrame",
    *,
    target_column: str = "event_label",
    feature_columns: Sequence[str] | None = None,
    fill_value: float = 0.0,
) -> tuple["np.ndarray", "np.ndarray", list[str]]:
    """Prepare ``X`` and ``y`` arrays from a dataframe.

    The helper keeps preprocessing lightweight:
    - chooses numeric columns by default
    - coerces feature values to numeric
    - fills missing values with ``fill_value``
    - normalizes labels to binary ints (0/1)
    """
    _ensure_dataframe_support()
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    selected = list(feature_columns) if feature_columns is not None else select_numeric_feature_columns(
        frame,
        target_column=target_column,
    )
    if not selected:
        raise ValueError("No feature columns provided")

    missing_features = [name for name in selected if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Feature columns missing from dataframe: {missing_features}")

    x_frame = frame.loc[:, selected].copy()
    for name in selected:
        x_frame[name] = pd.to_numeric(x_frame[name], errors="coerce")
    x_frame = x_frame.fillna(fill_value)

    y_series = pd.to_numeric(frame[target_column], errors="coerce").fillna(0).astype(int)
    y_series = y_series.clip(lower=0, upper=1)

    return x_frame.to_numpy(dtype=float), y_series.to_numpy(dtype=int), selected


@dataclass
class FallbackDetectorModel:
    """Very small fallback detector used when sklearn is unavailable."""

    majority_label: int = 0

    def fit(self, _: "np.ndarray", y: "np.ndarray") -> "FallbackDetectorModel":
        if y.size == 0:
            raise ValueError("Cannot fit fallback detector on empty labels")
        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        self.majority_label = 1 if positives >= negatives else 0
        return self

    def predict(self, x: "np.ndarray") -> "np.ndarray":
        if np is None:
            raise ImportError("numpy is required for fallback detector predictions")
        return np.full(shape=(x.shape[0],), fill_value=self.majority_label, dtype=int)

    def predict_proba(self, x: "np.ndarray") -> "np.ndarray":
        if np is None:
            raise ImportError("numpy is required for fallback detector predictions")
        proba_pos = float(self.majority_label)
        proba_neg = 1.0 - proba_pos
        return np.tile(np.array([proba_neg, proba_pos], dtype=float), (x.shape[0], 1))


class TabularDetectorBaseline:
    """Simple detector baseline for event detection from tabular features.

    Algorithms:
    - ``logreg``: LogisticRegression pipeline (default)
    - ``random_forest``: RandomForestClassifier
    """

    def __init__(
        self,
        *,
        algorithm: str = "logreg",
        random_state: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.random_state = random_state
        self.model: Any | None = None
        self.feature_columns_: list[str] = []

    def _build_model(self) -> Any:
        if not _HAS_SKLEARN:
            return FallbackDetectorModel()

        if self.algorithm == "random_forest":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            random_state=self.random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

        if self.algorithm != "logreg":
            raise ValueError(f"Unsupported detector algorithm: {self.algorithm}")

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, x: "np.ndarray", y: "np.ndarray") -> "TabularDetectorBaseline":
        """Fit detector from matrix inputs."""
        if np is None:
            raise ImportError("numpy is required for detector fitting")
        if x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        if y.ndim != 1:
            raise ValueError("y must be a 1D label array")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have matching row counts")

        self.model = self._build_model()
        self.model.fit(x, y)
        return self

    def fit_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
        feature_columns: Sequence[str] | None = None,
    ) -> "TabularDetectorBaseline":
        """Fit detector directly from a dataframe."""
        x, y, used_columns = prepare_feature_matrix(
            frame,
            target_column=target_column,
            feature_columns=feature_columns,
        )
        self.feature_columns_ = used_columns
        return self.fit(x, y)

    def predict(self, x: "np.ndarray") -> "np.ndarray":
        """Predict binary labels."""
        if self.model is None:
            raise RuntimeError("Detector is not fitted yet")
        return self.model.predict(x)

    def predict_proba(self, x: "np.ndarray") -> "np.ndarray":
        """Predict class probabilities for each row.

        Returns ``[p(class=0), p(class=1)]`` per row.
        """
        if self.model is None:
            raise RuntimeError("Detector is not fitted yet")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)

        # Safety fallback for models that do not expose predict_proba.
        pred = self.model.predict(x)
        if np is None:
            raise ImportError("numpy is required for probability conversion")
        return np.column_stack((1 - pred, pred))

    def predict_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
    ) -> "np.ndarray":
        """Predict labels from a dataframe using fitted feature columns."""
        if not self.feature_columns_:
            self.feature_columns_ = select_numeric_feature_columns(
                frame,
                target_column=target_column,
            )
        x, _, _ = prepare_feature_matrix(
            frame,
            target_column=target_column,
            feature_columns=self.feature_columns_,
        )
        return self.predict(x)


# ---------------------------------------------------------------------------
# Optional PyTorch import — available at module level for both new detectors
# ---------------------------------------------------------------------------
try:
    import torch as _torch
    import torch.nn as _nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - PyTorch is optional
    _torch = None  # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# P5: LSTM-Autoencoder sequence detector
# ---------------------------------------------------------------------------

class LSTMAEDetector:
    """LSTM-Autoencoder anomaly detector for time-series WDN data.

    Trains on normal-class sequences only (``event_label == 0``).  At inference,
    the mean per-timestep reconstruction error is used as the anomaly score.

    If PyTorch is not installed the detector transparently falls back to
    :class:`TabularDetectorBaseline` with ``algorithm='random_forest'``.

    Parameters
    ----------
    hidden_size:
        Number of LSTM hidden units.
    num_layers:
        Number of stacked LSTM layers.
    sequence_length:
        Number of timesteps per sequence window.  Sequences shorter than this
        are zero-padded on the right; longer ones are truncated.
    epochs:
        Training epochs.
    lr:
        Adam learning rate.
    batch_size:
        Mini-batch size during training.
    random_state:
        Seed for reproducibility (torch and numpy).
    device:
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        sequence_length: int = 24,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device
        self._model = None
        self._threshold: float | None = None
        self._feature_columns: list[str] = []
        self._fallback: TabularDetectorBaseline | None = None

        if not _HAS_TORCH:
            self._fallback = TabularDetectorBaseline(algorithm="random_forest")

    # ------------------------------------------------------------------
    # Internal PyTorch model (only instantiated when torch is available)
    # ------------------------------------------------------------------

    def _build_torch_model(self, n_features: int) -> "Any":
        """Construct the LSTM-AE nn.Module."""
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is not installed")

        hidden_size = self.hidden_size
        num_layers = self.num_layers

        class _LSTMAEModel(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = _nn.LSTM(
                    n_features,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.decoder = _nn.LSTM(
                    hidden_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.output_layer = _nn.Linear(hidden_size, n_features)

            def forward(self, x: "Any") -> "Any":
                # x: (batch, T, F)
                _, (hidden, cell) = self.encoder(x)
                # Use final hidden state, repeated T times, as decoder input
                T = x.shape[1]
                decoder_input = hidden[-1].unsqueeze(1).repeat(1, T, 1)
                decoded, _ = self.decoder(decoder_input, (hidden, cell))
                return self.output_layer(decoded)  # (batch, T, F)

        return _LSTMAEModel()

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------

    def _pad_or_truncate(self, arr: "np.ndarray") -> "np.ndarray":
        """Pad or truncate a (T, F) array to (sequence_length, F)."""
        T, F = arr.shape
        if T >= self.sequence_length:
            return arr[: self.sequence_length]
        pad = np.zeros((self.sequence_length - T, F), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    # ------------------------------------------------------------------
    # Public fit API
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: "np.ndarray",
        *,
        feature_columns: list[str] | None = None,
    ) -> "LSTMAEDetector":
        """Fit detector on stacked normal-class sequences.

        Parameters
        ----------
        sequences:
            Array of shape ``(N, T, F)`` — N runs, T timesteps, F features.
            Only normal-class (label=0) sequences should be passed here.
        feature_columns:
            Optional names for the F feature dimensions.
        """
        if np is None:
            raise ImportError("numpy is required for LSTMAEDetector")

        if feature_columns is not None:
            self._feature_columns = list(feature_columns)

        if self._fallback is not None:
            # PyTorch unavailable — build a flat 2-D matrix and delegate
            flat = sequences.reshape(sequences.shape[0], -1)
            dummy_y = np.zeros(flat.shape[0], dtype=int)
            self._fallback.fit(flat, dummy_y)
            return self

        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (N, T, F)")
        N, T, F = sequences.shape
        if N == 0:
            raise ValueError("Cannot fit on empty sequences")

        _torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._model = self._build_torch_model(F).to(self.device)
        optimizer = _torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = _nn.MSELoss()

        tensor = _torch.tensor(sequences, dtype=_torch.float32).to(self.device)

        self._model.train()
        for _ in range(self.epochs):
            indices = np.random.permutation(N)
            for start in range(0, N, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch = tensor[batch_idx]
                optimizer.zero_grad()
                reconstruction = self._model(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

        # Compute per-sample train reconstruction errors to set the threshold
        self._model.eval()
        with _torch.no_grad():
            recon = self._model(tensor)
            errors = (recon - tensor).pow(2).mean(dim=(1, 2)).cpu().numpy()
        self._threshold = float(np.percentile(errors, 95))
        return self

    def fit_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        scenario_id_col: str = "scenario_id",
        target_col: str = "event_label",
        feature_columns: list[str] | None = None,
    ) -> "LSTMAEDetector":
        """Fit from a tabular dataframe, grouping rows by scenario.

        Only rows with ``target_col == 0`` (normal class) are used.
        Each scenario group is padded/truncated to ``sequence_length`` timesteps.

        Parameters
        ----------
        frame:
            Aggregated or raw simulation dataframe.
        scenario_id_col:
            Column that identifies individual simulation runs.
        target_col:
            Binary label column (0 = normal, 1 = anomaly).
        feature_columns:
            Numeric sensor columns to use.  Auto-detected when ``None``.
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for LSTMAEDetector.fit_from_frame")

        normal = frame[frame[target_col] == 0].copy() if target_col in frame.columns else frame.copy()
        if normal.empty:
            raise ValueError("No normal-class rows found for fitting LSTMAEDetector")

        if feature_columns is None:
            feature_columns = [
                c for c in normal.columns
                if c not in {target_col, scenario_id_col, "time_seconds", "benchmark_id",
                             "scenario_kind", "split", "stage", "leak_node",
                             "disturbance_target", "event_label"}
                and pd.api.types.is_numeric_dtype(normal[c])
            ]
        if not feature_columns:
            raise ValueError("No numeric feature columns found in frame for LSTMAEDetector")

        self._feature_columns = list(feature_columns)

        # If the frame has a time_seconds column, sort by it within each group
        has_time = "time_seconds" in normal.columns

        seqs: list[np.ndarray] = []
        for _, group_df in normal.groupby(scenario_id_col):
            if has_time:
                group_df = group_df.sort_values("time_seconds")
            arr = group_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            seqs.append(self._pad_or_truncate(arr))

        if not seqs:
            raise ValueError("No sequences could be extracted from the frame")

        sequences = np.stack(seqs, axis=0)  # (N, T, F)

        # Fallback path: treat each timestep as an independent tabular sample
        if self._fallback is not None:
            return self.fit(sequences, feature_columns=feature_columns)

        return self.fit(sequences, feature_columns=feature_columns)

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------

    def _reconstruction_errors(self, x: "np.ndarray") -> "np.ndarray":
        """Return per-sample mean reconstruction error for shape (N, T, F)."""
        if self._fallback is not None:
            raise RuntimeError("_reconstruction_errors called on fallback path")
        if self._model is None:
            raise RuntimeError("LSTMAEDetector is not fitted yet")

        self._model.eval()
        with _torch.no_grad():
            tensor = _torch.tensor(x, dtype=_torch.float32).to(self.device)
            recon = self._model(tensor)
            errors = (recon - tensor).pow(2).mean(dim=(1, 2)).cpu().numpy()
        return errors

    def predict_proba(self, x: "np.ndarray") -> "np.ndarray":
        """Anomaly probability for each sample.

        Parameters
        ----------
        x:
            Array of shape ``(N, T, F)`` or ``(N, F)`` (treated as T=1 sequences).

        Returns
        -------
        np.ndarray
            Shape ``(N, 2)`` with columns ``[p_normal, p_anomaly]``.
        """
        if np is None:
            raise ImportError("numpy is required for LSTMAEDetector predictions")

        if self._fallback is not None:
            flat = x.reshape(x.shape[0], -1) if x.ndim == 3 else x
            return self._fallback.predict_proba(flat)

        if x.ndim == 2:
            x = x[:, np.newaxis, :]  # treat as single-timestep sequences

        errors = self._reconstruction_errors(x)
        threshold = self._threshold if self._threshold is not None else 1.0

        # Sigmoid-scaled score: 0.5 at threshold, rising above it
        scores = 1.0 / (1.0 + np.exp(-((errors - threshold) / max(threshold, 1e-9))))
        return np.column_stack((1.0 - scores, scores))

    def predict(self, x: "np.ndarray") -> "np.ndarray":
        """Return binary anomaly labels (1 = anomaly, 0 = normal).

        Parameters
        ----------
        x:
            Array of shape ``(N, T, F)`` or ``(N, F)``.
        """
        if np is None:
            raise ImportError("numpy is required for LSTMAEDetector predictions")

        if self._fallback is not None:
            flat = x.reshape(x.shape[0], -1) if x.ndim == 3 else x
            return self._fallback.predict(flat)

        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        scenario_id_col: str = "scenario_id",
        target_col: str = "event_label",
        feature_columns: list[str] | None = None,
    ) -> "np.ndarray":
        """Predict binary labels from a dataframe, grouping rows by scenario.

        Returns one prediction per *row* in ``frame``, in the original row order.
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for LSTMAEDetector.predict_from_frame")

        used_cols = feature_columns if feature_columns is not None else self._feature_columns
        if not used_cols:
            used_cols = [
                c for c in frame.columns
                if c not in {target_col, scenario_id_col, "time_seconds", "benchmark_id",
                             "scenario_kind", "split", "stage", "leak_node",
                             "disturbance_target", "event_label"}
                and pd.api.types.is_numeric_dtype(frame[c])
            ]

        has_time = "time_seconds" in frame.columns
        predictions = np.zeros(len(frame), dtype=int)
        original_index = frame.index.copy()

        for _, group_df in frame.groupby(scenario_id_col):
            if has_time:
                group_df = group_df.sort_values("time_seconds")
            arr = group_df[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            seq = self._pad_or_truncate(arr)[np.newaxis, ...]  # (1, T, F)
            pred = self.predict(seq)  # (1,)
            # Assign the single sequence-level prediction to every row in the group
            row_positions = [original_index.get_loc(idx) for idx in group_df.index]
            for pos in row_positions:
                predictions[pos] = int(pred[0])

        return predictions


# ---------------------------------------------------------------------------
# P8: WNTR Pressure Residual Domain Baseline Detector
# ---------------------------------------------------------------------------

class WNTRResidualDetector:
    """Domain engineering baseline: threshold anomaly detection on pressure residuals.

    Trains on normal-class baseline simulation data to learn expected pressure
    ranges per sensor column.  At inference, flags timesteps (or aggregated rows)
    where the observed pressure falls outside the learned bounds.

    This establishes the domain standard that ML methods must beat in order to
    claim practical value.

    Parameters
    ----------
    z_score_threshold:
        Number of standard deviations used to set bounds in z-score mode.
    use_iqr:
        When ``True`` (default), use ``Q1 - 1.5·IQR`` / ``Q3 + 1.5·IQR`` bounds
        (robust to outliers).  When ``False``, use ``mean ± z_score_threshold·std``.
    """

    def __init__(
        self,
        *,
        z_score_threshold: float = 3.0,
        use_iqr: bool = True,
    ) -> None:
        self.z_score_threshold = z_score_threshold
        self.use_iqr = use_iqr
        self._pressure_bounds: dict[str, tuple[float, float]] = {}
        self._global_mean: float = 0.0
        self._global_std: float = 1.0
        self.pressure_columns_: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_pressure_columns(self, frame: "pd.DataFrame") -> list[str]:
        """Auto-detect pressure-related columns from a dataframe."""
        return [c for c in frame.columns if "pressure" in c.lower() and pd.api.types.is_numeric_dtype(frame[c])]

    def _resolve_pressure_columns(
        self,
        frame: "pd.DataFrame",
        pressure_columns: list[str] | None,
    ) -> list[str]:
        if pressure_columns is not None:
            return [c for c in pressure_columns if c in frame.columns]
        if self.pressure_columns_:
            available = [c for c in self.pressure_columns_ if c in frame.columns]
            if available:
                return available
        return self._detect_pressure_columns(frame)

    # ------------------------------------------------------------------
    # Public fit API
    # ------------------------------------------------------------------

    def fit(
        self,
        frame: "pd.DataFrame",
        *,
        pressure_columns: list[str] | None = None,
        target_column: str = "event_label",
    ) -> "WNTRResidualDetector":
        """Learn pressure bounds from normal-class rows.

        Parameters
        ----------
        frame:
            Tabular simulation dataframe (aggregated or raw).
        pressure_columns:
            Columns to treat as pressure sensors.  Auto-detected when ``None``
            using any column whose name contains ``"pressure"``.
        target_column:
            Binary label column identifying normal (0) vs anomaly (1) rows.
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for WNTRResidualDetector")

        normal = frame[frame[target_column] == 0].copy() if target_column in frame.columns else frame.copy()
        if normal.empty:
            raise ValueError("No normal-class rows found for fitting WNTRResidualDetector")

        cols = self._resolve_pressure_columns(normal, pressure_columns)
        if not cols:
            raise ValueError("No pressure columns found in frame for WNTRResidualDetector")

        self.pressure_columns_ = list(cols)
        self._pressure_bounds = {}

        for col in cols:
            series = pd.to_numeric(normal[col], errors="coerce").dropna()
            if series.empty:
                self._pressure_bounds[col] = (-np.inf, np.inf)
                continue
            if self.use_iqr:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                iqr = q3 - q1
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
            else:
                mean = float(series.mean())
                std = float(series.std())
                if std < 1e-12:
                    std = 1e-12
                lo = mean - self.z_score_threshold * std
                hi = mean + self.z_score_threshold * std
            self._pressure_bounds[col] = (lo, hi)

        # Global stats for score normalisation (all normal-class pressure values)
        all_values = np.concatenate(
            [pd.to_numeric(normal[c], errors="coerce").dropna().to_numpy() for c in cols]
        )
        self._global_mean = float(np.mean(all_values)) if len(all_values) > 0 else 0.0
        self._global_std = float(np.std(all_values)) if len(all_values) > 0 else 1.0
        if self._global_std < 1e-12:
            self._global_std = 1.0

        return self

    def fit_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
        feature_columns: "Sequence[str] | None" = None,
    ) -> "WNTRResidualDetector":
        """Fit from a dataframe, matching the ``TabularDetectorBaseline`` interface.

        ``feature_columns`` is used as ``pressure_columns`` when provided.
        """
        pressure_cols = list(feature_columns) if feature_columns is not None else None
        return self.fit(frame, pressure_columns=pressure_cols, target_column=target_column)

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        frame: "pd.DataFrame",
        *,
        pressure_columns: list[str] | None = None,
    ) -> "np.ndarray":
        """Anomaly probability for each row in ``frame``.

        The anomaly score for a row is the fraction of pressure columns whose
        observed value lies outside the learned bounds.

        Returns
        -------
        np.ndarray
            Shape ``(N, 2)`` with columns ``[p_normal, p_anomaly]``.
        """
        if pd is None or np is None:
            raise ImportError("pandas and numpy are required for WNTRResidualDetector")
        if not self._pressure_bounds:
            raise RuntimeError("WNTRResidualDetector is not fitted yet")

        cols = self._resolve_pressure_columns(frame, pressure_columns)
        # Only keep columns that were seen during fitting
        cols = [c for c in cols if c in self._pressure_bounds]
        if not cols:
            # No pressure columns available — return zero anomaly score
            return np.column_stack((np.ones(len(frame)), np.zeros(len(frame))))

        violation_fracs = np.zeros(len(frame), dtype=float)
        for col in cols:
            lo, hi = self._pressure_bounds[col]
            values = pd.to_numeric(frame[col], errors="coerce").fillna(self._global_mean).to_numpy(dtype=float)
            outside = (values < lo) | (values > hi)
            violation_fracs += outside.astype(float)

        violation_fracs /= len(cols)
        return np.column_stack((1.0 - violation_fracs, violation_fracs))

    def predict(
        self,
        frame: "pd.DataFrame",
        *,
        pressure_columns: list[str] | None = None,
    ) -> "np.ndarray":
        """Return binary anomaly labels (1 = anomaly, 0 = normal).

        A row is flagged anomalous (1) when *any* pressure column lies outside
        its learned bounds.
        """
        if np is None:
            raise ImportError("numpy is required for WNTRResidualDetector predictions")

        proba = self.predict_proba(frame, pressure_columns=pressure_columns)
        return (proba[:, 1] > 0.0).astype(int)

    def predict_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
    ) -> "np.ndarray":
        """Predict labels from a dataframe, matching the ``TabularDetectorBaseline`` interface."""
        return self.predict(frame)


# ---------------------------------------------------------------------------
# Optional PyTorch Geometric import — for GNN detector
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data as _PyGData

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# GNN Detector Baseline
# ---------------------------------------------------------------------------

class GNNDetector:
    """GCN-based anomaly detector for tabular WDN features.

    Constructs a k-NN feature-similarity graph from training rows, then runs
    a 2-layer GCN for binary classification.  At inference, the test rows are
    appended to the training graph (inductive edges via k-NN to training nodes)
    and scores are extracted from the test node positions.

    Falls back to :class:`TabularDetectorBaseline` (RF) when PyTorch or
    PyTorch Geometric is unavailable.

    Parameters
    ----------
    hidden_dim:
        GCN hidden dimension.
    k_neighbors:
        Number of nearest neighbours for graph construction.
    epochs:
        Training epochs.
    lr:
        Adam learning rate.
    random_state:
        Seed for reproducibility.
    device:
        PyTorch device string.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        k_neighbors: int = 10,
        epochs: int = 100,
        lr: float = 1e-3,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.device = device
        self._model: Any = None
        self._scaler: Any = None
        self._train_x: "np.ndarray | None" = None
        self.feature_columns_: list[str] = []
        self._fallback: TabularDetectorBaseline | None = None

        if not (_HAS_TORCH and _HAS_PYG):
            self._fallback = TabularDetectorBaseline(algorithm="random_forest")

    @staticmethod
    def _knn_edge_index(x: "Any", k: int) -> "Any":
        """Build symmetric k-NN edge_index from a feature tensor."""
        # x: (N, F) torch tensor
        # Compute pairwise squared Euclidean distances
        dist = _torch.cdist(x, x, p=2)
        # For each node, find k nearest (excluding self)
        _, idx = dist.topk(k + 1, largest=False, dim=1)
        idx = idx[:, 1:]  # drop self
        n = x.shape[0]
        src = _torch.arange(n, device=x.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = idx.reshape(-1)
        # Make symmetric
        edge_index = _torch.stack([
            _torch.cat([src, dst]),
            _torch.cat([dst, src]),
        ], dim=0)
        return edge_index

    def _build_model(self, n_features: int) -> "Any":
        hidden_dim = self.hidden_dim

        class _GCNClassifier(_nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = GCNConv(n_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.classifier = _nn.Linear(hidden_dim, 2)

            def forward(self, x: "Any", edge_index: "Any") -> "Any":
                h = _torch.relu(self.conv1(x, edge_index))
                h = _torch.dropout(h, p=0.3, train=self.training)
                h = _torch.relu(self.conv2(h, edge_index))
                return self.classifier(h)

        return _GCNClassifier()

    def fit(self, x: "np.ndarray", y: "np.ndarray") -> "GNNDetector":
        if self._fallback is not None:
            self._fallback.fit(x, y)
            return self

        _torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Standardize
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        x_scaled = self._scaler.fit_transform(x)
        self._train_x = x_scaled.copy()

        x_t = _torch.tensor(x_scaled, dtype=_torch.float32, device=self.device)
        y_t = _torch.tensor(y, dtype=_torch.long, device=self.device)

        k = min(self.k_neighbors, x_t.shape[0] - 1)
        edge_index = self._knn_edge_index(x_t, k)

        self._model = self._build_model(x_t.shape[1]).to(self.device)
        optimizer = _torch.optim.Adam(self._model.parameters(), lr=self.lr)

        # Class weights for imbalanced data
        n_pos = max(1, int((y == 1).sum()))
        n_neg = max(1, int((y == 0).sum()))
        weight = _torch.tensor([1.0 / n_neg, 1.0 / n_pos], device=self.device)
        weight = weight / weight.sum() * 2.0
        loss_fn = _nn.CrossEntropyLoss(weight=weight)

        self._model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self._model(x_t, edge_index)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def fit_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
        feature_columns: "Sequence[str] | None" = None,
    ) -> "GNNDetector":
        x, y, used_columns = prepare_feature_matrix(
            frame, target_column=target_column, feature_columns=feature_columns,
        )
        self.feature_columns_ = used_columns
        return self.fit(x, y)

    def predict_proba(self, x: "np.ndarray") -> "np.ndarray":
        if self._fallback is not None:
            return self._fallback.predict_proba(x)

        x_scaled = self._scaler.transform(x)
        # Build inductive graph: training nodes + test nodes
        x_all = np.concatenate([self._train_x, x_scaled], axis=0)
        x_t = _torch.tensor(x_all, dtype=_torch.float32, device=self.device)

        k = min(self.k_neighbors, x_t.shape[0] - 1)
        edge_index = self._knn_edge_index(x_t, k)

        self._model.eval()
        with _torch.no_grad():
            logits = self._model(x_t, edge_index)
            proba = _torch.softmax(logits, dim=1).cpu().numpy()

        # Return only test node probabilities
        n_train = self._train_x.shape[0]
        return proba[n_train:]

    def predict(self, x: "np.ndarray") -> "np.ndarray":
        proba = self.predict_proba(x)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_from_frame(
        self,
        frame: "pd.DataFrame",
        *,
        target_column: str = "event_label",
    ) -> "np.ndarray":
        if not self.feature_columns_:
            self.feature_columns_ = select_numeric_feature_columns(
                frame, target_column=target_column,
            )
        x, _, _ = prepare_feature_matrix(
            frame, target_column=target_column, feature_columns=self.feature_columns_,
        )
        return self.predict(x)
