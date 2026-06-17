"""Schema definitions for water monitoring experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List


STRICT_REQUIRED_FEATURES = ["timestamp", "network_id", "pressure", "flow"]


@dataclass
class FeatureSchema:
    """Defines required and optional model features."""

    required: List[str] = field(default_factory=lambda: STRICT_REQUIRED_FEATURES.copy())
    optional: List[str] = field(
        default_factory=lambda: [
            "tank_level",
            "demand",
            "pump_state",
            "valve_state",
        ]
    )
    target: str = "event_label"

    def validate(self) -> None:
        missing = [name for name in STRICT_REQUIRED_FEATURES if name not in self.required]
        if missing:
            raise ValueError(f"Missing required core features: {missing}")

        if len(set(self.required)) != len(self.required):
            raise ValueError("Required features contain duplicates")

        if len(set(self.optional)) != len(self.optional):
            raise ValueError("Optional features contain duplicates")

        overlap = set(self.required).intersection(self.optional)
        if overlap:
            raise ValueError(f"Feature names overlap between required and optional: {sorted(overlap)}")

        if self.target in self.required or self.target in self.optional:
            raise ValueError("Target column must not appear in required/optional features")

    def validate_observed_columns(self, columns: Iterable[str]) -> None:
        observed = {name.strip() for name in columns}
        missing = [name for name in self.required if name not in observed]
        if missing:
            raise ValueError(f"Observed data is missing required features: {missing}")


@dataclass
class SplitConfig:
    """Defines split protocol used by experiments."""

    strategy: str = "network_holdout"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    def validate(self) -> None:
        fractions = [self.train_fraction, self.val_fraction, self.test_fraction]
        if any(value <= 0 for value in fractions):
            raise ValueError("Split fractions must be greater than zero")

        if abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0")


@dataclass
class SimulationConfig:
    """Defines smoke-experiment simulation settings."""

    smoke_splits: List[str] = field(default_factory=lambda: ["train", "validation"])
    leak_area_values: List[float] = field(default_factory=lambda: [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005])
    max_candidates_per_network: int = 5
    include_baseline: bool = True
