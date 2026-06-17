"""Model placeholders for detector and generator baselines."""

from .augmentation import (
    ADASYNAugmenter,
    CTGANAugmenter,
    GaussianMixtureAugmenter,
    PositiveClassAugmenter,
    QuantilePlausibilityFilter,
    SMOTEAugmenter,
    TVAEAugmenter,
)
from .baselines import (
    BASELINE_REGISTRY,
    BaselineSpec,
    LSTMAEDetector,
    TabularDetectorBaseline,
    WNTRResidualDetector,
    prepare_feature_matrix,
    select_numeric_feature_columns,
)
from .graph_cvae import GraphCVAEAugmenter
from .topology_gnn import TopologyAwareGNNDetector

__all__ = [
    "BASELINE_REGISTRY",
    "ADASYNAugmenter",
    "BaselineSpec",
    "CTGANAugmenter",
    "GaussianMixtureAugmenter",
    "GraphCVAEAugmenter",
    "LSTMAEDetector",
    "PositiveClassAugmenter",
    "QuantilePlausibilityFilter",
    "SMOTEAugmenter",
    "TVAEAugmenter",
    "TabularDetectorBaseline",
    "TopologyAwareGNNDetector",
    "WNTRResidualDetector",
    "prepare_feature_matrix",
    "select_numeric_feature_columns",
]

