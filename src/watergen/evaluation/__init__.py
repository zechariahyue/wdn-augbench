"""Evaluation utilities for detector and augmentation experiments."""

from .metrics import (
    BinaryMetrics,
    BinaryReport,
    binary_metrics,
    binary_report,
    confusion_counts,
    report_as_dict,
)
from .plausibility import (
    HydraulicResidualScorer,
    PlausibilityReport,
    RowwisePlausibilityResult,
    WNTRFeasibilityChecker,
    compare_augmenter_plausibility,
)
from .statistics import (
    MultiSeedSummary,
    SeedManager,
    aggregate_multi_seed_results,
    bootstrap_auprc_ci,
    spearman_plausibility_utility,
    wilcoxon_augmentation_test,
)
from .scenario_metrics import (
    ScenarioExample,
    ScenarioMetricReport,
    hierarchical_bootstrap_auprc,
    scenario_level_report,
    summarize_by_disturbance,
)
from .triage import (
    PhysicallyInformedTriageModel,
    TriageResult,
    TRIAGE_FEATURE_COLUMNS,
    build_triage_frame,
    heuristic_triage_score,
)

__all__ = [
    "BinaryMetrics",
    "BinaryReport",
    "HydraulicResidualScorer",
    "MultiSeedSummary",
    "PhysicallyInformedTriageModel",
    "PlausibilityReport",
    "RowwisePlausibilityResult",
    "ScenarioExample",
    "ScenarioMetricReport",
    "SeedManager",
    "TRIAGE_FEATURE_COLUMNS",
    "TriageResult",
    "WNTRFeasibilityChecker",
    "aggregate_multi_seed_results",
    "binary_metrics",
    "binary_report",
    "bootstrap_auprc_ci",
    "build_triage_frame",
    "compare_augmenter_plausibility",
    "confusion_counts",
    "heuristic_triage_score",
    "hierarchical_bootstrap_auprc",
    "report_as_dict",
    "scenario_level_report",
    "spearman_plausibility_utility",
    "summarize_by_disturbance",
    "wilcoxon_augmentation_test",
]
