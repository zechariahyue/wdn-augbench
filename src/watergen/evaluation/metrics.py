"""Small metric helpers without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

try:
    from sklearn.metrics import average_precision_score, roc_auc_score

    _HAS_SKLEARN_METRICS = True
except ImportError:  # pragma: no cover - optional dependency path
    _HAS_SKLEARN_METRICS = False


@dataclass
class BinaryMetrics:
    precision: float
    recall: float
    f1: float


@dataclass
class BinaryReport:
    """Extended binary classification report for first-pass experiments."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    balanced_accuracy: float
    tp: int
    fp: int
    tn: int
    fn: int
    support: int
    auroc: float | None = None
    auprc: float | None = None


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> tuple[int, int, int, int]:
    """Return confusion-matrix counts as ``(tp, fp, tn, fn)``."""
    true = list(y_true)
    pred = list(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")

    tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
    return tp, fp, tn, fn


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> BinaryMetrics:
    """Compute precision, recall, and F1 for binary labels."""
    tp, fp, _, fn = confusion_counts(y_true, y_pred)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return BinaryMetrics(precision=precision, recall=recall, f1=f1)


def binary_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
) -> BinaryReport:
    """Compute an extended binary report for experiment logging.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        y_score: Optional positive-class scores for AUROC/AUPRC.
    """
    true = list(y_true)
    pred = list(y_pred)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")

    tp, fp, tn, fn = confusion_counts(true, pred)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, len(true))
    balanced_accuracy = (recall + specificity) / 2.0

    auroc: float | None = None
    auprc: float | None = None
    if y_score is not None and _HAS_SKLEARN_METRICS:
        scores = list(y_score)
        if len(scores) != len(true):
            raise ValueError("y_score must have the same length as y_true")
        unique_labels = set(true)
        # AUROC/AUPRC are undefined for single-class truth vectors.
        if unique_labels == {0, 1}:
            auroc = float(roc_auc_score(true, scores))
            auprc = float(average_precision_score(true, scores))

    return BinaryReport(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        specificity=specificity,
        balanced_accuracy=balanced_accuracy,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        support=len(true),
        auroc=auroc,
        auprc=auprc,
    )


def report_as_dict(report: BinaryReport) -> dict[str, float | int | None]:
    """Convert a ``BinaryReport`` dataclass into a plain serializable dictionary."""
    return {
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "accuracy": report.accuracy,
        "specificity": report.specificity,
        "balanced_accuracy": report.balanced_accuracy,
        "tp": report.tp,
        "fp": report.fp,
        "tn": report.tn,
        "fn": report.fn,
        "support": report.support,
        "auroc": report.auroc,
        "auprc": report.auprc,
    }
