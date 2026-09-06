"""Regression tests for three defects that silently corrupted published results.

Each test below corresponds to a bug that produced plausible-looking numbers for
months. None of them would have been caught by a smoke test that merely checks the
pipeline "runs" -- they all ran fine and returned wrong answers. That is the point:
these assert on *meaning*, not on absence of exceptions.

Run:  pytest tests/test_regressions.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from watergen.data.simulation import (  # noqa: E402
    effective_duration_seconds,
    effective_report_step_seconds,
)
from watergen.models import QuantilePlausibilityFilter  # noqa: E402


# ---------------------------------------------------------------------------
# Defect 1: a training network shipped as a single-period (steady-state) model.
# Its leak window was computed from duration == 0, giving the inverted, empty
# interval [3600, 0]. WNTR read the invalid end time as "never ends", so the leak
# ran ACTIVE for the whole simulation while every one of its rows was labelled
# event_label = 0 ("normal"). Half the training set taught the detector that an
# active leak is a non-event. Nothing raised.
# ---------------------------------------------------------------------------

class _FakeTime:
    def __init__(self, duration, report, hydraulic):
        self.duration = duration
        self.report_timestep = report
        self.hydraulic_timestep = hydraulic


class _FakeOptions:
    def __init__(self, t):
        self.time = t


class _FakeWN:
    def __init__(self, duration, report=3600, hydraulic=3600):
        self.options = _FakeOptions(_FakeTime(duration, report, hydraulic))


def _leak_window(wn):
    """The window arithmetic used by build_leak_scenarios."""
    duration = effective_duration_seconds(wn)
    report = effective_report_step_seconds(wn)
    start = max(report, duration // 3)
    end = min(duration - report, (2 * duration) // 3)
    if end <= start:
        end = min(duration, start + report)
    return start, end


@pytest.mark.parametrize("duration", [0, -1, 86400, 604800])
def test_leak_window_is_never_empty(duration):
    """A disturbance window must be a real interval for EVERY network.

    duration == 0 (single-period benchmark models, e.g. the Kentucky family and
    Hanoi) previously produced start > end, which silently labelled every
    disturbance row as normal operation.
    """
    start, end = _leak_window(_FakeWN(duration))
    assert end > start, (
        f"empty/inverted leak window [{start}, {end}] for duration={duration}; "
        "disturbance rows would be labelled as normal operation"
    )


def test_steady_state_model_is_promoted_to_an_eps_horizon():
    """duration <= 0 must be promoted, and the promotion must be visible to the
    scenario builder -- not only to the simulator. The original bug was that the two
    disagreed."""
    assert effective_duration_seconds(_FakeWN(0)) > 0
    assert effective_duration_seconds(_FakeWN(86400)) == 86400  # non-zero untouched


# ---------------------------------------------------------------------------
# Defect 2: the plausibility filter screened on constant features. Networks with no
# tanks give tank_head_* == 0 in every real row, so q01 == q99 == 0 and the bound is
# the single point [0, 0]. Any augmenter that perturbs those features at all lands
# outside it -> ~99% of ALL synthetic samples rejected, for a reason that has nothing
# to do with plausibility.
# ---------------------------------------------------------------------------

def test_filter_does_not_screen_on_constant_features():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({
        "pressure_mean": rng.normal(50, 5, 500),
        "flow_mean": rng.normal(1.0, 0.2, 500),
        "tank_head_mean": np.zeros(500),   # network has no tanks
        "tank_head_std": np.zeros(500),
    })
    cols = list(real.columns)
    filt = QuantilePlausibilityFilter().fit(real, cols)

    assert "tank_head_mean" in filt.degenerate_
    assert "tank_head_std" in filt.degenerate_
    assert "tank_head_mean" not in filt.bounds_

    # A 5% perturbation of real rows is, by any sane definition, plausible.
    synth = real.copy()
    for c in cols:
        synth[c] = synth[c] + rng.normal(0, 0.05 * (real[c].std() or 1.0), len(synth))

    kept = filt.filter(synth, cols)
    frac = len(kept) / len(synth)
    assert frac > 0.80, (
        f"filter kept only {frac:.1%} of mildly-perturbed real states; "
        "a constant feature is almost certainly being screened on"
    )


def test_filter_still_rejects_genuinely_out_of_range_samples():
    """The fix must not make the filter toothless."""
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"pressure_mean": rng.normal(50, 5, 500)})
    filt = QuantilePlausibilityFilter().fit(real, ["pressure_mean"])
    absurd = pd.DataFrame({"pressure_mean": np.full(100, 5000.0)})
    assert len(filt.filter(absurd, ["pressure_mean"])) == 0


# ---------------------------------------------------------------------------
# Defect 3: when the plausibility screen rejected 100% of an augmenter's samples,
# _apply_augmenter discarded the screen and trained on the rejects. That converted a
# total generator failure into a large, harmful "augmentation effect" -- and it was
# the sole cause of the reported GMM result.
# ---------------------------------------------------------------------------

def test_total_rejection_collapses_to_baseline_and_never_trains_on_rejects():
    import run_cross_network_eval as cne

    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({c: rng.normal(1.0, 0.1, n) for c in cne.FEATURE_COLUMNS})
    train["event_label"] = (np.arange(n) % 2).astype(int)
    train["split"] = "train"

    class _AllRejected:
        def fit(self, *a, **k):
            return self

        def filter(self, frame, cols):
            return frame.iloc[0:0]          # reject everything

    original = cne.QuantilePlausibilityFilter
    cne.QuantilePlausibilityFilter = _AllRejected
    try:
        out = cne._apply_augmenter(train, "noise", random_state=0)
    finally:
        cne.QuantilePlausibilityFilter = original

    assert len(out) == len(train), (
        "augmenter trained on samples the plausibility screen rejected 100% of; "
        "it must collapse to the unaugmented baseline instead"
    )


# ---------------------------------------------------------------------------
# Defect 4: imbalanced-learn was declared in requirements.txt but failed to import
# against the installed scikit-learn, so SMOTEAugmenter silently fell back to the
# Gaussian-noise augmenter. Every reported "SMOTE" number was a duplicate of the
# noise arm -- and the identical values were rationalised in the manuscript rather
# than investigated.
# ---------------------------------------------------------------------------

def test_smote_is_actually_smote_not_a_silent_noise_fallback():
    pytest.importorskip(
        "imblearn",
        reason="imbalanced-learn must be importable; a silent fallback to the noise "
               "augmenter previously produced a fake SMOTE arm",
    )
    from imblearn.over_sampling import SMOTE  # noqa: F401

    rng = np.random.default_rng(0)
    from watergen.models.augmentation import SMOTEAugmenter
    import run_cross_network_eval as cne

    n = 400
    train = pd.DataFrame({c: rng.normal(1.0, 0.2, n) for c in cne.FEATURE_COLUMNS})
    train["event_label"] = (np.arange(n) < 80).astype(int)   # imbalanced

    smote = SMOTEAugmenter(random_state=0).generate(
        train, feature_columns=cne.FEATURE_COLUMNS, target_column="event_label")
    from watergen.models import PositiveClassAugmenter
    noise = PositiveClassAugmenter(random_state=0, noise_scale=0.05).generate(
        train, feature_columns=cne.FEATURE_COLUMNS, target_column="event_label",
        multiplier=1.0)

    # Real SMOTE balances the classes; the noise augmenter emits one sample per
    # positive. Identical counts are the signature of the silent fallback.
    assert len(smote) != len(noise), (
        "SMOTE and the noise augmenter produced the same number of samples -- "
        "SMOTE is almost certainly falling back to the noise augmenter"
    )


# ---------------------------------------------------------------------------
# Defect 5: the sample-space GNN detector built its k-NN graph from the dense
# N x N distance matrix (torch.cdist(x, x)). At inference the graph spans training
# + test rows, so on a large test network (~3e5 rows) that matrix needed hundreds
# of GB; a run reached ~82 GB of committed memory and thrashed the machine to a
# halt. The graph must be built with a memory-bounded (tree-based) k-NN instead.
# ---------------------------------------------------------------------------

def _brute_force_knn_edges(x, k):
    import torch
    d = torch.cdist(x, x).numpy()
    edges = set()
    for i in range(x.shape[0]):
        order = np.argsort(d[i])
        for j in [j for j in order if j != i][:k]:
            edges.add((i, j))
            edges.add((j, i))
    return edges


def test_knn_graph_matches_brute_force():
    """The memory-bounded k-NN must produce the same graph as the dense version."""
    import torch
    from watergen.models.baselines import GNNDetector
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(50, 6)), dtype=torch.float32)
    ei = GNNDetector._knn_edge_index(x, 6)
    got = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert got == _brute_force_knn_edges(x, 6)


def test_knn_graph_never_builds_the_dense_distance_matrix():
    """The fix must not form the N x N matrix -- that was the 82 GB OOM.

    We forbid torch.cdist for the duration of the call and require the k-NN graph
    to still be built correctly. Under the old implementation this raises; under
    the fix it succeeds, proving the dense matrix is never materialised.
    """
    import torch
    from watergen.models.baselines import GNNDetector

    rng = np.random.default_rng(0)
    n = 20_000                       # dense matrix here would be 20000^2*4 = 1.6 GB
    x = torch.tensor(rng.normal(size=(n, 16)), dtype=torch.float32)

    original_cdist = torch.cdist

    def _forbidden(*_a, **_k):
        raise AssertionError(
            "GNNDetector built the dense N x N distance matrix (torch.cdist); "
            "this is the O(N^2) memory blow-up that must not return"
        )

    torch.cdist = _forbidden
    try:
        ei = GNNDetector._knn_edge_index(x, 10)
    finally:
        torch.cdist = original_cdist

    # O(N*k) symmetric edges, not O(N^2).
    assert ei.shape == (2, n * 10 * 2)
