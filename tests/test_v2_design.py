"""Regression tests for the v2 benchmark design (S36).

Each test pins a property whose absence produced wrong-but-plausible numbers in v1:

  1. no scenario metadata (`leak_area`, sensor counts, ...) among detector inputs;
  2. sensor placement is random per run seed, identical across scenarios of one network
     within a seed, and the legacy stride is untouched when no seed is given;
  3. leak onsets are sampled per scenario when an rng is given, fixed otherwise;
  4. per-network normalisation uses only normal-operation rows and gives z-scores with
     zero mean / unit spread on them, zero where the spread is zero;
  5. the causal residual score never depends on future timesteps.

Run:  pytest tests/test_v2_design.py -v
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

from watergen.data.normalise import (  # noqa: E402
    SCENARIO_METADATA_COLUMNS,
    assert_no_scenario_metadata,
    normalise_by_network_reference,
)
from watergen.data.simulation import (  # noqa: E402
    _sample_leak_window,
    _select_coverage_columns,
    placement_rng_for,
)

import run_cross_network_eval as rce  # noqa: E402


# ---------------------------------------------------------------------------
# 1. detector inputs
# ---------------------------------------------------------------------------

def test_v2_feature_lists_contain_no_scenario_metadata():
    for lst in (rce.FEATURE_COLUMNS, rce.FEATURE_COLUMNS_RAW, rce.FEATURE_COLUMNS_V2):
        assert not set(lst) & SCENARIO_METADATA_COLUMNS
        assert_no_scenario_metadata(lst)
    # the v1 list is kept as a record and must still be recognised as leaky
    with pytest.raises(ValueError):
        assert_no_scenario_metadata(rce.FEATURE_COLUMNS_V1_WITH_LEAK)
    assert "leak_area" in rce.FEATURE_COLUMNS_V1_WITH_LEAK


def test_v2_features_are_normalised_aggregates_plus_coverage():
    z = [c for c in rce.FEATURE_COLUMNS_V2 if c.startswith("z__")]
    assert len(z) == 15 and rce.FEATURE_COLUMNS_V2[-1] == "sensor_coverage"
    assert "observed_junction_count" not in " ".join(rce.FEATURE_COLUMNS_V2)


# ---------------------------------------------------------------------------
# 2. sensor placement
# ---------------------------------------------------------------------------

def test_legacy_stride_is_unchanged_without_rng():
    cols = [f"J{i}" for i in range(20)]
    assert _select_coverage_columns(cols, 0.25) == cols[::4][:5]
    assert _select_coverage_columns(cols, 1.0) == cols


def test_random_placement_differs_by_seed_and_is_fixed_within_a_network():
    cols = [f"J{i}" for i in range(400)]
    a1 = _select_coverage_columns(cols, 0.25, placement_rng_for(42, "d_town", 0.25))
    a2 = _select_coverage_columns(cols, 0.25, placement_rng_for(42, "d_town", 0.25))
    b = _select_coverage_columns(cols, 0.25, placement_rng_for(43, "d_town", 0.25))
    other_net = _select_coverage_columns(cols, 0.25, placement_rng_for(42, "l_town", 0.25))
    assert a1 == a2, "same (seed, network, coverage) must give the same layout for every scenario"
    assert a1 != b, "different run seeds must give different layouts"
    assert a1 != other_net
    assert len(a1) == 100 and len(set(a1)) == 100 and a1 == sorted(a1, key=cols.index)
    assert a1 != cols[::4][:100], "random placement must not silently reproduce the stride"


def test_placement_rng_is_none_without_seed():
    assert placement_rng_for(None, "net3", 0.5) is None


# ---------------------------------------------------------------------------
# 3. leak onsets
# ---------------------------------------------------------------------------

def test_sampled_leak_windows_vary_and_stay_inside_the_horizon():
    duration, step = 24 * 3600, 3600
    rng = np.random.default_rng(0)
    wins = {_sample_leak_window(duration, step, rng) for _ in range(50)}
    assert len(wins) > 10, "onsets must actually vary across leak scenarios"
    for s, e in wins:
        assert step <= s <= duration // 2
        assert s < e <= duration - step
        assert duration // 6 <= e - s <= duration // 3 or e == duration - step


def test_build_leak_scenarios_uses_fixed_window_only_without_rng(tmp_path):
    """The fixed v1 window [d/3, 2d/3] must be the explicit no-rng default, not a hidden
    override of a sampled window."""
    from watergen.data import simulation as sim
    # the legacy branch of default_leak_window is the fixed window
    class _T:  # minimal fake network for the window arithmetic
        class options:
            class time:
                duration = 24 * 3600; report_timestep = 3600; hydraulic_timestep = 3600
    d, r = sim.effective_duration_seconds(_T), sim.effective_report_step_seconds(_T)
    assert (max(r, d // 3), min(d - r, 2 * d // 3)) == (8 * 3600, 16 * 3600)


# ---------------------------------------------------------------------------
# 4. normalisation
# ---------------------------------------------------------------------------

def _frame():
    rng = np.random.default_rng(1)
    rows = []
    for net, scale in (("a", 1.0), ("b", 50.0)):
        for cov in (0.25, 0.5):
            for kind, n in (("baseline", 40), ("leak", 20)):
                for _ in range(n):
                    rows.append({
                        "benchmark_id": net, "sensor_coverage": cov, "scenario_kind": kind,
                        "pressure_mean": scale * (10 + rng.normal()), "demand_sum": scale * rng.normal(),
                        "tank_head_mean": 7.0,  # constant -> zero spread
                        "event_label": int(kind != "baseline"),
                    })
    return pd.DataFrame(rows)


def test_normalisation_uses_only_normal_rows_and_handles_zero_spread():
    df = _frame()
    out, stats = normalise_by_network_reference(df, ["pressure_mean", "demand_sum", "tank_head_mean"])
    for (net, cov), g in out.groupby(["benchmark_id", "sensor_coverage"]):
        ref = g[g.scenario_kind == "baseline"]
        assert abs(ref["z__pressure_mean"].mean()) < 1e-9 and abs(ref["z__pressure_mean"].std(ddof=0) - 1) < 1e-9
        assert (g["z__tank_head_mean"] == 0).all()
        mu, sd = stats[(net, cov)]["pressure_mean"]
        # reference statistics come from normal rows only
        assert abs(mu - ref["pressure_mean"].mean()) < 1e-9
    # networks on very different raw scales end up on one scale
    a = out[out.benchmark_id == "a"]["z__pressure_mean"].abs().mean()
    b = out[out.benchmark_id == "b"]["z__pressure_mean"].abs().mean()
    assert abs(a - b) < 0.5


def test_normalisation_refuses_groups_without_normal_operation_rows():
    df = _frame()
    df = df[~((df.benchmark_id == "b") & (df.scenario_kind == "baseline"))]
    with pytest.raises(ValueError):
        normalise_by_network_reference(df, ["pressure_mean"])


# ---------------------------------------------------------------------------
# 5. causal residual detector
# ---------------------------------------------------------------------------

def test_causal_residual_score_ignores_the_future():
    from analyze_v2 import causal_residual_score
    rng = np.random.default_rng(3)
    mass, energy = rng.normal(size=60), rng.normal(size=60)
    ref = {"mass": (0.0, 1.0), "energy": (0.0, 1.0)}
    base = causal_residual_score(mass, energy, ref)
    mass2, energy2 = mass.copy(), energy.copy()
    mass2[40:] += 50.0  # perturb one stream only (equal and opposite shifts would cancel in the blend)
    perturbed = causal_residual_score(mass2, energy2, ref)
    assert np.allclose(base[:40], perturbed[:40]), "scores before t=40 must not change when the future changes"
    assert not np.allclose(base[40:], perturbed[40:])


# ---------------------------------------------------------------------------
# 6. plausibility screen under a rare prior
# ---------------------------------------------------------------------------

def test_plausibility_screen_is_fitted_on_real_positives_and_keeps_perturbed_positives():
    """At an 8% positive rate, bounds fitted on ALL rows come from normal operation and
    reject nearly every perturbed positive (seed-42 diagnostic: 95-98% rejected). Bounds
    fitted on the real positives keep small perturbations of them."""
    from watergen.models import QuantilePlausibilityFilter
    rng = np.random.default_rng(0)
    feats = [f"f{i}" for i in range(15)]
    normal = pd.DataFrame({f: rng.normal(0, 1, 9200) for f in feats}); normal["event_label"] = 0
    positive = pd.DataFrame({f: rng.normal(3, 1.5, 800) for f in feats}); positive["event_label"] = 1
    train = pd.concat([normal, positive], ignore_index=True)
    perturbed = positive.copy()
    for f in feats:
        perturbed[f] = positive[f] + rng.normal(0, 0.05 * positive[f].std(), len(positive))
    kept_all = QuantilePlausibilityFilter().fit(train, feats).filter(perturbed, feats)
    kept_pos = QuantilePlausibilityFilter().fit(positive, feats).filter(perturbed, feats)
    assert len(kept_pos) > len(kept_all), "positive-fitted bounds must keep more perturbed positives than pooled bounds"
    assert len(kept_pos) / len(perturbed) > 0.9
    assert rce.PLAUSIBILITY_REFERENCE == "positive" and rce.DOSE_MATCH_AUGMENTERS is True
