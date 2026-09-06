"""Do the augmenters synthesise hydraulically possible network states?

The benchmark's strongest result is that the Gaussian-mixture augmenter degrades
cross-network transfer on all eight held-out networks (p = 0.008), while noise and
SMOTE are nulls. That is currently a statistical fact with no physical explanation.

This script asks the hydraulic question: *what do the augmenters actually generate?*
Each augmenter's synthetic samples are scored against the physics residuals the
project already computes -- junction mass continuity and a Hazen-Williams energy
(headloss) proxy -- plus hard bounds on service pressure and pipe velocity. Real
simulated rows (which by construction satisfy EPANET's hydraulics) are the reference.

Hypothesis: SMOTE interpolates *between real hydraulic states*, so its samples stay
close to the manifold of physically realisable network conditions. A Gaussian mixture
fitted in feature space has no such constraint: it can place probability mass in
regions where the marginals look plausible but the joint state violates continuity --
a network configuration EPANET would never produce. If so, "GMM degrades transfer"
becomes a physical claim: augmentation that ignores hydraulic constraints synthesises
states the network cannot occupy, and training on them hurts.

Training split matches the LOLO protocol (anytown + hanoi), so the samples scored here
are the ones the benchmark's detectors were actually trained on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve()
ACTIVE = SCRIPT.parents[1]
sys.path.insert(0, str(ACTIVE / "src"))
sys.path.insert(0, str(ACTIVE / "scripts"))

import run_cross_network_eval as cne  # noqa: E402
from watergen.evaluation import HydraulicResidualScorer  # noqa: E402

OUT = ACTIVE / "artifacts" / "augmenter_physics"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
COVERAGE = [0.25, 0.50]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
AUGMENTERS = ["noise", "gmm", "smote"]

# Physical limits (same as the project's plausibility defaults)
MIN_PRESSURE_M = 10.0    # minimum service pressure
MAX_PRESSURE_M = 105.0   # pipe pressure rating


def bounds_violation_rate(df: pd.DataFrame) -> dict:
    """Fraction of rows outside physically admissible pressure ranges."""
    out = {}
    if "pressure_min" in df:
        out["below_min_service_pressure"] = float((df["pressure_min"] < MIN_PRESSURE_M).mean())
    if "pressure_max" in df:
        out["above_max_pipe_pressure"] = float((df["pressure_max"] > MAX_PRESSURE_M).mean())
    if {"pressure_min", "pressure_max"} <= set(df.columns):
        # A state where the minimum sensed pressure exceeds the maximum is not a
        # "bad" reading -- it is not a state at all.
        out["min_exceeds_max"] = float((df["pressure_min"] > df["pressure_max"]).mean())
    if {"pressure_mean", "pressure_min", "pressure_max"} <= set(df.columns):
        bad = (df["pressure_mean"] < df["pressure_min"]) | (df["pressure_mean"] > df["pressure_max"])
        out["mean_outside_min_max"] = float(bad.mean())
    return out


def summarise(scorer: HydraulicResidualScorer, df: pd.DataFrame, label: str) -> dict:
    rw = scorer.score_rows(df)
    mass_arr, ener_arr = rw.mass_residuals, rw.energy_residuals
    return {
        "label": label,
        "n_rows": int(len(df)),
        "mass_residual_mean": float(np.mean(mass_arr)),
        "mass_residual_std": float(np.std(mass_arr)),
        "mass_residual_p90": float(np.percentile(mass_arr, 90)),
        "energy_residual_mean": float(np.mean(ener_arr)),
        "energy_residual_std": float(np.std(ener_arr)),
        "energy_residual_p90": float(np.percentile(ener_arr, 90)),
        "bounds": bounds_violation_rate(df),
    }


def main() -> None:
    manifest = ACTIVE / "configs" / "benchmark_manifest.yaml"
    index = cne._build_network_index(manifest)

    per_seed = {}
    for seed in SEEDS:
        print(f"[seed {seed}] simulating training split (anytown + hanoi)...", flush=True)
        train = cne._simulate_network_group(
            cne.TRAIN_NETWORK_IDS, index, split_name="train",
            sensor_coverage_levels=COVERAGE, disturbance_types=DISTURBANCES,
        )
        if train.empty:
            print("[error] empty training frame", file=sys.stderr)
            return
        train = train.assign(split="train")

        # Reference: the real simulated rows. These satisfy EPANET's hydraulics by
        # construction, so their residuals are the floor any augmenter should match.
        normal = train.loc[train["event_label"] == 0]
        scorer = HydraulicResidualScorer().fit(normal if not normal.empty else train)

        rows = [summarise(scorer, train, "real (EPANET-simulated)")]

        for aug in AUGMENTERS:
            augmented = cne._apply_augmenter(train, aug, random_state=seed)
            # _apply_augmenter returns real rows (is_synthetic=0) + accepted synthetic.
            if "is_synthetic" in augmented.columns:
                synth = augmented.loc[augmented["is_synthetic"] != 0]
            else:
                synth = augmented.iloc[len(train):]
            if synth.empty:
                print(f"  [warn] {aug}: no synthetic rows retained", flush=True)
                continue
            rows.append(summarise(scorer, synth, f"{aug} (synthetic)"))
            print(f"  {aug}: {len(synth)} synthetic rows", flush=True)

        per_seed[f"seed_{seed}"] = rows

    (OUT / "augmenter_physics.json").write_text(json.dumps(per_seed, indent=2))

    # ---- aggregate across seeds ----
    labels = [r["label"] for r in per_seed[f"seed_{SEEDS[0]}"]]
    print("\n" + "=" * 78)
    print("HYDRAULIC PLAUSIBILITY OF AUGMENTER OUTPUT (5-seed mean)")
    print("Reference = real EPANET-simulated rows, which satisfy the hydraulics exactly.")
    print("=" * 78)
    print(f"{'sample source':<26} {'mass resid':>12} {'energy resid':>14} {'P<10m':>8} {'P>105m':>8}")
    agg = {}
    for lab in labels:
        m = [r["mass_residual_mean"] for s in per_seed.values() for r in s if r["label"] == lab]
        e = [r["energy_residual_mean"] for s in per_seed.values() for r in s if r["label"] == lab]
        lo = [r["bounds"].get("below_min_service_pressure", 0.0)
              for s in per_seed.values() for r in s if r["label"] == lab]
        hi = [r["bounds"].get("above_max_pipe_pressure", 0.0)
              for s in per_seed.values() for r in s if r["label"] == lab]
        agg[lab] = {
            "mass_residual": float(np.mean(m)), "mass_residual_std": float(np.std(m)),
            "energy_residual": float(np.mean(e)), "energy_residual_std": float(np.std(e)),
            "below_min_service_pressure": float(np.mean(lo)),
            "above_max_pipe_pressure": float(np.mean(hi)),
        }
        print(f"{lab:<26} {np.mean(m):>12.4f} {np.mean(e):>14.4f} "
              f"{np.mean(lo):>7.1%} {np.mean(hi):>8.1%}")

    ref = agg["real (EPANET-simulated)"]
    print("\nInflation over real simulated rows (x):")
    for lab, v in agg.items():
        if lab.startswith("real"):
            continue
        fm = v["mass_residual"] / ref["mass_residual"] if ref["mass_residual"] > 0 else float("nan")
        fe = v["energy_residual"] / ref["energy_residual"] if ref["energy_residual"] > 0 else float("nan")
        print(f"  {lab:<24} mass x{fm:>6.2f}   energy x{fe:>6.2f}")

    (OUT / "augmenter_physics_summary.json").write_text(json.dumps(agg, indent=2))
    print(f"\nSaved: {OUT / 'augmenter_physics_summary.json'}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
