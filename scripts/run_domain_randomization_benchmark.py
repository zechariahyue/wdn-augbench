"""Domain randomization benchmark for LOLO cross-network transfer.

Compares five training strategies on held-out D-Town and L-TOWN:
  1. baseline      – no augmentation
  2. noise         – PositiveClassAugmenter (Gaussian noise)
  3. smote         – SMOTE augmentation (noise fallback)
  4. domain_rand   – clean train + demand-noise re-simulation concatenated
  5. graph_cvae    – GraphCVAEAugmenter trained on clean data

All strategies fit a Random Forest detector and evaluate under LOLO protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT   = SCRIPT_PATH.parents[3]
SRC_ROOT    = ACTIVE_ROOT / "src"
for p in [str(SRC_ROOT), str(SCRIPT_PATH.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from run_cross_network_eval import (  # noqa: E402
    _apply_augmenter,
    _build_network_index,
    _build_split_for_networks,
    _evaluate_on_network,
    _simulate_network_group,
    FEATURE_COLUMNS,
    NETWORK_NODE_COUNTS,
    TRAIN_NETWORK_IDS,
    TEST_NETWORK_IDS,
    size_weighted_auprc,
)
from watergen.data import build_leak_scenarios  # noqa: E402
from watergen.data.simulation import simulate_scenario  # noqa: E402
from watergen.models import (  # noqa: E402
    GraphCVAEAugmenter,
    QuantilePlausibilityFilter,
    TabularDetectorBaseline,
)

MANIFEST   = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_DIR = ACTIVE_ROOT / "artifacts" / "domain_rand"

# Strategies handled in this benchmark (order preserved in output)
STRATEGIES = ["baseline", "noise", "smote", "domain_rand", "graph_cvae"]

# Simulation hyperparameters
SENSOR_COVERAGE_LEVELS = [0.25, 0.50]
DISTURBANCE_TYPES      = ["leak", "pipe_closure", "pump_outage"]
SEED                   = 42


# ---------------------------------------------------------------------------
# Domain-randomization simulation helper
# ---------------------------------------------------------------------------

def _simulate_with_noise(
    network_ids: list[str],
    network_index: dict[str, Any],
    *,
    split_name: str,
    sensor_coverage_levels: list[float],
    disturbance_types: list[str],
) -> pd.DataFrame:
    """Re-simulate training networks with stochastic demand noise (±20 %).

    Mirrors the structure of ``_simulate_network_group`` but passes
    ``demand_noise=True`` to every ``simulate_scenario`` call.  This
    produces a perturbed copy of the training distribution that, when
    concatenated with the clean training data, gives the model exposure
    to demand-shifted operating conditions without any extra network data.
    """
    resolved = _build_split_for_networks(
        network_ids, network_index, split_name=split_name
    )

    specs = build_leak_scenarios(
        resolved,
        smoke_splits=[split_name],
        leak_area_values=[0.0001, 0.0005],
        max_candidates_per_network=2,
        disturbance_types=disturbance_types,
        include_baseline=True,
    )

    frames: list[pd.DataFrame] = []
    for spec in specs:
        try:
            df = simulate_scenario(
                spec,
                sensor_coverage_levels=sensor_coverage_levels,
                demand_noise=True,
                demand_noise_fraction=0.20,
            )
            frames.append(df)
        except Exception as exc:
            print(
                f"[warn] noise-sim failed for {spec.scenario_id}: {exc}",
                file=sys.stderr,
            )

    if not frames:
        print("[warn] Domain-rand noise simulation produced no frames.", file=sys.stderr)
        return pd.DataFrame()

    noisy_df = pd.concat(frames, ignore_index=True)
    print(f"[info] Domain-rand noisy re-simulation: {len(noisy_df)} rows")
    return noisy_df


# ---------------------------------------------------------------------------
# GraphCVAE augmentation helper
# ---------------------------------------------------------------------------

def _apply_graph_cvae(
    train_df: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit a GraphCVAEAugmenter and return the augmented training frame."""
    augmenter = GraphCVAEAugmenter(
        random_state=random_state,
        epochs=30,
        lambda_physics=0.5,
        warmup_epochs=10,
    )

    try:
        augmenter.fit(train_df, feature_columns=FEATURE_COLUMNS, target_column="event_label")
        synthetic = augmenter.generate(
            train_df,
            feature_columns=FEATURE_COLUMNS,
            target_column="event_label",
        )
    except Exception as exc:
        print(f"[warn] GraphCVAEAugmenter failed ({exc}); using unaugmented train.", file=sys.stderr)
        return train_df.copy()

    if synthetic.empty:
        print("[warn] GraphCVAE produced no synthetic rows; using unaugmented train.", file=sys.stderr)
        return train_df.copy()

    # Apply plausibility filter
    plausibility_filter = QuantilePlausibilityFilter().fit(train_df, FEATURE_COLUMNS)
    filtered = plausibility_filter.filter(synthetic, FEATURE_COLUMNS)
    if filtered.empty:
        print("[warn] All GraphCVAE rows filtered by plausibility; using unfiltered.", file=sys.stderr)
        filtered = synthetic.copy()

    augmented = pd.concat(
        [train_df.assign(is_synthetic=0), filtered],
        axis=0,
        ignore_index=True,
    )
    return augmented


# ---------------------------------------------------------------------------
# Markdown table writer
# ---------------------------------------------------------------------------

def _write_markdown_table(results: dict[str, Any], output_path: Path) -> None:
    """Write strategy × test-network AUPRC table in markdown format."""
    test_networks = TEST_NETWORK_IDS
    header    = "| Strategy     | " + " | ".join(test_networks) + " | Weighted AUPRC |"
    separator = "|--------------|" + "|".join(["----------"] * len(test_networks)) + "|----------------|"

    lines = [
        "# Domain Randomization Benchmark — LOLO Results",
        "",
        header,
        separator,
    ]
    for strategy in STRATEGIES:
        strat_data = results.get("per_strategy", {}).get(strategy, {})
        per_net    = strat_data.get("per_network", {})
        weighted   = strat_data.get("weighted_auprc", "n/a")
        w_str      = f"{weighted:.3f}" if isinstance(weighted, (int, float)) else str(weighted)

        cells = []
        for nid in test_networks:
            auprc = per_net.get(nid, {}).get("auprc")
            cells.append(f"{auprc:.3f}" if isinstance(auprc, (int, float)) else "n/a")

        lines.append(f"| {strategy:<12} | " + " | ".join(cells) + f" | {w_str} |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Markdown table saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Force UTF-8 on Windows to avoid encoding errors from WNTR
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    if not MANIFEST.exists():
        print(f"[error] Manifest not found: {MANIFEST}", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Manifest:    {MANIFEST}")
    print(f"[info] Output dir:  {OUTPUT_DIR}")

    # ----------------------------------------------------------------
    # 1. Build network index
    # ----------------------------------------------------------------
    print("[info] Building network index...")
    network_index = _build_network_index(MANIFEST)

    available_train = [nid for nid in TRAIN_NETWORK_IDS if nid in network_index]
    available_test  = [nid for nid in TEST_NETWORK_IDS  if nid in network_index]
    print(f"[info] Train networks ({len(available_train)}): {available_train}")
    print(f"[info] Test networks  ({len(available_test)}):  {available_test}")

    # ----------------------------------------------------------------
    # 2. Simulate clean training data
    # ----------------------------------------------------------------
    print("[info] Simulating clean TRAIN data...")
    train_df = _simulate_network_group(
        available_train,
        network_index,
        split_name="train",
        sensor_coverage_levels=SENSOR_COVERAGE_LEVELS,
        disturbance_types=DISTURBANCE_TYPES,
    )
    print(f"[info] Clean train dataset: {len(train_df)} rows")

    if train_df.empty:
        print("[error] Clean training simulation produced no data — aborting.", file=sys.stderr)
        return 1

    # Normalise split column
    train_split = train_df[train_df["split"] == "train"].copy() if "split" in train_df.columns else train_df.copy()
    if train_split.empty:
        train_split = train_df.copy()
        train_split["split"] = "train"

    # ----------------------------------------------------------------
    # 3. Simulate domain-rand noisy training data (once, shared across run)
    # ----------------------------------------------------------------
    print("[info] Simulating domain-rand noisy TRAIN data...")
    noisy_train_df = _simulate_with_noise(
        available_train,
        network_index,
        split_name="train",
        sensor_coverage_levels=SENSOR_COVERAGE_LEVELS,
        disturbance_types=DISTURBANCE_TYPES,
    )

    # ----------------------------------------------------------------
    # 4. Simulate test data per held-out network
    # ----------------------------------------------------------------
    print("[info] Simulating TEST data per held-out network...")
    test_frames: dict[str, pd.DataFrame] = {}
    for nid in available_test:
        print(f"[info]   Simulating '{nid}'...")
        df = _simulate_network_group(
            [nid],
            network_index,
            split_name="test",
            sensor_coverage_levels=SENSOR_COVERAGE_LEVELS,
            disturbance_types=DISTURBANCE_TYPES,
        )
        test_frames[nid] = df
        print(f"[info]   '{nid}': {len(df)} rows")

    # ----------------------------------------------------------------
    # 5. Evaluate each strategy
    # ----------------------------------------------------------------
    results: dict[str, Any] = {
        "protocol": "LOLO",
        "train_networks": available_train,
        "test_networks": available_test,
        "sensor_coverage_levels": SENSOR_COVERAGE_LEVELS,
        "disturbance_types": DISTURBANCE_TYPES,
        "seed": SEED,
        "clean_train_rows": int(len(train_split)),
        "noisy_train_rows": int(len(noisy_train_df)),
        "per_strategy": {},
    }

    for strategy in STRATEGIES:
        print(f"\n[info] === Strategy: {strategy!r} ===")

        # --- Build the augmented training frame for this strategy ---
        try:
            if strategy == "domain_rand":
                # Concatenate clean + noisy simulations; no synthetic rows
                if noisy_train_df.empty:
                    print("[warn] Noisy frame is empty; domain_rand falls back to clean train.", file=sys.stderr)
                    augmented_train = train_split.copy()
                else:
                    # Align columns: keep only columns present in both
                    common_cols = [c for c in train_split.columns if c in noisy_train_df.columns]
                    augmented_train = pd.concat(
                        [train_split[common_cols], noisy_train_df[common_cols]],
                        axis=0,
                        ignore_index=True,
                    )
                    # Tag origin so downstream can inspect if needed
                    augmented_train["is_synthetic"] = 0
                print(f"[info] domain_rand train rows: {len(augmented_train)}")

            elif strategy == "graph_cvae":
                augmented_train = _apply_graph_cvae(train_split, random_state=SEED)
                print(f"[info] graph_cvae train rows: {len(augmented_train)}")

            else:
                # baseline / noise / smote handled by existing helper
                augmented_train = _apply_augmenter(train_split, strategy, random_state=SEED)
                print(f"[info] {strategy} train rows: {len(augmented_train)}")

        except Exception as exc:
            print(f"[warn] Strategy '{strategy}' augmentation failed: {exc} — using baseline.", file=sys.stderr)
            augmented_train = train_split.copy()

        # --- Fit detector ---
        try:
            detector = TabularDetectorBaseline(algorithm="random_forest", random_state=SEED)
            detector.fit_from_frame(augmented_train, feature_columns=FEATURE_COLUMNS)
        except Exception as exc:
            print(f"[warn] Detector fit failed for '{strategy}': {exc}", file=sys.stderr)
            results["per_strategy"][strategy] = {"error": str(exc)}
            continue

        # --- Evaluate on each held-out test network ---
        per_network: dict[str, dict[str, Any]] = {}
        for nid, test_df in test_frames.items():
            net_result = _evaluate_on_network(detector, test_df, nid)
            per_network[nid] = net_result
            auprc = net_result.get("auprc")
            auprc_str = f"{auprc:.3f}" if isinstance(auprc, (int, float)) else str(auprc)
            print(f"[info]   '{nid}' AUPRC={auprc_str}")

        weighted = size_weighted_auprc(per_network)
        results["per_strategy"][strategy] = {
            "strategy": strategy,
            "augmented_train_rows": int(len(augmented_train)),
            "per_network": per_network,
            "weighted_auprc": weighted,
        }
        print(f"[info]   Weighted AUPRC = {weighted:.3f}")

    # ----------------------------------------------------------------
    # 6. Save outputs
    # ----------------------------------------------------------------
    json_path = OUTPUT_DIR / "domain_rand_results.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n[info] JSON results saved: {json_path}")

    md_path = OUTPUT_DIR / "domain_rand_table.md"
    _write_markdown_table(results, md_path)

    print("[info] Domain randomization benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
