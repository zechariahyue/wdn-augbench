"""Filter ablation for Graph-CVAE LOLO.

Runs Graph-CVAE across 5 seeds under three filter conditions:
  1. no_filter  — filter disabled entirely (all synthetic samples used)
  2. relaxed    — quantile bounds 5–95 percent instead of 1–99 percent
  3. strict     — current default (1–99 percent); reproduces Table 3 numbers

This addresses the 100 percent rejection problem observed in the Table 3
Graph-CVAE row by diagnosing where the plausibility filter is binding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(ACTIVE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ACTIVE_ROOT / "scripts"))

from watergen.data import (  # noqa: E402
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_path,
    simulate_many,
)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (  # noqa: E402
    GraphCVAEAugmenter,
    QuantilePlausibilityFilter,
    TabularDetectorBaseline,
    prepare_feature_matrix,
)

SEEDS = [42, 43, 44, 45, 46]
MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_ROOT = ACTIVE_ROOT / "artifacts" / "graph_cvae_filter_ablation"

TRAIN_NETWORK_IDS = ["anytown", "hanoi"]
TEST_NETWORK_IDS = ["net3", "d_town", "l_town"]

FEATURE_COLUMNS = [
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "demand_mean", "demand_sum", "demand_std",
    "flow_mean", "flow_abs_mean", "flow_std",
    "tank_head_mean", "tank_head_std",
    "sensor_coverage", "observed_junction_count", "observed_link_count",
    "leak_area",
]

SENSOR_COVERAGE_LEVELS = [0.25, 0.50]
DISTURBANCE_TYPES = ["leak", "pipe_closure", "pump_outage"]


FILTER_CONFIGS = {
    "no_filter": None,
    "relaxed": {"lower_quantile": 0.05, "upper_quantile": 0.95, "margin_scale": 0.20},
    "strict": {"lower_quantile": 0.01, "upper_quantile": 0.99, "margin_scale": 0.10},
}


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_network_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_yaml(manifest_path)
    entries = collect_benchmark_entries(manifest)
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        nid = str(entry.get("id", "")).strip()
        if not nid:
            continue
        resolved = resolve_path(str(entry.get("path", "")), root=REPO_ROOT)
        index[nid] = {**entry, "resolved_path": resolved}
    return index


def _build_split(network_ids: list[str], network_index: dict, split_name: str) -> dict:
    entries = []
    for nid in network_ids:
        if nid not in network_index:
            continue
        resolved = network_index[nid]["resolved_path"]
        if not resolved.exists():
            continue
        entries.append({"id": nid, "stage": "train", "path": str(resolved),
                        "split": split_name, "split_role": split_name})
    return {"partitions": {split_name: entries}}


def _simulate_networks(network_ids: list[str], network_index: dict, split_name: str) -> pd.DataFrame:
    resolved_split = _build_split(network_ids, network_index, split_name)
    frames: list[pd.DataFrame] = []
    for entry in resolved_split["partitions"].get(split_name, []):
        nid = entry["id"]
        try:
            specs = build_leak_scenarios(
                resolved_split,
                smoke_splits=[split_name],
                leak_area_values=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
                max_candidates_per_network=5,
                disturbance_types=DISTURBANCE_TYPES,
                include_baseline=True,
            )
            network_specs = [s for s in specs if s.benchmark_id == nid]
            if not network_specs:
                continue
            df = simulate_many(network_specs, sensor_coverage_levels=SENSOR_COVERAGE_LEVELS)
            frames.append(df)
        except Exception as exc:
            print(f"[warn] Sim failed '{nid}': {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _evaluate(detector: TabularDetectorBaseline, test_df: pd.DataFrame, network_id: str) -> dict[str, Any]:
    if test_df.empty:
        return {"network_id": network_id, "auprc": None}
    available = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    if not available:
        return {"network_id": network_id, "auprc": None}
    try:
        x, y, _ = prepare_feature_matrix(test_df, target_column="event_label",
                                          feature_columns=available)
        proba = detector.predict_proba(x)[:, 1]
        pred = detector.predict(x)
        rep = binary_report(y, pred, proba)
        out = report_as_dict(rep)
        out["network_id"] = network_id
        out["rows"] = int(len(test_df))
        out["positive_rows"] = int(test_df["event_label"].sum())
        return out
    except Exception as exc:
        return {"network_id": network_id, "error": str(exc), "auprc": None}


def run_variant(filter_config: dict | None, seed: int,
                 train_split: pd.DataFrame, test_frames: dict,
                 benchmark_path: str | None) -> dict[str, Any]:
    """Run Graph-CVAE once with the given filter config."""
    available_feats = [c for c in FEATURE_COLUMNS if c in train_split.columns]

    augmenter = GraphCVAEAugmenter(
        benchmark_path=benchmark_path,
        latent_dim=64, hidden_dim=128, epochs=100, lr=1e-3,
        batch_size=32, random_state=seed,
        lambda_physics=0.5, warmup_epochs=25, device="cpu",
    )
    augmenter.fit(train_split, feature_columns=available_feats, target_column="event_label")

    synthetic = augmenter.generate(
        train_split, feature_columns=available_feats, target_column="event_label",
    )
    n_generated = len(synthetic)

    if filter_config is None:
        # Skip filter entirely
        n_kept = n_generated
    else:
        plausibility_filter = QuantilePlausibilityFilter(
            lower_quantile=filter_config["lower_quantile"],
            upper_quantile=filter_config["upper_quantile"],
            margin_scale=filter_config["margin_scale"],
        ).fit(train_split, available_feats)
        if not synthetic.empty:
            synthetic = plausibility_filter.filter(synthetic, available_feats)
        n_kept = len(synthetic)

    if synthetic.empty:
        augmented_train = train_split.copy()
    else:
        augmented_train = pd.concat(
            [train_split.assign(is_synthetic=0), synthetic],
            ignore_index=True,
        )

    detector = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    detector.fit_from_frame(augmented_train, feature_columns=available_feats)

    per_network: dict[str, dict[str, Any]] = {}
    for nid, test_df in test_frames.items():
        per_network[nid] = _evaluate(detector, test_df, nid)

    valid = [v["auprc"] for v in per_network.values() if isinstance(v.get("auprc"), (int, float))]
    macro = float(np.mean(valid)) if valid else None

    return {
        "per_network": per_network,
        "macro_auprc": macro,
        "n_generated": n_generated,
        "n_kept_after_filter": n_kept,
        "filter_rejection_rate": (1.0 - n_kept / n_generated) if n_generated > 0 else None,
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("[info] Building network index...")
    network_index = _build_network_index(MANIFEST)
    available_train = [n for n in TRAIN_NETWORK_IDS if n in network_index]
    available_test = [n for n in TEST_NETWORK_IDS if n in network_index]

    # Simulate once — reused across seeds and filter configs
    print("[info] Simulating TRAIN...")
    train_df = _simulate_networks(available_train, network_index, "train")
    train_split = train_df[train_df["split"] == "train"].copy()
    if train_split.empty:
        train_split = train_df.copy()
    print(f"[info] Train rows: {len(train_split)}")

    print("[info] Simulating TEST per network...")
    test_frames: dict[str, pd.DataFrame] = {}
    for nid in available_test:
        test_frames[nid] = _simulate_networks([nid], network_index, "test")
        print(f"[info]  '{nid}': {len(test_frames[nid])} rows")

    benchmark_path: str | None = None
    for nid in available_train:
        rp = network_index[nid]["resolved_path"]
        if rp.exists():
            benchmark_path = str(rp)
            break

    all_results: dict[str, dict] = {}
    for variant_name, filter_config in FILTER_CONFIGS.items():
        print(f"\n{'='*60}\n  VARIANT: {variant_name}\n{'='*60}\n")
        seed_results = []
        for seed in SEEDS:
            print(f"--- seed {seed} ---")
            result = run_variant(filter_config, seed, train_split, test_frames, benchmark_path)
            seed_results.append({**result, "seed": seed})
            print(f"  n_generated={result['n_generated']}, n_kept={result['n_kept_after_filter']}, macro={result['macro_auprc']:.3f}" if result['macro_auprc'] else f"  n_generated={result['n_generated']}, n_kept={result['n_kept_after_filter']}")

        # Aggregate
        agg = {nid: [] for nid in TEST_NETWORK_IDS}
        agg["macro"] = []
        agg["n_kept"] = []
        agg["n_generated"] = []
        for r in seed_results:
            for nid in TEST_NETWORK_IDS:
                v = r["per_network"].get(nid, {}).get("auprc")
                if isinstance(v, (int, float)):
                    agg[nid].append(float(v))
            if isinstance(r.get("macro_auprc"), (int, float)):
                agg["macro"].append(float(r["macro_auprc"]))
            agg["n_kept"].append(r["n_kept_after_filter"])
            agg["n_generated"].append(r["n_generated"])

        summary = {"variant": variant_name, "per_seed": seed_results, "aggregated": {}}
        for key, vals in agg.items():
            if key in ("n_kept", "n_generated"):
                summary["aggregated"][key] = {"mean": float(np.mean(vals)), "values": vals}
            else:
                if vals:
                    summary["aggregated"][key] = {
                        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                        "n": len(vals), "values": vals,
                    }
                else:
                    summary["aggregated"][key] = {"mean": None, "std": None, "n": 0}

        all_results[variant_name] = summary

    out_path = OUTPUT_ROOT / "filter_ablation_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\n[info] Saved: {out_path}")

    # Print summary table
    print(f"\n{'Variant':15s} {'Net3':>14s} {'D-Town':>14s} {'L-TOWN':>14s} {'Macro':>14s} {'n_kept':>8s}")
    print("-" * 90)
    for name, data in all_results.items():
        agg = data["aggregated"]
        parts = []
        for nid in TEST_NETWORK_IDS + ["macro"]:
            m = agg[nid]["mean"]
            s = agg[nid]["std"]
            parts.append(f"{m:.3f}±{s:.3f}" if m is not None else "---")
        nk = agg["n_kept"]["mean"]
        ng = agg["n_generated"]["mean"]
        print(f"{name:15s} {parts[0]:>14s} {parts[1]:>14s} {parts[2]:>14s} {parts[3]:>14s} {nk:>6.0f}/{ng:>3.0f}")


if __name__ == "__main__":
    main()
