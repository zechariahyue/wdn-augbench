"""Topology-aware GNN detector LOLO evaluation, 5 seeds.

Trains a ``TopologyAwareGNNDetector`` on the union of two training networks
and evaluates on three held-out networks, swapping the reference WDN
topology to each test network at inference time.  Produces per-network and
macro-AUPRC numbers directly comparable to the augmenter rows in Table 3.
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

from watergen.data import (  # noqa: E402
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_path,
    simulate_many,
)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (  # noqa: E402
    TopologyAwareGNNDetector,
    prepare_feature_matrix,
)

SEEDS = [42, 43, 44, 45, 46]
MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUTPUT_ROOT = ACTIVE_ROOT / "artifacts" / "topology_gnn_lolo"

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


def _evaluate(detector: TopologyAwareGNNDetector, test_df: pd.DataFrame,
              available_feats: list[str], network_id: str) -> dict[str, Any]:
    if test_df.empty:
        return {"network_id": network_id, "auprc": None}
    if not available_feats:
        return {"network_id": network_id, "auprc": None}
    try:
        x, y, _ = prepare_feature_matrix(test_df, target_column="event_label",
                                          feature_columns=available_feats)
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


def run_seed(seed: int, train_split: pd.DataFrame, test_frames: dict,
             train_topology_path: str, test_topology_paths: dict) -> dict[str, Any]:
    available_feats = [c for c in FEATURE_COLUMNS if c in train_split.columns]

    detector = TopologyAwareGNNDetector(
        benchmark_path=train_topology_path,
        hidden_dim=64, epochs=60, lr=1e-3, batch_size=32,
        random_state=seed, device="cpu",
    )
    detector.fit_from_frame(train_split, feature_columns=available_feats,
                             target_column="event_label")

    per_network: dict[str, dict[str, Any]] = {}
    for nid, test_df in test_frames.items():
        # Swap topology to the test network before scoring
        topology_path = test_topology_paths.get(nid)
        if topology_path is not None and detector._fallback is None:
            ok = detector.set_topology(topology_path)
            if not ok:
                per_network[nid] = {"network_id": nid, "auprc": None,
                                     "error": "topology swap failed"}
                continue
        per_network[nid] = _evaluate(detector, test_df, available_feats, nid)

    valid = [v["auprc"] for v in per_network.values() if isinstance(v.get("auprc"), (int, float))]
    macro = float(np.mean(valid)) if valid else None

    return {"per_network": per_network, "macro_auprc": macro}


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("[info] Building network index...")
    network_index = _build_network_index(MANIFEST)
    available_train = [n for n in TRAIN_NETWORK_IDS if n in network_index]
    available_test = [n for n in TEST_NETWORK_IDS if n in network_index]

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

    # Training topology: use the first available train network
    train_topology_path: str | None = None
    for nid in available_train:
        rp = network_index[nid]["resolved_path"]
        if rp.exists():
            train_topology_path = str(rp)
            print(f"[info] Using train topology from: {nid}")
            break

    test_topology_paths: dict[str, str] = {}
    for nid in available_test:
        rp = network_index[nid]["resolved_path"]
        if rp.exists():
            test_topology_paths[nid] = str(rp)

    if train_topology_path is None:
        print("[error] No training topology available; aborting.")
        sys.exit(1)

    all_results = []
    print(f"\n{'='*60}\n  TOPOLOGY-AWARE GNN LOLO (5 seeds)\n{'='*60}\n")
    for seed in SEEDS:
        print(f"--- seed {seed} ---")
        result = run_seed(seed, train_split, test_frames,
                           train_topology_path, test_topology_paths)
        macro = result.get("macro_auprc")
        macro_str = f"{macro:.3f}" if macro is not None else "n/a"
        print(f"  macro_auprc = {macro_str}")
        for nid, stats in result["per_network"].items():
            a = stats.get("auprc")
            a_str = f"{a:.3f}" if isinstance(a, (int, float)) else "n/a"
            print(f"    {nid}: auprc={a_str}")
        all_results.append({**result, "seed": seed})

    # Aggregate
    agg = {nid: [] for nid in TEST_NETWORK_IDS}
    agg["macro"] = []
    for r in all_results:
        for nid in TEST_NETWORK_IDS:
            v = r["per_network"].get(nid, {}).get("auprc")
            if isinstance(v, (int, float)):
                agg[nid].append(float(v))
        if isinstance(r.get("macro_auprc"), (int, float)):
            agg["macro"].append(float(r["macro_auprc"]))

    summary = {"per_seed": all_results, "aggregated": {}}
    for key, vals in agg.items():
        if vals:
            summary["aggregated"][key] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals), "values": vals,
            }
        else:
            summary["aggregated"][key] = {"mean": None, "std": None, "n": 0}

    out_path = OUTPUT_ROOT / "topology_gnn_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[info] Saved: {out_path}")

    print(f"\n{'Network':12s} {'AUPRC (mean ± std, n=5)':>30s}")
    print("-" * 45)
    for nid in TEST_NETWORK_IDS + ["macro"]:
        stats = summary["aggregated"][nid]
        m, s = stats["mean"], stats["std"]
        if m is not None:
            print(f"{nid:12s} {f'{m:.3f} ± {s:.3f}':>30s}")
        else:
            print(f"{nid:12s} {'---':>30s}")


if __name__ == "__main__":
    main()
