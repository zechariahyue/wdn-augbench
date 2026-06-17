"""Graph-CVAE LOLO-3net cross-network evaluation.

Trains GraphCVAEAugmenter on Anytown + Hanoi, then evaluates zero-shot
transfer to Net3, D-Town, L-TOWN — matching the exact protocol used in
run_cross_network_eval.py so results are directly comparable.

Usage::

    python scripts/run_graph_cvae_lolo.py --seed 0 \
        --output-dir dev/active/artifacts/graph_cvae_lolo3/seed_0

Run once per seed (0-4).  Results are aggregated by
aggregate_graph_cvae_lolo.py.
"""

from __future__ import annotations

import argparse
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

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.data import (          # noqa: E402
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_path,
    simulate_many,
)
from watergen.evaluation import binary_report, report_as_dict  # noqa: E402
from watergen.models import (         # noqa: E402
    GraphCVAEAugmenter,
    QuantilePlausibilityFilter,
    TabularDetectorBaseline,
    prepare_feature_matrix,
)

# ---------------------------------------------------------------------------
# Protocol constants — must match run_cross_network_eval.py exactly
# ---------------------------------------------------------------------------
TRAIN_NETWORK_IDS = ["anytown", "hanoi"]
TEST_NETWORK_IDS  = ["net3", "d_town", "l_town"]

FEATURE_COLUMNS = [
    "pressure_mean", "pressure_min", "pressure_max", "pressure_std",
    "demand_mean",   "demand_sum",   "demand_std",
    "flow_mean",     "flow_abs_mean","flow_std",
    "tank_head_mean","tank_head_std",
    "sensor_coverage","observed_junction_count","observed_link_count",
    "leak_area",
]

SENSOR_COVERAGE_LEVELS = [0.25, 0.50]
DISTURBANCE_TYPES      = ["leak", "pipe_closure", "pump_outage"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _build_network_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = _load_yaml(manifest_path)
    entries  = collect_benchmark_entries(manifest)
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
            print(f"[warn] '{nid}' not in manifest — skipping", file=sys.stderr)
            continue
        resolved = network_index[nid]["resolved_path"]
        if not resolved.exists():
            print(f"[warn] Path missing: {resolved} — skipping", file=sys.stderr)
            continue
        entries.append({"id": nid, "stage": "train", "path": str(resolved),
                         "split": split_name, "split_role": split_name})
    return {"partitions": {split_name: entries}}


def _simulate_networks(network_ids: list[str], network_index: dict,
                        split_name: str) -> pd.DataFrame:
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
                print(f"[warn] No scenarios for '{nid}'", file=sys.stderr)
                continue
            df = simulate_many(network_specs, sensor_coverage_levels=SENSOR_COVERAGE_LEVELS)
            frames.append(df)
            print(f"[info]  '{nid}': {len(df)} rows")
        except Exception as exc:
            print(f"[warn] Simulation failed for '{nid}': {exc}", file=sys.stderr)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _evaluate(detector: TabularDetectorBaseline,
              test_df: pd.DataFrame, network_id: str) -> dict[str, Any]:
    if test_df.empty:
        return {"network_id": network_id, "error": "empty test frame", "auprc": None}
    available = [c for c in FEATURE_COLUMNS if c in test_df.columns]
    if not available:
        return {"network_id": network_id, "error": "no features", "auprc": None}
    try:
        x, y, _ = prepare_feature_matrix(test_df, target_column="event_label",
                                          feature_columns=available)
        proba = detector.predict_proba(x)[:, 1]
        pred  = detector.predict(x)
        rep   = binary_report(y, pred, proba)
        out   = report_as_dict(rep)
        out["network_id"]    = network_id
        out["rows"]          = int(len(test_df))
        out["positive_rows"] = int(test_df["event_label"].sum())
        return out
    except Exception as exc:
        return {"network_id": network_id, "error": str(exc), "auprc": None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(manifest_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Seed: {seed}")
    print("[info] Building network index...")
    network_index = _build_network_index(manifest_path)

    available_train = [n for n in TRAIN_NETWORK_IDS if n in network_index]
    available_test  = [n for n in TEST_NETWORK_IDS  if n in network_index]

    # ---- Simulate training data ----
    print(f"[info] Simulating TRAIN: {available_train}")
    train_df = _simulate_networks(available_train, network_index, "train")
    if train_df.empty:
        print("[error] Empty training data", file=sys.stderr)
        return {"error": "empty training data"}
    print(f"[info] Train rows: {len(train_df)}, "
          f"positive: {int(train_df['event_label'].sum())}")

    # ---- Simulate test data ----
    print(f"[info] Simulating TEST: {available_test}")
    test_frames: dict[str, pd.DataFrame] = {}
    for nid in available_test:
        df = _simulate_networks([nid], network_index, "test")
        test_frames[nid] = df
        print(f"[info]  '{nid}': {len(df)} rows")

    train_split = train_df[train_df["split"] == "train"].copy()
    if train_split.empty:
        train_split = train_df.copy()

    available_feats = [c for c in FEATURE_COLUMNS if c in train_split.columns]

    # ---- Find benchmark path for graph construction (first train network) ----
    benchmark_path: str | None = None
    for nid in available_train:
        rp = network_index[nid]["resolved_path"]
        if rp.exists():
            benchmark_path = str(rp)
            break

    # ---- Train Graph-CVAE ----
    print(f"[info] Training GraphCVAEAugmenter (seed={seed}, "
          f"benchmark={Path(benchmark_path).name if benchmark_path else 'tabular'})...")
    augmenter = GraphCVAEAugmenter(
        benchmark_path=benchmark_path,
        latent_dim=64,
        hidden_dim=128,
        epochs=100,
        lr=1e-3,
        batch_size=32,
        random_state=seed,
        lambda_physics=0.5,
        warmup_epochs=25,
        device="cpu",
    )
    augmenter.fit(train_split, feature_columns=available_feats, target_column="event_label")

    # ---- Generate synthetic positives ----
    print("[info] Generating synthetic samples...")
    try:
        plausibility_filter = QuantilePlausibilityFilter().fit(train_split, available_feats)
        synthetic = augmenter.generate(
            train_split,
            feature_columns=available_feats,
            target_column="event_label",
        )
        if not synthetic.empty:
            synthetic = plausibility_filter.filter(synthetic, available_feats)
        if synthetic.empty:
            print("[warn] All synthetic rows filtered — using unaugmented train", file=sys.stderr)
            augmented_train = train_split.copy()
        else:
            augmented_train = pd.concat(
                [train_split.assign(is_synthetic=0), synthetic],
                ignore_index=True,
            )
        print(f"[info] Augmented train rows: {len(augmented_train)} "
              f"(+{len(synthetic)} synthetic)")
    except Exception as exc:
        print(f"[warn] Graph-CVAE generation failed: {exc} — using baseline train", file=sys.stderr)
        augmented_train = train_split.copy()

    # ---- Train RF detector on augmented data ----
    print("[info] Training RF detector on Graph-CVAE augmented data...")
    detector = TabularDetectorBaseline(algorithm="random_forest", random_state=seed)
    try:
        detector.fit_from_frame(augmented_train, feature_columns=available_feats)
    except Exception as exc:
        print(f"[error] Detector training failed: {exc}", file=sys.stderr)
        return {"error": str(exc)}

    # ---- Evaluate on test networks ----
    per_network: dict[str, dict[str, Any]] = {}
    for nid, test_df in test_frames.items():
        result = _evaluate(detector, test_df, nid)
        per_network[nid] = result
        auprc_str = f"{result['auprc']:.3f}" if isinstance(result.get("auprc"), (int, float)) else "n/a"
        print(f"[info]  '{nid}' AUPRC={auprc_str}")

    # Equal-weight macro AUPRC (matches aggregate_lolo_seeds.py)
    valid_auprcs = [v["auprc"] for v in per_network.values()
                    if isinstance(v.get("auprc"), (int, float))]
    macro_auprc = sum(valid_auprcs) / len(valid_auprcs) if valid_auprcs else None

    results: dict[str, Any] = {
        "protocol":       "LOLO-3net",
        "augmenter":      "graph_cvae",
        "seed":           seed,
        "train_networks": available_train,
        "test_networks":  available_test,
        "train_rows":     int(len(train_split)),
        "augmented_rows": int(len(augmented_train)),
        "synthetic_rows": int(len(synthetic)) if not synthetic.empty else 0,
        "per_network":    per_network,
        "macro_auprc":    macro_auprc,
        "augmenter_note": (
            "graph_cvae: GraphCVAEAugmenter with GATv2Conv encoder + "
            "Hazen-Williams physics loss. Trained on Anytown+Hanoi, "
            "zero-shot evaluation on Net3/D-Town/L-TOWN."
        ),
    }

    _write_json(results, output_dir / "graph_cvae_lolo_results.json")
    print(f"[info] Results -> {output_dir / 'graph_cvae_lolo_results.json'}")
    print(f"[info] Macro AUPRC (equal-weight): {macro_auprc}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-CVAE LOLO-3net cross-network evaluation (one seed)."
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ACTIVE_ROOT / "artifacts" / "graph_cvae_lolo3" / "seed_0",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()

    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = (REPO_ROOT / manifest).resolve()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    if not manifest.exists():
        print(f"[error] Manifest not found: {manifest}", file=sys.stderr)
        return 1

    result = run(manifest, output_dir, seed=args.seed)
    if "error" in result:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
