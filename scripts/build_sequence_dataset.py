"""Build scenario-level sequence dataset from EPANET benchmark simulations.

Outputs one .npz per scenario and an index.csv with metadata.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = ACTIVE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.data import build_leak_scenarios, collect_benchmark_entries, load_yaml_file, resolve_path, simulate_scenario_raw  # noqa: E402


def _build_network_index(manifest_path: Path) -> dict[str, dict]:
    manifest = load_yaml_file(manifest_path)
    entries = collect_benchmark_entries(manifest)
    out: dict[str, dict] = {}
    for entry in entries:
        nid = str(entry.get("id", "")).strip()
        if not nid:
            continue
        out[nid] = {
            **entry,
            "resolved_path": resolve_path(str(entry.get("path", "")), root=REPO_ROOT),
        }
    return out


def _make_split(network_ids: list[str], network_index: dict[str, dict], split_name: str) -> dict:
    entries = []
    for nid in network_ids:
        if nid not in network_index:
            continue
        entry = network_index[nid]
        if not Path(entry["resolved_path"]).exists():
            continue
        entries.append(
            {
                "id": nid,
                "stage": entry.get("stage", "unknown"),
                "path": str(entry["resolved_path"]),
                "role": entry.get("role", ""),
                "notes": entry.get("notes", ""),
                "split": split_name,
                "split_role": split_name,
            }
        )
    return {"partitions": {split_name: entries}}


def build_dataset(manifest_path: Path, output_dir: Path, *, coverages: list[float], disturbance_types: list[str], train_ids: list[str], test_ids: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    network_index = _build_network_index(manifest_path)

    rows: list[dict[str, object]] = []
    for split_name, ids in [("train", train_ids), ("test", test_ids)]:
        resolved = _make_split(ids, network_index, split_name)
        specs = build_leak_scenarios(
            resolved,
            smoke_splits=[split_name],
            leak_area_values=[0.0001, 0.0005],
            max_candidates_per_network=2,
            disturbance_types=disturbance_types,
            include_baseline=True,
        )
        for spec in specs:
            try:
                df = simulate_scenario_raw(spec, sensor_coverage_levels=coverages)
            except Exception as exc:
                print(f"[warn] Failed scenario {spec.scenario_id}: {exc}", file=sys.stderr)
                continue
            feature_cols = [c for c in df.columns if c.startswith("pressure__") or c.startswith("demand__") or c.startswith("flow__")]
            for coverage, cov_df in df.groupby("sensor_coverage"):
                cov_df = cov_df.sort_values("time_seconds").reset_index(drop=True)
                x = cov_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                y_t = cov_df["event_label"].to_numpy(dtype=np.int64)
                y = int(y_t.max())
                if y == 0 and spec.kind != "baseline":
                    print(
                        f"[warn] Skipping {spec.scenario_id}: non-baseline scenario produced 0 positive rows",
                        file=sys.stderr,
                    )
                    continue
                out_name = f"{spec.scenario_id}__cov_{float(coverage):.2f}.npz".replace("/", "_")
                out_path = output_dir / out_name
                np.savez_compressed(
                    out_path,
                    x=x,
                    y_t=y_t,
                    scenario_id=np.array(spec.scenario_id),
                    network_id=np.array(spec.benchmark_id),
                    disturbance=np.array(spec.kind),
                    coverage=np.array(float(coverage), dtype=np.float32),
                    feature_names=np.array(feature_cols, dtype=object),
                )
                rows.append(
                    {
                        "file": out_name,
                        "scenario_id": spec.scenario_id,
                        "network_id": spec.benchmark_id,
                        "disturbance": spec.kind,
                        "coverage": float(coverage),
                        "split": split_name,
                        "timesteps": int(len(cov_df)),
                        "channels": int(len(feature_cols)),
                        "scenario_label": y,
                    }
                )
                print(f"[info] Saved {out_name} -> x={x.shape} y={y}")

    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"[info] Wrote index: {index_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scenario-level sequence dataset")
    parser.add_argument("--manifest", type=Path, default=ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml")
    parser.add_argument("--output-dir", type=Path, default=ACTIVE_ROOT / "artifacts" / "datasets" / "sequence")
    parser.add_argument("--coverages", type=float, nargs="*", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--disturbance-types", nargs="*", default=["leak", "pipe_closure", "pump_outage"])
    args = parser.parse_args()

    train_ids = ["net1", "anytown", "hanoi", "net3"]
    test_ids = ["d_town", "l_town"]
    build_dataset(args.manifest, args.output_dir, coverages=list(args.coverages), disturbance_types=list(args.disturbance_types), train_ids=train_ids, test_ids=test_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
