"""Validation package runner for hybrid benchmark + triage paper.

Sweeps seeds and selected evaluation settings, producing a canonical validation
artifact package:
- experiment_manifest.json
- per-run output directories
- summary pointers for aggregation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _run(cmd: list[str], *, cwd: Path) -> int:
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation package sweeps")
    parser.add_argument("--output-dir", type=Path, default=ACTIVE_ROOT / "artifacts" / "validation_package")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--coverages", type=float, nargs="*", default=[0.25, 0.5])
    parser.add_argument("--disturbances", nargs="*", default=["leak", "pipe_closure", "pump_outage"])
    parser.add_argument("--min-timesteps", type=int, nargs="*", default=[24, 100])
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_name": "hybrid_benchmark_triage_validation",
        "benchmark_protocol": "LOLO-3net corrected benchmark + scenario-level validation",
        "train_networks": ["anytown", "hanoi"],
        "test_networks": ["net3", "d_town", "l_town"],
        "seeds": args.seeds,
        "models": ["rf", "lstm_ae", "triage_reranker"],
        "coverages": args.coverages,
        "disturbances": args.disturbances,
        "window_configs": [{"min_timesteps": x} for x in args.min_timesteps],
        "runs": [],
    }

    # Row-level LOLO sweeps
    for seed in args.seeds:
        run_dir = output_dir / "cross_network" / f"seed_{seed}"
        cmd = [
            sys.executable,
            str(ACTIVE_ROOT / "scripts" / "run_cross_network_eval.py"),
            "--seed", str(seed),
            "--output-dir", str(run_dir),
            "--sensor-coverage", *[str(x) for x in args.coverages],
            "--disturbance-types", *list(args.disturbances),
        ]
        code = _run(cmd, cwd=REPO_ROOT)
        manifest["runs"].append({
            "kind": "cross_network",
            "seed": seed,
            "output_dir": str(run_dir),
            "return_code": code,
        })

    # Scenario-level validation sweeps
    for seed in args.seeds:
        for min_timesteps in args.min_timesteps:
            run_dir = output_dir / "scenario_eval" / f"seed_{seed}__mint_{min_timesteps}"
            cmd = [
                sys.executable,
                str(ACTIVE_ROOT / "scripts" / "run_hierarchical_eval.py"),
                "--dataset-dir", str(ACTIVE_ROOT / "artifacts" / "datasets" / "sequence"),
                "--output-dir", str(run_dir),
                "--seed", str(seed),
                "--coverages", *[str(x) for x in args.coverages],
                "--disturbances", *list(args.disturbances),
                "--min-timesteps", str(min_timesteps),
            ]
            code = _run(cmd, cwd=REPO_ROOT)
            manifest["runs"].append({
                "kind": "scenario_eval",
                "seed": seed,
                "min_timesteps": min_timesteps,
                "output_dir": str(run_dir),
                "return_code": code,
            })

    _write_json(manifest, output_dir / "experiment_manifest.json")
    print(f"[info] Manifest -> {output_dir / 'experiment_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
