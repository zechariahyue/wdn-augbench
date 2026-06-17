"""Standalone Graph-CVAE pre-training script.

Loads training networks from a benchmark manifest, runs WNTR simulations to
obtain training data, trains the GraphCVAEAugmenter, and saves both the model
weights and the training curve.

Usage::

    python scripts/train_graph_cvae.py \\
        --manifest dev/active/configs/benchmark_manifest.yaml \\
        --networks net1 anytown hanoi net3 \\
        --epochs 100 \\
        --output-dir dev/active/artifacts/models/graph_cvae

Output artefacts written to *--output-dir*:

- ``graph_cvae_weights.pt``      — trained model weights (torch.save)
- ``training_curve.json``        — loss history per epoch (list of dicts)
- ``training_curve.png``         — matplotlib figure (loss vs epoch)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve source root so the script can be run from any working directory.
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = ACTIVE_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from watergen.data import (  # noqa: E402
    build_leak_scenarios,
    collect_benchmark_entries,
    resolve_split_manifest,
    simulate_many,
)
from watergen.models import GraphCVAEAugmenter  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at top level: {path}")
    return data


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _save_training_curve(history: list[dict], output_dir: Path) -> None:
    """Persist training history as JSON and as a matplotlib figure."""
    _write_json(history, output_dir / "training_curve.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = list(range(1, len(history) + 1))
        total = [h.get("total", 0.0) for h in history]
        recon = [h.get("recon", 0.0) for h in history]
        kl = [h.get("kl", 0.0) for h in history]
        phys = [h.get("phys", 0.0) for h in history]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.suptitle("Graph-CVAE Training Curve", fontsize=13)

        axes[0, 0].plot(epochs, total, color="steelblue")
        axes[0, 0].set_title("Total loss")
        axes[0, 0].set_xlabel("Epoch")

        axes[0, 1].plot(epochs, recon, color="darkorange")
        axes[0, 1].set_title("Reconstruction loss")
        axes[0, 1].set_xlabel("Epoch")

        axes[1, 0].plot(epochs, kl, color="seagreen")
        axes[1, 0].set_title("KL divergence")
        axes[1, 0].set_xlabel("Epoch")

        axes[1, 1].plot(epochs, phys, color="firebrick")
        axes[1, 1].set_title("Physics constraint loss")
        axes[1, 1].set_xlabel("Epoch")

        plt.tight_layout()
        fig_path = output_dir / "training_curve.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[info] Training curve saved -> {fig_path}")
    except Exception as exc:
        print(f"[warn] Could not save training curve figure: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------

def train(
    manifest_path: Path,
    *,
    network_ids: list[str] | None,
    epochs: int,
    output_dir: Path,
    lr: float,
    latent_dim: int,
    hidden_dim: int,
    lambda_physics: float,
    warmup_epochs: int,
    batch_size: int,
    device: str,
) -> None:
    """Load data, train GraphCVAEAugmenter, and save artefacts.

    Args:
        manifest_path: Path to the benchmark manifest YAML.
        network_ids: Optional whitelist of benchmark IDs.  ``None`` means all.
        epochs: Training epochs.
        output_dir: Directory for output artefacts.
        lr: Adam learning rate.
        latent_dim: Latent space dimension.
        hidden_dim: Hidden layer width.
        lambda_physics: Physics loss weight.
        warmup_epochs: Epoch at which physics lambda reaches 1.0.
        batch_size: Rows per gradient step.
        device: Torch device string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # repo_root is 3 levels up from scripts/ (scripts -> active -> dev -> repo)
    # but also handle running from arbitrary paths by using the absolute manifest path
    repo_root = manifest_path.resolve().parent
    while repo_root.name not in ("", "LLM water") and repo_root != repo_root.parent:
        if (repo_root / "dev").exists():
            break
        repo_root = repo_root.parent

    manifest = _load_yaml(manifest_path)
    all_entries = collect_benchmark_entries(manifest)

    if network_ids:
        allowed = {nid.strip() for nid in network_ids}
        all_entries = [e for e in all_entries if str(e.get("id", "")).strip() in allowed]

    if not all_entries:
        print("[error] No benchmark entries found after filtering.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Training on {len(all_entries)} network(s).")

    # ---- Build scenario specs and simulate --------------------------------
    # Use a simple stub split manifest so simulate_many works.
    stub_split: dict = {
        "partitions": {
            "train": [
                {
                    "id": str(e.get("id", "unknown")),
                    "path": str(_resolve_path(str(e.get("path", "")), root=repo_root)),
                    "stage": str(e.get("stage", "train")),
                }
                for e in all_entries
                if e.get("path")
            ]
        }
    }

    try:
        scenario_specs = build_leak_scenarios(
            stub_split,
            smoke_splits=["train"],
            leak_area_values=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005],
            max_candidates_per_network=5,
            disturbance_types=["leak", "pipe_closure", "pump_outage"],
            include_baseline=True,
        )
    except Exception as exc:
        print(f"[error] Scenario building failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not scenario_specs:
        print("[error] No scenarios generated.", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Simulating {len(scenario_specs)} scenarios…")
    try:
        dataset = simulate_many(scenario_specs, sensor_coverage_levels=[1.0])
    except Exception as exc:
        print(f"[error] Simulation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[info] Dataset: {len(dataset)} rows, "
          f"{int(dataset['event_label'].sum())} positive.")

    feature_columns = [
        "pressure_mean",
        "pressure_min",
        "pressure_max",
        "pressure_std",
        "demand_mean",
        "demand_sum",
        "demand_std",
        "flow_mean",
        "flow_abs_mean",
        "flow_std",
        "tank_head_mean",
        "tank_head_std",
        "sensor_coverage",
        "observed_junction_count",
        "observed_link_count",
        "leak_area",
    ]
    feature_columns = [c for c in feature_columns if c in dataset.columns]

    # ---- Pick a representative benchmark path for graph construction ------
    benchmark_path: str | None = None
    if all_entries:
        raw = str(all_entries[0].get("path", ""))
        if raw:
            resolved = _resolve_path(raw, root=repo_root)
            if resolved.exists():
                benchmark_path = str(resolved)

    if benchmark_path:
        print(f"[info] Graph-CVAE using benchmark: {benchmark_path}")
    else:
        print("[info] No benchmark path resolved; using tabular CVAE mode.")

    # ---- Instantiate and train augmenter ----------------------------------
    augmenter = GraphCVAEAugmenter(
        benchmark_path=benchmark_path,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        random_state=42,
        lambda_physics=lambda_physics,
        warmup_epochs=warmup_epochs,
        device=device,
    )

    augmenter.fit(dataset, feature_columns=feature_columns, target_column="event_label")

    # ---- Persist weights --------------------------------------------------
    weights_path = output_dir / "graph_cvae_weights.pt"
    try:
        augmenter.save(weights_path)
        print(f"[info] Model weights saved -> {weights_path}")
    except Exception as exc:
        print(f"[warn] Could not save weights: {exc}", file=sys.stderr)

    # ---- Persist training curve -------------------------------------------
    _save_training_curve(augmenter.training_history_, output_dir)
    print(f"[info] Training curve JSON -> {output_dir / 'training_curve.json'}")
    print("[info] Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train Graph-CVAE augmenter on a set of EPANET networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to benchmark manifest YAML (e.g. dev/active/configs/benchmark_manifest.yaml).",
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        metavar="ID",
        default=None,
        help="Whitelist of benchmark IDs to train on.  Omit to use all entries.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ACTIVE_ROOT / "artifacts" / "models" / "graph_cvae",
        help="Directory for output artefacts (weights, curves).",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent space dimension.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer width.")
    parser.add_argument(
        "--lambda-physics", type=float, default=0.5, help="Physics constraint loss weight."
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=25,
        help="Epoch at which physics lambda warmup reaches 1.0.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device (cpu/cuda).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.manifest.exists():
        print(f"[error] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    train(
        manifest_path=args.manifest,
        network_ids=args.networks,
        epochs=args.epochs,
        output_dir=args.output_dir,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_physics=args.lambda_physics,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
