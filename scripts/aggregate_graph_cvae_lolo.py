"""Aggregate Graph-CVAE LOLO-3net results across 5 seeds.

Reads seed_{0-4}/graph_cvae_lolo_results.json and produces:
  - aggregated_graph_cvae_lolo_results.json   (mean ± std per network)
  - aggregated_graph_cvae_lolo_table.md       (LaTeX-ready markdown)

Usage::
    python scripts/aggregate_graph_cvae_lolo.py \
        --input-dir dev/active/artifacts/graph_cvae_lolo3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def aggregate(input_dir: Path) -> dict:
    seed_dirs = sorted(input_dir.glob("seed_*"))
    if not seed_dirs:
        print(f"[error] No seed_* directories found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    per_network_auprcs: dict[str, list[float]] = {}
    n_valid_seeds: dict[str, int] = {}
    seeds_found: list[int] = []

    for sd in seed_dirs:
        result_file = sd / "graph_cvae_lolo_results.json"
        if not result_file.exists():
            print(f"[warn] Missing: {result_file}", file=sys.stderr)
            continue
        data = _load_json(result_file)
        seed = data.get("seed", "?")
        seeds_found.append(seed)

        for nid, net_result in data.get("per_network", {}).items():
            auprc = net_result.get("auprc")
            if isinstance(auprc, (int, float)) and auprc == auprc:  # not NaN
                per_network_auprcs.setdefault(nid, []).append(float(auprc))
                n_valid_seeds[nid] = n_valid_seeds.get(nid, 0) + 1

    print(f"[info] Seeds aggregated: {sorted(seeds_found)}")

    # Compute mean ± std per network
    network_stats: dict[str, dict] = {}
    all_means: list[float] = []
    for nid, auprcs in per_network_auprcs.items():
        mean = float(np.mean(auprcs))
        std  = float(np.std(auprcs, ddof=0))  # population std (matches existing aggregation)
        network_stats[nid] = {
            "mean":       round(mean, 3),
            "std":        round(std,  3),
            "n_seeds":    n_valid_seeds[nid],
            "auprc_values": [round(v, 4) for v in auprcs],
        }
        all_means.append(mean)
        print(f"[info]  {nid}: AUPRC={mean:.3f} ± {std:.3f}  (n={n_valid_seeds[nid]} seeds)")

    macro_auprc = float(np.mean(all_means)) if all_means else None
    macro_std   = float(np.std(all_means, ddof=0)) if len(all_means) > 1 else 0.0

    # Bootstrap 95% CI across network means (matches aggregate_lolo_seeds.py)
    rng = np.random.default_rng(0)
    if all_means:
        boot = [float(np.mean(rng.choice(all_means, size=len(all_means), replace=True)))
                for _ in range(2000)]
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    else:
        ci_lo, ci_hi = 0.0, 0.0

    result = {
        "augmenter":     "graph_cvae",
        "seeds":         sorted(seeds_found),
        "per_network":   network_stats,
        "macro_auprc":   round(macro_auprc, 3) if macro_auprc else None,
        "macro_std":     round(macro_std, 3),
        "ci_95":         [round(ci_lo, 3), round(ci_hi, 3)],
        "note": (
            "Graph-CVAE (GATv2Conv encoder, Hazen-Williams physics loss). "
            "Trained on Anytown + Hanoi. Zero-shot test on Net3, D-Town, L-TOWN. "
            "Macro AUPRC = equal-weight average across test networks. "
            "D-Town note: EpanetSimulator instability may produce empty test frames "
            "for some seeds — n_seeds reported per network."
        ),
    }

    return result


def write_markdown_table(agg: dict, output_path: Path) -> None:
    net_order = ["net3", "d_town", "l_town"]
    pn = agg.get("per_network", {})

    lines = [
        "# Graph-CVAE LOLO-3net Aggregated Results",
        "",
        "> Note: Graph-CVAE (GATv2Conv encoder + Hazen-Williams physics loss).",
        "> Trained on Anytown + Hanoi. Zero-shot evaluation on Net3, D-Town, L-TOWN.",
        "> D-Town: EpanetSimulator instability may reduce effective seed count.",
        "",
        "| Method | Net3 | D-Town† | L-TOWN | Macro AUPRC | 95% CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    cells = []
    for nid in net_order:
        stats = pn.get(nid, {})
        m   = stats.get("mean", None)
        s   = stats.get("std",  None)
        n   = stats.get("n_seeds", "?")
        dagger = "†" if nid == "d_town" else ""
        if m is not None:
            cells.append(f"{m:.3f} ±{s:.3f}" + (f" (n={n})" if nid == "d_town" else ""))
        else:
            cells.append("n/a")

    macro = agg.get("macro_auprc")
    ci    = agg.get("ci_95", [None, None])
    macro_str = f"{macro:.3f}" if macro is not None else "n/a"
    ci_str    = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a"

    row = f"| Graph-CVAE | {' | '.join(cells)} | {macro_str} | {ci_str} |"
    lines.append(row)
    lines.append("")
    lines.append("†D-Town: n_seeds may be <5 due to EpanetSimulator instability.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[info] Markdown table -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Graph-CVAE LOLO-3net results across seeds."
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("dev/active/artifacts/graph_cvae_lolo3"),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        # resolve relative to repo root (3 levels up from scripts/)
        script_root = Path(__file__).resolve().parents[3]
        input_dir = (script_root / input_dir).resolve()

    print(f"[info] Aggregating from: {input_dir}")

    agg = aggregate(input_dir)

    out_json = input_dir / "aggregated_graph_cvae_lolo_results.json"
    _write_json(agg, out_json)
    print(f"[info] JSON -> {out_json}")

    write_markdown_table(agg, input_dir / "aggregated_graph_cvae_lolo_table.md")

    print(f"\n[summary] Graph-CVAE Macro AUPRC: {agg['macro_auprc']} "
          f"95%CI {agg['ci_95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
