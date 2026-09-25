# WDN-AugBench

Reproducibility code and summary artifacts for the paper:

> **Cross-Network Monitoring of Rare Disturbance Scenarios in Water Distribution Networks:
> Why Synthetic Augmentation Fails and Event-Level Accuracy Overstates Early Warning**
> (Y. Zhu and Q. Liu, submitted to the *Journal of Hydroinformatics*, 2026).

The paper introduces **Row-Scenario Inflation (RSI)** — a diagnostic for when per-timestep
(row-level) method comparisons fail to carry to the per-event (scenario) level — and
**WDN-AugBench**, a held-out-network transfer benchmark over EPANET water distribution
networks. This repository contains the evaluation code, the RSI reference implementation,
the simulation/experiment configurations, and the summary result artifacts that the
manuscript's tables and figures are computed from.

## Repository layout

```
src/        Python package: simulation, feature extraction, detectors, augmenters,
            the Graph-CVAE, the RSI metric, and the hierarchical-bootstrap helper.
scripts/    Runnable experiment drivers (one per result; see the map below).
configs/    Benchmark manifest and experiment configuration files.
artifacts/  Summary / aggregated / results JSON files (the values reported in the paper).
            NOTE: the full per-row artifacts (~8.3 GB; see "Data and artifacts") are NOT
            committed here — they regenerate from the scripts.
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

Hydraulic simulation uses [WNTR](https://github.com/USEPA/WNTR) (bundles EPANET). A CUDA GPU
is optional and used only by the graph models.

## Reproducing the paper

Each result maps to a driver script (run from the repo root):

| Manuscript result | Script(s) |
|---|---|
| LOLO transfer benchmark, 8 held-out networks (Table: `tab:lolo`) | `run_expanded_lolo.py` |
| Original 3-network LOLO + per-row triage records | `run_lolo_5seed.py`, `aggregate_lolo_seeds.py` |
| Balanced-endpoint RSI of the aggregate-residual detector (Table: `tab:rsi-balanced`) | `run_nonsat_hydraulic_rsi.py` |
| Balanced non-saturated RSI (Table: `tab:nonsat-rsi`) | `run_nonsat_endpoint.py`, `compute_nonsat_rsi.py` |
| Saturated-label RSI cross-check (Table: `tab:meanpool-rsi`) | `compute_meanpool_rsi.py` |
| Leak-only difficulty sweep | `run_leak_only_lolo.py` |
| Intra-scale (large-to-large) LOLO fold | `run_intrascale_lolo.py` |
| Classifier-generality check (RF vs LR) | `run_lolo_classifier_check.py` |
| Sensitivity analysis | `run_sensitivity_analysis.py` |
| Early-warning / temporal-localization endpoint | `compute_early_warning_endpoint.py` |
| GCN / Graph-CVAE transfer | `run_graph_cvae_lolo_5seed.py`, `run_topology_gnn_lolo.py` |
| Post-hoc triage diagnostics (exploratory) | `run_triage_validation.py` |
| Provenance table (script / artifact / sha256 / freshness per table and figure) | `build_provenance.py` |
| Network-pool screen (held-out expansion) | network-expansion screen under `artifacts/network_expansion_screen/` |

All experiments use seeds 42–46. The **RSI metric** and its hierarchical-bootstrap confidence
intervals are implemented in `src/` and exercised by `compute_nonsat_rsi.py` (importable for
application to your own per-row scores, labels, and scenario index).

## Data and artifacts

**Inputs (network models).** Experiments run on EPANET `.inp` network files from the
[EPANET-Benchmarks](https://github.com/OpenWaterAnalytics) suite, [WNTR](https://github.com/USEPA/WNTR),
the BattLeDIM / BATADAL competitions, and a curated set of synthetic/mined topologies. These
network files are publicly available from those sources and are not redistributed here.

**Generated artifacts (~8.3 GB total).** Running the pipeline regenerates a large set of
intermediate and result files. This is *generated* output (from WNTR Extended-Period
Simulations → per-row feature/score extraction → detector training/evaluation across
networks × seeds × augmenters × disturbance types), **not** downloaded data. The size is
dominated by per-row CSV records:

| Type | Size | What it is |
|---|---|---|
| per-row CSV | ~8.1 GB (155 files) | one row per timestep × sensor: features, detector scores, hydraulic residuals, labels — in `per_disturbance_lolo/`, `lolo_5seed_revised/`, `intrascale_lolo/`, `validation_package_full/` |
| simulated datasets + `.npz` | ~0.6 GB | simulated scenario tensors and generated-sample arrays |
| model weights (`.pt`), logs, figures | ~0.02 GB | trained detector/generator checkpoints, run logs, plots |
| summary / aggregated JSON | ~0.02 GB | **shipped in this repo** — the exact values the paper reports |

To keep the repository lightweight, only the **summary/aggregated/results JSON** files
(~20 MB) are committed; the full ~8.3 GB regenerates deterministically by running the
scripts above, and the complete archive will be deposited on Zenodo for the camera-ready
release (DOI minted at acceptance).

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code or the RSI diagnostic, please cite the paper above. A BibTeX entry will
be added with the published version.
