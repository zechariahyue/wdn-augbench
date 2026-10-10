# WDN-AugBench

Reproducibility code and summary artifacts for the paper:

> **What Transfers Across Water Distribution Networks: Normalised Features, Not Synthetic Data**
> (Y. Zhu and Q. Liu, submitted to the *Journal of Hydroinformatics*, 2026).

WDN-AugBench is a held-out-network transfer benchmark for disturbance detection (leaks, pipe
closures, pump outages) on EPANET water distribution networks. **This is the v2 benchmark
(2026-10-04).** The first version is withdrawn: its detector feature frame contained
`leak_area` (the true emitter area of the simulated scenario) and two network-size counts,
and that single column reproduced its headline results (`scripts/run_leak_area_ablation.py`,
`artifacts/leak_area_ablation/`). The v1 scripts and summary artifacts are retained under
their original names for the record and are not used by any v2 result.

## v2 protocol (what the paper reports)

- **Split.** Train on Anytown + Hanoi; test on eight held-out networks (Net3, D-Town, L-TOWN,
  ky2, ky3, ky5, ky6, ky7). Fixed small-to-large split, fitted once per seed.
- **Normals.** Every network carries **N = 100 normal-operation scenarios** with distinct seeded
  ±20 % demand noise; disturbance scenarios (30 leaks = 5 junctions × 6 emitter areas, 3 pipe
  closures, ≤ 3 pump outages) receive the same seeded noise. Row prior ≈ 0.08–0.10, scenario
  prior ≈ 0.25–0.27.
- **Sensors.** Uniform random junction/link subset per (seed, network, coverage), shared by all
  scenarios of that network; coverages 0.25 and 0.50; seeds 42–46 (seed std includes placement).
- **Onsets.** Leak onset uniform in [1 step, H/2], duration uniform in [H/6, H/3]; closures and
  outages are active from t = 0 (no onset; excluded from WSTL/TTD).
- **Detector inputs (`FEATURE_COLUMNS_V2`).** 15 aggregate sensor features — pressure
  mean/min/max/std, demand mean/sum/std, flow mean/|mean|/std, tank-head mean/std, pressure
  trend, cumulative pressure trend, demand trend — **z-scored against the same network's
  normal-operation scenarios at the same coverage** (`src/watergen/data/normalise.py`,
  label-free), plus the coverage fraction. No scenario metadata; `assert_no_scenario_metadata`
  guards every feature list and `tests/test_v2_design.py` enforces it. `rf_raw_baseline` is the
  identical Random Forest on the un-normalised aggregates (the normalisation ablation).
- **Arms.** RF (200 trees, balanced) with no augmentation / Gaussian noise / GMM (K = 2, full
  covariance) / SMOTE (k = 5), each augmenter adding one synthetic positive per real positive
  behind a 1–99 % quantile plausibility screen (+10 % margin) fitted on the real positive rows;
  logistic regression; sample-space k-NN GCN; and a label-free residual comparator
  (mass/energy residual proxies standardised by the network's normals, causal trailing z-score,
  4-step warm-up, 3-step persistence) built in `analyze_v2.py`.
- **Metrics.** Row AUPRC and scenario AUPRC (max-pool; mean-pool as a check) with their priors;
  per-disturbance slices with slice priors; WSTL and first-3h/TTD on leak scenarios at ≤ 1 alarm
  per normal scenario; RSI with a pooled scenario bootstrap; paired Wilcoxon over the 8 networks,
  Kentucky-family check, network bootstrap, MDE; scenario-prior sweep (20/40/100 normals).

Headline (5 seeds, equal-weight macro over 8 networks, row prior 0.091): normalised RF 0.235;
RF on raw features 0.131; RF + noise 0.256, + GMM 0.234, + SMOTE 0.233 (all nulls); LR 0.226;
GCN 0.225; residual comparator 0.092.

## Repository layout

```
src/        Python package: simulation, feature extraction, per-network normalisation,
            detectors, augmenters, and the v1 models retained for the record.
scripts/    Runnable experiment drivers. v2: run_v2_benchmark.py, analyze_v2.py,
            make_v2_tables.py, run_v2_splitcal.py, run_leak_area_ablation.py, build_provenance.py (see the map below).
configs/    Benchmark manifest and experiment configuration files.
artifacts/  v2_benchmark/analysis/v2_summary.json + v2_tables.md (every number the paper
            reports) and per-seed summaries; leak_area_ablation/ (the v1 leak diagnostic);
            the v1 summary JSONs under their original names. The per-seed v2 record files
            (records_seed{42..46}.parquet, ~20 MB each) and the v1 per-row CSVs are NOT
            committed — they regenerate from the scripts.
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -r requirements.txt
```

Hydraulic simulation uses [WNTR](https://github.com/USEPA/WNTR) (bundles EPANET). A CUDA GPU
is optional and used only by the graph models.

## Reproducing the paper

Three commands regenerate every table and macro in the manuscript (run from the repo root):

| Step | Script | Output |
|---|---|---|
| 1. Simulate, normalise, fit every arm, score the 8 test networks (5 seeds; resumable) | `scripts/run_v2_benchmark.py` | `artifacts/v2_benchmark/records_seed{42..46}.parquet`, `summary_seed*.json` |
| 2. Residual comparator, all metrics, paired tests, RSI bootstrap, prior sweep | `scripts/analyze_v2.py` | `artifacts/v2_benchmark/analysis/v2_summary.json`, `v2_tables.md` |
| 3. LaTeX table bodies + prose macros `\input` by the manuscript | `scripts/make_v2_tables.py` | `manuscript/tables_v2/*.tex` |
| 4. Split-calibration check (disjoint normal reference; Supplementary S7). Re-simulates the test networks, ~2–3 h with 3 workers; re-run any file whose canary fails serially | `scripts/run_v2_splitcal.py simulate` then `analyze` | `artifacts/v2_splitcal/splitcal_summary.json` |
| Metadata-leak diagnostic on the withdrawn v1 (quoted only as superseded) | `scripts/run_leak_area_ablation.py` | `artifacts/leak_area_ablation/` |
| Provenance table (script / artifact / sha256 / freshness per table) | `scripts/build_provenance.py` | `PROVENANCE.md`, `PROVENANCE.json` |

| Manuscript table | Source |
|---|---|
| Row AUPRC + prior, all arms (T2); paired deltas, Wilcoxon, bootstrap, MDE (T3); scenario AUPRC + prior (T4); early warning (T5) | `analyze_v2.py` → `make_v2_tables.py` |
| Per-disturbance slices (S1), max- vs mean-pool (S2), RSI (S3), prior sweep (S4), seed-level (S5) | `analyze_v2.py` → `make_v2_tables.py` |
| Meta-audit ledger (S6), network metadata (T1) | manual ledger / EPANET `.inp` headers |

`python -m pytest tests -q` runs the regression tests: the v1 data-pipeline defects, the
k-NN graph memory fix, and the v2 design guards (no scenario metadata in any feature list,
normalisation against normals only, random placement, sampled onsets, dose-matched augmenters,
positive-row plausibility bounds).

The v1 drivers (`run_expanded_lolo.py`, `run_nonsat_*.py`, `compute_*_rsi.py`,
`run_intrascale_lolo.py`, `run_graph_cvae_*.py`, `run_topology_gnn_lolo.py`,
`run_triage_validation.py`, ...) are retained for the record; they use the withdrawn v1 feature
list and reproduce the withdrawn v1 tables only.

## Data and artifacts

**Inputs (network models).** Experiments run on EPANET `.inp` network files from the
[EPANET-Benchmarks](https://github.com/OpenWaterAnalytics) suite, [WNTR](https://github.com/USEPA/WNTR),
the BattLeDIM / BATADAL competitions, and a curated set of synthetic/mined topologies. These
network files are publicly available from those sources and are not redistributed here.

**Generated artifacts.** The v2 pipeline writes ~110 MB of per-seed record files plus the
summary JSON shipped here. The withdrawn v1 pipeline (~8.3 GB total) regenerated a large set of
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
