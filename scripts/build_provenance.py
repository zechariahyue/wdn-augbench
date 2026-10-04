"""Generate a per-result provenance table for the manuscript (Codex S29 FATAL 2 / Q1).

For every numeric table and figure in `sn-article.tex` and `supplementary.tex` (the
Supplementary Material, S35), this records the generating
script, the artifact it reads, that artifact's sha256 and modification time, and --
the point of the exercise -- whether the artifact was produced AFTER the two data
defects found in S29 were fixed. A reviewer (or we) can then tell at a glance which
results are computed on corrected data and which are still stale.

The two fixes and the boundary each result must clear:

  SIM_FIX  2026-07-10 22:32  hanoi leak-label corruption (simulation.py). Affects
                             every result derived from the training simulation.
  AUG_FIX  2026-07-13 15:25  augmenter pipeline (degenerate-bounds filter, 100%-
                             rejection fallback, silent SMOTE import failure). Affects
                             only results that involve an augmenter arm.

An artifact is CURRENT iff its mtime is at or after the boundary that applies to it
(AUG_FIX for augmenter-dependent results, SIM_FIX otherwise); STALE if older; MISSING
if absent. Re-run `rerun_after_hanoi_fix.py` and regenerate this file; STALE rows
become CURRENT as their artifacts land.

S36c (2026-10-04): every manuscript table now derives from the v2 benchmark
(`v2_benchmark/records_seed{42..46}.parquet` -> `analysis/v2_summary.json`), which
postdates both boundaries, so every computed row is CURRENT by construction; the
boundaries are kept so that any v1 artifact accidentally wired in would show as STALE.

Usage:  python dev/active/scripts/build_provenance.py
Writes: manuscript/PROVENANCE.md, manuscript/PROVENANCE.json
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

def _resolve():
    """Locate the artifacts tree and an output dir, in both the working tree
    (`dev/active/artifacts`, output to `manuscript/`) and the released repo
    (`artifacts/` at the repo root, output alongside).

    Note on mtimes: the CURRENT/STALE classification reads artifact modification
    times, which are only meaningful in the tree that generated them. Copying an
    artifact resets its mtime, so running this in the released repo (where every
    file was copied at once) will misreport status. The authoritative record is
    the PROVENANCE.md generated in the working tree and shipped with the repo;
    regenerate it there after the re-run, then re-ship.
    """
    here = Path(__file__).resolve()
    # Working-tree layout: <root>/dev/active/artifacts, output to <root>/manuscript.
    for parent in here.parents:
        cand = parent / "dev" / "active" / "artifacts"
        if cand.is_dir():
            return cand, parent / "manuscript"
    # Released-repo layout (standalone clone): <root>/artifacts, output to <root>.
    for parent in here.parents:
        cand = parent / "artifacts"
        if cand.is_dir():
            return cand, parent
    raise SystemExit("build_provenance: could not locate an artifacts/ directory")


ART, _OUTDIR = _resolve()
OUT_MD = _OUTDIR / "PROVENANCE.md"
OUT_JSON = _OUTDIR / "PROVENANCE.json"

# Fix boundaries (local time). Hardcoded rather than read from source-file mtimes,
# which change on copy; these are the semantic fix times documented in HANDOFF_STATE.
SIM_FIX = dt.datetime(2026, 7, 10, 22, 32)
AUG_FIX = dt.datetime(2026, 7, 13, 15, 25)

# (label, description, scripts, primary_artifact, extra_artifacts, augmenter_dependent, note)
# `augmenter_dependent` picks the boundary. A None artifact means "no computed
# artifact" (metadata/literature tables) -- reported as N/A, never stale.
PARQUET = ['v2_benchmark/records_seed42.parquet', 'v2_benchmark/records_seed43.parquet', 'v2_benchmark/records_seed44.parquet', 'v2_benchmark/records_seed45.parquet', 'v2_benchmark/records_seed46.parquet']

ROWS = [
    ('tab:row-auprc', 'Row AUPRC per network + prior, all arms, 5-seed mean/std (main T2)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('tab:deltas', 'Paired deltas vs rf_baseline: wins, Wilcoxon, network bootstrap, 2.5 SE_Delta (main T3)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('tab:scen-auprc', 'Scenario AUPRC (max-pool) + scenario prior, all arms (main T4)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('tab:early-warning', 'WSTL / first-3h / median TTD, rf_baseline vs residual comparator (main T5)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, 'no augmenter arm; residual comparator built in analyze_v2.py from recorded residual proxies'),
    ('tab:slices', 'Per-disturbance slices vs all normals with slice priors (S1)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, ''),
    ('tab:pooling', 'Max-pool vs mean-pool scenario AUPRC (S2)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, ''),
    ('tab:rsi', 'RSI per network/pair with pooled scenario bootstrap (S3)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('tab:prior-sweep', 'Scenario-prior sweep by subsampling normals 20/40/100 (S4)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, ''),
    ('tab:seed-level', 'Per-seed row AUPRC, rf_baseline and rf_raw_baseline (S5)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, ''),
    ('tab:family', 'Family-level paired deltas vs rf_baseline: Net3, D-Town, L-TOWN, Kentucky mean (S6)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, False, ''),
    ('fig:forest', 'Per-network paired deltas vs rf_baseline, forest plot (main Fig. 1)',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('macros_v2 (prose numbers)', 'Headline macros/deltas/p-values quoted in abstract, intro, results, conclusion',
     ['run_v2_benchmark.py', 'analyze_v2.py', 'make_v2_tables.py'], 'v2_benchmark/analysis/v2_summary.json',
     PARQUET, True, ''),
    ('what-changed (S8.2)', 'Metadata-leak diagnostic on the withdrawn v1 (quoted only as superseded)',
     ['run_leak_area_ablation.py'], 'leak_area_ablation/ablation_summary.json',
     [], False, 'v1 artifacts retained under their original names; not used by any v2 table'),
    ("tab:meta-audit", "Evaluation-unit audit of 16 papers",
     ["(manual literature classification ledger)"], None, [], False,
     "not computed from simulation; released as the classification ledger"),
    ("tab:networks", "Network metadata (junctions/pipes/pumps/...)",
     ["(EPANET .inp headers)"], None, [], False, "static metadata"),
]


def _sha_mtime(rel: str):
    p = ART / rel
    if not p.exists():
        return None, None
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    m = dt.datetime.fromtimestamp(p.stat().st_mtime)
    return h, m


def _classify(mtime, aug_dep):
    if mtime is None:
        return "MISSING"
    boundary = AUG_FIX if aug_dep else SIM_FIX
    return "CURRENT" if mtime >= boundary else "STALE"


def main() -> int:
    records = []
    for label, desc, scripts, primary, extra, aug_dep, note in ROWS:
        if primary is None:
            records.append(dict(
                label=label, description=desc, scripts=scripts, artifact=None,
                sha256=None, mtime=None, status="N/A", augmenter_dependent=aug_dep,
                note=note))
            continue
        sha, mtime = _sha_mtime(primary)
        status = _classify(mtime, aug_dep)
        # A row is only as current as its stalest artifact.
        for e in extra:
            _, em = _sha_mtime(e)
            if _classify(em, aug_dep) == "STALE" and status == "CURRENT":
                status = "STALE"
            if em is None:
                status = "MISSING" if status != "STALE" else status
        records.append(dict(
            label=label, description=desc, scripts=scripts, artifact=primary,
            sha256=sha, mtime=mtime.isoformat(timespec="seconds") if mtime else None,
            status=status, augmenter_dependent=aug_dep, note=note))

    n_current = sum(r["status"] == "CURRENT" for r in records)
    n_stale = sum(r["status"] == "STALE" for r in records)
    n_missing = sum(r["status"] == "MISSING" for r in records)

    lines = []
    lines.append("# Result provenance\n")
    lines.append(
        "Auto-generated by `dev/active/scripts/build_provenance.py`. For every "
        "numeric table and figure, this records the generating script, the artifact "
        "read, its sha256 and mtime, and whether that artifact postdates the two S29 "
        "data-defect fixes.\n")
    lines.append(
        f"**Fix boundaries.** SIM_FIX `{SIM_FIX:%Y-%m-%d %H:%M}` (hanoi leak-label, "
        "`simulation.py`) — applies to every simulation-derived result. AUG_FIX "
        f"`{AUG_FIX:%Y-%m-%d %H:%M}` (augmenter pipeline: degenerate-bounds filter, "
        "100%-rejection fallback, silent SMOTE import) — applies to results with an "
        "augmenter arm. A result is **CURRENT** iff its artifact is at or after the "
        "boundary that applies to it.\n")
    lines.append(
        f"**Status: {n_current} CURRENT, {n_stale} STALE, {n_missing} MISSING** "
        f"(of {len(records)} tracked results).\n")
    lines.append(
        "_This record was generated in the working tree, where artifact mtimes are "
        "authoritative. Copying an artifact resets its mtime, so regenerate this file "
        "in the tree that produced the artifacts (after the re-run) rather than "
        "re-deriving status from copied files._\n")
    lines.append("| Result | Artifact | Boundary | mtime | sha256 (12) | Status |")
    lines.append("|---|---|:--:|---|---|:--:|")
    for r in records:
        bnd = "—" if r["artifact"] is None else ("AUG" if r["augmenter_dependent"] else "SIM")
        art = r["artifact"] or "_(not computed)_"
        mt = (r["mtime"] or "—").replace("T", " ")
        sh = (r["sha256"][:12] if r["sha256"] else "—")
        badge = {"CURRENT": "✅ CURRENT", "STALE": "⚠️ STALE",
                 "MISSING": "❌ MISSING", "N/A": "— N/A"}[r["status"]]
        lines.append(f"| {r['label']} | `{art}` | {bnd} | {mt} | `{sh}` | {badge} |")
    lines.append("")
    lines.append("## Scripts")
    lines.append("| Result | Script(s) | Note |")
    lines.append("|---|---|---|")
    for r in records:
        s = ", ".join(f"`{x}`" for x in r["scripts"])
        lines.append(f"| {r['label']} | {s} | {r['note']} |")
    lines.append("")
    if n_stale or n_missing:
        lines.append("## Not yet current")
        for r in records:
            if r["status"] in ("STALE", "MISSING"):
                lines.append(f"- **{r['status']}** {r['label']} — `{r['artifact']}` "
                             f"({r['note'] or 'regenerate before submission'})")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    OUT_JSON.write_text(json.dumps(dict(
        generated_boundary=dict(sim_fix=SIM_FIX.isoformat(), aug_fix=AUG_FIX.isoformat()),
        summary=dict(current=n_current, stale=n_stale, missing=n_missing, total=len(records)),
        records=records), indent=2), encoding="utf-8")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print(f"{n_current} CURRENT, {n_stale} STALE, {n_missing} MISSING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
