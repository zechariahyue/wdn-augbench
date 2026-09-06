"""Phase 1: expand the LOLO held-out pool from 3 to 9 test networks.

The manuscript's negative augmentation finding (Table 8, tab:lolo) rests on
n=3 held-out networks, where the Wilcoxon signed-rank sign-floor is 0.125 — the
finding cannot be tested at alpha=0.05 no matter how consistent it is. Phase 0
(S25b) curated six further vetted, topologically independent benchmark networks.

This script runs *exactly the Table 8 protocol* with the enlarged held-out pool:

  train = anytown, hanoi          (UNCHANGED from Table 8)
  test  = net3, d_town, l_town    (the original three)
          + example3, ky2, ky3, ky5, ky6, ky7   (Phase 0 additions)
  seeds = 42..46
  arms  = baseline, noise, gmm, smote   (RF downstream)

Holding the training set fixed is the point. An earlier cut of this script used
a leave-one-out design (train on 8 of the 9, test on the 9th). That trains on
~200k rows from 8 diverse networks, which solves the detection task outright:
every arm lands at AUPRC 0.996 +/- 0.0001, leaving no headroom for augmentation
to help or hurt, and producing a "no augmenter beats baseline" result that is
true but vacuous. It is also not comparable to Table 8, so it could not be used
to re-tier the Table 8 claim. Keeping train = {anytown, hanoi} preserves both the
difficulty (baseline macro ~0.58) and the comparability.

Wilcoxon is paired over the 9 held-out networks, so the sign-floor moves
0.125 -> 0.0039 (two-sided) and the negative finding becomes testable.

The GCN arm is dropped (its k-NN graph OOMs on the larger test frames); the
augmentation finding is the RF arm, so this is not a loss.

Resumable: writes one JSON per seed and skips completed seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SCRIPT_PATH = Path(__file__).resolve()
ACTIVE_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(ACTIVE_ROOT / "src"))
sys.path.insert(0, str(ACTIVE_ROOT / "scripts"))

import run_cross_network_eval as cne  # noqa: E402

cne.RUN_GNN_BASELINE = False

MANIFEST = ACTIVE_ROOT / "configs" / "benchmark_manifest.yaml"
OUT = ACTIVE_ROOT / "artifacts" / "expanded_lolo"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]
COVERAGE = [0.25, 0.50]
DISTURBANCES = ["leak", "pipe_closure", "pump_outage"]
AUGMENTERS = ["baseline", "noise", "gmm", "smote"]

TRAIN = ["anytown", "hanoi"]
ORIGINAL_TEST = ["net3", "d_town", "l_town"]
# example3 is EXCLUDED: it is the same network as net3 (identical junction and
# pipe name sets; EPANET's "Example 3" IS Net3). Phase 0 deduplicated candidates
# against each other but not against the existing held-out pool, so it slipped
# through. Including it would double-count net3 and break the independence
# assumption of the paired Wilcoxon. Held-out pool is therefore 3 -> 8, not 3 -> 9.
ADDED_TEST = ["ky2", "ky3", "ky5", "ky6", "ky7"]
TEST = ORIGINAL_TEST + ADDED_TEST

# Table 8 baseline AUPRC on the original three, for the regression check. If these
# do not reproduce, the S28 simulation fixes perturbed something they should not
# have and the expanded numbers cannot be trusted.
TABLE8_BASELINE = {"net3": 0.589, "d_town": 0.651, "l_town": 0.540}

KEEP_TRIAGE_CSV = False


def run_one(seed: int) -> dict:
    out_json = OUT / f"expanded_seed{seed}.json"
    if out_json.exists():
        print(f"[skip] seed {seed} already done", flush=True)
        return json.loads(out_json.read_text())

    cne.TRAIN_NETWORK_IDS = list(TRAIN)
    cne.TEST_NETWORK_IDS = list(TEST)
    seed_dir = OUT / f"expanded_seed{seed}_run"

    t0 = time.time()
    print(f"[run] seed {seed} (train={TRAIN}, {len(TEST)} test networks)", flush=True)
    res = cne.run_cross_network_eval(
        MANIFEST, seed_dir,
        sensor_coverage_levels=COVERAGE,
        disturbance_types=DISTURBANCES,
        seed=seed,
    )
    res["_elapsed_sec"] = round(time.time() - t0, 1)
    res["_train_networks"] = list(TRAIN)
    res["_test_networks"] = list(TEST)
    out_json.write_text(json.dumps(res, indent=2))

    if not KEEP_TRIAGE_CSV:
        csv = seed_dir / "cross_network_triage_records.csv"
        if csv.exists():
            mb = csv.stat().st_size / 1e6
            csv.unlink()
            print(f"[clean] dropped {csv.name} ({mb:.0f} MB)", flush=True)

    print(f"[done] seed {seed} in {res['_elapsed_sec']}s", flush=True)
    return res


def _collect() -> dict[str, dict[str, list[float]]]:
    """per_net[network][augmenter] -> list of AUPRC, one per completed seed."""
    per_net = {n: {a: [] for a in AUGMENTERS} for n in TEST}
    for seed in SEEDS:
        p = OUT / f"expanded_seed{seed}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        for n in TEST:
            for a in AUGMENTERS:
                v = (res.get("per_augmenter", {}).get(a, {})
                        .get("per_network", {}).get(n, {}).get("auprc"))
                if v is not None:
                    per_net[n][a].append(float(v))
    return per_net


def aggregate() -> dict:
    per_net = _collect()
    summary: dict = {
        "protocol": "LOLO (train=anytown+hanoi), held-out pool expanded 3 -> 9",
        "train_networks": TRAIN,
        "test_networks": TEST,
        "per_network": {},
        "macro_eqwt": {},
        "macro_nodewt": {},
        "wilcoxon_vs_baseline": {},
        "regression_check_vs_table8": {},
    }

    for n in TEST:
        summary["per_network"][n] = {
            a: {"mean": float(np.mean(v)) if v else None,
                "std": float(np.std(v)) if v else None,
                "n_seeds": len(v)}
            for a, v in per_net[n].items()
        }

    for a in AUGMENTERS:
        means, weights = [], []
        for n in TEST:
            m = summary["per_network"][n][a]["mean"]
            if m is not None:
                means.append(m)
                weights.append(float(cne.NETWORK_NODE_COUNTS.get(n, 1)))
        summary["macro_eqwt"][a] = float(np.mean(means)) if means else None
        summary["macro_nodewt"][a] = float(np.average(means, weights=weights)) if means else None

    # The point of Phase 1: paired signed-rank over the 8 held-out networks.
    # n=8 -> two-sided sign-floor 2/2^8 = 0.0078 (was 0.125 at n=3).
    for a in AUGMENTERS:
        if a == "baseline":
            continue
        base_v, aug_v = [], []
        for n in TEST:
            b = summary["per_network"][n]["baseline"]["mean"]
            x = summary["per_network"][n][a]["mean"]
            if b is not None and x is not None:
                base_v.append(b)
                aug_v.append(x)
        entry: dict = {"n_networks": len(base_v)}
        if len(base_v) >= 2 and not np.allclose(base_v, aug_v):
            try:
                stat, p = wilcoxon(aug_v, base_v)
                entry["statistic"] = float(stat)
                entry["p_value"] = float(p)
            except Exception as exc:
                entry["error"] = repr(exc)
        else:
            entry["note"] = "insufficient or identical paired observations"
        entry["mean_delta_vs_baseline"] = (
            float(np.mean(np.array(aug_v) - np.array(base_v))) if base_v else None
        )
        entry["n_networks_augmenter_wins"] = (
            int(np.sum(np.array(aug_v) > np.array(base_v))) if base_v else None
        )
        summary["wilcoxon_vs_baseline"][a] = entry

    # Regression check: the original three must still reproduce Table 8.
    for n, expected in TABLE8_BASELINE.items():
        got = summary["per_network"][n]["baseline"]["mean"]
        if got is not None:
            summary["regression_check_vs_table8"][n] = {
                "table8": expected, "rerun": round(got, 4), "delta": round(got - expected, 4),
            }

    n_done = sum(1 for s in SEEDS if (OUT / f"expanded_seed{s}.json").exists())
    summary["_progress"] = {"completed_seeds": n_done, "total_seeds": len(SEEDS)}
    (OUT / "expanded_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Expanded LOLO ({n_done}/{len(SEEDS)} seeds done) ===")
    print("Baseline per network (mean over seeds):")
    for n in TEST:
        m = summary["per_network"][n]["baseline"]["mean"]
        tag = ""
        if n in TABLE8_BASELINE and m is not None:
            tag = f"   [Table 8: {TABLE8_BASELINE[n]:.3f}  delta {m - TABLE8_BASELINE[n]:+.3f}]"
        elif n in ADDED_TEST:
            tag = "   [new]"
        print(f"  {n:<10}: {m:.4f}{tag}" if m is not None else f"  {n:<10}: (pending)")

    print("\nMacro AUPRC (equal-weight / node-weighted):")
    for a in AUGMENTERS:
        e, nw = summary["macro_eqwt"][a], summary["macro_nodewt"][a]
        print(f"  {a:<10}: {e:.4f} / {nw:.4f}" if e is not None else f"  {a:<10}: (pending)")

    print("\nWilcoxon signed-rank vs baseline (paired over held-out networks):")
    for a, e in summary["wilcoxon_vs_baseline"].items():
        p, d, w = e.get("p_value"), e.get("mean_delta_vs_baseline"), e.get("n_networks_augmenter_wins")
        pstr = f"p={p:.4f}" if p is not None else e.get("note", e.get("error", "n/a"))
        dstr = f"delta={d:+.4f}" if d is not None else ""
        wstr = f"wins={w}/{e['n_networks']}" if w is not None else ""
        print(f"  {a:<10}: n={e['n_networks']}  {dstr}  {wstr}  {pstr}")
    return summary


def main() -> None:
    t0 = time.time()
    print(f"[start] train={TRAIN}  test={len(TEST)} networks: {TEST}")
    print(f"[start] {len(SEEDS)} seeds; GNN arm enabled: {cne.RUN_GNN_BASELINE}")
    for seed in SEEDS:
        try:
            run_one(seed)
        except Exception as exc:
            print(f"[error] seed {seed}: {exc!r}", flush=True)
        aggregate()
    print(f"\n[ALL DONE] total {round(time.time()-t0, 1)}s")
    aggregate()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
