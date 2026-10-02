#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Release tool: read-only recompute of the published primary metrics from OOF.

Reads ``oof_predictions.parquet`` of a Stage-15b run and recomputes the pooled
window-level metric bundle with the SAME code path that produced the published
numbers (``psg_audio_benchmark.modeling.metrics.compute_metrics`` from this
repository's ``src/``), for the pre-registered cells persisted in the OOF store:

  * primary battery, main label (overlap_ge_10s), causal lag_0, all six models;
  * airflow secondary (core_restricted vs airflow_enhanced) and its AUROC delta.

(Note: the frozen runner persists OOF rows only for the ``primary_battery`` and
``airflow`` tasks; the label-sensitivity and lag-sensitivity OOF frames were not
persisted and cannot be recomputed from the parquet — rerun the analysis to
verify those cells.)

Comparison semantics, reported explicitly per value:

  * ``published_4dp``  — round(recomputed, 4) == round(CSV value, 4): the precision
    used in the manuscript. Must be EXACT.
  * ``float64_agreement`` — relative difference vs the CSV text. The CSV writer
    stores ~16 significant digits, so bit-exact equality can fail on the 17th
    digit purely from text serialization. Agreement within rel <= 1e-12 passes;
    anything larger is a genuine MISMATCH.

Nothing is written; the run directory is only read. Exit code 0 = all cells
pass both checks; 1 = any failure (printed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark.modeling.metrics import compute_metrics  # noqa: E402

METRICS = ["auroc", "auprc", "brier", "log_loss", "sensitivity", "specificity"]
REL_TOL = 1e-12


def _pooled(d: pd.DataFrame) -> dict:
    """Mirror of bspc_revision.runner.evaluate_oof pooled computation."""
    thr = float(d["threshold"].iloc[0]) if len(d) else 0.5
    return compute_metrics(y_true=d["y_true"].to_numpy(), y_prob=d["y_prob"].to_numpy(),
                           threshold=thr)


def _same(a, b) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (b is not None and np.isnan(b))
    if b is None:
        return False
    return float(a) == float(b)


def _close(a, b) -> bool:
    if a is None or b is None:
        return _same(a, b)
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    return bool(np.isclose(float(a), float(b), rtol=REL_TOL, atol=0.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Read-only recompute + agreement check of primary metrics from frozen OOF.")
    ap.add_argument("--run-dir", required=True,
                    help="Stage-15b results run dir containing oof_predictions.parquet")
    args = ap.parse_args()
    run = Path(args.run_dir)

    oof = pd.read_parquet(run / "oof_predictions.parquet")
    summary = pd.read_csv(run / "summary_metrics.csv")
    af = pd.read_csv(run / "airflow_paired_summary.csv")
    results = []
    max_rel_diff = 0.0

    def _compare(cell_label, d, frozen_row):
        nonlocal max_rel_diff
        rec = _pooled(d)
        ok = True
        keys = METRICS + ["n"]
        col_of = {"n": "n_windows"}
        for k in keys:
            a, b = rec.get(k), frozen_row[col_of.get(k, k)]
            b_f = None if (b is None or pd.isna(b)) else float(b)
            pub = (_same(round(float(a), 4), round(b_f, 4)) if a is not None and b_f is not None
                   else _same(a, b_f))
            rel = (abs(float(a) - b_f) / max(abs(float(b_f)), 1e-300)
                   if (a is not None and b_f is not None and not np.isnan(a) and not np.isnan(b_f)
                       and float(b_f) != float(a)) else 0.0)
            max_rel_diff = max(max_rel_diff, rel)
            close = _close(a, b_f)
            good = pub and close
            results.append({"cell": cell_label, "metric": k, "recomputed": a, "csv": b_f,
                            "published_4dp_match": bool(pub),
                            "float64_agreement_le_1e-12": bool(close),
                            "rel_diff": rel, "pass": bool(good)})
            if not good:
                ok = False
        print(f"[{'OK' if ok else 'MISMATCH'}] {cell_label}")

    def _cell(task, variant, lag, model):
        return oof[(oof.task == task) & (oof.variant == variant)
                   & (oof.lag_label == lag) & (oof.model == model)]

    for model in ["dummy_prior", "simple_spo2_min_lr", "hr_only_lr", "spo2_only_lr",
                  "hr_spo2_lr", "hr_spo2_hgb"]:
        d = _cell("primary_battery", "main", "lag_0", model)
        fr = summary[(summary.task == "primary_battery") & (summary.variant == "main")
                     & (summary.lag_label == "lag_0") & (summary.model == model)].iloc[0]
        _compare(f"primary_battery|main|lag_0|{model}", d, fr)

    aurocs = {}
    af_map = {r.feature_set: r for r in af.itertuples()}
    for role in ["core_restricted", "airflow_enhanced"]:
        d = _cell("airflow_secondary", "main", "lag_0", f"airflow_{role}")
        fr = af[af.feature_set == role].iloc[0]
        _compare(f"airflow_secondary|{role}", d, fr)
        aurocs[role] = _pooled(d)["auroc"]
    delta = aurocs["airflow_enhanced"] - aurocs["core_restricted"]
    frozen_delta = float(af_map["airflow_enhanced"].auroc) - float(af_map["core_restricted"].auroc)
    pub = round(delta, 4) == round(frozen_delta, 4)
    close = np.isclose(delta, frozen_delta, rtol=REL_TOL, atol=0.0)
    results.append({"cell": "airflow_secondary|delta_auroc", "metric": "delta",
                    "recomputed": delta, "csv": frozen_delta,
                    "published_4dp_match": bool(pub),
                    "float64_agreement_le_1e-12": bool(close),
                    "rel_diff": abs(delta - frozen_delta) / max(abs(frozen_delta), 1e-300),
                    "pass": bool(pub and close)})
    print(f"[{'OK' if (pub and close) else 'MISMATCH'}] airflow_secondary delta AUROC "
          f"recomputed={delta!r} csv={frozen_delta!r}")

    n_bad = sum(1 for r in results if not r["pass"])
    print(f"\ncomparisons={len(results)} pass={len(results) - n_bad} fail={n_bad} "
          f"max_rel_diff={max_rel_diff:.3e} (tol {REL_TOL:.0e})")
    for r in results:
        if not r["pass"]:
            print(f"  FAIL {r['cell']}.{r['metric']}: recomputed={r['recomputed']!r} "
                  f"csv={r['csv']!r} rel_diff={r['rel_diff']:.3e}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
