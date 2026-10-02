#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Release tool: export figure-input CSVs from a Stage-15b analysis run.

Adapted WITHOUT definitional changes from the Stage-21 read-only verification
recompute (submission_package run stage21-final-editorial-consistency-2026-10-01T140832Z,
work/round1_text/verify_recompute.py). The frozen Stage-15b definitions are
replicated EXACTLY:

  - candidate events = merged strictly-contiguous (30-s) positive-predicted windows
  - GT = scored respiratory events with overlaps_awake_interval == False
  - GT scope = per-patient min(window_start) .. max(window_end) envelope of
    analysable windows
  - matching = positive interval intersection, many-to-many

Outputs (into --out-dir):
  fig2_curve_data.csv            ROC/PR curve points for Figure 2
  sf2_calibration_bins.csv       10-bin calibration data for SF2
  fold_auroc_verification.csv    per-outer-fold primary metrics (SF3)
  event_counts_verification.csv  event-level recount (verification evidence)
  per_patient_auroc_primary.csv  per-patient primary AUROC (single-class excluded)
  per_patient_excluded_single_class.csv
  verification_summary.json      summary incl. pooled primary AUROC/AP

This tool READS ONLY; it never modifies the run directory or any frozen artifact.
Use it after re-running the analysis pipeline on the official dataset to produce
the deterministic figure inputs consumed by tools/make_figures.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export figure-input CSVs from a Stage-15b run (read-only).")
    p.add_argument("--project-root", required=True,
                   help="project root containing splits/ and annotations/ (frozen pipeline products)")
    p.add_argument("--run-dir", required=True,
                   help="the Stage-15b results run dir containing oof_predictions.parquet")
    p.add_argument("--out-dir", required=True, help="directory to write the exported CSVs into")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = str(args.project_root)
    f15 = str(args.run_dir)
    out = str(args.out_dir)
    os.makedirs(out, exist_ok=True)

    oof = pd.read_parquet(os.path.join(f15, "oof_predictions.parquet"))
    wm = pd.read_parquet(os.path.join(root, "splits", "core_cohort_window_membership.parquet"))
    ev = pd.read_parquet(os.path.join(root, "annotations", "parsed_events.parquet"))
    wm_small = wm[["window_id", "window_start_relative_to_record_start",
                   "window_end_relative_to_record_start"]]

    gt = ev[(ev.is_any_scored_respiratory_event == True) & (ev.overlaps_awake_interval == False)]  # noqa: E712
    gt = gt[["patient_id", "event_start_relative_to_record_start",
             "event_end_relative_to_record_start"]]

    frozen_events = pd.read_csv(os.path.join(f15, "event_level_metrics.csv"))

    rows = []
    for model in ["dummy_prior", "simple_spo2_min_lr", "hr_only_lr", "spo2_only_lr",
                  "hr_spo2_lr", "hr_spo2_hgb"]:
        d = oof[(oof.model == model) & (oof.variant == "main") & (oof.lag_label == "lag_0")].merge(
            wm_small, on="window_id", how="left")
        assert d.window_start_relative_to_record_start.notna().all()
        n_cand = n_tp = 0
        n_gt_scope = n_gt_matched = 0
        for pid, g in d.groupby("patient_id"):
            pos = g[g.y_pred == 1].sort_values("window_start_relative_to_record_start")
            cands = []
            for ws, we in zip(pos.window_start_relative_to_record_start,
                              pos.window_end_relative_to_record_start):
                if cands and abs(ws - cands[-1][1]) < 1e-6:
                    cands[-1] = (cands[-1][0], we)
                else:
                    cands.append((ws, we))
            lo = g.window_start_relative_to_record_start.min()
            hi = g.window_end_relative_to_record_start.max()
            pgt = gt[gt.patient_id == pid]
            in_scope = pgt[(pgt.event_start_relative_to_record_start < hi)
                           & (pgt.event_end_relative_to_record_start > lo)]
            n_gt_scope += len(in_scope)
            for s, e, in in_scope[["event_start_relative_to_record_start",
                                   "event_end_relative_to_record_start"]].itertuples(index=False, name=None):
                if any(min(e, ce) - max(s, cs) > 0 for cs, ce in cands):
                    n_gt_matched += 1
            for cs, ce in cands:
                n_cand += 1
                if any(min(ce, e) - max(cs, s) > 0 for s, e in zip(
                        pgt.event_start_relative_to_record_start,
                        pgt.event_end_relative_to_record_start)):
                    n_tp += 1
        n_fp = n_cand - n_tp
        fr = frozen_events[(frozen_events.model == model) & (frozen_events.variant == "main")]
        fe = fr.iloc[0]
        rows.append(dict(model=model, n_gt_events_in_scope=n_gt_scope, n_gt_events_matched=n_gt_matched,
                         n_candidate_events=n_cand, n_tp_candidates=n_tp, n_fp_candidates=n_fp,
                         matched_candidates=n_tp, false_positive_candidates=n_fp,
                         recomputed_sensitivity=n_gt_matched / n_gt_scope if n_gt_scope else np.nan,
                         recomputed_precision=n_tp / n_cand if n_cand else np.nan,
                         frozen_sensitivity=fe.event_sensitivity, frozen_precision=fe.event_precision,
                         frozen_n_candidates=int(fe.n_candidate_events),
                         frozen_n_matched=int(fe.n_gt_events_matched),
                         agrees=(n_gt_scope == int(fe.n_gt_events_in_scope))
                         and (n_gt_matched == int(fe.n_gt_events_matched))
                         and (n_cand == int(fe.n_candidate_events))))
    pd.DataFrame(rows).to_csv(os.path.join(out, "event_counts_verification.csv"), index=False)

    # ---- per-outer-fold AUROC (primary) ----
    p = oof[(oof.model == "hr_spo2_lr") & (oof.variant == "main") & (oof.lag_label == "lag_0")]
    fold_rows = []
    for f, g in p.groupby("outer_fold"):
        fold_rows.append(dict(outer_fold=int(f), n_patients=g.patient_id.nunique(), n_windows=len(g),
                              n_positive=int(g.y_true.sum()), auroc=roc_auc_score(g.y_true, g.y_prob),
                              auprc=average_precision_score(g.y_true, g.y_prob)))
    pd.DataFrame(fold_rows).to_csv(os.path.join(out, "fold_auroc_verification.csv"), index=False)

    # ---- per-patient AUROC (primary; single-observed-class exclusion) ----
    pp_rows, excl = [], []
    for pid, g in p.groupby("patient_id"):
        if g.y_true.nunique() < 2:
            excl.append(dict(patient_id=pid, n_windows=len(g), n_positive=int(g.y_true.sum())))
            continue
        pp_rows.append(dict(patient_id=pid, auroc=roc_auc_score(g.y_true, g.y_prob),
                            auprc=average_precision_score(g.y_true, g.y_prob)))
    pp = pd.DataFrame(pp_rows)
    pp.to_csv(os.path.join(out, "per_patient_auroc_primary.csv"), index=False)
    pd.DataFrame(excl).to_csv(os.path.join(out, "per_patient_excluded_single_class.csv"), index=False)

    # ---- ROC / PR curve data for Figure 2 ----
    curve_rows = []
    for model in ["simple_spo2_min_lr", "hr_only_lr", "spo2_only_lr", "hr_spo2_lr",
                  "hr_spo2_hgb", "dummy_prior"]:
        d = oof[(oof.model == model) & (oof.variant == "main") & (oof.lag_label == "lag_0")]
        fpr, tpr, _ = roc_curve(d.y_true, d.y_prob)
        for a, b in zip(fpr, tpr):
            curve_rows.append(dict(model=model, curve="roc", x=a, y=b))
        pre, rec, _ = precision_recall_curve(d.y_true, d.y_prob)
        for a, b in zip(rec, pre):
            curve_rows.append(dict(model=model, curve="pr", x=a, y=b))
    pd.DataFrame(curve_rows).to_csv(os.path.join(out, "fig2_curve_data.csv"), index=False)

    # ---- SF2 calibration bins (primary) ----
    edges = np.linspace(0, 1, 11)
    bins = []
    ece = 0.0
    for i in range(10):
        lo_e, hi_e = edges[i], edges[i + 1]
        if i < 9:
            m = (p.y_prob >= lo_e) & (p.y_prob < hi_e)
        else:
            m = (p.y_prob >= lo_e) & (p.y_prob <= hi_e)
        cnt = int(m.sum())
        if cnt == 0:
            bins.append(dict(bin=i, lo=lo_e, hi=hi_e, n=0, mean_pred=np.nan, obs_rate=np.nan))
            continue
        mp, ob = float(p.loc[m, "y_prob"].mean()), float(p.loc[m, "y_true"].mean())
        ece += cnt / len(p) * abs(ob - mp)
        bins.append(dict(bin=i, lo=lo_e, hi=hi_e, n=cnt, mean_pred=mp, obs_rate=ob))
    pd.DataFrame(bins).assign(ece_count_weighted=ece).to_csv(
        os.path.join(out, "sf2_calibration_bins.csv"), index=False)

    # ---- summary JSON ----
    summary = dict(
        event_verification_all_agree=bool(all(r["agrees"] for r in rows)),
        primary=dict(n_candidates=rows[4]["n_candidate_events"],
                     matched_candidates=rows[4]["matched_candidates"],
                     false_positive_candidates=rows[4]["false_positive_candidates"],
                     gt_in_scope=rows[4]["n_gt_events_in_scope"],
                     gt_matched=rows[4]["n_gt_events_matched"],
                     ece_recomputed=round(ece, 10)),
        fold_auroc=[round(r["auroc"], 6) for r in fold_rows],
        fold_mean=round(np.mean([r["auroc"] for r in fold_rows]), 6),
        patient_macro_auroc=round(pp.auroc.mean(), 10), n_patients_included=len(pp),
        n_excluded=len(excl),
        pooled_auroc=round(roc_auc_score(p.y_true, p.y_prob), 10),
        pooled_ap=round(average_precision_score(p.y_true, p.y_prob), 10))
    with open(os.path.join(out, "verification_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
