"""Stage 15b — patient-macro, event-level and calibration metrics.

These extend the Stage-7 window-level metrics with the BSPC-required multi-level
evaluation. Window metrics are reused from ``modeling.metrics``. None of these functions
conflate window-level with patient-level or event-level roles.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from ..modeling.metrics import safe_auroc, safe_auprc

_EPS = 1e-9


# --------------------------------------------------------------------------- patient macro
def patient_macro(*, oof: pd.DataFrame) -> Dict[str, object]:
    """Per-patient AUROC/AUPRC over patients with BOTH classes in their OOF, then macro-averaged.

    Single-class patients are excluded with an explicit NA reason; never conflated with a
    window macro.
    """
    per_patient: Dict[str, List[float]] = {"auroc": [], "auprc": []}
    excluded: List[str] = []
    for pid, g in oof.groupby("patient_id"):
        y = g["y_true"].to_numpy()
        p = g["y_prob"].to_numpy()
        if len(np.unique(y)) < 2:
            excluded.append(f"{pid}:single_class_{np.unique(y).tolist()}")
            continue
        au, _ = safe_auroc(y, p)
        ap, _ = safe_auprc(y, p)
        if au is not None:
            per_patient["auroc"].append(float(au))
        if ap is not None:
            per_patient["auprc"].append(float(ap))
    out: Dict[str, object] = {
        "n_patients_total": int(oof["patient_id"].nunique()),
        "n_patients_included": max(len(per_patient["auroc"]), len(per_patient["auprc"])),
        "n_patients_excluded": len(excluded),
        "exclusion_reasons": ";".join(excluded[:20]),
    }
    for m in ("auroc", "auprc"):
        vals = per_patient[m]
        out[f"patient_macro_{m}"] = float(np.mean(vals)) if vals else None
        out[f"patient_macro_{m}_n"] = len(vals)
        out[f"patient_macro_{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- event level
def _candidate_events(oof: pd.DataFrame, starts: Dict[str, float]) -> List[tuple]:
    """Merge adjacent positive windows (30 s contiguous in time, same patient) into candidate
    events; return list of (patient_id, ev_start, ev_end)."""
    cands: List[tuple] = []
    for pid, g in oof[oof["y_pred"] == 1].sort_values("window_start").groupby("patient_id"):
        ws = g["window_start"].to_numpy(dtype=float)
        we = g["window_end"].to_numpy(dtype=float)
        if len(ws) == 0:
            continue
        cur_s, cur_e = ws[0], we[0]
        for i in range(1, len(ws)):
            if abs(ws[i] - cur_e) < 1e-6:   # temporally contiguous (30 s step)
                cur_e = we[i]
            else:
                cands.append((str(pid), float(cur_s), float(cur_e)))
                cur_s, cur_e = ws[i], we[i]
        cands.append((str(pid), float(cur_s), float(cur_e)))
    return cands


def event_level(*, oof: pd.DataFrame, retained_events: pd.DataFrame,
                analyzable_hours: float) -> Dict[str, object]:
    """Event-level detection metrics. A predicted candidate event matches a retained (non-awake
    scored) ground-truth event if their intervals overlap (intersection > 0)."""
    cands = _candidate_events(oof, {})
    gt = retained_events[retained_events["is_any_scored_respiratory_event"].fillna(False)
                         & (~retained_events["overlaps_awake_interval"].fillna(False).astype(bool))].copy()

    # restrict GT to events overlapping the analyzable time span per patient (fair denominator)
    amin = oof.groupby("patient_id")["window_start"].min()
    amax = oof.groupby("patient_id")["window_end"].max()
    gt_in_scope = 0
    matched_gt = 0
    for pid, g in gt.groupby(gt["patient_id"].astype(str)):
        if str(pid) not in amin.index:
            continue
        lo, hi = float(amin.loc[str(pid)]), float(amax.loc[str(pid)])
        es = g["event_start_relative_to_record_start"].to_numpy(dtype=float)
        ee = g["event_end_relative_to_record_start"].to_numpy(dtype=float)
        for s, e in zip(es, ee):
            if min(e, hi) - max(s, lo) > 0:   # overlaps analyzable span
                gt_in_scope += 1
                # matched if any candidate of this patient overlaps [s, e)
                if any(cp == str(pid) and min(ce, e) - max(cs, s) > 0 for cp, cs, ce in cands):
                    matched_gt += 1

    tp_cand = 0
    for cp, cs, ce in cands:
        g = gt[gt["patient_id"].astype(str) == cp]
        if g.empty:
            continue
        es = g["event_start_relative_to_record_start"].to_numpy(dtype=float)
        ee = g["event_end_relative_to_record_start"].to_numpy(dtype=float)
        if any(min(ce, e) - max(cs, s) > 0 for s, e in zip(es, ee)):
            tp_cand += 1
    n_cand = len(cands)
    fp_cand = n_cand - tp_cand
    sens = matched_gt / gt_in_scope if gt_in_scope else None
    prec = tp_cand / n_cand if n_cand else None
    f1 = (2 * prec * sens / (prec + sens)) if (prec is not None and sens is not None and prec + sens > 0) else None
    fp_per_hour = fp_cand / analyzable_hours if analyzable_hours > 0 else None
    return {
        "n_gt_events_in_scope": int(gt_in_scope),
        "n_gt_events_matched": int(matched_gt),
        "n_candidate_events": int(n_cand),
        "n_candidate_events_tp": int(tp_cand),
        "n_candidate_events_fp": int(fp_cand),
        "event_sensitivity": sens,
        "event_precision": prec,
        "event_f1": f1,
        "false_positives_per_hour": fp_per_hour,
        "analyzable_hours": float(analyzable_hours),
        "not_patient_diagnosis": True,
    }


# --------------------------------------------------------------------------- calibration
def calibration(*, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """Brier, Cox calibration intercept/slope, ECE and 10-bin reliability."""
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS)
    brier = float(brier_score_loss(y, p))
    # Cox recalibration: logistic regression of y on logit(p) -> intercept, slope
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        intercept, slope = None, None
    else:
        lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
        lr.fit(logit, y)
        intercept, slope = float(lr.intercept_[0]), float(lr.coef_[0][0])
    # ECE + reliability bins
    edges = np.linspace(0, 1, n_bins + 1)
    rel: List[Dict[str, float]] = []
    ece_num = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        mean_p = float(p[m].mean())
        mean_y = float(y[m].mean())
        ece_num += cnt * abs(mean_y - mean_p)
        rel.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": cnt,
                    "mean_prob": mean_p, "mean_observed": mean_y})
    ece = float(ece_num / max(len(y), 1))
    return {
        "brier": brier, "calibration_intercept": intercept, "calibration_slope": slope,
        "ece": ece, "n_bins": n_bins, "reliability": rel,
    }


__all__ = ["patient_macro", "event_level", "calibration"]
