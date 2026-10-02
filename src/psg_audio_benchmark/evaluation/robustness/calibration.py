"""Stage-9 2.4 -- probability-calibration sensitivity (LR only).

Two pre-registered calibrators are fit **per outer fold** on that fold's
inner-OOF predictions and applied to that fold's outer-test OOF probabilities:

1. ``sigmoid_platt``  -- a 1-D logistic regression of ``y_inner ~ logit(p_inner)``.
2. ``isotonic``       -- monotonic ``IsotonicRegression`` on ``(p_inner, y_inner)``.

A ``raw_uncalibrated`` row carries the un-calibrated probabilities for
comparison. Calibration is **never** fit on outer-test labels; Brier / log loss /
ECE / calibration intercept / calibration slope on the outer-test are
evaluation only. Isotonic is marked ``not interpretable`` when any fold's
inner-OOF has fewer than the configured minimum positives or negatives.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ...modeling import schema as MS
from ...modeling.metrics import safe_brier, safe_logloss
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import FoldRerun, RerunBundle

_EPS = 1e-6
_CAL_METRICS = ("brier", "log_loss", "ece", "calibration_intercept", "calibration_slope")
_VARIANTS = (S.CAL_RAW, S.CAL_SIGMOID_PLATT, S.CAL_ISOTONIC)


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    pc = _clip(p)
    return np.log(pc / (1.0 - pc))


# --- calibrator fit / apply -------------------------------------------------

def _fit_sigmoid(inner_y: np.ndarray, inner_p: np.ndarray) -> LogisticRegression:
    s = _logit(inner_p).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    lr.fit(s, inner_y)
    return lr


def _apply_sigmoid(lr: LogisticRegression, p: np.ndarray) -> np.ndarray:
    s = _logit(p).reshape(-1, 1)
    return np.clip(lr.predict_proba(s)[:, 1], 0.0, 1.0)


def _fit_isotonic(inner_y: np.ndarray, inner_p: np.ndarray) -> IsotonicRegression:
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    ir.fit(inner_p, inner_y)
    return ir


def _apply_isotonic(ir: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    return np.clip(ir.transform(p), 0.0, 1.0)


# --- calibration evaluation metrics ----------------------------------------

def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = float(len(y))
    if n == 0:
        return float("nan")
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        m = int(mask.sum())
        if m == 0:
            continue
        ece += (m / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def _slope_intercept(y: np.ndarray, p: np.ndarray) -> Dict[str, Optional[float]]:
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": None, "calibration_slope": None}
    s = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    lr.fit(s, y)
    return {"calibration_intercept": float(lr.intercept_[0]),
            "calibration_slope": float(lr.coef_[0][0])}


def _eval(y: np.ndarray, p: np.ndarray) -> Dict[str, Optional[float]]:
    out = {"brier": safe_brier(y, p)[0], "log_loss": safe_logloss(y, p)[0],
           "ece": _ece(y, p)}
    out.update(_slope_intercept(y, p))
    return out


# --- per-fold pooled calibrated probabilities ------------------------------

def _pooled_calibrated(
    folds: List[FoldRerun], *, method: str, min_pos: int, min_neg: int,
) -> Dict[str, object]:
    """Return pooled (y, p) for a calibration variant + an interpretability flag."""
    order = sorted(folds, key=lambda f: f.outer_fold)
    ys, ps = [], []
    interpretable = True
    reason = "ok"
    for f in order:
        inner_y = f.inner_oof_y
        inner_p = f.inner_oof_p
        if method == S.CAL_RAW:
            ps.append(f.proba)
        elif method == S.CAL_SIGMOID_PLATT:
            if inner_y is None or inner_p is None or len(np.unique(inner_y)) < 2:
                ps.append(f.proba); interpretable = False; reason = "inner_oof_single_class"
                continue
            lr = _fit_sigmoid(inner_y, inner_p)
            ps.append(_apply_sigmoid(lr, f.proba))
        elif method == S.CAL_ISOTONIC:
            if inner_y is None or inner_p is None:
                ps.append(f.proba); interpretable = False; reason = "inner_oof_missing"
                continue
            n_pos = int(np.sum(inner_y == 1)); n_neg = int(np.sum(inner_y == 0))
            if n_pos < min_pos or n_neg < min_neg:
                ps.append(f.proba); interpretable = False
                reason = (f"isotonic_min_counts_unmet_pos{n_pos}_neg{n_neg}")
                continue
            ir = _fit_isotonic(inner_y, inner_p)
            ps.append(_apply_isotonic(ir, f.proba))
        ys.append(f.y_test)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return {"y": y, "p": p, "interpretable": interpretable, "reason": reason}


def _row(run_id, cohort, feature_set, model, role, method, metric, value,
         interpretable, reason):
    return {
        "run_id": run_id, "analysis": S.A_CALIBRATION,
        "analysis_role": S.ROLE_SENSITIVITY, "cohort": cohort,
        "feature_set": feature_set, "model": model, "model_role": role,
        "metric_kind": "calibration", "metric": metric, "variant": method,
        "outer_fold": "pooled",
        "value": (None if value is None else float(value)),
        "frozen_value": None, "abs_diff": None,
        "interpretable": bool(interpretable), "reason": reason,
    }


def calibration_sensitivity(
    *, bundle: RerunBundle, run_id: str, mc: ResolvedRobustnessConfig,
) -> pd.DataFrame:
    """Calibration sensitivity for LR across the core + airflow cohorts."""
    cal_cfg = mc.calibration_sensitivity
    min_pos = int(cal_cfg.get("isotonic_min_pos_per_inner_fold", 5))
    min_neg = int(cal_cfg.get("isotonic_min_neg_per_inner_fold", 5))
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    rows: List[Dict[str, object]] = []
    for cohort, feature_set in targets:
        folds = bundle.folds(cohort=cohort, feature_set=feature_set,
                             model=MS.MODEL_LOGISTIC_REGRESSION)
        if not folds:
            continue
        role = MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION]
        for method in _VARIANTS:
            pooled = _pooled_calibrated(folds, method=method,
                                        min_pos=min_pos, min_neg=min_neg)
            ev = _eval(pooled["y"], pooled["p"])
            interp = pooled["interpretable"]
            reason = pooled["reason"]
            for metric in _CAL_METRICS:
                v = ev.get(metric)
                rows.append(_row(run_id, cohort, feature_set,
                                 MS.MODEL_LOGISTIC_REGRESSION, role, method,
                                 metric, v, interp, reason))
    return pd.DataFrame(rows, columns=list(S.METRIC_ROW_COLUMNS))


__all__ = ["calibration_sensitivity"]
