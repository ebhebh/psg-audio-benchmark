"""Stage-9 2.3 -- classification threshold sensitivity.

The main threshold rule stays ``inner_oof_max_youden_j``. Two pre-registered
**descriptive** sensitivities are added, both derived ONLY from that outer fold's
inner out-of-fold predictions: a fixed 0.5 threshold and the inner-OOF
balanced-accuracy-maximizing threshold. Threshold-**free** metrics
(AUROC/AUPRC/Brier/log loss) must be **identical** across variants. Outer-test
labels never influence any threshold.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ...modeling import schema as MS
from ...modeling.metrics import safe_auroc, safe_auprc, safe_brier, safe_logloss
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import FoldRerun, RerunBundle

_METRICS_TF = ("auroc", "auprc", "brier", "log_loss")
_METRICS_TB = ("sensitivity", "specificity", "precision", "f1", "balanced_accuracy")


def balanced_accuracy_threshold(
    *, y_true: np.ndarray, y_prob: np.ndarray,
) -> Optional[float]:
    """Inner-OOF balanced-accuracy-maximizing threshold (tie-break lowest prob)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2 or np.unique(p).size <= 1:
        return None
    fpr, tpr, thr = roc_curve(y, p)
    ba = (tpr + (1.0 - fpr)) / 2.0
    best = np.argmax(ba)
    best_ba = ba[best]
    tol = 1e-12
    tied = np.where(ba >= best_ba - tol)[0]
    chosen = tied[int(np.argmin(thr[tied]))]
    return float(thr[chosen])


def _confusion_metrics(y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * sens / (prec + sens)) if (
        prec is not None and sens is not None and (prec + sens) > 0) else None
    bal = (sens + spec) / 2.0 if (sens is not None and spec is not None) else None
    return {"sensitivity": sens, "specificity": spec, "precision": prec,
            "f1": f1, "balanced_accuracy": bal,
            "_tp": tp, "_fp": fp, "_tn": tn, "_fn": fn}


def _per_window_thresholds(
    folds: List[FoldRerun],
) -> Dict[str, np.ndarray]:
    """Three per-window threshold vectors (youden / fixed 0.5 / balanced-acc)."""
    youden, fixed05, balacc = [], [], []
    for f in sorted(folds, key=lambda g: g.outer_fold):
        n = f.y_test.size
        youden.append(np.full(n, f.threshold_youden))
        fixed05.append(np.full(n, 0.5))
        if f.inner_oof_y is not None and f.inner_oof_p is not None:
            t = balanced_accuracy_threshold(y_true=f.inner_oof_y, y_prob=f.inner_oof_p)
            t = 0.5 if t is None else t
        else:  # dummy_prior: no inner OOF -> fixed 0.5 only
            t = 0.5
        balacc.append(np.full(n, t))
    return {"youden": np.concatenate(youden),
            "fixed_0_5": np.concatenate(fixed05),
            "balacc": np.concatenate(balacc)}


def _threshold_free(y: np.ndarray, p: np.ndarray) -> Dict[str, Optional[float]]:
    return {"auroc": safe_auroc(y, p)[0], "auprc": safe_auprc(y, p)[0],
            "brier": safe_brier(y, p)[0], "log_loss": safe_logloss(y, p)[0]}


def _block(
    *, run_id: str, cohort: str, feature_set: str, model: str, role: str,
    folds: List[FoldRerun], n_windows_per_variant_check: bool,
) -> List[Dict[str, object]]:
    pooled_y = np.concatenate([f.y_test for f in sorted(folds, key=lambda g: g.outer_fold)])
    pooled_p = np.concatenate([f.proba for f in sorted(folds, key=lambda g: g.outer_fold)])
    tf = _threshold_free(pooled_y, pooled_p)  # invariant across variants
    pwt = _per_window_thresholds(folds)
    variants = {
        S.THRESHOLD_YOUDEN: pwt["youden"],
        S.THRESHOLD_FIXED_05: pwt["fixed_0_5"],
        S.THRESHOLD_BAL_ACC: pwt["balacc"],
    }
    rows: List[Dict[str, object]] = []
    # threshold-free metrics emitted for every variant (must be identical)
    for vname, _thr in variants.items():
        for metric in _METRICS_TF:
            rows.append(_row(run_id, cohort, feature_set, model, role,
                             "threshold_free", metric, vname, tf[metric]))
    # threshold-based metrics per variant (these DO change with the threshold)
    for vname, thr in variants.items():
        cm = _confusion_metrics(pooled_y, (pooled_p >= thr).astype(int))
        for metric in _METRICS_TB:
            rows.append(_row(run_id, cohort, feature_set, model, role,
                             "threshold_based", metric, vname, cm[metric]))
    return rows


def _row(run_id, cohort, feature_set, model, role, kind, metric, variant, value):
    return {
        "run_id": run_id, "analysis": S.A_THRESHOLD,
        "analysis_role": S.ROLE_SENSITIVITY, "cohort": cohort,
        "feature_set": feature_set, "model": model, "model_role": role,
        "metric_kind": kind, "metric": metric, "variant": variant,
        "outer_fold": "pooled", "value": (None if value is None else float(value)),
        "frozen_value": None, "abs_diff": None, "interpretable": True,
        "reason": "ok",
    }


def threshold_sensitivity(
    *, bundle: RerunBundle, run_id: str, mc: ResolvedRobustnessConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    for cohort, feature_set in targets:
        for model_name in MS.MODEL_FAMILIES:
            folds = bundle.folds(cohort=cohort, feature_set=feature_set, model=model_name)
            if not folds:
                continue
            rows.extend(_block(
                run_id=run_id, cohort=cohort, feature_set=feature_set,
                model=model_name, role=MS.MODEL_ROLE[model_name], folds=folds,
                n_windows_per_variant_check=True))
    df = pd.DataFrame(rows, columns=list(S.METRIC_ROW_COLUMNS))

    # invariant guard: threshold-free metrics must not vary across variants
    _assert_threshold_free_invariant(df)
    return df


def _assert_threshold_free_invariant(df: pd.DataFrame) -> None:
    tf = df[df["metric_kind"] == "threshold_free"]
    for (cohort, fs, model, metric), g in tf.groupby(["cohort", "feature_set", "model", "metric"]):
        vals = g["value"].astype(float).to_numpy()
        if vals.size and not np.allclose(vals, vals[0], atol=1e-12, equal_nan=True):
            raise RuntimeError(
                f"threshold-free metric {metric} varied across threshold variants "
                f"({cohort}/{fs}/{model}): {vals.tolist()}")


__all__ = ["threshold_sensitivity", "balanced_accuracy_threshold"]
