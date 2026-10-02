"""Stage-7 metric computation (see ``docs/metric_contract.md``).

All metrics take ``y_true`` (int 0/1), ``y_prob`` (positive-class P(y=1)) and,
for threshold-dependent metrics, a fixed ``threshold``. Implementations use
scikit-learn 1.7.2.

Degenerate inputs (single true class, undefined precision/F1, etc.) yield
``None`` (NA) with an explicit ``reason`` — **never** a crash and **never** a
fabricated 0. The :func:`compute_metrics` bundle returns a dict with both the
threshold-free and threshold-dependent metrics plus a confusion matrix.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

_NA: Optional[float] = None

_EPS = 1e-15


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], str]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2:
        return None, f"single true class {np.unique(y).tolist()}; AUROC undefined"
    if np.unique(p).size <= 1:
        return None, "constant probability; AUROC undefined"
    return float(roc_auc_score(y, p)), "ok"


def safe_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], str]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2:
        return None, f"single true class {np.unique(y).tolist()}; AUPRC undefined"
    if np.unique(p).size <= 1:
        # average_precision is still defined for constant prob, but uninformative
        # report it as the positive rate (the constant's AP)
        return float(average_precision_score(y, p)), "constant probability"
    return float(average_precision_score(y, p)), "ok"


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], str]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    return float(brier_score_loss(y, p)), "ok"


def safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Optional[float], str]:
    y = np.asarray(y_true).astype(int)
    p = _clip(y_prob)
    return float(log_loss(y, p, labels=[0, 1])), "ok"


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(np.sum((yp == 1) & (y == 1)))
    fp = int(np.sum((yp == 1) & (y == 0)))
    tn = int(np.sum((yp == 0) & (y == 0)))
    fn = int(np.sum((yp == 0) & (y == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(
    *, y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
) -> Dict[str, object]:
    """Compute the full metric bundle at a fixed ``threshold``."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    out: Dict[str, object] = {}

    out["n"] = int(len(y))
    out["n_positive"] = int(np.sum(y == 1))
    out["n_negative"] = int(np.sum(y == 0))
    out["positive_rate"] = float(np.mean(y)) if len(y) else None

    auroc, r = safe_auroc(y, p); out["auroc"], out["auroc_reason"] = auroc, r
    auprc, r = safe_auprc(y, p); out["auprc"], out["auprc_reason"] = auprc, r
    brier, r = safe_brier(y, p); out["brier"], out["brier_reason"] = brier, r
    ll, r = safe_logloss(y, p); out["log_loss"], out["log_loss_reason"] = ll, r

    y_pred = (p >= threshold).astype(int)
    cm = _confusion(y, y_pred)
    out["confusion_matrix"] = cm
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * sens / (prec + sens)) if (prec is not None and sens is not None
                                               and (prec + sens) > 0) else None
    bal = (sens + spec) / 2.0 if (sens is not None and spec is not None) else None
    out["threshold"] = float(threshold)
    out["sensitivity"] = sens
    out["specificity"] = spec
    out["precision"] = prec
    out["f1"] = f1
    out["balanced_accuracy"] = bal
    out["single_class"] = bool(len(np.unique(y)) < 2)
    return out


def pooled_metric(
    metric: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
) -> Optional[float]:
    """Recompute a single metric on a (possibly resampled) pooled set."""
    res = compute_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    val = res.get(metric)
    return None if val is None else (float(val) if not isinstance(val, (int, dict)) else val)


__all__ = [
    "safe_auroc",
    "safe_auprc",
    "safe_brier",
    "safe_logloss",
    "compute_metrics",
    "pooled_metric",
]
