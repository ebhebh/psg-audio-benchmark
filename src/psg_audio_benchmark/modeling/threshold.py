"""Stage-7 threshold selection.

Thresholds for threshold-dependent metrics are chosen by the pre-registered
rule:

* data-driven models (LR, HGB): **maximum Youden J** = argmax_t (TPR(t) −
  FPR(t)), computed on that outer fold's **inner out-of-fold validation
  predictions** of the selected candidate. Ties are broken by the **lowest**
  probability (more conservative) and are fully deterministic. The chosen
  threshold is then FIXED and applied to the outer-test predictions. The
  outer-test labels NEVER influence the threshold.
* ``dummy_prior``: a fixed 0.5 threshold (distinct, labelled).

Degenerate inputs (single true class, or a constant probability) yield a
threshold of 0.5 with an explicit ``reason`` rather than a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import roc_curve

from . import schema as S


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    rule: str
    reason: str = ""


def youden_threshold(
    *, y_true: np.ndarray, y_prob: np.ndarray, model_name: str,
) -> ThresholdResult:
    """Max-Youden-J threshold from inner OOF predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if model_name == S.MODEL_DUMMY_PRIOR:
        return ThresholdResult(
            threshold=S.THRESHOLD_DUMMY_VALUE, rule=S.THRESHOLD_RULE_DUMMY,
            reason="dummy_prior uses a fixed 0.5 threshold.",
        )
    classes = np.unique(y_true)
    if len(classes) < 2:
        return ThresholdResult(
            threshold=0.5, rule=S.THRESHOLD_RULE_DATA_DRIVEN,
            reason=f"single true class {classes.tolist()}; defaulting to 0.5.",
        )
    if np.unique(y_prob).size <= 1:
        return ThresholdResult(
            threshold=0.5, rule=S.THRESHOLD_RULE_DATA_DRIVEN,
            reason="constant inner-OOF probability; defaulting to 0.5.",
        )
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    # tie-break: highest J, then LOWEST threshold (more conservative / fewer FP)
    best = np.argmax(j)  # first max by default (deterministic)
    best_j = j[best]
    # among all thresholds within tolerance of best_j, pick the smallest threshold
    tol = 1e-12
    tied = np.where(j >= best_j - tol)[0]
    chosen = tied[int(np.argmin(thr[tied]))]
    return ThresholdResult(
        threshold=float(thr[chosen]), rule=S.THRESHOLD_RULE_DATA_DRIVEN,
        reason=f"max Youden J={best_j:.6f} at threshold={thr[chosen]:.6f} "
               f"(tie-break lowest probability among {len(tied)} tied).",
    )


__all__ = ["ThresholdResult", "youden_threshold"]
