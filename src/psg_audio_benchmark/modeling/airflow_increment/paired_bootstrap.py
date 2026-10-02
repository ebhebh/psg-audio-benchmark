"""Stage-8 paired patient-cluster bootstrap (primary LR increment only).

Quantifies the uncertainty of the airflow INCREMENT (airflow_enhanced -
core_restricted) on the SAME windows. The pairing is preserved by resampling
**patients** (the independent cluster unit) ONCE and bringing, for every drawn
patient, ALL of that patient's pooled out-of-fold windows for BOTH feature sets
simultaneously. The increment is then computed within the identical resampled
cohort.

Hard contract (enforced structurally; this module is incapable of violating it):

* resample unit = patient, never window IID. Windows within a patient are
  correlated, so a window-level IID bootstrap is forbidden;
* the SAME patient draw feeds BOTH feature sets (``prob_core`` and
  ``prob_enhanced`` index the same windows in the same order), so the increment
  is a within-cohort paired difference -- never a difference of two independent
  bootstraps, never a cross-cohort pairing;
* a resample whose true labels are a single class is **skipped and counted**
  (it makes AUROC/AUPRC undefined); the whole resample is skipped so every
  metric's CI is built on the identical kept-draw set.

Threshold-free metrics (AUROC, AUPRC, Brier) use the raw probabilities.
Threshold-based metrics (sensitivity, specificity, F1, balanced accuracy) use
each feature set's PER-WINDOW inner-OOF Youden threshold (applied identically
inside every resample). Probabilities are RAW; no calibration is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..metrics import safe_auroc, safe_auprc, safe_brier

_THRESHOLD_FREE = {"auroc": safe_auroc, "auprc": safe_auprc, "brier": safe_brier}


@dataclass
class PairedBootstrapCI:
    metric: str
    kind: str                       # threshold_free | threshold_based
    direction: str                  # enhanced_minus_core
    delta_point_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    core_point_estimate: Optional[float]
    enhanced_point_estimate: Optional[float]
    n_resamples: int
    n_used: int
    n_single_class_skipped: int
    ci_level: float
    unit: str

    def as_record(self, *, run_id: str, model: str) -> dict:
        return {
            "run_id": run_id,
            "model": model,
            "metric": self.metric,
            "kind": self.kind,
            "direction": self.direction,
            "delta_point_estimate": self.delta_point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "core_point_estimate": self.core_point_estimate,
            "enhanced_point_estimate": self.enhanced_point_estimate,
            "n_resamples": int(self.n_resamples),
            "n_used": int(self.n_used),
            "n_single_class_skipped": int(self.n_single_class_skipped),
            "ci_level": float(self.ci_level),
            "unit": self.unit,
        }


def _confusion_metrics(y: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    y = np.asarray(y).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(np.sum((yp == 1) & (y == 1)))
    fp = int(np.sum((yp == 1) & (y == 0)))
    tn = int(np.sum((yp == 0) & (y == 0)))
    fn = int(np.sum((yp == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * sens / (prec + sens)) if (
        prec is not None and sens is not None and (prec + sens) > 0) else None
    bal = (sens + spec) / 2.0 if (sens is not None and spec is not None) else None
    return {"sensitivity": sens, "specificity": spec, "f1": f1,
            "balanced_accuracy": bal}


def _metric_value(
    metric: str, y: np.ndarray, p: np.ndarray, thr_per_window: np.ndarray,
) -> Optional[float]:
    """Value of one metric on (y, p) at the per-window threshold (if relevant)."""
    if metric in _THRESHOLD_FREE:
        val, _reason = _THRESHOLD_FREE[metric](y, p)
        return val
    # threshold-based: apply the per-window threshold, then the confusion metric
    y_pred = (np.asarray(p) >= np.asarray(thr_per_window)).astype(int)
    return _confusion_metrics(y, y_pred).get(metric)


def _resample_patient_indices(
    *, patient_ids: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Draw patients WITH REPLACEMENT once, expand to the full paired window index
    list (every window of every drawn patient). The SAME index list applies to
    both feature sets (identical window order)."""
    unique_pids = np.array(sorted(set(patient_ids.tolist())))
    draw = rng.choice(unique_pids, size=unique_pids.size, replace=True)
    pid_to_idx: Dict[str, List[int]] = {}
    for i, p in enumerate(patient_ids):
        pid_to_idx.setdefault(str(p), []).append(int(i))
    out: List[int] = []
    for p in draw:
        out.extend(pid_to_idx[str(p)])
    return np.asarray(out, dtype=int)


def paired_patient_cluster_bootstrap(
    *, y_true: np.ndarray, prob_core: np.ndarray, prob_enhanced: np.ndarray,
    patient_ids: np.ndarray, threshold_core: np.ndarray,
    threshold_enhanced: np.ndarray, metrics: Sequence[str], n_resamples: int = 1000,
    seed: int = 20250714, ci_level: float = 0.95,
) -> List[PairedBootstrapCI]:
    """Paired patient-cluster percentile bootstrap CI for each requested metric's
    increment (enhanced - core).

    ``y_true``, ``prob_core``, ``prob_enhanced``, ``patient_ids``,
    ``threshold_core`` and ``threshold_enhanced`` must all index the SAME windows
    in the SAME order (asserted). The same patient draw feeds both feature sets.
    """
    n = len(y_true)
    arrays = (y_true, prob_core, prob_enhanced, patient_ids,
              threshold_core, threshold_enhanced)
    if not all(len(a) == n for a in arrays):
        raise ValueError("paired bootstrap: input arrays must share one length")
    if set(metrics) - (set(_THRESHOLD_FREE) | {"sensitivity", "specificity",
                                               "f1", "balanced_accuracy"}):
        raise ValueError(f"paired bootstrap: unknown metric(s): {metrics}")

    rng = np.random.default_rng(seed)
    # classify metrics
    tf_metrics = [m for m in metrics if m in _THRESHOLD_FREE]
    tb_metrics = [m for m in metrics if m not in _THRESHOLD_FREE]

    # pre-collect kept-draw increments per metric
    increments: Dict[str, List[float]] = {m: [] for m in metrics}
    skipped = 0

    for _ in range(n_resamples):
        idx = _resample_patient_indices(patient_ids=patient_ids, rng=rng)
        yb = y_true[idx]
        # single true class -> AUROC/AUPRC undefined -> skip the whole draw
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        pc = prob_core[idx]
        pe = prob_enhanced[idx]
        tcore = threshold_core[idx]
        tenh = threshold_enhanced[idx]
        for m in tf_metrics:
            vc = _metric_value(m, yb, pc, tcore)
            ve = _metric_value(m, yb, pe, tenh)
            if vc is not None and ve is not None:
                increments[m].append(float(ve) - float(vc))
        for m in tb_metrics:
            vc = _metric_value(m, yb, pc, tcore)
            ve = _metric_value(m, yb, pe, tenh)
            if vc is not None and ve is not None:
                increments[m].append(float(ve) - float(vc))

    alpha = (1.0 - ci_level) / 2.0
    results: List[PairedBootstrapCI] = []
    for m in metrics:
        kind = "threshold_free" if m in _THRESHOLD_FREE else "threshold_based"
        core_pt = _metric_value(m, y_true, prob_core, threshold_core)
        enh_pt = _metric_value(m, y_true, prob_enhanced, threshold_enhanced)
        delta_pt = (float(enh_pt) - float(core_pt)) if (
            core_pt is not None and enh_pt is not None) else None
        draws = increments[m]
        if draws:
            lo = float(np.percentile(draws, 100.0 * alpha))
            hi = float(np.percentile(draws, 100.0 * (1.0 - alpha)))
            n_used = len(draws)
        else:
            lo = hi = None
            n_used = 0
        results.append(PairedBootstrapCI(
            metric=m, kind=kind, direction="enhanced_minus_core",
            delta_point_estimate=delta_pt, ci_low=lo, ci_high=hi,
            core_point_estimate=core_pt, enhanced_point_estimate=enh_pt,
            n_resamples=int(n_resamples), n_used=int(n_used),
            n_single_class_skipped=int(skipped), ci_level=float(ci_level),
            unit="patient",
        ))
    return results


__all__ = ["PairedBootstrapCI", "paired_patient_cluster_bootstrap"]
