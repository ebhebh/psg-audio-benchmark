"""Stage-7 patient-cluster bootstrap (main model only).

Uncertainty is quantified by resampling **patients** (the independent cluster
unit) with replacement and bringing ALL of each drawn patient's pooled
out-of-fold windows — never by resampling windows IID (windows within a patient
are correlated). A resample that contains a single true class is **skipped and
counted**, not scored. Report the percentile 95% CI of the kept draws.

This module enforces the patient-cluster contract; it is incapable of doing a
window-level IID bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .metrics import safe_auroc, safe_auprc


@dataclass
class BootstrapCI:
    metric: str
    point_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    n_resamples: int
    n_used: int
    n_single_class_skipped: int
    ci_level: float
    unit: str

    def as_record(self, *, run_id: str, model_name: str) -> dict:
        return {
            "run_id": run_id,
            "model": model_name,
            "metric": self.metric,
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_resamples": int(self.n_resamples),
            "n_used": int(self.n_used),
            "n_single_class_skipped": int(self.n_single_class_skipped),
            "ci_level": float(self.ci_level),
            "unit": self.unit,
        }


def _resample_indices(
    *, patient_ids: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Patient-cluster resample: draw patients with replacement, expand to the
    full window index list (every window of every drawn patient)."""
    unique_pids = np.array(sorted(set(patient_ids.tolist())))
    draw = rng.choice(unique_pids, size=unique_pids.size, replace=True)
    pid_to_idx: Dict[str, List[int]] = {}
    for i, p in enumerate(patient_ids):
        pid_to_idx.setdefault(str(p), []).append(int(i))
    out: List[int] = []
    for p in draw:
        out.extend(pid_to_idx[str(p)])
    return np.asarray(out, dtype=int)


def _metric_fn(metric: str):
    return {"auroc": safe_auroc, "auprc": safe_auprc}[metric]


def patient_cluster_bootstrap(
    *, y_true: np.ndarray, y_prob: np.ndarray, patient_ids: np.ndarray,
    metrics: Sequence[str], n_resamples: int = 1000, seed: int = 20250714,
    ci_level: float = 0.95,
) -> List[BootstrapCI]:
    """Patient-cluster percentile bootstrap CI for each requested metric."""
    if len(y_true) != len(y_prob) or len(y_true) != len(patient_ids):
        raise ValueError("y_true / y_prob / patient_ids length mismatch")
    rng = np.random.default_rng(seed)
    results: List[BootstrapCI] = []
    for metric in metrics:
        fn = _metric_fn(metric)
        draws: List[float] = []
        skipped = 0
        for _ in range(n_resamples):
            idx = _resample_indices(patient_ids=patient_ids, rng=rng)
            yb = y_true[idx]
            pb = y_prob[idx]
            val, _reason = fn(yb, pb)
            if val is None:
                skipped += 1
                continue
            draws.append(float(val))
        point, _ = fn(y_true, y_prob)
        if draws:
            alpha = (1.0 - ci_level) / 2.0
            lo = float(np.percentile(draws, 100.0 * alpha))
            hi = float(np.percentile(draws, 100.0 * (1.0 - alpha)))
            n_used = len(draws)
        else:
            lo = hi = None
            n_used = 0
        results.append(BootstrapCI(
            metric=metric, point_estimate=point, ci_low=lo, ci_high=hi,
            n_resamples=int(n_resamples), n_used=int(n_used),
            n_single_class_skipped=int(skipped), ci_level=float(ci_level),
            unit="patient",
        ))
    return results


__all__ = ["BootstrapCI", "patient_cluster_bootstrap"]
