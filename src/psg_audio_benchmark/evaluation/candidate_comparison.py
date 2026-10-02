"""v1-vs-v2 candidate comparison for Stage 6B.

Computes per-fold + cohort balance metrics from a patient->fold assignment using
ONLY patient-level aggregates (no feature values), and builds a tidy v1-vs-v2
comparison table (``split_candidate_comparison.csv``) plus a summary dict.

``positive_rate_spread`` = max-min of per-fold test positive rate.
``window_count_cv`` = std(ddof=0)/mean of per-fold test window counts.
``max_abs_deviation`` = max over folds of |fold positive rate - cohort rate|.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .balance_optimizer import PatientBurden
from .split_balance_config import ResolvedSplitBalanceConfig


@dataclass
class FoldMetrics:
    per_fold: List[Dict[str, Any]] = field(default_factory=list)
    positive_rate_spread: float = 0.0
    window_count_cv: float = 0.0
    overall_positive_rate: float = 0.0
    max_abs_deviation: float = 0.0
    n_patients_total: int = 0
    n_windows_total: int = 0
    n_positive_total: int = 0


def _metrics_from_perfold(per_fold: List[Dict[str, Any]], overall_rate: float) -> FoldMetrics:
    rates = np.array([pf["positive_rate"] for pf in per_fold], dtype=np.float64)
    wins = np.array([pf["n_windows"] for pf in per_fold], dtype=np.float64)
    spread = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0
    mean_w = float(wins.mean()) if len(wins) else 0.0
    wcv = float(wins.std(ddof=0) / mean_w) if mean_w > 0 else 0.0
    abs_dev = float(np.max(np.abs(rates - overall_rate))) if len(rates) else 0.0
    return FoldMetrics(
        per_fold=per_fold, positive_rate_spread=spread, window_count_cv=wcv,
        overall_positive_rate=overall_rate, max_abs_deviation=abs_dev,
        n_patients_total=int(sum(pf["n_patients"] for pf in per_fold)),
        n_windows_total=int(sum(pf["n_windows"] for pf in per_fold)),
        n_positive_total=int(sum(pf["n_positive"] for pf in per_fold)),
    )


def compute_assignment_metrics(
    *, patient_to_fold: Dict[str, int], burden_by_pid: Dict[str, PatientBurden],
    n_folds: int,
) -> FoldMetrics:
    """Metrics for a patient->fold assignment over a fixed cohort."""
    fw = np.zeros(n_folds, dtype=np.int64)
    fp = np.zeros(n_folds, dtype=np.int64)
    counts = np.zeros(n_folds, dtype=np.int64)
    for pid, fold in patient_to_fold.items():
        b = burden_by_pid[pid]
        fw[fold] += b.n_windows; fp[fold] += b.n_positive; counts[fold] += 1
    total_w = int(fw.sum()); total_p = int(fp.sum())
    overall = float(total_p / total_w) if total_w else 0.0
    per_fold: List[Dict[str, Any]] = []
    for f in range(n_folds):
        n = int(fw[f]); pos = int(fp[f])
        per_fold.append({
            "fold": f, "n_patients": int(counts[f]), "n_windows": n,
            "n_positive": pos, "n_negative": n - pos,
            "positive_rate": float(pos / n) if n else 0.0,
        })
    return _metrics_from_perfold(per_fold, overall)


def compute_v1_metrics_from_outer_csv(
    path: Path | str, *, n_folds: int = 5,
) -> FoldMetrics:
    """Metrics for the v1 split from its ``outer_patient_folds_core.csv``.

    (The v1 product is the immutable comparator; this only reads it.)
    """
    import pandas as pd
    df = pd.read_csv(path, dtype={"patient_id": str})
    total_w = int(df["n_windows"].sum()); total_p = int(df["n_positive"].sum())
    overall = float(total_p / total_w) if total_w else 0.0
    per_fold: List[Dict[str, Any]] = []
    g = df.groupby("outer_fold")
    for f in range(n_folds):
        if f not in g.groups:
            per_fold.append({"fold": f, "n_patients": 0, "n_windows": 0,
                             "n_positive": 0, "n_negative": 0, "positive_rate": 0.0})
            continue
        sub = g.get_group(f)
        n = int(sub["n_windows"].sum()); pos = int(sub["n_positive"].sum())
        per_fold.append({
            "fold": f, "n_patients": int(sub["patient_id"].nunique()),
            "n_windows": n, "n_positive": pos, "n_negative": n - pos,
            "positive_rate": float(pos / n) if n else 0.0,
        })
    return _metrics_from_perfold(per_fold, overall)


def build_comparison_rows(v1: FoldMetrics, v2: FoldMetrics) -> List[Dict[str, Any]]:
    """Tidy long-form rows: ``scope, fold, metric, v1, v2, delta``."""
    rows: List[Dict[str, Any]] = []
    nf = max(len(v1.per_fold), len(v2.per_fold))
    per_fold_metrics = ("n_patients", "n_windows", "n_positive", "n_negative",
                        "positive_rate", "abs_deviation_from_overall")
    for f in range(nf):
        v1f = v1.per_fold[f] if f < len(v1.per_fold) else {}
        v2f = v2.per_fold[f] if f < len(v2.per_fold) else {}
        for m in per_fold_metrics:
            if m == "abs_deviation_from_overall":
                a = abs(float(v1f.get("positive_rate", 0.0)) - v1.overall_positive_rate)
                b = abs(float(v2f.get("positive_rate", 0.0)) - v2.overall_positive_rate)
            else:
                a = v1f.get(m); b = v2f.get(m)
            rows.append({"scope": "fold", "fold": f, "metric": m,
                         "v1": a, "v2": b, "delta": (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None})
    for m, a, b in (
        ("positive_rate_spread", v1.positive_rate_spread, v2.positive_rate_spread),
        ("window_count_cv", v1.window_count_cv, v2.window_count_cv),
        ("max_abs_deviation", v1.max_abs_deviation, v2.max_abs_deviation),
        ("overall_positive_rate", v1.overall_positive_rate, v2.overall_positive_rate),
        ("n_patients_total", v1.n_patients_total, v2.n_patients_total),
        ("n_windows_total", v1.n_windows_total, v2.n_windows_total),
        ("n_positive_total", v1.n_positive_total, v2.n_positive_total),
    ):
        rows.append({"scope": "cohort", "fold": "", "metric": m,
                     "v1": a, "v2": b, "delta": (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None})
    return rows


def build_comparison_summary(
    v1: FoldMetrics, v2: FoldMetrics, config: ResolvedSplitBalanceConfig,
) -> Dict[str, Any]:
    return {
        "v1_positive_rate_spread": v1.positive_rate_spread,
        "v2_positive_rate_spread": v2.positive_rate_spread,
        "v1_window_count_cv": v1.window_count_cv,
        "v2_window_count_cv": v2.window_count_cv,
        "v1_max_abs_deviation": v1.max_abs_deviation,
        "v2_max_abs_deviation": v2.max_abs_deviation,
        "overall_positive_rate": v2.overall_positive_rate,
        "gate_spread_max": config.gate_relative_positive_rate_spread_max,
        "wcv_ceiling_factor": config.gate_v2_wcv_ceiling_factor,
        "v2_spread_below_threshold": v2.positive_rate_spread <= config.gate_relative_positive_rate_spread_max,
        "v2_spread_strictly_below_v1": v2.positive_rate_spread < v1.positive_rate_spread,
        "v2_wcv_not_worse_than_v1_ceiling": v2.window_count_cv <= v1.window_count_cv * config.gate_v2_wcv_ceiling_factor + 1e-12,
    }


__all__ = [
    "FoldMetrics",
    "compute_assignment_metrics",
    "compute_v1_metrics_from_outer_csv",
    "build_comparison_rows",
    "build_comparison_summary",
]
