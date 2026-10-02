"""Pure, interpretable physiology feature functions (Stage 5).

Every function here operates ONLY on raw CSV sample arrays (times + values) and
window bounds -- never on labels, annotations, awake flags, sleep stage or any
audio field. This is what makes the label-permutation regression test trivial:
because no label is ever an input, permuting the labels cannot change a feature.

Only a small, transparent baseline set is implemented (prompt section 3). No
black-box embeddings, no deep features, no audio features, no unexplained
feature selection. Statistics are computed over the FINITE samples that fall
inside the half-open window; missing/non-finite samples are counted, never
imputed to a physiological value.

These functions are numpy-based and unit-testable in isolation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _as_float_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr


def finite_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of finite (non-NaN, non-Inf) entries."""
    return np.isfinite(values)


def common_statistics(
    t_win: np.ndarray, v_win: np.ndarray, *, min_samples: int
) -> Dict[str, Optional[float]]:
    """Mean/median/std(ddof=1)/min/max/range/iqr over finite (t, v) pairs.

    ``t_win`` and ``v_win`` are the samples whose time falls in the window (same
    length). Returns a dict of Optional[float]; all None when fewer than
    ``min_samples`` finite values are present.
    """
    out: Dict[str, Optional[float]] = {
        "mean": None, "median": None, "std_ddof1": None,
        "min": None, "max": None, "range": None, "iqr": None,
    }
    v = _as_float_array(v_win)
    finite = finite_mask(v)
    if int(finite.sum()) < max(2, min_samples):
        return out
    vf = v[finite]
    out["mean"] = float(np.mean(vf))
    out["median"] = float(np.median(vf))
    out["std_ddof1"] = float(np.std(vf, ddof=1))
    out["min"] = float(np.min(vf))
    out["max"] = float(np.max(vf))
    out["range"] = float(out["max"] - out["min"])
    q1, q3 = np.percentile(vf, [25, 75])
    out["iqr"] = float(q3 - q1)
    return out


def linear_slope(
    t_win: np.ndarray, v_win: np.ndarray, *, min_samples: int
) -> Optional[float]:
    """Ordinary least-squares slope of value vs window-time (value per second).

    Computed over finite (t, v) pairs. Returns None when fewer than
    ``min_samples`` finite values are present, or when window-time has zero
    variance (a constant time axis).
    """
    v = _as_float_array(v_win)
    t = _as_float_array(t_win)
    finite = finite_mask(v) & finite_mask(t)
    if int(finite.sum()) < max(2, min_samples):
        return None
    tf = t[finite]
    vf = v[finite]
    t_var = float(np.var(tf))
    if t_var == 0.0:
        return None
    # slope = cov(t, v) / var(t)
    cov = float(np.mean((tf - tf.mean()) * (vf - vf.mean())))
    return float(cov / t_var)


def rms(values: np.ndarray) -> Optional[float]:
    """Root-mean-square of the finite samples (airflow)."""
    v = _as_float_array(values)
    finite = finite_mask(v)
    if int(finite.sum()) < 2:
        return None
    vf = v[finite]
    return float(np.sqrt(np.mean(vf ** 2)))


def zero_crossings(
    values: np.ndarray, *, duration_seconds: float, min_samples: int
) -> Tuple[Optional[int], Optional[float]]:
    """Zero-crossing count and rate (airflow).

    Definition: a zero crossing is a sign change between two CONSECUTIVE FINITE
    samples. A sample exactly equal to 0 is treated as positive (>= 0); a
    transition from >= 0 to < 0 (or vice versa) counts as one crossing.

    Returns ``(count, rate)`` where ``rate = count / duration_seconds``. Both are
    None when fewer than ``min_samples`` finite values are present or the
    duration is non-positive.
    """
    v = _as_float_array(values)
    finite = finite_mask(v)
    if int(finite.sum()) < max(2, min_samples) or duration_seconds <= 0:
        return None, None
    vf = v[finite]
    sign = np.where(vf >= 0, 1, -1)
    # crossings = number of adjacent sign differences
    diffs = np.diff(sign)
    count = int(np.count_nonzero(diffs != 0))
    rate = float(count / duration_seconds)
    return count, rate


def n_below_threshold(values: np.ndarray, threshold: float) -> int:
    """DESCRIPTIVE count of finite samples strictly below a threshold (SpO2).

    This is a descriptive statistic only; it is NOT a clinical desaturation
    diagnosis and uses only the config-declared threshold.
    """
    v = _as_float_array(values)
    finite = finite_mask(v)
    if int(finite.sum()) == 0:
        return 0
    return int(np.count_nonzero(v[finite] < threshold))


__all__ = [
    "finite_mask",
    "common_statistics",
    "linear_slope",
    "rms",
    "zero_crossings",
    "n_below_threshold",
]
