"""Coverage, quality status, warnings and the availability gate (Stage 5).

Given the raw CSV samples that fall inside a window, this module decides whether
a modality is AVAILABLE for that window (coverage >= min_coverage and enough
finite samples) and assembles the quality fields + the (possibly all-NULL)
feature row. Low-coverage modalities are NULL + a quality code, NEVER
zero-filled, interpolated or imputed.

The assembly calls the pure functions in :mod:`.features`; it never touches
labels/annotations/audio. A modality absent for a patient (e.g. airflow for
16/50) yields ``modality_not_available_for_patient`` and does not affect the core
HR/SpO2 windows.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import features as F
from .schema import (
    AUDIO_FEATURES_PRESENT,
    MOD_AIRFLOW,
    MOD_HEART_RATE,
    MOD_SPO2,
    MODALITY_FEATURE_VERSION,
    MODALITY_VALUE_UNIT,
    Q_ALL_MISSING,
    Q_AVAILABLE,
    Q_INSUFFICIENT_SAMPLES,
    Q_LOW_COVERAGE,
    Q_MODALITY_NOT_AVAILABLE_FOR_PATIENT,
    Q_TIME_AXIS_UNAVAILABLE,
    WARN_ABNORMAL_TIME,
    WARN_CONSTANT_SIGNAL,
    WARN_NON_FINITE,
    WARN_RANGE,
    FeatureRow,
    ResolvedPhysiologyFeatureConfig,
)

# modality key -> physiological_range_warnings config key
_RANGE_KEY = {MOD_HEART_RATE: "heart_rate_bpm", MOD_SPO2: "spo2_pct"}


def compute_coverage(
    n_expected: int, n_observed: int, n_finite: int
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(coverage_fraction, missing_fraction)``.

    coverage = n_finite / n_expected ; missing = 1 - coverage. Both None when
    n_expected is not positive (time axis / fs unavailable).
    """
    if n_expected <= 0:
        return None, None
    cov = float(n_finite) / float(n_expected)
    return cov, float(1.0 - cov)


def decide_quality_status(
    *,
    modality_present_for_patient: bool,
    time_axis_ok: bool,
    n_finite: int,
    coverage_fraction: Optional[float],
    min_coverage: float,
    min_samples: int,
) -> str:
    """Decide the mutually-exclusive quality verdict for a modality/window."""
    if not modality_present_for_patient:
        return Q_MODALITY_NOT_AVAILABLE_FOR_PATIENT
    if not time_axis_ok or coverage_fraction is None:
        return Q_TIME_AXIS_UNAVAILABLE
    if n_finite == 0:
        return Q_ALL_MISSING
    if coverage_fraction < min_coverage:
        return Q_LOW_COVERAGE
    if n_finite < min_samples:
        return Q_INSUFFICIENT_SAMPLES
    return Q_AVAILABLE


def is_available(quality_status: str) -> bool:
    return quality_status == Q_AVAILABLE


def collect_warnings(
    v_win: np.ndarray,
    t_win: np.ndarray,
    *,
    modality: str,
    config: ResolvedPhysiologyFeatureConfig,
) -> List[str]:
    """Non-blocking quality warnings; never delete/clip raw data."""
    warnings: List[str] = []
    v = np.asarray(v_win, dtype=float)
    t = np.asarray(t_win, dtype=float)
    finite = F.finite_mask(v)
    n_obs = int(v.shape[0])
    n_fin = int(finite.sum())
    if n_obs > n_fin:
        warnings.append(WARN_NON_FINITE)
    if n_fin >= 2:
        vf = v[finite]
        if float(np.std(vf)) == 0.0:
            warnings.append(WARN_CONSTANT_SIGNAL)
        # physiological range warning (descriptive only)
        if config.physiological_range_warnings_enabled and modality in _RANGE_KEY:
            rng = config.physiological_ranges.get(_RANGE_KEY[modality])
            if isinstance(rng, tuple) and len(rng) == 2:
                lo, hi = float(rng[0]), float(rng[1])
                if bool(np.any(vf < lo)) or bool(np.any(vf > hi)):
                    warnings.append(WARN_RANGE)
    # abnormal time: non-monotonic sample times inside the window
    tf = t[F.finite_mask(t)]
    if tf.shape[0] >= 2 and bool(np.any(np.diff(tf) < 0)):
        warnings.append(WARN_ABNORMAL_TIME)
    return warnings


def build_feature_row(
    *,
    run_id: str,
    window_id: str,
    patient_id: str,
    window_index: int,
    modality: str,
    window_start_rel: float,
    window_end_rel: float,
    duration_seconds: float,
    t_win: np.ndarray,
    v_win: np.ndarray,
    n_expected: int,
    modality_present_for_patient: bool,
    time_axis_ok: bool,
    config: ResolvedPhysiologyFeatureConfig,
) -> FeatureRow:
    """Assemble one feature row with quality-aware NULLing.

    When the modality is unavailable for this window, every numeric feature is
    NULL and the quality fields explain why.
    """
    v = np.asarray(v_win, dtype=float)
    finite = F.finite_mask(v)
    n_observed = int(v.shape[0])
    n_finite = int(finite.sum())

    coverage, missing = compute_coverage(n_expected, n_observed, n_finite)
    status = decide_quality_status(
        modality_present_for_patient=modality_present_for_patient,
        time_axis_ok=time_axis_ok,
        n_finite=n_finite,
        coverage_fraction=coverage,
        min_coverage=config.min_coverage_for(modality),
        min_samples=config.min_samples_for_statistics,
    )
    warnings = collect_warnings(v, t_win, modality=modality, config=config)
    available = is_available(status)

    if available:
        stats = F.common_statistics(t_win, v, min_samples=config.min_samples_for_statistics)
        slope = F.linear_slope(t_win, v, min_samples=config.min_samples_for_slope)
        mean = stats["mean"]; median = stats["median"]; std = stats["std_ddof1"]
        mn = stats["min"]; mx = stats["max"]; rng = stats["range"]; iqr = stats["iqr"]
        rms_val: Optional[float] = None
        zc_count: Optional[int] = None
        zc_rate: Optional[float] = None
        if modality == MOD_AIRFLOW:
            rms_val = F.rms(v)
            zc_count, zc_rate = F.zero_crossings(
                v, duration_seconds=duration_seconds,
                min_samples=config.min_samples_for_zcr,
            )
    else:
        mean = median = std = mn = mx = rng = iqr = slope = None
        rms_val = zc_count = zc_rate = None

    # SpO2 optional descriptive desaturation count (off by default).
    n_below: Optional[int] = None
    desat_thr: Optional[float] = None
    if modality == MOD_SPO2 and config.spo2_desaturation_enabled and available:
        n_below = F.n_below_threshold(v, config.spo2_desaturation_threshold_pct)
        desat_thr = config.spo2_desaturation_threshold_pct

    return FeatureRow(
        run_id=run_id,
        window_id=window_id,
        patient_id=patient_id,
        window_index=window_index,
        modality=modality,
        window_start_relative_to_record_start=float(window_start_rel),
        window_end_relative_to_record_start=float(window_end_rel),
        duration_seconds=float(duration_seconds),
        modality_feature_version=MODALITY_FEATURE_VERSION[modality],
        audio_features_present=AUDIO_FEATURES_PRESENT,
        n_expected=int(n_expected),
        n_observed=n_observed,
        n_finite=n_finite,
        coverage_fraction=(None if coverage is None else float(coverage)),
        missing_fraction=(None if missing is None else float(missing)),
        quality_status=status,
        quality_warnings=";".join(warnings),
        mean=mean, median=median, std_ddof1=std, min=mn, max=mx, range=rng, iqr=iqr,
        slope_per_second=slope,
        value_unit=MODALITY_VALUE_UNIT[modality],
        rms=rms_val, zero_crossing_count=zc_count, zero_crossing_rate=zc_rate,
        n_below_desaturation_threshold=n_below,
        desaturation_threshold_pct=desat_thr,
    )


__all__ = [
    "compute_coverage",
    "decide_quality_status",
    "is_available",
    "collect_warnings",
    "build_feature_row",
]
