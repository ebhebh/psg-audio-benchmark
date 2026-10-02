"""Pure core-signal coverage test for Stage-4 windows (prompt section 3).

A window is **core-coverage-complete** iff its entire ``[start, end)`` lies
inside BOTH verified core extents (heart_rate and spo2). ``airflow`` is optional:
it only reports ``has_airflow_coverage`` and never gates label eligibility.
``sleep_structure`` is observational context only.

These functions take plain numbers / coverage tuples so they are decoupled from
the Stage-3 reader and unit-testable independently. Nothing reads files.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .schema import PatientSignalCoverage


def _inside(start: float, end: float, first_rel: Optional[float],
            last_rel: Optional[float]) -> bool:
    """True iff ``[start, end)`` is fully within ``[first_rel, last_rel]``."""
    if first_rel is None or last_rel is None:
        return False
    return start >= first_rel and end <= last_rel


def core_heart_rate_coverage(
    start_rel: float, end_rel: float, cov: PatientSignalCoverage
) -> bool:
    """True iff the window is fully covered by the verified heart_rate extent."""
    if not cov.hr_verified:
        return False
    return _inside(start_rel, end_rel, cov.hr_first_rel, cov.hr_last_rel)


def core_spo2_coverage(
    start_rel: float, end_rel: float, cov: PatientSignalCoverage
) -> bool:
    """True iff the window is fully covered by the verified spo2 extent."""
    if not cov.spo2_verified:
        return False
    return _inside(start_rel, end_rel, cov.spo2_first_rel, cov.spo2_last_rel)


def core_coverage_complete(
    start_rel: float, end_rel: float, cov: PatientSignalCoverage
) -> bool:
    """True iff BOTH core modalities fully cover ``[start, end)``."""
    return core_heart_rate_coverage(start_rel, end_rel, cov) and core_spo2_coverage(
        start_rel, end_rel, cov
    )


def airflow_coverage(
    start_rel: float, end_rel: float, cov: PatientSignalCoverage
) -> bool:
    """Optional: True iff airflow is verified and fully covers the window.

    Never affects label eligibility (it only enriches the window with
    ``has_airflow_coverage``).
    """
    if not cov.airflow_verified:
        return False
    return _inside(start_rel, end_rel, cov.airflow_first_rel, cov.airflow_last_rel)


def coverage_flags(
    start_rel: float, end_rel: float, cov: PatientSignalCoverage
) -> Tuple[bool, bool, bool, bool]:
    """Return ``(hr, spo2, airflow, complete)`` coverage flags for a window."""
    hr = core_heart_rate_coverage(start_rel, end_rel, cov)
    spo2 = core_spo2_coverage(start_rel, end_rel, cov)
    af = airflow_coverage(start_rel, end_rel, cov)
    return hr, spo2, af, (hr and spo2)


__all__ = [
    "core_heart_rate_coverage",
    "core_spo2_coverage",
    "core_coverage_complete",
    "airflow_coverage",
    "coverage_flags",
]
