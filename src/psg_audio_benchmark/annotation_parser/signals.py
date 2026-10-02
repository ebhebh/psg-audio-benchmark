"""CSV signal time-range parsing + event/signal availability (prompt 4.1).

For every physiological / sleep-structure CSV this module streams the file once
(never loading it whole, never imputing) and, for each patient/signal, computes:

* first & last **absolute** clock time (cumulative + relative-to-record-start);
* first & last **relative** time and the relative duration;
* sample count;
* absolute vs relative duration, cross-midnight, day offsets;
* a verification status (Section 5 of the time-axis contract) and a data-quality
  status, with the consistency error in seconds.

It also offers the per-event <-> per-signal availability test used to annotate
``parsed_events.overlaps_signals``.
"""

from __future__ import annotations

import csv as csvlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..data_audit import csv_probe
from .schema import (
    SignalTimeRange,
    SyncWarning,
    WARN_ABS_REL_DURATION_MISMATCH,
    WARN_NON_MONOTONIC_CSV,
    WARN_NO_ABSOLUTE_COLUMN,
    WARN_NO_RELATIVE_COLUMN,
    WARN_SIGNAL_STARTS_BEFORE_RECORD_START,
)
from .time_axis import (
    SECONDS_PER_DAY,
    TimePoint,
    clock_timepoint,
    parse_hhmmss_seconds,
    segment_in_range,
)

#: Max rows we will ever iterate (safety valve against pathological files).
_MAX_ROWS = 5_000_000


def _find_column(columns: List[str], *needles: str) -> Optional[int]:
    """First column whose lower-cased name contains every needle AND is a time
    column (name suggests position/time)."""
    for i, col in enumerate(columns):
        low = col.lower()
        if all(n in low for n in needles) and csv_probe._is_time_named(col):
            return i
    return None


def compute_signal_time_range(
    path,
    *,
    run_id: str,
    patient_id: str,
    modality: str,
    source_relpath: str,
    rs_point: Optional[TimePoint],
    duration_tolerance_s: float,
) -> Tuple[Optional[SignalTimeRange], List[SyncWarning]]:
    """Stream one CSV and build its :class:`SignalTimeRange` (never raises)."""
    path = Path(path)
    warnings: List[SyncWarning] = []
    if not path.is_file():
        return None, warnings
    try:
        if path.stat().st_size == 0:
            return None, warnings
    except OSError:
        return None, warnings

    encoding = csv_probe._detect_encoding(path)
    try:
        handle = open(path, "r", encoding=encoding, newline="", errors="replace")
    except OSError:
        return None, warnings

    with handle:
        first_line = handle.readline()
        delimiter = csv_probe._detect_delimiter(first_line)
        handle.seek(0)
        reader = csvlib.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return None, warnings
        columns = [c.strip() for c in header]

        abs_idx = _find_column(columns, "absolute")
        rel_idx = _find_column(columns, "relative")
        has_abs = abs_idx is not None
        has_rel = rel_idx is not None

        # Stream-track only first/last/prev + running day_offset (no per-row
        # list) so even multi-million-row CSVs stay light and fast.
        first_abs_sod: Optional[float] = None
        last_abs_sod: Optional[float] = None
        prev_abs: Optional[float] = None
        non_monotonic_abs = False
        running_day_offset = 0
        first_day_offset = 0
        wraps = 0
        n_abs = 0
        first_rel: Optional[float] = None
        last_rel: Optional[float] = None
        n_rel = 0
        n_rows = 0
        rs_sod = rs_point.seconds_of_day if rs_point is not None else 0.0
        for row in reader:
            n_rows += 1
            if n_rows > _MAX_ROWS:
                warnings.append(_w(run_id, patient_id, source_relpath, "csv_row_iter_capped"))
                break
            if has_abs and abs_idx is not None and len(row) > abs_idx:
                v = parse_hhmmss_seconds(row[abs_idx])
                if v is not None:
                    if n_abs == 0:
                        running_day_offset = 0 if v >= rs_sod else 1
                        first_day_offset = running_day_offset
                        first_abs_sod = v
                    else:
                        prev = prev_abs if prev_abs is not None else v
                        if v < prev:
                            drop = prev - v
                            if drop > SECONDS_PER_DAY / 2:
                                # legitimate single midnight crossing
                                running_day_offset += 1
                                wraps += 1
                                if wraps > 1:
                                    # >1 wrap in a <24h recording is anomalous
                                    non_monotonic_abs = True
                            else:
                                # small backward jump = clock glitch, not midnight
                                non_monotonic_abs = True
                    prev_abs = v
                    last_abs_sod = v
                    n_abs += 1
            if has_rel and rel_idx is not None and len(row) > rel_idx:
                v = parse_hhmmss_seconds(row[rel_idx])
                if v is not None:
                    if n_rel == 0:
                        first_rel = v
                    last_rel = v
                    n_rel += 1

    last_day_offset = running_day_offset

    # Absolute cumulative timeline via the rollover rule (streamed above).
    first_abs_rel = last_abs_rel = None
    first_abs_cum = last_abs_cum = None
    cross_midnight = False
    abs_duration = None
    if has_abs and first_abs_sod is not None and last_abs_sod is not None:
        first_tp = clock_timepoint(first_abs_sod, first_day_offset)
        last_tp = clock_timepoint(last_abs_sod, last_day_offset)
        first_abs_cum = first_tp.cumulative_seconds
        last_abs_cum = last_tp.cumulative_seconds
        abs_duration = last_abs_cum - first_abs_cum
        cross_midnight = last_day_offset > first_day_offset
        if rs_point is not None:
            first_abs_rel = round(first_tp.relative_to(rs_point), 6)
            last_abs_rel = round(last_tp.relative_to(rs_point), 6)

    rel_duration = None
    if has_rel and first_rel is not None and last_rel is not None:
        rel_duration = round(last_rel - first_rel, 6)
        first_rel = round(first_rel, 6)
        last_rel = round(last_rel, 6)

    # Verification (time-axis contract Section 5).
    verification_status = "not_verified"
    data_quality = "ok"
    alignment_error: Optional[float] = None
    warn_codes: List[str] = []

    if not has_abs:
        warn_codes.append(WARN_NO_ABSOLUTE_COLUMN)
        data_quality = "no_absolute_time_column"
    if not has_rel:
        warn_codes.append(WARN_NO_RELATIVE_COLUMN)
        if data_quality == "ok":
            data_quality = "no_relative_time_column"

    if has_abs and has_rel and abs_duration is not None and rel_duration is not None:
        alignment_error = round(abs(abs_duration - rel_duration), 6)
        if alignment_error > duration_tolerance_s:
            warn_codes.append(WARN_ABS_REL_DURATION_MISMATCH)
            data_quality = "absolute_relative_duration_mismatch"

    if non_monotonic_abs:
        warn_codes.append(WARN_NON_MONOTONIC_CSV)
        if data_quality == "ok":
            data_quality = "non_monotonic_absolute_time"

    if first_abs_rel is not None and first_abs_rel < -duration_tolerance_s:
        warn_codes.append(WARN_SIGNAL_STARTS_BEFORE_RECORD_START)
        if data_quality == "ok":
            data_quality = "signal_starts_before_record_start"

    if (
        has_abs
        and data_quality == "ok"
    ):
        verification_status = "verified"

    for code in warn_codes:
        warnings.append(_w(run_id, patient_id, source_relpath, code))

    rng = SignalTimeRange(
        run_id=run_id,
        patient_id=patient_id,
        modality=modality,
        source_relpath=source_relpath,
        time_axis_kind=("hhmmss" if (has_abs or has_rel) else "none"),
        has_absolute_column=bool(has_abs),
        has_relative_column=bool(has_rel),
        first_absolute_cumulative_seconds=_r(first_abs_cum),
        last_absolute_cumulative_seconds=_r(last_abs_cum),
        first_absolute_relative_to_record_start=first_abs_rel,
        last_absolute_relative_to_record_start=last_abs_rel,
        first_relative_seconds=first_rel,
        last_relative_seconds=last_rel,
        sample_count=n_rows,
        absolute_duration_seconds=_r(abs_duration),
        relative_duration_seconds=_r(rel_duration),
        cross_midnight=bool(cross_midnight),
        first_day_offset=int(first_day_offset),
        last_day_offset=int(last_day_offset),
        verification_status=verification_status,
        data_quality_status=data_quality,
        alignment_error_seconds=alignment_error,
        warnings=";".join(warn_codes),
    )
    return rng, warnings


def _r(v):
    return None if v is None else round(float(v), 6)


def _w(run_id, patient_id, relpath, code) -> SyncWarning:
    return SyncWarning(
        run_id=run_id,
        patient_id=patient_id,
        scope="csv",
        source_relpath=relpath,
        warning_code=code,
        detail="",
    )


def event_signal_overlap(
    event_start_rel: float,
    event_end_rel: float,
    ranges_by_modality: Dict[str, SignalTimeRange],
) -> Tuple[str, int]:
    """Return (semicolon-joined modalities covering the event, count)."""
    covering: List[str] = []
    for modality, rng in ranges_by_modality.items():
        if rng.first_absolute_relative_to_record_start is None:
            continue
        if rng.last_absolute_relative_to_record_start is None:
            continue
        if segment_in_range(
            event_start_rel,
            event_end_rel,
            rng.first_absolute_relative_to_record_start,
            rng.last_absolute_relative_to_record_start,
        ):
            covering.append(modality)
    return ";".join(sorted(covering)), len(covering)


__all__ = ["compute_signal_time_range", "event_signal_overlap"]
