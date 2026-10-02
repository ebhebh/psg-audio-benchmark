"""Awake-interval normalization: raw -> canonical (prompt section 3.3).

Produces two tables per run:

* ``awake_intervals_raw.parquet`` keeps every interval in original order/index.
* ``awake_intervals_canonical.parquet`` applies, within each patient:
    1. cross-midnight normalization on the cumulative timeline;
    2. sort by start;
    3. exact-duplicate removal (identical ``(start, end)`` pairs collapsed, with
       a source-index map so nothing is silently lost);
    4. overlap / adjacent (touching) merge.

A raw->canonical source-index map and per-patient dedup/merge/cross-midnight
counts are returned. This stage does NOT delete wake events and does NOT build
label windows; it only computes event/awake overlap for the next stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .schema import (
    AwakeCanonical,
    AwakeRaw,
    SyncWarning,
    WARN_AWAKE_BEFORE_RECORD_START,
)
from .time_axis import TimePoint, annotation_timepoint


@dataclass
class AwakeCanonicalizationResult:
    raw_rows: List[AwakeRaw]
    canonical_rows: List[AwakeCanonical]
    warnings: List[SyncWarning]
    stats: Dict[str, int]


def canonicalize_awake(
    *,
    run_id: str,
    patient_id: str,
    source_annotation_relpath: str,
    awake_raw: List[Tuple[int, float, float]],
    rs_point: TimePoint,
) -> AwakeCanonicalizationResult:
    """Canonicalize one patient's awake intervals."""
    raw_rows: List[AwakeRaw] = []
    warnings: List[SyncWarning] = []
    total_raw = len(awake_raw)

    # Build raw rows + validated entries on the cumulative timeline.
    entries: List[_Entry] = []
    n_negative_or_zero = 0
    for raw_idx, s_raw, e_raw in awake_raw:
        raw_rows.append(
            AwakeRaw(
                run_id=run_id,
                patient_id=patient_id,
                source_annotation_relpath=source_annotation_relpath,
                raw_order_index=raw_idx,
                start_raw_seconds=float(s_raw),
                end_raw_seconds=float(e_raw),
            )
        )
        if e_raw <= s_raw:
            n_negative_or_zero += 1
            warnings.append(
                SyncWarning(
                    run_id=run_id,
                    patient_id=patient_id,
                    scope="awake",
                    source_relpath=source_annotation_relpath,
                    warning_code="awake_nonpositive_duration",
                    detail=f"raw_index={raw_idx} start>={end_repr(e_raw)}",
                )
            )
            continue
        sp = annotation_timepoint(s_raw)
        ep = annotation_timepoint(e_raw)
        s_rel = sp.relative_to(rs_point)
        e_rel = ep.relative_to(rs_point)
        if e_rel < 0 or s_rel < 0:
            warnings.append(
                SyncWarning(
                    run_id=run_id,
                    patient_id=patient_id,
                    scope="awake",
                    source_relpath=source_annotation_relpath,
                    warning_code=WARN_AWAKE_BEFORE_RECORD_START,
                    detail=f"raw_index={raw_idx} before record_start",
                )
            )
        entries.append(
            _Entry(
                raw_indices=[raw_idx],
                start_raw=s_raw,
                end_raw=e_raw,
                start_point=sp,
                end_point=ep,
                start_rel=s_rel,
                end_rel=e_rel,
            )
        )

    # 1) exact-duplicate removal keyed by (start_raw, end_raw)
    dedup: Dict[Tuple[float, float], _Entry] = {}
    order: List[Tuple[float, float]] = []
    for e in entries:
        key = (e.start_raw, e.end_raw)
        if key in dedup:
            dedup[key].raw_indices.extend(e.raw_indices)
        else:
            dedup[key] = _Entry(
                raw_indices=list(e.raw_indices),
                start_raw=e.start_raw,
                end_raw=e.end_raw,
                start_point=e.start_point,
                end_point=e.end_point,
                start_rel=e.start_rel,
                end_rel=e.end_rel,
            )
            order.append(key)
    unique = [dedup[k] for k in order]
    exact_duplicates_removed = total_raw - len(unique)

    # 2) sort by start_rel (then end_rel)
    unique.sort(key=lambda e: (e.start_rel, e.end_rel))

    # 3) overlap / adjacent (touching) merge: next.start_rel <= cur.end_rel
    merged: List[_Entry] = []
    for e in unique:
        if merged and e.start_rel <= merged[-1].end_rel:
            cur = merged[-1]
            if e.end_rel > cur.end_rel:
                cur.end_rel = e.end_rel
                cur.end_point = e.end_point
                cur.end_raw = e.end_raw
            cur.raw_indices.extend(e.raw_indices)
        else:
            merged.append(_Entry(
                raw_indices=list(e.raw_indices),
                start_raw=e.start_raw,
                end_raw=e.end_raw,
                start_point=e.start_point,
                end_point=e.end_point,
                start_rel=e.start_rel,
                end_rel=e.end_rel,
            ))
    overlaps_or_adjacent_merged = len(unique) - len(merged)

    # Build canonical rows + cross-midnight accounting.
    canonical_rows: List[AwakeCanonical] = []
    cross_midnight_intervals = 0
    for e in merged:
        cross = e.start_point.day_offset != e.end_point.day_offset
        if cross:
            cross_midnight_intervals += 1
        canonical_rows.append(
            AwakeCanonical(
                run_id=run_id,
                patient_id=patient_id,
                start_seconds_of_day=round(e.start_point.seconds_of_day, 6),
                start_day_offset=e.start_point.day_offset,
                start_relative_to_record_start=round(e.start_rel, 6),
                end_seconds_of_day=round(e.end_point.seconds_of_day, 6),
                end_day_offset=e.end_point.day_offset,
                end_relative_to_record_start=round(e.end_rel, 6),
                duration_seconds=round(e.end_rel - e.start_rel, 6),
                cross_midnight=bool(cross),
                source_raw_indices=";".join(str(i) for i in sorted(set(e.raw_indices))),
                n_source_intervals=len(set(e.raw_indices)),
            )
        )

    stats = {
        "raw_total": total_raw,
        "canonical_total": len(canonical_rows),
        "exact_duplicates_removed": exact_duplicates_removed,
        "overlaps_or_adjacent_merged": overlaps_or_adjacent_merged,
        "cross_midnight_intervals": cross_midnight_intervals,
        "nonpositive_dropped": n_negative_or_zero,
    }
    return AwakeCanonicalizationResult(
        raw_rows=raw_rows,
        canonical_rows=canonical_rows,
        warnings=warnings,
        stats=stats,
    )


def end_repr(v: float) -> str:
    return repr(float(v))


@dataclass
class _Entry:
    raw_indices: List[int]
    start_raw: float
    end_raw: float
    start_point: TimePoint
    end_point: TimePoint
    start_rel: float
    end_rel: float


__all__ = ["AwakeCanonicalizationResult", "canonicalize_awake"]
