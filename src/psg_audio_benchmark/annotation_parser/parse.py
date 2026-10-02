"""Controlled annotation parsing with original-fidelity + rejection recording
(prompt section 3.1).

Reads each annotation JSON and extracts ``record_start``, ``awake_intervals``
and ``events`` while:

* preserving ``patient_id``, ``source_annotation_relpath``,
  ``source_event_index`` and the raw field names/values as a *safe* (structured,
  non-identifying) representation;
* tolerating the ``evnet_start`` typo AND the canonical ``event_start`` field;
  if both are present and *conflict*, recording ``conflicting_event_start_fields``
  rather than silently picking one;
* sending every problem (missing field, non-numeric, NaN/Inf, negative duration,
  end-before-start, unknown event type, unparseable JSON) to the rejection list
  with a reason + severity, so the pipeline NEVER crashes on a bad event.

Outputs never copy full event JSON, recognizable speech or long free text; only
the structured fields and counts needed for research.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .event_mapping import EventTypeMapping
from .schema import (
    ParsedEvent,
    RejectionRecord,
    SEV_ERROR,
    SEV_WARNING,
)
from .time_axis import (
    TimePoint,
    annotation_timepoint,
    is_negative_duration,
    parse_float_seconds,
)

#: Start-field candidates in priority order (canonical first, typo second).
START_FIELD_CANDIDATES = ("event_start", "evnet_start")


@dataclass
class AnnotationParseResult:
    """Outcome of parsing one annotation file."""

    patient_id: str
    source_annotation_relpath: str
    record_start_point: Optional[TimePoint]
    record_start_present: bool
    awake_raw: List[Tuple[int, float, float]] = field(default_factory=list)
    parsed_events: List[ParsedEvent] = field(default_factory=list)
    rejections: List[RejectionRecord] = field(default_factory=list)
    file_anomalies: List[str] = field(default_factory=list)
    n_events_in_file: int = 0


# ---------------------------------------------------------------------------
# JSON loading (never raises out of the public API)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Tuple[Optional[dict], List[str]]:
    """Return ``(parsed_dict_or_None, anomalies)``. Never raises."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
        except OSError as exc:
            return None, [f"read_failed:{type(exc).__name__}"]
    except OSError as exc:
        return None, [f"read_failed:{type(exc).__name__}"]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"json_decode_error:{exc}"]
    if not isinstance(data, dict):
        return None, ["top_level_not_object"]
    return data, []


# ---------------------------------------------------------------------------
# Start-field resolution (prompt 3.1: priority + conflict)
# ---------------------------------------------------------------------------

def resolve_event_start(
    event: dict,
) -> Tuple[Optional[float], str, bool, Optional[str]]:
    """Resolve the event start seconds.

    Returns ``(start_seconds_or_None, field_used, conflict_flag, reason)``.
    ``field_used`` is the field name actually consulted ("" if none).
    ``reason`` is a rejection reason code or "" on success.
    """
    canonical = event.get("event_start")
    typo = event.get("evnet_start")
    has_canonical = "event_start" in event
    has_typo = "evnet_start" in event

    if has_canonical and has_typo:
        # Both present: only a conflict if they differ.
        cval, _ = parse_float_seconds(canonical)
        tval, _ = parse_float_seconds(typo)
        if cval is not None and tval is not None and cval != tval:
            return None, "", True, "conflicting_event_start_fields"
        # equal (or unparseable-but-both-present): prefer canonical
        val, code = parse_float_seconds(canonical)
        if val is None:
            return None, "event_start", True, f"event_start_{code}"
        return val, "event_start", True, ""
    if has_canonical:
        val, code = parse_float_seconds(canonical)
        if val is None:
            return None, "event_start", False, f"event_start_{code}"
        return val, "event_start", False, ""
    if has_typo:
        val, code = parse_float_seconds(typo)
        if val is None:
            return None, "evnet_start", False, f"evnet_start_{code}"
        return val, "evnet_start", False, ""
    return None, "", False, "missing_event_start_field"


# ---------------------------------------------------------------------------
# Main parse
# ---------------------------------------------------------------------------

def parse_annotation_file(
    path: Path,
    *,
    run_id: str,
    patient_id: str,
    source_annotation_relpath: str,
    mapping: EventTypeMapping,
) -> AnnotationParseResult:
    """Parse one annotation JSON into events + rejections (never raises)."""
    path = Path(path)
    data, anomalies = _load_json(path)
    result = AnnotationParseResult(
        patient_id=patient_id,
        source_annotation_relpath=source_annotation_relpath,
        record_start_point=None,
        record_start_present=False,
        file_anomalies=list(anomalies),
    )
    if data is None:
        # Whole file unparseable: nothing more to do; anomalies carry the reason.
        return result

    # ---- record_start (the per-patient timeline anchor) ----
    rs_present = "record_start" in data
    result.record_start_present = rs_present
    rs_point: Optional[TimePoint] = None
    rs_ok = True
    if rs_present:
        val, code = parse_float_seconds(data.get("record_start"))
        if val is None:
            result.file_anomalies.append(f"record_start_{code}")
            rs_ok = False
        else:
            rs_point = annotation_timepoint(val)
            result.record_start_point = rs_point
    else:
        result.file_anomalies.append("record_start_missing")
        rs_ok = False

    # ---- awake_intervals (raw only here; normalization is in awake.py) ----
    aw = data.get("awake_intervals")
    if isinstance(aw, list):
        for idx, item in enumerate(aw):
            sb, sb_code = _pair_value(item, 0) if isinstance(item, (list, tuple)) and len(item) >= 2 else (None, "awake_not_pair")
            eb, eb_code = _pair_value(item, 1) if isinstance(item, (list, tuple)) and len(item) >= 2 else (None, "awake_not_pair")
            if sb is None or eb is None:
                result.file_anomalies.append(f"awake_interval_{idx}_unparseable")
                continue
            result.awake_raw.append((idx, sb, eb))
    elif aw is None:
        result.file_anomalies.append("awake_intervals_missing")
    else:
        result.file_anomalies.append("awake_intervals_not_list")

    # ---- events ----
    events = data.get("events")
    if not isinstance(events, list):
        if events is None:
            result.file_anomalies.append("events_missing")
        else:
            result.file_anomalies.append("events_not_list")
        return result

    result.n_events_in_file = len(events)
    for idx, ev in enumerate(events):
        _parse_one_event(
            ev,
            idx=idx,
            run_id=run_id,
            patient_id=patient_id,
            relpath=source_annotation_relpath,
            mapping=mapping,
            rs_point=rs_point,
            rs_ok=rs_ok,
            result=result,
        )
    return result


def _pair_value(pair: Any, pos: int) -> Tuple[Optional[float], str]:
    try:
        return parse_float_seconds(pair[pos])
    except (IndexError, TypeError):
        return None, "awake_not_pair"


def _parse_one_event(
    ev: Any,
    *,
    idx: int,
    run_id: str,
    patient_id: str,
    relpath: str,
    mapping: EventTypeMapping,
    rs_point: Optional[TimePoint],
    rs_ok: bool,
    result: AnnotationParseResult,
) -> None:
    """Validate + standardize one event, appending to parsed/rejection lists."""
    base_reject = lambda reason, severity=SEV_ERROR, hint="": result.rejections.append(
        RejectionRecord(
            run_id=run_id,
            source_annotation_relpath=relpath,
            patient_id=patient_id,
            source_event_index=idx,
            reason=reason,
            severity=severity,
            raw_field_hint=hint,
        )
    )

    if not isinstance(ev, dict):
        base_reject("event_not_object")
        return

    # event_type
    raw_type = ev.get("event_type")
    if raw_type is None or not isinstance(raw_type, str) or not raw_type.strip():
        base_reject("missing_or_nonstring_event_type")
        return
    raw_type = raw_type.strip()
    standardized, map_status, is_scored = mapping.standardize(raw_type)
    if map_status == "unknown_type":
        base_reject(f"unknown_event_type:{raw_type}", hint="event_type")
        return

    # start
    start_val, field_used, conflict, reason = resolve_event_start(ev)
    if conflict and reason == "conflicting_event_start_fields":
        base_reject("conflicting_event_start_fields", hint="event_start/evnet_start")
        return
    if start_val is None:
        base_reject(reason or "event_start_unparseable", hint=field_used)
        return
    if start_val < 0:
        base_reject("negative_event_start", hint=field_used)
        return

    # duration
    if "event_duration" not in ev:
        base_reject("missing_event_duration", hint="event_duration")
        return
    dur_val, dur_code = parse_float_seconds(ev.get("event_duration"))
    if dur_val is None:
        base_reject(f"event_duration_{dur_code}", hint="event_duration")
        return
    if is_negative_duration(start_val, dur_val):
        base_reject("negative_event_duration", hint="event_duration")
        return

    end_val = start_val + dur_val
    if end_val < start_val:
        base_reject("end_before_start", hint="event_duration")
        return

    # If record_start is unavailable we cannot place the event on the timeline.
    if not rs_ok or rs_point is None:
        base_reject("record_start_unavailable_for_relative_time")
        return

    start_point = annotation_timepoint(start_val)
    start_rel = start_point.relative_to(rs_point)
    end_rel = start_rel + dur_val
    if start_rel < 0:
        # Event precedes record_start: still parse, but record a warning.
        result.rejections.append(
            RejectionRecord(
                run_id=run_id,
                source_annotation_relpath=relpath,
                patient_id=patient_id,
                source_event_index=idx,
                reason="event_start_before_record_start",
                severity=SEV_WARNING,
                raw_field_hint=field_used,
            )
        )

    sleep_stage = ev.get("sleep_stage", "")
    sleep_stage = sleep_stage if isinstance(sleep_stage, str) else ""

    result.parsed_events.append(
        ParsedEvent(
            run_id=run_id,
            patient_id=patient_id,
            source_annotation_relpath=relpath,
            source_event_index=idx,
            event_type_raw=raw_type,
            event_type_standardized=standardized,
            mapping_status=map_status,
            is_any_scored_respiratory_event=bool(is_scored),
            event_start_raw_seconds=float(start_val),
            event_start_seconds_of_day=start_point.seconds_of_day,
            event_start_day_offset=start_point.day_offset,
            event_start_relative_to_record_start=round(start_rel, 6),
            event_duration_seconds=float(dur_val),
            event_end_relative_to_record_start=round(end_rel, 6),
            overlaps_awake_interval=False,
            awake_overlap_seconds=0.0,
            awake_overlap_fraction=0.0,
            sleep_stage_raw=sleep_stage,
            conflicting_event_start_fields=bool(conflict),
        )
    )


__all__ = [
    "START_FIELD_CANDIDATES",
    "AnnotationParseResult",
    "parse_annotation_file",
    "resolve_event_start",
]
