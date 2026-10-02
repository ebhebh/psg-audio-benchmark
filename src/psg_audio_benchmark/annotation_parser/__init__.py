"""Controlled annotation parsing, standardization and time-axis synchronization.

Stage 3 responsibility: parse the real JSON annotations (preserving fidelity and
recording rejections), conservatively standardize event types WITHOUT inventing
apnea subtypes, normalize awake intervals (raw -> canonical), compute CSV signal
time ranges on a unified, auditable timeline, and judge every audio file's time
anchor strictly (evidence-first: unresolved unless a verifiable anchor exists).

This stage does NOT window, extract features, train, build label windows, or
assume any audio acquisition start. See ``docs/time_axis_contract.md``.
"""

from __future__ import annotations

from .event_mapping import EventTypeMapping, load_event_type_mapping
from .parse import AnnotationParseResult, parse_annotation_file, resolve_event_start
from .qc import Stage3ContaminationError
from .runner import Stage3Options, Stage3Runner
from .schema import Stage3Summary
from .time_axis import (
    SECONDS_PER_DAY,
    TimePoint,
    annotation_timepoint,
    assign_clock_day_offsets,
    clock_timepoint,
    parse_float_seconds,
    parse_hhmmss_seconds,
)

__all__ = [
    "EventTypeMapping",
    "load_event_type_mapping",
    "AnnotationParseResult",
    "parse_annotation_file",
    "resolve_event_start",
    "Stage3ContaminationError",
    "Stage3Options",
    "Stage3Runner",
    "Stage3Summary",
    "SECONDS_PER_DAY",
    "TimePoint",
    "annotation_timepoint",
    "assign_clock_day_offsets",
    "clock_timepoint",
    "parse_float_seconds",
    "parse_hhmmss_seconds",
]
