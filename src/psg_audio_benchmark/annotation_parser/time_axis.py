"""Unified time representation for Stage 3 (prompt section 4.1).

Implements the explicit time-axis contract documented in
``docs/time_axis_contract.md``. The rules here are deliberately pure and
side-effect-free so they can be unit-tested exhaustively (prompt section 6).

Two source representations are reconciled into one monotonic timeline:

* **Annotation values** (``record_start``, event start, awake-interval bounds)
  are stored as float seconds that are *already cumulative* from the midnight
  beginning the recording-start day (they may exceed 86,400).
* **CSV clock strings** (``HH:MM:SS(.ms)``) wrap every midnight and need the
  single ``+86400`` rollover rule to become cumulative.

Nothing in this module reads files, touches raw data, or assumes any audio
acquisition start. It never compares time as a fuzzy string.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: Seconds in one day.
SECONDS_PER_DAY = 86_400

#: ``HH:MM:SS`` with optional ``.ms`` fraction. Hours may exceed 23 only when
#: the value is already a relative position exceeding 24 h (we still parse it).
_HHMMSS_RE = re.compile(
    r"^\s*(\d{1,3}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$"
)


@dataclass(frozen=True)
class TimePoint:
    """Auditable, normalized representation of a single time value."""

    raw_value: float
    raw_repr: str
    seconds_of_day: float
    day_offset: int
    cumulative_seconds: float

    def relative_to(self, anchor: "TimePoint") -> float:
        """Seconds relative to ``anchor`` on the cumulative timeline."""
        return self.cumulative_seconds - anchor.cumulative_seconds


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------

def parse_float_seconds(value) -> Tuple[Optional[float], str]:
    """Parse an annotation numeric value to float seconds.

    Returns ``(seconds_or_None, reason)``. ``reason`` is empty on success and a
    short code (``not_numeric`` / ``nan_or_inf`` / ``negative``) otherwise.
    """
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        return None, "not_numeric"
    if not isinstance(value, (int, float)):
        return None, "not_numeric"
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        return None, "nan_or_inf"
    return f, ""


def parse_hhmmss_seconds(text: str) -> Optional[float]:
    """Parse ``HH:MM:SS(.ms)`` -> seconds (may exceed 86,400 for relative)."""
    if text is None:
        return None
    m = _HHMMSS_RE.match(str(text))
    if not m:
        return None
    h, mi, s, ms = m.groups()
    frac = 0.0
    if ms:
        frac = float("0." + ms)
    return int(h) * 3600 + int(mi) * 60 + int(s) + frac


def is_negative_duration(start: float, duration: float) -> bool:
    return duration < 0


# ---------------------------------------------------------------------------
# Normalization to the unified timeline
# ---------------------------------------------------------------------------

def annotation_timepoint(raw_seconds: float) -> TimePoint:
    """Normalize an annotation value (already cumulative) to a :class:`TimePoint`.

    ``day_offset = floor(raw / 86400)``, ``seconds_of_day = raw mod 86400``.
    """
    raw = float(raw_seconds)
    day_offset = int(math.floor(raw / SECONDS_PER_DAY)) if raw >= 0 else 0
    seconds_of_day = raw - day_offset * SECONDS_PER_DAY
    return TimePoint(
        raw_value=raw,
        raw_repr=_repr_float(raw),
        seconds_of_day=seconds_of_day,
        day_offset=max(day_offset, 0),
        cumulative_seconds=raw,
    )


def _repr_float(v: float) -> str:
    """Compact, stable repr that survives CSV round-tripping."""
    if float(v).is_integer():
        return f"{v:.1f}"
    return repr(float(v))


def assign_clock_day_offsets(
    seconds_of_day_seq: Sequence[float], record_start_sod: float
) -> List[int]:
    """Assign ``day_offset`` to a CSV clock column using the single rollover rule.

    ``seconds_of_day_seq`` are parsed ``HH:MM:SS`` values **in file order**.
    ``record_start_sod`` is ``record_start`` seconds-of-day (the anchor).

    Rule (see contract Section 3):
      * first sample: ``day_offset = 0`` if ``s0 >= record_start_sod`` else ``1``;
      * subsequent: increment by 1 whenever the clock wraps backward
        (``s_i < s_{i-1}``).
    """
    if not seconds_of_day_seq:
        return []
    offsets: List[int] = []
    first = seconds_of_day_seq[0]
    offsets.append(0 if first >= record_start_sod else 1)
    prev = first
    for s in seconds_of_day_seq[1:]:
        if s < prev:
            offsets.append(offsets[-1] + 1)
        else:
            offsets.append(offsets[-1])
        prev = s
    return offsets


def clock_timepoint(
    seconds_of_day: float, day_offset: int
) -> TimePoint:
    """Build a :class:`TimePoint` for a CSV clock value with a known day_offset."""
    sod = float(seconds_of_day)
    do = int(day_offset)
    cum = sod + do * SECONDS_PER_DAY
    return TimePoint(
        raw_value=cum,
        raw_repr=_format_hhmmss(cum),
        seconds_of_day=sod,
        day_offset=do,
        cumulative_seconds=cum,
    )


def _format_hhmmss(cumulative: float) -> str:
    """Render cumulative seconds as ``HH:MM:SS.mmm`` (hours may exceed 23)."""
    sign = "-" if cumulative < 0 else ""
    c = abs(cumulative)
    h = int(c // 3600)
    rem = c - h * 3600
    mi = int(rem // 60)
    s = rem - mi * 60
    return f"{sign}{h:02d}:{mi:02d}:{s:06.3f}"


def segment_overlaps_awake(
    seg_start_rel: float,
    seg_end_rel: float,
    awake_bounds_rel: Sequence[Tuple[float, float]],
) -> Tuple[bool, float, float]:
    """Overlap of ``[seg_start_rel, seg_end_rel]`` with the union of awake bounds.

    Returns ``(overlaps, covered_seconds, covered_fraction)`` where the fraction
    is relative to the segment duration (``seg_end - seg_start``); 0.0 if the
    segment has non-positive duration.
    """
    dur = seg_end_rel - seg_start_rel
    if dur <= 0:
        return False, 0.0, 0.0
    covered = 0.0
    overlaps = False
    for (a, b) in awake_bounds_rel:
        if b <= a:
            continue
        lo = max(seg_start_rel, a)
        hi = min(seg_end_rel, b)
        if hi > lo:
            overlaps = True
            covered += hi - lo
    return overlaps, round(covered, 6), round(covered / dur, 6)


def segment_in_range(
    seg_start_rel: float, seg_end_rel: float, range_start_rel: float, range_end_rel: float
) -> bool:
    """True iff ``[seg_start, seg_end]`` intersects ``[range_start, range_end]``."""
    if seg_end_rel < seg_start_rel or range_end_rel < range_start_rel:
        return False
    return not (seg_end_rel < range_start_rel or seg_start_rel > range_end_rel)


__all__ = [
    "SECONDS_PER_DAY",
    "TimePoint",
    "parse_float_seconds",
    "parse_hhmmss_seconds",
    "is_negative_duration",
    "annotation_timepoint",
    "assign_clock_day_offsets",
    "clock_timepoint",
    "segment_overlaps_awake",
    "segment_in_range",
]
