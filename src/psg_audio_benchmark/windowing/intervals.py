"""Pure interval math for Stage-4 windowing.

Side-effect-free helpers for the half-open window contract and the exact
awake-overlap computation. Nothing here reads files, touches raw data, rounds
away precision, or assumes any audio time anchor. These primitives are unit
tested exhaustively (prompt section 6, boundary cases 1-5).
"""

from __future__ import annotations

from typing import Sequence, Tuple

#: A half-open interval [start, end) in seconds relative to record_start.
Interval = Tuple[float, float]


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def window_contains(window_start: float, window_end: float, point: float) -> bool:
    """Half-open containment: ``window_start <= point < window_end``.

    A point exactly on a window boundary belongs to the window that *starts*
    there, never to the one that *ends* there (prompt section 6 case 1).
    """
    return window_start <= point < window_end


def awake_overlap_fraction(
    window_start: float,
    window_end: float,
    awake_bounds: Sequence[Interval],
) -> float:
    """Exact fraction of ``[window_start, window_end)`` covered by awake bounds.

    Returns a value in ``[0.0, 1.0]``. ``0.0`` means no overlap; ``1.0`` means
    the window is fully inside awake intervals. A non-positive window duration
    yields ``0.0``. Overlap is summed across all bounds (the awake set is a
    union of canonical intervals which are themselves non-overlapping, but the
    arithmetic is correct even if inputs overlap).
    """
    dur = window_end - window_start
    if dur <= 0:
        return 0.0
    covered = 0.0
    for (a, b) in awake_bounds:
        if b <= a:
            continue
        lo = a if a > window_start else window_start
        hi = b if b < window_end else window_end
        if hi > lo:
            covered += hi - lo
    if covered <= 0:
        return 0.0
    if covered >= dur:
        return 1.0
    return round(covered / dur, 6)


def awake_overlap_is_excluded(
    overlap_fraction: float, max_awake_overlap_fraction: float
) -> bool:
    """True iff the window must be excluded for awake overlap.

    With the default ``max_awake_overlap_fraction == 0.0`` any strictly positive
    overlap excludes the window. ``>`` (not ``>=``) is used so that exactly
    ``0.0`` is allowed and exactly the threshold is allowed (prompt case 3).
    """
    return overlap_fraction > max_awake_overlap_fraction


__all__ = [
    "Interval",
    "clamp",
    "window_contains",
    "awake_overlap_fraction",
    "awake_overlap_is_excluded",
]
