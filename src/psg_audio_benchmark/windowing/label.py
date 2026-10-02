"""Pure research-label assignment for Stage 4 (prompt section 3, case 6).

Assigns the mutually-exclusive label status to each window:

* ``excluded``  -- incomplete core coverage OR awake overlap. ``binary_event_label``
  is empty (NULL). An excluded window is NEVER recorded as negative.
* ``positive``  -- core-complete AND awake-qualified window with >= 1 retained
  (linked) event start. ``binary_event_label = 1``.
* ``negative``  -- core-complete AND awake-qualified window with 0 retained event
  starts. ``binary_event_label = 0``.

Eligibility (core-complete AND awake-qualified) is decided first; the retained
event counts come from :mod:`event_link`. These functions are pure and tested.
"""

from __future__ import annotations

from typing import Optional

from .schema import (
    BINARY_NEGATIVE,
    BINARY_POSITIVE,
    EXC_AWAKE_OVERLAP,
    EXC_INCOMPLETE_CORE,
    LABEL_EXCLUDED,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
)


def is_excluded(
    core_coverage_complete: bool,
    awake_overlap_fraction: float,
    max_awake_overlap_fraction: float,
) -> tuple:
    """Return ``(excluded: bool, reason: str)`` for a window's coverage/awake state.

    Core-coverage incompleteness takes precedence over awake overlap (a window
    outside the core domain is excluded for coverage regardless of awake state).
    """
    if not core_coverage_complete:
        return True, EXC_INCOMPLETE_CORE
    if awake_overlap_fraction > max_awake_overlap_fraction:
        return True, EXC_AWAKE_OVERLAP
    return False, ""


def assign_label(
    excluded: bool,
    n_retained_events: int,
) -> tuple:
    """Return ``(label_status, binary_event_label)`` for an eligible/excluded window.

    ``binary_event_label`` is ``None`` for excluded windows -- it is never 0 for
    an excluded window (boundary case 6: excluded never becomes negative).
    """
    if excluded:
        return LABEL_EXCLUDED, None
    if n_retained_events >= 1:
        return LABEL_POSITIVE, BINARY_POSITIVE
    return LABEL_NEGATIVE, BINARY_NEGATIVE


def binary_value(label_status: str) -> Optional[int]:
    """Map a label status to its binary value (None for excluded)."""
    if label_status == LABEL_POSITIVE:
        return BINARY_POSITIVE
    if label_status == LABEL_NEGATIVE:
        return BINARY_NEGATIVE
    return None


__all__ = [
    "is_excluded",
    "assign_label",
    "binary_value",
]
