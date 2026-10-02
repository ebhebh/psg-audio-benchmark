"""Pure event <-> window association for Stage 4 (prompt section 3/6).

Rules implemented here (all side-effect-free, all unit-tested):

* An event is associated to a window by its **start point** under the half-open
  ``[start, end)`` rule (boundary case 1: a point on a boundary belongs to the
  window that starts there, and to only one window).
* A cross-boundary event is associated only to its start window; its
  ``duration``/``end`` are carried on the link so it is never double-counted
  (boundary case 2).
* Every event is classified into exactly one destiny (boundary case 6) so that
  awake-overlap, out-of-domain and no-valid-window events are accounted
  separately and never hidden:
    - ``linked``               start in a label-eligible window (retained)
    - ``awake_overlap_event``  event itself overlaps awake (set aside)
    - ``event_start_outside_core_domain``  no candidate window contains the start
    - ``event_in_excluded_or_absent_window`` start landed in an excluded window

The window eligibility (core-complete AND awake-qualified) is computed BEFORE
events are linked, so there is no circular dependency with labeling.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .intervals import window_contains
from .schema import (
    EVT_AWAKE,
    EVT_LINKED,
    EVT_NO_VALID_WINDOW,
    EVT_OUT_OF_DOMAIN,
    EXC_INCOMPLETE_CORE,
    LINK_EVENT_OVERLAPS_AWAKE,
    LINK_IN_WINDOW_EXCLUDED_AWAKE,
    LINK_IN_WINDOW_EXCLUDED_COVERAGE,
    LINK_START_IN_LABEL_WINDOW,
    LINK_START_OUTSIDE_GRID,
    WindowEventLink,
)


@dataclass(frozen=True)
class WindowSlot:
    """Intermediate window descriptor used during event linking.

    ``eligible`` is True iff the window is core-coverage-complete AND
    awake-qualified (i.e. it will receive a positive/negative label).
    """

    window_index: int
    start: float
    end: float
    eligible: bool
    exclusion_reason: str


@dataclass(frozen=True)
class EventInput:
    """Minimal, side-effect-free event descriptor consumed by the linker."""

    source_event_index: int
    event_start_rel: float
    event_duration_seconds: float
    event_end_rel: float
    event_type_standardized: str
    overlaps_awake: bool


def find_containing_slot(
    event_start_rel: float, slots: Sequence[WindowSlot]
) -> Optional[int]:
    """Return the ``window_index`` of the slot whose ``[start, end)`` contains the
    event start, or ``None``.

    ``slots`` must be sorted ascending by ``start``. Uses bisect on the start
    array then verifies the half-open end, so it is correct even if the grid has
    gaps (a start inside a gap yields None).
    """
    if not slots:
        return None
    starts = [s.start for s in slots]
    i = bisect_right(starts, event_start_rel) - 1
    if i < 0:
        return None
    slot = slots[i]
    if window_contains(slot.start, slot.end, event_start_rel):
        return slot.window_index
    return None


def classify_event_destiny(
    event_overlaps_awake: bool,
    containing_window_index: Optional[int],
    slots_by_index: Dict[int, WindowSlot],
) -> Tuple[str, str]:
    """Classify one event into ``(destiny, link_reason)``.

    Order matters: an awake-overlapping event is set aside even if its start is
    in an eligible window (it creates no positive label and is never a negative).
    """
    if event_overlaps_awake:
        return EVT_AWAKE, LINK_EVENT_OVERLAPS_AWAKE
    if containing_window_index is None:
        return EVT_OUT_OF_DOMAIN, LINK_START_OUTSIDE_GRID
    slot = slots_by_index.get(containing_window_index)
    if slot is None:
        # start resolved to a window index that no longer exists (defensive)
        return EVT_OUT_OF_DOMAIN, LINK_START_OUTSIDE_GRID
    if not slot.eligible:
        if slot.exclusion_reason == EXC_INCOMPLETE_CORE:
            return EVT_NO_VALID_WINDOW, LINK_IN_WINDOW_EXCLUDED_COVERAGE
        return EVT_NO_VALID_WINDOW, LINK_IN_WINDOW_EXCLUDED_AWAKE
    return EVT_LINKED, LINK_START_IN_LABEL_WINDOW


def link_patient_events(
    *,
    run_id: str,
    patient_id: str,
    slots: Sequence[WindowSlot],
    events: Sequence[EventInput],
) -> Tuple[List[WindowEventLink], Dict[int, int]]:
    """Link all of one patient's events to its window slots.

    Returns ``(links, retained_counts)`` where ``retained_counts`` maps
    ``window_index -> number of retained (linked) event starts`` (only eligible
    windows appear, and only with a positive count). Every event produces
    exactly one link row, so unlinked events are visible, not hidden.
    """
    slots_sorted = sorted(slots, key=lambda s: s.start)
    slots_by_index: Dict[int, WindowSlot] = {s.window_index: s for s in slots}
    links: List[WindowEventLink] = []
    retained: Dict[int, int] = {}

    for ev in events:
        containing = find_containing_slot(ev.event_start_rel, slots_sorted)
        destiny, reason = classify_event_destiny(
            ev.overlaps_awake, containing, slots_by_index
        )
        links.append(
            WindowEventLink(
                run_id=run_id,
                patient_id=patient_id,
                window_index=containing if containing is not None else -1,
                source_event_index=ev.source_event_index,
                event_type_standardized=ev.event_type_standardized,
                event_start_relative_to_record_start=ev.event_start_rel,
                event_duration_seconds=ev.event_duration_seconds,
                event_end_relative_to_record_start=ev.event_end_rel,
                event_overlaps_awake_interval=bool(ev.overlaps_awake),
                event_destiny=destiny,
                link_reason=reason,
            )
        )
        if destiny == EVT_LINKED and containing is not None:
            retained[containing] = retained.get(containing, 0) + 1

    return links, retained


__all__ = [
    "WindowSlot",
    "EventInput",
    "find_containing_slot",
    "classify_event_destiny",
    "link_patient_events",
]
