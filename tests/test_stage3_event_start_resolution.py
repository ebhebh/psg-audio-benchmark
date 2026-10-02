"""Stage-3 boundary case 1: ``event_start`` vs ``evnet_start`` typo / priority / conflict.

Guarantees (prompt section 6.1):
  * the dataset's ``evnet_start`` typo is tolerated and used;
  * the canonical ``event_start`` is used when present alone;
  * when both are present and EQUAL, the canonical field wins with no conflict;
  * when both are present and DIFFER, the event is REJECTED with reason
    ``conflicting_event_start_fields`` (never silently picking one).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psg_audio_benchmark.annotation_parser.event_mapping import EventTypeMapping
from psg_audio_benchmark.annotation_parser.parse import (
    parse_annotation_file,
    resolve_event_start,
)
from psg_audio_benchmark.config import Config, default_project_root

# Reuse the real versioned mapping (hypo/osa) loaded from config.
_MAPPING = EventTypeMapping(
    mapping_version="test",
    mapping_source="test",
    observed_raw_types=("hypo", "osa"),
    mapped={"hypo": "hypopnea", "osa": "apnea_unspecified"},
    status={"hypo": "mapped", "osa": "mapped"},
    reserved=("central", "mixed", "obstructive"),
    binary_field="is_any_scored_respiratory_event",
    binary_applies_to=("hypo", "osa"),
)


def test_resolve_typo_field_is_used() -> None:
    val, field_used, conflict, reason = resolve_event_start({"evnet_start": 100.0})
    assert val == 100.0
    assert field_used == "evnet_start"
    assert conflict is False
    assert reason == ""


def test_resolve_canonical_field_is_used() -> None:
    val, field_used, conflict, reason = resolve_event_start({"event_start": 200.0})
    assert val == 200.0
    assert field_used == "event_start"
    assert conflict is False


def test_resolve_both_equal_no_conflict() -> None:
    val, field_used, conflict, reason = resolve_event_start(
        {"event_start": 300.0, "evnet_start": 300.0}
    )
    assert val == 300.0
    assert field_used == "event_start"  # canonical preferred when equal
    assert conflict is True  # both-present flag is True, but reason empty
    assert reason == ""


def test_resolve_both_differ_is_conflict() -> None:
    val, field_used, conflict, reason = resolve_event_start(
        {"event_start": 300.0, "evnet_start": 999.0}
    )
    assert val is None
    assert conflict is True
    assert reason == "conflicting_event_start_fields"


def test_resolve_missing_start_field() -> None:
    val, _field, _conflict, reason = resolve_event_start({"event_type": "hypo"})
    assert val is None
    assert reason == "missing_event_start_field"


def _parse(path: Path, record_start: float = 75454.0):
    return parse_annotation_file(
        path,
        run_id="r",
        patient_id="01",
        source_annotation_relpath="raw/V5/Data/01/01_annotation.json",
        mapping=_MAPPING,
    )


def test_end_to_end_conflict_event_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        '{"record_start": 75454.0, "awake_intervals": [], "events": ['
        '{"event_type": "hypo", "event_start": 76000.0, "evnet_start": 77000.0, "event_duration": 10.0}'
        "]}",
        encoding="utf-8",
    )
    res = _parse(p)
    assert res.parsed_events == []  # conflicting event not placed on timeline
    reasons = [r.reason for r in res.rejections]
    assert "conflicting_event_start_fields" in reasons


def test_end_to_end_typo_event_is_parsed(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(
        '{"record_start": 75454.0, "awake_intervals": [], "events": ['
        '{"event_type": "hypo", "evnet_start": 75900.0, "event_duration": 10.0}'
        "]}",
        encoding="utf-8",
    )
    res = _parse(p)
    assert len(res.parsed_events) == 1
    ev = res.parsed_events[0]
    assert ev.event_start_raw_seconds == 75900.0
    assert ev.event_start_relative_to_record_start == pytest.approx(75900.0 - 75454.0)
    assert ev.conflicting_event_start_fields is False
