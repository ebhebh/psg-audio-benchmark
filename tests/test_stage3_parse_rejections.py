"""Stage-3 boundary case 4: malformed events are REJECTED, never crash the parser.

Guarantees (prompt section 6.4):
  * negative event duration -> rejected;
  * non-numeric start -> rejected;
  * unknown event type -> rejected (never a fabricated subtype);
  * missing event_duration -> rejected;
  * event not a dict / record_start unavailable -> rejected;
  * the public API never raises on any malformed input.
"""

from __future__ import annotations

import json
from pathlib import Path

from psg_audio_benchmark.annotation_parser.event_mapping import EventTypeMapping
from psg_audio_benchmark.annotation_parser.parse import parse_annotation_file

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

RS = 75454.0


def _write(tmp_path: Path, events: list, *, record_start: float = RS) -> Path:
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"record_start": record_start, "awake_intervals": [], "events": events}),
                 encoding="utf-8")
    return p


def _parse(p: Path):
    return parse_annotation_file(
        p, run_id="r", patient_id="01",
        source_annotation_relpath="raw/V5/Data/01/01_annotation.json",
        mapping=_MAPPING,
    )


def test_negative_duration_rejected(tmp_path: Path) -> None:
    res = _parse(_write(tmp_path, [
        {"event_type": "hypo", "evnet_start": 76000.0, "event_duration": -5.0}
    ]))
    assert res.parsed_events == []
    assert "negative_event_duration" in [r.reason for r in res.rejections]


def test_non_numeric_start_rejected(tmp_path: Path) -> None:
    res = _parse(_write(tmp_path, [
        {"event_type": "hypo", "evnet_start": "abc", "event_duration": 10.0}
    ]))
    assert res.parsed_events == []
    reasons = [r.reason for r in res.rejections]
    assert any(r.startswith("evnet_start_not_numeric") or r == "evnet_start_not_numeric" for r in reasons)


def test_unknown_event_type_rejected(tmp_path: Path) -> None:
    res = _parse(_write(tmp_path, [
        {"event_type": "central", "evnet_start": 76000.0, "event_duration": 10.0}
    ]))
    assert res.parsed_events == []
    reasons = [r.reason for r in res.rejections]
    assert any(r.startswith("unknown_event_type") for r in reasons)
    # and no fabricated subtype label leaks into any standardized output
    assert all("central_apnea" != r.reason for r in res.rejections)


def test_missing_duration_rejected(tmp_path: Path) -> None:
    res = _parse(_write(tmp_path, [
        {"event_type": "hypo", "evnet_start": 76000.0}
    ]))
    assert res.parsed_events == []
    assert "missing_event_duration" in [r.reason for r in res.rejections]


def test_event_not_dict_rejected(tmp_path: Path) -> None:
    res = _parse(_write(tmp_path, ["not-a-dict", 42, None]))
    assert res.parsed_events == []
    assert [r.reason for r in res.rejections].count("event_not_object") == 3


def test_record_start_unavailable_rejects_events(tmp_path: Path) -> None:
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({"awake_intervals": [], "events": [
        {"event_type": "hypo", "evnet_start": 76000.0, "event_duration": 10.0}
    ]}), encoding="utf-8")
    res = _parse(p)
    assert res.parsed_events == []
    assert "record_start_unavailable_for_relative_time" in [r.reason for r in res.rejections]


def test_parser_never_raises_on_garbage(tmp_path: Path) -> None:
    # Unparseable JSON, non-object top level, and a broken binary file.
    for name, content in (("bad.json", "{not json"), ("arr.json", "[1,2,3]"),
                          ("bin.json", b"\x00\x01\x02\x03")):
        p = tmp_path / name
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8")
        res = _parse(p)  # must not raise
        assert res.parsed_events == []
        assert res.file_anomalies  # something recorded
