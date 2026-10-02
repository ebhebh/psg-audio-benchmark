"""Annotation structure-probe tests (prompt section 4.3).

JSON/TXT exploration is structure-only and PRELIMINARY: it must not produce
final standardized labels, must flag the ``evnet_start`` variant, and must
desensitise values.
"""

from __future__ import annotations

import json
from pathlib import Path

from psg_audio_benchmark.data_audit import annotation_probe

from _stage2_fixtures import write_annotation_json


def test_json_structure_and_variant(tmp_path: Path) -> None:
    p = write_annotation_json(tmp_path / "01_annotation.json")
    probe = annotation_probe.probe_json(p)
    assert probe.format == "json"
    assert probe.record_start_present is True
    assert probe.awake_intervals_count == 1
    assert probe.events_count == 2
    assert "event_type" in probe.event_field_set
    assert "evnet_start" in probe.event_field_set  # original (typo) spelling present
    # variant reported
    assert any("evnet_start" in v and "event_start" in v for v in probe.field_variants)
    assert probe.raw_event_type_freq == {"hypo": 1, "osa": 1}
    assert "preliminary_structure_audit" in probe.preliminary_note


def test_json_desensitises_values(tmp_path: Path) -> None:
    p = write_annotation_json(tmp_path / "02_annotation.json", record_start=75454.0)
    probe = annotation_probe.probe_json(p)
    skeleton = probe.structure_example_desensitized
    # type placeholders, never the raw float value echoed as-is in the skeleton
    assert "<float>" in skeleton or "..." in skeleton
    assert "75454.0" not in skeleton


def test_json_missing_events_reported_not_crashing(tmp_path: Path) -> None:
    p = tmp_path / "03_annotation.json"
    p.write_text(json.dumps({"record_start": 1.0, "awake_intervals": []}), encoding="utf-8")
    probe = annotation_probe.probe_json(p)
    assert probe.events_present is False
    assert probe.events_count == 0
    assert probe.raw_event_type_freq == {}


def test_malformed_json_recorded(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    probe = annotation_probe.probe_json(p)
    assert any("json_decode_error" in a for a in probe.anomalies)
    assert probe.events_count == 0


def test_txt_probe_structure(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("event_start: 10\nevent_type: hypo\n", encoding="utf-8")
    probe = annotation_probe.probe_annotation(p)
    assert probe.format == "txt"
    assert probe.txt_line_count == 2
    assert probe.txt_head_pattern


def test_probe_does_not_modify_raw(tmp_path: Path) -> None:
    p = write_annotation_json(tmp_path / "04_annotation.json")
    before = p.read_bytes()
    annotation_probe.probe_json(p)
    after = p.read_bytes()
    assert before == after  # read-only
