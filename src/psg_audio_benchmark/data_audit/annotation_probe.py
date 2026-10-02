"""Annotation structure exploration (Stage 2, prompt 4.3).

A *structure-only* probe of JSON / TXT annotation files. It records shapes,
keys, array sizes, event-object field sets and the **raw** event-type frequency
-- explicitly tagged ``preliminary_structure_audit``. It does NOT produce a
standardised event table, clinical conclusions, or a final parse rule; that is
a later stage's job.

Desensitisation: actual patient values are never echoed. Structure examples use
type placeholders (``<float>``, ``<str>``). Event-type *labels* (e.g.
``hypo``/``osa``) are categorical, not identifying, so their counts are kept.

Field variants are reported explicitly (prompt 4.3): the dataset's event object
spells the start time ``evnet_start`` -- a typo of ``event_start`` -- so the
probe flags whichever variant is present rather than assuming the canonical
name.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Canonical event-object field names expected by the literature.
_CANONICAL_EVENT_FIELDS = (
    "event_type",
    "event_start",
    "event_duration",
    "sleep_stage",
)

#: Known variant spellings seen in this dataset (typos the parser must tolerate).
_KNOWN_VARIANTS = {
    "evnet_start": "event_start",  # typo observed in the wild
}


@dataclass
class AnnotationProbe:
    """Structure summary for one annotation file."""

    format: str = ""  # 'json' | 'txt'
    encoding: str = "utf-8"
    top_level_keys: List[str] = field(default_factory=list)
    record_start_present: bool = False
    record_start_value_type: str = ""
    awake_intervals_present: bool = False
    awake_intervals_count: int = 0
    events_present: bool = False
    events_count: int = 0
    event_field_set: List[str] = field(default_factory=list)
    event_field_types: Dict[str, str] = field(default_factory=dict)
    field_variants: List[str] = field(default_factory=list)
    raw_event_type_freq: Dict[str, int] = field(default_factory=dict)
    structure_example_desensitized: str = ""
    txt_line_count: int = 0
    txt_head_pattern: str = ""
    txt_candidate_fields: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    preliminary_note: str = (
        "preliminary_structure_audit; NOT final labels or clinical conclusions"
    )

    def as_dict(self) -> dict:
        return {
            "annotation_format": self.format,
            "annotation_encoding": self.encoding,
            "annotation_top_level_keys": "; ".join(self.top_level_keys),
            "annotation_record_start_present": self.record_start_present,
            "annotation_record_start_value_type": self.record_start_value_type,
            "annotation_awake_intervals_present": self.awake_intervals_present,
            "annotation_awake_intervals_count": self.awake_intervals_count,
            "annotation_events_present": self.events_present,
            "annotation_events_count": self.events_count,
            "annotation_event_field_set": "; ".join(self.event_field_set),
            "annotation_event_field_types": "; ".join(
                f"{k}:{v}" for k, v in self.event_field_types.items()
            ),
            "annotation_field_variants": "; ".join(self.field_variants),
            "annotation_raw_event_type_freq": "; ".join(
                f"{k}={v}" for k, v in sorted(self.raw_event_type_freq.items())
            ),
            "annotation_structure_example_desensitized": self.structure_example_desensitized,
            "annotation_txt_line_count": self.txt_line_count,
            "annotation_txt_head_pattern": self.txt_head_pattern,
            "annotation_txt_candidate_fields": "; ".join(self.txt_candidate_fields),
            "annotation_anomalies": "; ".join(self.anomalies),
            "annotation_preliminary_note": self.preliminary_note,
        }


# ---------------------------------------------------------------------------
# JSON probe
# ---------------------------------------------------------------------------

def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _field_types_from_events(events: List[Any], cap: int = 500) -> Dict[str, str]:
    """Union of event-object fields -> a representative type, sampled."""
    types: Dict[str, str] = {}
    for ev in events[:cap]:
        if not isinstance(ev, dict):
            continue
        for k, v in ev.items():
            # Don't overwrite a known type with 'NoneType'.
            tn = _type_name(v)
            if k not in types or types[k] == "NoneType":
                types[k] = tn
    return types


def _desensitize_json_skeleton(obj: Any, depth: int = 0) -> str:
    """Render a shape-only skeleton (keys + type placeholders, no values)."""
    if depth > 3:
        return "..."
    if isinstance(obj, dict):
        parts = []
        for k, v in list(obj.items())[:8]:
            parts.append(f'"{k}": {_desensitize_json_skeleton(v, depth + 1)}')
        return "{" + ", ".join(parts) + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        return "[" + _desensitize_json_skeleton(obj[0], depth + 1) + ", ...]"
    return "<" + _type_name(obj) + ">"


def probe_json(path: Path) -> AnnotationProbe:
    probe = AnnotationProbe(format="json")
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="latin-1")
            probe.encoding = "latin-1"
        except OSError as exc:
            probe.anomalies.append(f"read_failed:{type(exc).__name__}:{exc}")
            return probe
    except OSError as exc:
        probe.anomalies.append(f"read_failed:{type(exc).__name__}:{exc}")
        return probe

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        probe.anomalies.append(f"json_decode_error:{exc}")
        return probe

    if not isinstance(data, dict):
        probe.anomalies.append("top_level_not_object")
        probe.structure_example_desensitized = _desensitize_json_skeleton(data)
        return probe

    probe.top_level_keys = list(data.keys())
    probe.structure_example_desensitized = _desensitize_json_skeleton(data)

    # record_start
    rs = data.get("record_start")
    probe.record_start_present = "record_start" in data
    probe.record_start_value_type = _type_name(rs) if "record_start" in data else ""

    # awake_intervals
    aw = data.get("awake_intervals")
    probe.awake_intervals_present = "awake_intervals" in data
    if isinstance(aw, list):
        probe.awake_intervals_count = len(aw)
    elif aw is not None:
        probe.anomalies.append("awake_intervals_not_list")

    # events
    evs = data.get("events")
    probe.events_present = "events" in data
    if isinstance(evs, list):
        probe.events_count = len(evs)
        types = _field_types_from_events(evs)
        probe.event_field_types = types
        probe.event_field_set = list(types.keys())
        # Variant detection: canonical vs typo spellings.
        present = set(types.keys())
        for canon in _CANONICAL_EVENT_FIELDS:
            if canon in present:
                continue
            for variant, maps_to in _KNOWN_VARIANTS.items():
                if maps_to == canon and variant in present:
                    probe.field_variants.append(
                        f"{variant!r} present instead of canonical {canon!r}"
                    )
        # Raw event-type frequency (preliminary structure audit only).
        type_counter: Counter = Counter()
        for ev in evs:
            if isinstance(ev, dict) and "event_type" in ev:
                type_counter[str(ev["event_type"])] += 1
        probe.raw_event_type_freq = dict(type_counter)
        if not type_counter and evs:
            probe.anomalies.append("events_present_but_no_event_type_key")
    elif evs is not None:
        probe.anomalies.append("events_not_list")

    return probe


# ---------------------------------------------------------------------------
# TXT probe
# ---------------------------------------------------------------------------

def probe_txt(path: Path) -> AnnotationProbe:
    probe = AnnotationProbe(format="txt")
    path = Path(path)
    head_lines: List[str] = []
    n_lines = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                n_lines += 1
                if len(head_lines) < 6:
                    head_lines.append(line.rstrip("\n"))
    except OSError as exc:
        probe.anomalies.append(f"read_failed:{type(exc).__name__}:{exc}")
        return probe

    probe.encoding = "utf-8"
    probe.txt_line_count = n_lines
    probe.txt_head_pattern = (
        " | ".join(f"<line{i+1}:{len(ln)}chars>" for i, ln in enumerate(head_lines))
    )
    # candidate fields: tokens that look like keys on lines containing ':' / '='.
    cand: List[str] = []
    for ln in head_lines:
        for tok in ln.replace("=", ":").split(":"):
            tok = tok.strip()
            if tok and 1 <= len(tok) <= 40 and tok.isascii():
                cand.append(tok)
    # de-dup, keep order
    seen = set()
    for c in cand:
        if c not in seen:
            seen.add(c)
            probe.txt_candidate_fields.append(c)
        if len(probe.txt_candidate_fields) >= 20:
            break
    return probe


def probe_annotation(path: Path) -> AnnotationProbe:
    """Dispatch to :func:`probe_json` / :func:`probe_txt` by extension."""
    ext = Path(path).suffix.lower()
    if ext == ".json":
        return probe_json(path)
    if ext in (".txt", ".text"):
        return probe_txt(path)
    p = AnnotationProbe()
    p.anomalies.append(f"unhandled_annotation_extension:{ext}")
    return p


__all__ = [
    "AnnotationProbe",
    "probe_json",
    "probe_txt",
    "probe_annotation",
]
