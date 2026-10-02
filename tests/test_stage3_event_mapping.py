"""Stage-3 boundary case 5: conservative event-type mapping.

Guarantees (prompt section 6.5):
  * ``hypo`` -> ``hypopnea`` and ``osa`` -> ``apnea_unspecified``;
  * both are scored respiratory events (binary candidate True);
  * ``osa`` is NOT mapped to ``obstructive_apnea``;
  * no central/mixed/obstructive subtype is generated from observed data;
  * unknown raw types -> ``unknown_type`` (rejected), never a subtype;
  * the versioned config validates observed == mapped keys.
"""

from __future__ import annotations

import pytest

from psg_audio_benchmark.annotation_parser.event_mapping import (
    EventTypeMapping,
    load_event_type_mapping,
)
from psg_audio_benchmark.config import Config, default_project_root


@pytest.fixture(scope="module")
def mapping() -> EventTypeMapping:
    return load_event_type_mapping(Config(project_root=default_project_root()))


def test_hypo_maps_to_hypopnea(mapping: EventTypeMapping) -> None:
    label, status, scored = mapping.standardize("hypo")
    assert label == "hypopnea"
    assert status == "mapped"
    assert scored is True


def test_osa_maps_to_apnea_unspecified(mapping: EventTypeMapping) -> None:
    label, status, scored = mapping.standardize("osa")
    assert label == "apnea_unspecified"
    assert status == "mapped"
    assert scored is True


def test_osa_is_not_obstructive(mapping: EventTypeMapping) -> None:
    label, _, _ = mapping.standardize("osa")
    assert label != "obstructive_apnea"
    assert "central" not in label and "mixed" not in label


def test_reserved_subtypes_not_generated_from_data(mapping: EventTypeMapping) -> None:
    # The reserved labels are documented but never produced by standardize().
    produced = {mapping.standardize(t)[0] for t in mapping.observed_raw_types}
    for forbidden in ("central_apnea", "mixed_apnea", "obstructive_apnea"):
        assert forbidden not in produced, f"{forbidden} must not be generated from observed types"


def test_unknown_type_is_unknown(mapping: EventTypeMapping) -> None:
    label, status, scored = mapping.standardize("central")
    assert label == "unknown"
    assert status == "unknown_type"
    assert scored is False
    label2, _, _ = mapping.standardize("ca")
    assert label2 == "unknown"


def test_mapping_versioned_and_consistent(mapping: EventTypeMapping) -> None:
    assert mapping.mapping_version
    # observed set must equal the mapped keys (config self-consistency)
    assert set(mapping.observed_raw_types) == set(mapping.mapped)
    assert set(mapping.binary_applies_to).issubset(set(mapping.observed_raw_types))
    assert mapping.binary_field == "is_any_scored_respiratory_event"
