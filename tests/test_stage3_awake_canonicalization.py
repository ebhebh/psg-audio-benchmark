"""Stage-3 boundary case 3: awake-interval raw -> canonical normalization.

Guarantees (prompt section 6.3):
  * exact-duplicate intervals collapse with merged source-index provenance;
  * overlapping intervals merge;
  * adjacent (touching) intervals merge;
  * a cross-midnight interval is flagged;
  * non-positive-duration intervals are dropped (warned, never crash);
  * the raw table still keeps every interval in original order.
"""

from __future__ import annotations

from psg_audio_benchmark.annotation_parser.awake import canonicalize_awake
from psg_audio_benchmark.annotation_parser.time_axis import annotation_timepoint

RS = annotation_timepoint(75454.0)


def _canon(raw):
    return canonicalize_awake(
        run_id="r",
        patient_id="01",
        source_annotation_relpath="raw/V5/Data/01/01_annotation.json",
        awake_raw=raw,
        rs_point=RS,
    )


def test_exact_duplicates_collapsed_with_provenance() -> None:
    raw = [(0, 75500.0, 75600.0), (1, 75500.0, 75600.0), (2, 75700.0, 75800.0)]
    res = _canon(raw)
    assert res.stats["raw_total"] == 3
    assert res.stats["canonical_total"] == 2
    assert res.stats["exact_duplicates_removed"] == 1
    by_start = {c.start_relative_to_record_start: c for c in res.canonical_rows}
    merged = by_start[75500.0 - 75454.0]
    assert merged.source_raw_indices == "0;1"
    assert merged.n_source_intervals == 2


def test_overlapping_intervals_merge() -> None:
    raw = [(0, 75500.0, 75600.0), (1, 75550.0, 75700.0)]
    res = _canon(raw)
    assert res.stats["canonical_total"] == 1
    assert res.stats["overlaps_or_adjacent_merged"] == 1
    c = res.canonical_rows[0]
    assert c.start_relative_to_record_start == 75500.0 - 75454.0
    assert c.end_relative_to_record_start == 75700.0 - 75454.0
    # provenance from both source intervals
    assert set(c.source_raw_indices.split(";")) == {"0", "1"}


def test_adjacent_touching_intervals_merge() -> None:
    # [75500,75600] touches [75600,75700] at the boundary -> merge
    raw = [(0, 75500.0, 75600.0), (1, 75600.0, 75700.0)]
    res = _canon(raw)
    assert res.stats["canonical_total"] == 1
    assert res.stats["overlaps_or_adjacent_merged"] == 1


def test_cross_midnight_interval_flagged() -> None:
    # interval from 23:59:00 (86340) to 00:05:00 next day (86340 + 360 = 86700)
    raw = [(0, 86340.0, 86700.0)]
    res = _canon(raw)
    assert res.stats["cross_midnight_intervals"] == 1
    assert res.canonical_rows[0].cross_midnight is True


def test_nonpositive_duration_dropped_not_crash() -> None:
    raw = [(0, 75500.0, 75500.0), (1, 75600.0, 75500.0), (2, 75700.0, 75800.0)]
    res = _canon(raw)
    assert res.stats["nonpositive_dropped"] == 2
    assert res.stats["canonical_total"] == 1
    assert any(w.warning_code == "awake_nonpositive_duration" for w in res.warnings)


def test_raw_table_preserves_original_order() -> None:
    raw = [(0, 75500.0, 75600.0), (1, 75400.0, 75450.0)]  # out of start order
    res = _canon(raw)
    assert [r.raw_order_index for r in res.raw_rows] == [0, 1]
    # canonical is sorted by start regardless of raw order
    assert res.canonical_rows[0].start_relative_to_record_start < \
           res.canonical_rows[-1].start_relative_to_record_start
