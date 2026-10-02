"""Stage-3 boundary case 2: cross-midnight rollover + unified time axis.

Guarantees (prompt section 6.2):
  * the single rollover rule assigns the correct day offsets;
  * annotation values already cumulative normalize with seconds_of_day/day_offset;
  * a CSV absolute clock that crosses midnight ONCE verifies (NOT flagged
    non-monotonic), while a genuine small backward glitch IS flagged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psg_audio_benchmark.annotation_parser.signals import compute_signal_time_range
from psg_audio_benchmark.annotation_parser.time_axis import (
    SECONDS_PER_DAY,
    annotation_timepoint,
    assign_clock_day_offsets,
    clock_timepoint,
)
from _stage3_fixtures import write_cross_midnight_csv, write_monotonic_csv


def test_day_offset_rule_first_sample_before_record_start() -> None:
    # record_start at 23:59:00 (=86340); first CSV sample at 23:58:00 (< rs) -> offset 1
    offsets = assign_clock_day_offsets([86280.0, 86340.0, 40.0], record_start_sod=86340.0)
    assert offsets == [1, 1, 2]  # wrap at index 2 (40 < 86340)


def test_day_offset_rule_first_sample_at_or_after_record_start() -> None:
    offsets = assign_clock_day_offsets([86340.0, 86400 % 86400 + 60, 120.0], record_start_sod=86340.0)
    # first >= rs -> 0; subsequent wrap increments once
    assert offsets[0] == 0


def test_annotation_timepoint_cumulative_over_day() -> None:
    tp = annotation_timepoint(75454.0)
    assert tp.day_offset == 0
    assert tp.seconds_of_day == 75454.0
    assert tp.cumulative_seconds == 75454.0

    tp2 = annotation_timepoint(75454.0 + SECONDS_PER_DAY)
    assert tp2.day_offset == 1
    assert tp2.seconds_of_day == 75454.0
    assert tp2.cumulative_seconds == 75454.0 + SECONDS_PER_DAY
    # relative_to is purely cumulative subtraction
    assert tp2.relative_to(tp) == SECONDS_PER_DAY


def test_cross_midnight_csv_verifies_not_flagged(tmp_path: Path) -> None:
    """A single legitimate midnight wrap must verify, NOT be flagged non-monotonic."""
    p = tmp_path / "01_SpO2.csv"
    write_cross_midnight_csv(p, start_sod=86340.0, step=60.0, n=5)
    rs = clock_timepoint(86340.0, 0)
    rng, warns = compute_signal_time_range(
        p, run_id="r", patient_id="01", modality="spo2",
        source_relpath="raw/V5/Data/01/01_SpO2.csv",
        rs_point=rs, duration_tolerance_s=1.0,
    )
    assert rng is not None
    assert rng.cross_midnight is True
    assert rng.data_quality_status == "ok", warns
    assert rng.verification_status == "verified"
    assert all(w.warning_code != "non_monotonic_csv_absolute_time" for w in warns)
    # absolute vs relative durations agree (single rollover applied correctly)
    assert rng.absolute_duration_seconds == pytest.approx(rng.relative_duration_seconds)
    assert rng.alignment_error_seconds is not None
    assert rng.alignment_error_seconds < 1.0


def test_monotonic_same_day_csv_verifies(tmp_path: Path) -> None:
    p = tmp_path / "01_SpO2.csv"
    write_monotonic_csv(p, start_sod=75454.0, step=60.0, n=5)
    rs = clock_timepoint(75454.0, 0)
    rng, _ = compute_signal_time_range(
        p, run_id="r", patient_id="01", modality="spo2",
        source_relpath="raw/V5/Data/01/01_SpO2.csv",
        rs_point=rs, duration_tolerance_s=1.0,
    )
    assert rng is not None
    assert rng.cross_midnight is False
    assert rng.verification_status == "verified"


def test_genuine_clock_glitch_flagged_non_monotonic(tmp_path: Path) -> None:
    """A small backward jump that is NOT a midnight wrap must be flagged.

    The absolute clock has a 70 s back-jump (drop << half-day, so NOT a midnight
    wrap); the relative column is shaped so abs-duration == rel-duration, so the
    ONLY quality issue that fires is the non-monotonic flag.
    """
    p = tmp_path / "01_SpO2.csv"
    lines = ['relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),OSat ("%")']
    abs_times = [3600.0, 3660.0, 3720.0, 3650.0, 3780.0]  # back-jump at idx 3
    rel_times = [0.0, 60.0, 120.0, 110.0, 180.0]          # span 180s == abs span
    for at, rt in zip(abs_times, rel_times):
        h = int(at // 3600); m = int((at % 3600) // 60); s = at - h * 3600 - m * 60
        abst = f"{h:02d}:{m:02d}:{s:06.3f}"
        rh = int(rt // 3600); rm = int((rt % 3600) // 60); rs_ = rt - rh * 3600 - rm * 60
        lines.append(f"{rh:02d}:{rm:02d}:{rs_:06.3f},{abst},97")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rs = clock_timepoint(3600.0, 0)
    rng, warns = compute_signal_time_range(
        p, run_id="r", patient_id="01", modality="spo2",
        source_relpath="raw/V5/Data/01/01_SpO2.csv",
        rs_point=rs, duration_tolerance_s=1.0,
    )
    assert rng is not None
    assert rng.data_quality_status == "non_monotonic_absolute_time"
    assert any(w.warning_code == "non_monotonic_csv_absolute_time" for w in warns)
