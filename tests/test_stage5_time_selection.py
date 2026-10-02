"""Stage 5 boundary case: half-open [start, end) selection by TIME, not row index.

The feature time contract (prompt) requires selecting samples whose Stage-3-clock
time ``t = relative_position_seconds + offset`` falls in a half-open window
``[start, end)`` -- a boundary sample belongs to exactly ONE window, and selection
is by the inherited Stage-3 clock (not the CSV row index).
"""

from __future__ import annotations

import math

import pandas as pd

from _stage5_fixtures import (
    _patient_spec,
    build_stage5_world,
    features_out,
    make_isolated_cfg,
    run_stage5,
)


def _hr_row(tmp_path, window_id):
    df = pd.read_parquet(features_out(tmp_path) / "hr_window_features.parquet")
    return df[df["window_id"] == window_id].iloc[0]


def test_half_open_boundary_sample_belongs_to_exactly_one_window(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    # 1 Hz HR at relative 0..5: values 10,20,30,40,50,60
    patients = [_patient_spec(patient_id="01", hr=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0])]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
        {"patient_id": "01", "window_index": 1, "start_rel": 3.0, "end_rel": 6.0, "label_status": "negative"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _runner, _opt = run_stage5(cfg, tmp_path, **ids)

    w0 = _hr_row(tmp_path, "01-00000")
    w1 = _hr_row(tmp_path, "01-00001")

    # Half-open: the sample at relative 3.0 (value 40) is EXCLUDED from [0,3)
    # and INCLUDED in [3,6). Means therefore exclude / include 40 respectively.
    assert w0["n_observed"] == 3 and w0["n_finite"] == 3
    assert math.isclose(w0["mean"], 20.0)          # mean(10,20,30)
    assert w1["n_observed"] == 3 and w1["n_finite"] == 3
    assert math.isclose(w1["mean"], 50.0)          # mean(40,50,60)
    assert w0["coverage_fraction"] == 1.0 and w0["quality_status"] == "available"


def test_selection_uses_inherited_stage3_clock_offset_not_row_index(tmp_path):
    """offset = 100 -> t = relative + 100; window [100,103) selects the FIRST
    three CSV rows (relative 0,1,2), proving time-based selection."""
    cfg = make_isolated_cfg(tmp_path)
    patients = [_patient_spec(patient_id="02", offset=100.0, hr=[1.0, 2.0, 3.0, 4.0, 5.0])]
    windows = [
        {"patient_id": "02", "window_index": 0, "start_rel": 100.0, "end_rel": 103.0, "label_status": "positive"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _runner, _opt = run_stage5(cfg, tmp_path, **ids)

    row = _hr_row(tmp_path, "02-00000")
    assert row["n_observed"] == 3                  # relative 0,1,2 (not rows 100..102)
    assert math.isclose(row["mean"], 2.0)          # mean(1,2,3)
    assert row["n_expected"] == 3
    assert row["quality_status"] == "available"


def test_window_outside_signal_coverage_yields_no_samples(tmp_path):
    """A window whose clock range has no samples is all-missing (null features),
    never zero-filled."""
    cfg = make_isolated_cfg(tmp_path)
    patients = [_patient_spec(patient_id="01", hr=[60.0] * 4)]   # relative 0..3
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 100.0, "end_rel": 103.0, "label_status": "negative"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _runner, _opt = run_stage5(cfg, tmp_path, **ids)

    row = _hr_row(tmp_path, "01-00000")
    assert row["n_observed"] == 0 and row["n_finite"] == 0
    assert row["quality_status"] == "all_missing"
    assert row["coverage_fraction"] == 0.0
    assert pd.isna(row["mean"])                    # NULL, not zero-filled
