"""Stage 5 boundary cases: excluded-window handling and window-key integrity.

(5) Stage-4 EXCLUDED windows are NOT feature candidates: they never appear in a
    feature table and are NEVER faked as negative. Each is logged once in
    feature_exclusions.csv with scope=stage4_window_not_candidate.
    A candidate window whose core HR/SpO2 quality is insufficient is logged with
    scope=core_hr_spo2_feature_quality (also not faked as negative).
(8) Exactly one feature row per (modality, window_id); no patient cross-
    contamination -- each patient's features reflect only its own samples.
"""

from __future__ import annotations

import math

import pandas as pd

from _stage5_fixtures import (
    build_stage5_world,
    features_out,
    make_isolated_cfg,
    run_stage5,
)


def test_excluded_windows_are_not_feature_candidates_and_not_faked_negative(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    patients = [{"patient_id": "01", "hr": [60.0] * 10, "spo2": [97.0] * 10}]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
        {"patient_id": "01", "window_index": 1, "start_rel": 3.0, "end_rel": 6.0, "label_status": "negative"},
        {"patient_id": "01", "window_index": 2, "start_rel": 6.0, "end_rel": 9.0,
         "label_status": "excluded", "exclusion_reason": "incomplete_core_coverage"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    hr = pd.read_parquet(features_out(tmp_path) / "hr_window_features.parquet")
    excl = pd.read_csv(features_out(tmp_path) / "feature_exclusions.csv")

    # Feature tables hold ONLY the two candidate windows (positive+negative).
    assert set(hr["window_id"]) == {"01-00000", "01-00001"}
    assert "01-00002" not in set(hr["window_id"])    # excluded -> not a feature row

    # The excluded window is logged once, as not-a-candidate (not as negative).
    s4_excl = excl[excl["scope"] == "stage4_window_not_candidate"]
    assert set(s4_excl["window_id"]) == {"01-00002"}
    assert (s4_excl["reason"].iloc[0] == "incomplete_core_coverage")


def test_core_quality_exclusion_is_logged_not_faked_negative(tmp_path):
    """A candidate window whose HR is low_coverage is excluded from the core
    feature set and logged with scope=core_hr_spo2_feature_quality."""
    cfg = make_isolated_cfg(tmp_path)
    # HR heavily missing so [0,5) -> 1/5 = 0.2 coverage -> low_coverage (core fails)
    hr = [60.0, None, None, None, None, 60.0, 60.0, 60.0, 60.0, 60.0]
    patients = [{"patient_id": "01", "hr": hr, "spo2": [97.0] * 10}]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 5.0, "label_status": "positive"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    excl = pd.read_csv(features_out(tmp_path) / "feature_exclusions.csv")
    av = pd.read_parquet(features_out(tmp_path) / "physiology_feature_availability.parquet")
    core_excl = excl[excl["scope"] == "core_hr_spo2_feature_quality"]
    assert "01-00000" in set(core_excl["window_id"])
    assert bool(av[av["window_id"] == "01-00000"].iloc[0]["core_hr_spo2_available"]) is False


def test_no_duplicate_window_key_and_no_cross_patient_contamination(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    patients = [
        {"patient_id": "01", "hr": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "spo2": [97.0] * 6},
        {"patient_id": "02", "hr": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0], "spo2": [97.0] * 6},
    ]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
        {"patient_id": "02", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    hr = pd.read_parquet(features_out(tmp_path) / "hr_window_features.parquet")
    # Unique window_id per modality; no duplicates.
    assert hr.duplicated("window_id").sum() == 0
    assert set(hr["window_id"]) == {"01-00000", "02-00000"}
    # Each patient's mean reflects its OWN samples only (no cross-contamination).
    m01 = hr[hr["window_id"] == "01-00000"].iloc[0]["mean"]
    m02 = hr[hr["window_id"] == "02-00000"].iloc[0]["mean"]
    assert math.isclose(m01, 20.0)     # mean(10,20,30)
    assert math.isclose(m02, 200.0)    # mean(100,200,300)
