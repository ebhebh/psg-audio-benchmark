"""Stage 5 boundary cases: the coverage gate (>= 0.80 available; below -> NULL +
quality code, never zero-filled) and the optional-airflow rule.

(3) At coverage exactly 0.80 a modality is AVAILABLE; below 0.80 it is
    low_coverage with every numeric feature NULL (no zero-fill / imputation).
    all_missing and insufficient_samples are exercised too.
(4) Airflow is OPTIONAL: a patient without it yields
    modality_not_available_for_patient for airflow only, and the core HR/SpO2
    availability is unaffected. When airflow IS present it gets RMS + zero
    crossing rate; HR/SpO2 never do.
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


def _hr(tmp_path):
    return pd.read_parquet(features_out(tmp_path) / "hr_window_features.parquet")


def _af(tmp_path):
    return pd.read_parquet(features_out(tmp_path) / "airflow_window_features.parquet")


def _av(tmp_path):
    return pd.read_parquet(features_out(tmp_path) / "physiology_feature_availability.parquet")


def test_coverage_boundary_0p80_available_below_low_coverage_null(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    # 1 Hz HR, 10 samples. Window [0,5) -> [60,_,70,80,90] = 4/5 = 0.80 (available).
    # Window [5,10) -> [60,_,_,80,90] = 3/5 = 0.60 (low_coverage -> null).
    hr = [60.0, None, 70.0, 80.0, 90.0, 60.0, None, None, 80.0, 90.0]
    patients = [{"patient_id": "01", "hr": hr, "spo2": [97.0] * 10}]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 5.0, "label_status": "positive"},
        {"patient_id": "01", "window_index": 1, "start_rel": 5.0, "end_rel": 10.0, "label_status": "negative"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    df = _hr(tmp_path)
    avail = df[df["window_id"] == "01-00000"].iloc[0]
    low = df[df["window_id"] == "01-00001"].iloc[0]

    # Exactly 0.80 -> available with real statistics.
    assert math.isclose(avail["coverage_fraction"], 0.80)
    assert avail["quality_status"] == "available"
    assert math.isclose(avail["mean"], 75.0)          # mean(60,70,80,90)
    assert avail["n_expected"] == 5 and avail["n_finite"] == 4

    # Below 0.80 -> low_coverage, ALL numeric features NULL (not zero-filled).
    assert low["quality_status"] == "low_coverage"
    assert math.isclose(low["coverage_fraction"], 0.60)
    for col in ("mean", "median", "std_ddof1", "min", "max", "range", "iqr", "slope_per_second"):
        assert pd.isna(low[col]), f"{col} must be NULL under low_coverage, got {low[col]}"


def test_all_missing_and_insufficient_samples_quality_codes(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    patients = [
        # all-missing window: every sample in range is "-"
        {"patient_id": "01", "hr": [None] * 6, "spo2": [97.0] * 6},
        # insufficient_samples: n_expected=1, n_finite=1 (< min_samples=2) but coverage 1.0
        {"patient_id": "02", "hr": [55.0], "spo2": [97.0]},
    ]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "negative"},
        {"patient_id": "02", "window_index": 0, "start_rel": 0.0, "end_rel": 1.0, "label_status": "negative"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    df = _hr(tmp_path)
    am = df[df["window_id"] == "01-00000"].iloc[0]
    ins = df[df["window_id"] == "02-00000"].iloc[0]
    assert am["quality_status"] == "all_missing" and pd.isna(am["mean"])
    assert ins["quality_status"] == "insufficient_samples" and pd.isna(ins["mean"])


def test_airflow_optional_absence_does_not_affect_core(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    # alternating-sign airflow at 2 Hz for patient 04; NO airflow for patient 05
    af = [(-1.0) ** i for i in range(12)]
    patients = [
        {"patient_id": "04", "hr": [60.0] * 6, "spo2": [97.0] * 6, "airflow": af},
        {"patient_id": "05", "hr": [60.0] * 6, "spo2": [97.0] * 6, "airflow": None},
    ]
    windows = [
        {"patient_id": "04", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
        {"patient_id": "05", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    af_df = _af(tmp_path)
    av = _av(tmp_path)
    present = af_df[af_df["window_id"] == "04-00000"].iloc[0]
    absent = af_df[af_df["window_id"] == "05-00000"].iloc[0]

    # Present: available with RMS + zero-crossing computed.
    assert present["quality_status"] == "available"
    assert present["n_expected"] == 6                              # 3 s * 2 Hz
    assert not pd.isna(present["rms"]) and present["zero_crossing_count"] == 5
    assert math.isclose(present["zero_crossing_rate"], 5.0 / 3.0)

    # Absent: modality_not_available_for_patient, NULL airflow features.
    assert absent["quality_status"] == "modality_not_available_for_patient"
    assert pd.isna(absent["rms"]) and pd.isna(absent["mean"])

    # Core HR+SpO2 still available for the no-airflow patient.
    core05 = av[av["window_id"] == "05-00000"].iloc[0]
    assert bool(core05["core_hr_spo2_available"]) is True
    assert bool(core05["airflow_modality_present_for_patient"]) is False
    assert bool(core05["airflow_available"]) is False


def test_hr_spo2_never_carry_airflow_extras(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    patients = [{"patient_id": "01", "hr": [60.0] * 6, "spo2": [97.0] * 6, "airflow": [0.1] * 12}]
    windows = [{"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"}]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    hr = _hr(tmp_path)
    sp = pd.read_parquet(features_out(tmp_path) / "spo2_window_features.parquet")
    # RMS / zero-crossing columns must NOT exist on HR/SpO2 tables.
    assert "rms" not in hr.columns and "zero_crossing_count" not in hr.columns
    assert "rms" not in sp.columns and "zero_crossing_count" not in sp.columns
