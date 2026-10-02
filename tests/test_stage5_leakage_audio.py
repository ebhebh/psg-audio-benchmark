"""Stage 5 boundary cases: label-permutation invariance (anti-leakage) and the
hard audio block.

(2) Features must be invariant under label permutation: the feature functions
    take ONLY (sample times, sample values, fs, config); label/annotation/sleep
    columns are on a deny-list and never become features. Permuting a window's
    label must leave its features unchanged, and no feature table may carry a
    label/annotation column.
(7) audio_features_present is hard False everywhere; the CSV reader refuses any
    ``*.wav`` path with AudioAccessForbidden; the resolved config pins the block.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from _stage5_fixtures import (
    build_stage5_world,
    features_out,
    make_isolated_cfg,
    run_stage5,
)
from psg_audio_benchmark.feature_extraction.csv_reader import (
    AudioAccessForbidden,
    read_modality_csv,
)
from psg_audio_benchmark.feature_extraction.schema import FORBIDDEN_FEATURE_INPUTS

_FORBIDDEN_COLS = {
    "binary_event_label", "label_status", "exclusion_reason", "n_linked_retained_events",
    "event_type_standardized", "awake_overlap_fraction",
    "sleep_structure_observational_status", "annotation", "any_audio_field",
}


def test_forbidden_inputs_denylist_covers_label_annotation_sleep_audio():
    # The hard deny-list must include the label/annotation/sleep/audio names.
    for required in (
        "binary_event_label", "label_status", "annotation",
        "awake_overlap_fraction", "sleep_structure_observational_status",
    ):
        assert required in FORBIDDEN_FEATURE_INPUTS


def test_label_permutation_leaves_features_unchanged(tmp_path):
    """Two windows over the SAME signal + SAME [start,end) but different labels
    produce IDENTICAL HR features (label cannot enter feature computation)."""
    cfg = make_isolated_cfg(tmp_path)
    patients = [{"patient_id": "01", "hr": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "spo2": [97.0] * 6}]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
        {"patient_id": "01", "window_index": 1, "start_rel": 0.0, "end_rel": 3.0, "label_status": "negative"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, _runner, _opt = run_stage5(cfg, tmp_path, **ids)

    hr = pd.read_parquet(features_out(tmp_path) / "hr_window_features.parquet")
    a = hr[hr["window_id"] == "01-00000"].iloc[0]
    b = hr[hr["window_id"] == "01-00001"].iloc[0]
    for col in ("mean", "median", "std_ddof1", "min", "max", "range", "iqr",
                "slope_per_second", "n_observed", "n_finite", "coverage_fraction"):
        assert a[col] == b[col], f"feature {col} differs under label permutation"
    # ... and no label/annotation column leaked into any feature table.
    for name in ("hr_window_features.parquet", "spo2_window_features.parquet",
                 "airflow_window_features.parquet"):
        df = pd.read_parquet(features_out(tmp_path) / name)
        assert not (_FORBIDDEN_COLS & set(df.columns)), f"forbidden col in {name}"


def test_feature_row_schema_has_no_label_field():
    from psg_audio_benchmark.feature_extraction.schema import FeatureRow
    fields = {f.name for f in __import__("dataclasses").fields(FeatureRow)}
    assert not (_FORBIDDEN_COLS & fields)


def test_wav_path_is_refused_by_csv_reader(tmp_path):
    wav = tmp_path / "01_phone.wav"
    wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    with pytest.raises(AudioAccessForbidden):
        read_modality_csv(wav, patient_id="01", modality="heart_rate",
                          source_relpath="x/01_phone.wav", offset=0.0)


def test_audio_features_present_false_everywhere_and_blocked_in_config(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    patients = [{"patient_id": "01", "hr": [60.0] * 6, "spo2": [97.0] * 6}]
    windows = [
        {"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"},
    ]
    ids = build_stage5_world(cfg, patients=patients, windows=windows)
    summary, runner, _opt = run_stage5(cfg, tmp_path, **ids)

    fout = features_out(tmp_path)
    av = pd.read_parquet(fout / "physiology_feature_availability.parquet")
    for name in ("hr_window_features.parquet", "spo2_window_features.parquet",
                 "airflow_window_features.parquet"):
        df = pd.read_parquet(fout / name)
        assert set(df["audio_features_present"].unique()) == {False}
    assert set(av["audio_features_present"].unique()) == {False}
    # resolved config pins the block.
    cfg_obj = runner.feature_config
    assert cfg_obj.audio_features_present is False
    assert cfg_obj.read_wav_forbidden is True
