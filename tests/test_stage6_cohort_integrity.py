"""(4) Cohort integrity: stable-key join correctness + no relabelling.

* Stage-4 ``excluded`` windows never enter the core cohort (and are not faked as
  negatives);
* every core member has ``label_status`` in {positive,negative}, a binary label
  in {0,1} and ``core_hr_spo2_available == True``;
* no duplicate ``window_id`` in the membership;
* Stage-4 vs Stage-5 time bounds agree (a divergence is a leakage signal);
* a candidate window with insufficient HR quality is excluded, not relabelled;
* binary labels are never rewritten (core label == Stage-4 label per window).
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import build_stage6_world, make_isolated_cfg, run_stage6, splits_out


def _world_windows():
    """8-patient world with: an excluded window (A-00002), an HR-insufficient
    positive candidate (B-00000), an all-negative patient (E), and an airflow
    subcohort (F/G/H)."""
    return [
        {"patient_id": "A", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "A", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "A", "window_index": 2, "label_status": "excluded", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "B", "window_index": 0, "label_status": "positive", "hr_available": False, "spo2_available": True,  "airflow_available": False},  # candidate, not core
        {"patient_id": "B", "window_index": 1, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "B", "window_index": 2, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "B", "window_index": 3, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "C", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "C", "window_index": 1, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "C", "window_index": 2, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "C", "window_index": 3, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "D", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "D", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "D", "window_index": 2, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "D", "window_index": 3, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "E", "window_index": 0, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "E", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "E", "window_index": 2, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "E", "window_index": 3, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": False},
        {"patient_id": "F", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "F", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "G", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "G", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "H", "window_index": 0, "label_status": "positive", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
        {"patient_id": "H", "window_index": 1, "label_status": "negative", "hr_available": True,  "spo2_available": True,  "airflow_available": True},
    ]


def _run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=_world_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return cfg, ids


def test_stage4_excluded_absent_from_core(tmp_path):
    _cfg, _ids = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    excl = pd.read_csv(splits_out(tmp_path) / "cohort_exclusions.csv")
    core_wids = set(core["window_id"])
    # the Stage-4 excluded window A-00002 is not a core member...
    assert "A-00002" not in core_wids
    # ...and every Stage-4 excluded window is absent from core
    stage4_excl = set(excl.loc[excl.scope == "stage4_window_excluded", "window_id"])
    assert stage4_excl.isdisjoint(core_wids)
    # excluded windows are recorded with the stage4 scope, never as negatives
    assert "A-00002" in stage4_excl


def test_core_members_have_valid_label_and_core_availability(tmp_path):
    _cfg, _ids = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    assert set(core["binary_event_label"].unique()).issubset({0, 1})
    assert set(core["label_status"].unique()).issubset({"positive", "negative"})
    assert bool(core["core_hr_spo2_available"].all())
    # no duplicate window_id
    assert int(core["window_id"].duplicated().sum()) == 0


def test_hr_insufficient_candidate_excluded_not_relabelled(tmp_path):
    """B-00000 is a positive candidate with hr_available=False -> not core, and
    recorded as an HR-quality exclusion (never turned into a negative)."""
    _cfg, _ids = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    excl = pd.read_csv(splits_out(tmp_path) / "cohort_exclusions.csv")
    assert "B-00000" not in set(core["window_id"])
    hr_excl = excl[(excl.window_id == "B-00000") & (excl.scope == "hr_feature_quality_insufficient")]
    assert len(hr_excl) == 1


def test_time_bounds_match_stage4_and_labels_not_rewritten(tmp_path):
    _cfg, ids = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    s4 = pd.read_parquet(
        _cfg.path("data_manifests") / "runs" / ids["stage4_run_id"] / "csv_window_index.parquet"
    )
    s4["patient_id"] = s4["patient_id"].astype(str)
    s4["window_id"] = s4["patient_id"] + "-" + s4["window_index"].astype(int).map(lambda i: f"{i:05d}")
    m = core.merge(s4[["window_id", "binary_event_label", "label_status",
                       "start_relative_to_record_start", "end_relative_to_record_start"]],
                   on="window_id", suffixes=("", "_s4"))
    # labels never rewritten
    assert int((m["binary_event_label"] != m["binary_event_label_s4"]).sum()) == 0
    # time bounds agree to tolerance
    assert ((m["window_start_relative_to_record_start"] - m["start_relative_to_record_start"]).abs() < 1e-6).all()
    assert ((m["window_end_relative_to_record_start"] - m["end_relative_to_record_start"]).abs() < 1e-6).all()


def test_audio_features_present_false_in_membership(tmp_path):
    _cfg, _ids = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    assert "audio_features_present" in core.columns
    assert bool((core["audio_features_present"] == False).all())
