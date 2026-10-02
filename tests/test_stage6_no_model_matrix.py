"""(7) No model matrix: Stage 6 emits only membership / assignment / audit tables.

* none of the forbidden feature-value columns appear in ANY Stage-6 output;
* ``binary_event_label`` appears ONLY in the two cohort membership tables;
* the resolved summary carries all the ``no_*`` downstream-deferral flags True.
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6, splits_out
from psg_audio_benchmark.evaluation.schema import ALLOWED_LABEL_COLUMN, STAGE6_FORBIDDEN_COLUMNS

_MEMBERSHIP_FILES = {"core_cohort_window_membership.parquet", "airflow_cohort_window_membership.parquet"}


def _run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return summary


def test_no_forbidden_columns_in_any_output(tmp_path):
    _summary = _run(tmp_path)
    offenders = []
    for p in splits_out(tmp_path).iterdir():
        if p.suffix == ".parquet":
            cols = set(pd.read_parquet(p).columns)
        elif p.suffix == ".csv":
            cols = set(pd.read_csv(p, nrows=0).columns)
        else:
            continue
        bad = cols & set(STAGE6_FORBIDDEN_COLUMNS)
        if bad:
            offenders.append((p.name, sorted(bad)))
    assert offenders == [], offenders


def test_binary_label_only_in_membership_tables(tmp_path):
    _summary = _run(tmp_path)
    for p in splits_out(tmp_path).iterdir():
        if p.suffix == ".parquet":
            cols = set(pd.read_parquet(p).columns)
        elif p.suffix == ".csv":
            cols = set(pd.read_csv(p, nrows=0).columns)
        else:
            continue
        if ALLOWED_LABEL_COLUMN in cols:
            assert p.name in _MEMBERSHIP_FILES, f"{p.name} must not carry {ALLOWED_LABEL_COLUMN}"
    # and the membership tables DO carry it
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    assert ALLOWED_LABEL_COLUMN in core.columns


def test_no_downstream_fitting_flags_true(tmp_path):
    summary = _run(tmp_path)
    for flag in (
        "no_model_matrix", "no_imputation", "no_normalization", "no_feature_selection",
        "no_class_resampling", "no_threshold_optimization", "no_performance_metrics",
    ):
        assert getattr(summary, flag) is True
