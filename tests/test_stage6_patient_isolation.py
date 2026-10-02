"""(1)(2) Patient-level isolation: no window or patient leaks across folds.

* every window of a patient lands in exactly one outer test fold;
* per outer fold the train/test patient sets are disjoint;
* within each outer fold the inner-train / inner-val patient sets are disjoint;
* an outer-TEST patient for fold f never appears in ANY inner row for fold f
  (the load-bearing nested-CV leakage guard).
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6, splits_out

_RUN = "stage6-synth-20260808T000000Z-abcd1234"


def _run_world(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _runner, _o = run_stage6(cfg, tmp_path, ids=ids, run_id=_RUN)
    assert summary.overall_status == "PASS", summary.anomalies
    return cfg, ids, summary


def test_every_window_of_a_patient_in_one_outer_fold(tmp_path):
    _cfg, _ids, _summary = _run_world(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    core["patient_id"] = core["patient_id"].astype(str)

    m = core[["window_id", "patient_id"]].merge(
        outer[["patient_id", "outer_fold"]], on="patient_id"
    )
    # each window resolves to exactly one outer fold
    per_win = m.groupby("window_id")["outer_fold"].nunique()
    assert int((per_win > 1).sum()) == 0
    # and every membership patient has a fold assigned
    assert set(core["patient_id"]) == set(outer["patient_id"])


def test_outer_train_test_patient_sets_disjoint(tmp_path):
    _cfg, _ids, _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    for f in range(5):
        test_p = set(outer.loc[outer.outer_fold == f, "patient_id"])
        train_p = set(outer.loc[outer.outer_fold != f, "patient_id"])
        assert test_p.isdisjoint(train_p)
        assert test_p | train_p == set(outer["patient_id"])


def test_inner_train_val_patient_sets_disjoint(tmp_path):
    _cfg, _ids, _summary = _run_world(tmp_path)
    inner = pd.read_parquet(splits_out(tmp_path) / "inner_patient_folds_core.parquet")
    inner["patient_id"] = inner["patient_id"].astype(str)
    for f in range(5):
        sub = inner[inner.outer_fold == f]
        for i in range(4):
            val_p = set(sub.loc[sub.inner_validation_fold == i, "patient_id"])
            train_p = set(sub.loc[sub.inner_validation_fold != i, "patient_id"])
            assert val_p.isdisjoint(train_p)


def test_outer_test_patients_never_in_inner(tmp_path):
    """The load-bearing nested-CV guard: outer-test patients for fold f appear
    in NO inner row for that fold."""
    _cfg, _ids, _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    inner = pd.read_parquet(splits_out(tmp_path) / "inner_patient_folds_core.parquet")
    inner["patient_id"] = inner["patient_id"].astype(str)
    for f in range(5):
        outer_test_p = set(outer.loc[outer.outer_fold == f, "patient_id"])
        inner_p = set(inner.loc[inner.outer_fold == f, "patient_id"])
        assert outer_test_p.isdisjoint(inner_p)
        # inner universe for fold f == outer-TRAIN patients for fold f
        outer_train_p = set(outer.loc[outer.outer_fold != f, "patient_id"])
        assert inner_p == outer_train_p


def test_no_anomalies_reported(tmp_path):
    """A clean run records zero cohort/split anomalies (balance warnings allowed)."""
    _cfg, _ids, summary = _run_world(tmp_path)
    assert summary.anomalies == []
