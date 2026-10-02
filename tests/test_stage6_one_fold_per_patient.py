"""(2) Each patient is assigned to exactly one fold at each nesting level.

* every core patient appears in exactly ONE outer test fold (count == 1);
* every outer-train patient appears in exactly ONE inner validation fold within
  its outer fold (count == 1);
* the outer-fold partition covers all core patients exactly once.
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6, splits_out


def _run_world(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return summary


def test_each_patient_in_exactly_one_outer_fold(tmp_path):
    _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    counts = outer["patient_id"].value_counts()
    assert len(counts) == 10                       # all 10 patients present
    assert int(counts.max()) == 1 and int(counts.min()) == 1  # exactly one row each
    # folds are the expected 0..4
    assert set(outer["outer_fold"]) == {0, 1, 2, 3, 4}


def test_each_outer_train_patient_in_exactly_one_inner_fold(tmp_path):
    _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    inner = pd.read_parquet(splits_out(tmp_path) / "inner_patient_folds_core.parquet")
    inner["patient_id"] = inner["patient_id"].astype(str)

    for f in range(5):
        sub = inner[inner.outer_fold == f]
        # each outer-train patient appears exactly once in this fold's inner table
        counts = sub["patient_id"].value_counts()
        assert int(counts.max()) == 1 and int(counts.min()) == 1
        # the inner patients for fold f are exactly the outer-TRAIN patients
        outer_train = set(outer.loc[outer.outer_fold != f, "patient_id"])
        assert set(sub["patient_id"]) == outer_train
        # inner validation folds are 0..3
        assert set(sub["inner_validation_fold"]) == {0, 1, 2, 3}


def test_outer_partition_covers_all_core_patients(tmp_path):
    _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    core["patient_id"] = core["patient_id"].astype(str)
    assert set(outer["patient_id"]) == set(core["patient_id"])
