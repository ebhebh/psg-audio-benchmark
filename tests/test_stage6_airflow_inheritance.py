"""(3) The airflow subcohort inherits the core outer fold (no independent split).

* airflow patients are a strict subset of core patients;
* each airflow patient's outer fold equals its core outer fold exactly;
* there is no separate airflow split artefact -- the inheritance table is the
  only airflow assignment, and its per-fold counts match the airflow membership.
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6, splits_out

# default airflow patients (see _stage6_fixtures.default_windows): 01/02/05/08/10
_EXPECTED_AIRFLOW_PATIENTS = {"01", "02", "05", "08", "10"}


def _run_world(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return summary


def test_airflow_patients_are_subset_of_core(tmp_path):
    _summary = _run_world(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    af = pd.read_parquet(splits_out(tmp_path) / "airflow_cohort_window_membership.parquet")
    core["patient_id"] = core["patient_id"].astype(str)
    af["patient_id"] = af["patient_id"].astype(str)
    assert set(af["patient_id"]).issubset(set(core["patient_id"]))
    assert set(af["patient_id"]) == _EXPECTED_AIRFLOW_PATIENTS


def test_airflow_outer_fold_equals_core_outer_fold(tmp_path):
    _summary = _run_world(tmp_path)
    outer = pd.read_csv(splits_out(tmp_path) / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    inh = pd.read_csv(splits_out(tmp_path) / "airflow_outer_fold_inheritance.csv", dtype={"patient_id": str})
    merged = inh.merge(outer, on="patient_id", suffixes=("_airflow", "_core"))
    # every airflow patient's inherited fold matches its core fold
    assert int((merged.outer_fold_airflow != merged.outer_fold_core).sum()) == 0
    # the inheritance table holds exactly the airflow patients (no extra split)
    assert set(inh["patient_id"]) == _EXPECTED_AIRFLOW_PATIENTS


def test_no_independent_airflow_split(tmp_path):
    """Only one airflow assignment artefact exists; it is the inheritance table,
  and its per-fold patient/window counts match the airflow membership grouped
  by the (inherited) core fold."""
    _summary = _run_world(tmp_path)
    sdir = splits_out(tmp_path)
    names = {p.name for p in sdir.iterdir()}
    # the inheritance CSV is the sole airflow assignment; no airflow inner folds
    assert "airflow_outer_fold_inheritance.csv" in names
    assert not any("airflow" in n and "inner" in n for n in names)

    af = pd.read_parquet(sdir / "airflow_cohort_window_membership.parquet")
    inh = pd.read_csv(sdir / "airflow_outer_fold_inheritance.csv", dtype={"patient_id": str})
    af["patient_id"] = af["patient_id"].astype(str)
    # per-fold airflow window count from membership == inheritance table's count
    mem_by_fold = (
        af.merge(inh[["patient_id", "outer_fold"]], on="patient_id")
          .groupby("outer_fold")["window_id"].count()
    )
    inh_by_fold = inh.groupby("outer_fold")["n_airflow_windows"].sum()
    for f in mem_by_fold.index:
        assert int(mem_by_fold.loc[f]) == int(inh_by_fold.loc[f])
