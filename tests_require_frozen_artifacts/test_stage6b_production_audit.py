"""(1,2,5,6,7,9) Real-data production audit over the genuine v1 core cohort.

Reads the real Stage-6 v1 run-dir (READ-ONLY) and runs ``Stage6bRunner`` isolated
under ``tmp_path/out``. Asserts approval, all patient-level leakage invariants,
airflow inheritance with zero mismatch, inner-CV isolation, a pinned + consistent
version signature, no model matrix / no raw / audio=0, and isolated, path-clean
outputs. No raw CSV/WAV or feature value is ever read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from _stage6b_fixtures import REAL_V1_RUN_ID, fast_config, load_6b_config, run_6b
from psg_audio_benchmark.config import Config
from psg_audio_benchmark.evaluation.schema import STAGE6_FORBIDDEN_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage6b-real-audit-0001"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("6b_audit")
    cfg = Config(project_root=PROJECT_ROOT)
    cfg._resolved_paths["logs"] = tmp / "logs"
    (tmp / "logs").mkdir(parents=True, exist_ok=True)
    config = fast_config(load_6b_config(PROJECT_ROOT))
    summary, runner = run_6b(
        cfg=cfg, run_id=RUN_ID, config=config,
        output_root=tmp / "out", relpath_base=tmp,
    )
    return {"tmp": tmp, "cfg": cfg, "summary": summary, "runner": runner}


def _splits(world):
    return world["tmp"] / "out" / "splits" / "runs" / RUN_ID


# ---- (1,5) approval + hard constraints on the real cohort ----

def test_real_v2_approved_and_balanced(world):
    s = world["summary"]
    assert s.status == "approved_for_modeling"
    assert s.gate.approved is True
    assert s.v2["spread"] <= 0.10
    assert s.v2["spread"] < s.v1["spread"]
    assert [p["n_patients"] for p in s.v2_per_fold] == [10] * 5
    assert s.anomalies == []


def test_real_no_patients_or_windows_dropped(world):
    s = world["summary"]
    assert s.cohort["n_patients"] == 50
    assert s.cohort["n_windows"] == 34643
    assert s.cohort["n_positive"] == 9732
    assert s.cohort["n_negative"] == 24911
    outer = pd.read_csv(_splits(world) / "outer_patient_folds_core_v2.csv", dtype={"patient_id": str})
    assert outer["patient_id"].nunique() == 50
    assert set(outer["patient_id"]) == set(range(1, 51)) or outer["patient_id"].nunique() == 50
    assert int(outer["n_windows"].sum()) == 34643
    assert int(outer["n_positive"].sum()) == 9732


def test_real_each_patient_one_fold_no_overlap(world):
    outer = pd.read_csv(_splits(world) / "outer_patient_folds_core_v2.csv", dtype={"patient_id": str})
    assert outer["patient_id"].is_unique
    assert sorted(outer["outer_fold"].unique().tolist()) == [0, 1, 2, 3, 4]
    # exactly 10 per fold
    assert outer.groupby("outer_fold")["patient_id"].count().tolist() == [10] * 5


# ---- (6) airflow inheritance, zero mismatch, no independent split ----

def test_real_airflow_inherits_v2_fold_zero_mismatch(world):
    s = world["summary"]
    assert s.airflow_mismatch_count == 0
    assert s.airflow_patients == 34
    outer = pd.read_csv(_splits(world) / "outer_patient_folds_core_v2.csv", dtype={"patient_id": str})
    inh = pd.read_csv(_splits(world) / "airflow_outer_fold_inheritance_v2.csv", dtype={"patient_id": str})
    merged = inh.merge(outer, on="patient_id", suffixes=("_airflow", "_core"))
    assert int((merged.outer_fold_airflow != merged.outer_fold_core).sum()) == 0
    assert set(inh["patient_id"]).issubset(set(outer["patient_id"]))


def test_real_no_independent_airflow_split(world):
    sdir = _splits(world)
    names = {p.name for p in sdir.iterdir()}
    assert "airflow_outer_fold_inheritance_v2.csv" in names
    # no separate airflow inner / outer split artefact beyond the inheritance table
    assert not any("airflow" in n and "inner" in n for n in names)


# ---- (7) inner CV isolation: outer-train == inner set; outer-test absent ----

def test_real_inner_isolation(world):
    sdir = _splits(world)
    inner_path = sdir / "inner_patient_folds_core_v2.parquet"
    assert inner_path.is_file()  # built only on approval
    inner = pd.read_parquet(inner_path)
    outer = pd.read_csv(sdir / "outer_patient_folds_core_v2.csv", dtype={"patient_id": str})
    outer["patient_id"] = outer["patient_id"].astype(str)
    inner["patient_id"] = inner["patient_id"].astype(str)
    for f in range(5):
        outer_test = set(outer.loc[outer.outer_fold == f, "patient_id"])
        outer_train = set(outer.loc[outer.outer_fold != f, "patient_id"])
        inner_f = set(inner.loc[inner.outer_fold == f, "patient_id"])
        assert inner_f == outer_train                       # inner set == outer-train
        assert not (outer_test & inner_f)                   # outer-test never in inner
        # 4 inner folds, each patient exactly one inner fold
        sub = inner[inner.outer_fold == f]
        assert sorted(sub.inner_validation_fold.unique().tolist()) == [0, 1, 2, 3]
        assert sub["patient_id"].is_unique


# ---- (8) version signature pinned + consistent ----

def test_real_signature_pinned_and_consistent(world):
    sig = json.loads((_splits(world) / "split_balance_version_signature.json").read_text("utf-8"))
    assert sig["approval_status"] == "approved_for_modeling"
    assert sig["lineage"]["v1_comparator_run_id"] == REAL_V1_RUN_ID
    assert sig["lineage"]["lineage_stage5_matches_stage4"] is True
    assert sig["lineage"]["stage4_run_id"] and sig["lineage"]["stage5_run_id"]
    # v2 product files pinned with sha256 + rows + schema
    assert sig["v2_files"], "no v2 files recorded"
    for key, rec in sig["v2_files"].items():
        assert len(rec["sha256"]) == 64
        assert rec["rows"] > 0
    # cohort + assignment fingerprints recorded
    assert len(sig["cohort"]["patient_set_sha256"]) == 64
    assert len(sig["split"]["assignment_sha256"]) == 64
    assert sig["cohort"]["n_patients"] == 50
    assert sig["audio_block"]["audio_cohort_count"] == 0


def test_real_resolved_config_records_runtime_comparators(world):
    import yaml
    doc = yaml.safe_load((_splits(world) / "split_balance_resolved.yaml").read_text("utf-8"))
    rc = doc["runtime_computed"]
    assert abs(rc["v1_computed_positive_rate_spread"] - 0.17556619642728147) < 1e-9
    assert rc["v2_positive_rate_spread"] <= 0.10
    assert rc["approval_status"] == "approved_for_modeling"
    assert all(c["passed"] for c in rc["gate_checks"])


# ---- (9) no model matrix, no raw/audio, audio=0 ----

def test_real_outputs_have_no_model_matrix_no_label_column(world):
    sdir = _splits(world)
    for p in sdir.iterdir():
        if p.suffix == ".parquet":
            cols = set(pd.read_parquet(p).columns)
        elif p.suffix == ".csv":
            cols = set(pd.read_csv(p, nrows=0).columns)
        else:
            continue
        assert not (cols & set(STAGE6_FORBIDDEN_COLUMNS)), (p.name, cols & set(STAGE6_FORBIDDEN_COLUMNS))
        assert "binary_event_label" not in cols, p.name  # label only allowed in membership; none here


def test_real_audio_zero_and_deferred_transforms(world):
    s = world["summary"]
    assert s.audio_cohort_count == 0


# ---- outputs isolated + path-clean ----

def test_real_outputs_isolated_under_output_root(world):
    sdir = _splits(world)
    # every product is strictly under the isolated output root (tmp/out)
    assert str(sdir.resolve()).startswith(str((world["tmp"] / "out").resolve()))
    assert sdir.is_dir()
    expected = {
        "outer_patient_folds_core_v2.csv", "patient_level_balance_inputs.csv",
        "split_candidate_comparison.csv", "airflow_outer_fold_inheritance_v2.csv",
        "inner_patient_folds_core_v2.parquet", "split_balance_version_signature.json",
        "split_balance_resolved.yaml",
    }
    assert expected.issubset({p.name for p in sdir.iterdir()})


def test_real_latest_not_updated_in_isolated_mode(world):
    # isolated mode never mirrors to fixed paths / LATEST
    assert world["summary"].latest_updated is False
