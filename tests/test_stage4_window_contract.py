"""Stage-4 boundary cases 1-6: window contract + research labels (pure + runner).

Covers (prompt section 6):
  1. half-open boundary: an event start exactly on a window endpoint links to
     exactly one window;
  2. a cross-boundary event is counted once and its duration/end are retained;
  3. 0% / partial / 100% awake overlap and the exclusion threshold;
  4. HR/SpO2 intersection, airflow optional, and insufficient core coverage;
  5. cross-midnight time coordinates; multiple patients never mix;
  6. positive / negative / excluded labels are mutually exclusive and an excluded
     window is never a negative.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from psg_audio_benchmark.windowing import Stage4Options, Stage4Runner
from psg_audio_benchmark.windowing import intervals as intervals_mod

from _stage4_fixtures import (
    DEFAULT_INPUT_RUN_ID,
    _patient_spec,
    build_input_run,
    make_isolated_cfg,
)


def _run(tmp_path: Path, patients, run_id: str):
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(cfg, DEFAULT_INPUT_RUN_ID, patients)
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": run_id, "config_hash": "realhash1234"},
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    assert summary.input_run_verified is True
    mdir = tmp_path / "outputs" / "data" / "manifests" / "runs" / run_id
    return cfg, summary, mdir


# ---------------------------------------------------------------------------
# Case 1: half-open boundary -- event start on an endpoint links to ONE window
# ---------------------------------------------------------------------------

def test_case1_half_open_boundary_single_link(tmp_path: Path) -> None:
    p = _patient_spec(
        patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0),
        events=[{"source_event_index": 0, "start": 60.0, "dur": 10.0, "end": 70.0}],
    )
    cfg, summary, mdir = _run(tmp_path, [p], "iso-stage4-case1")
    links = pd.read_parquet(mdir / "window_event_links.parquet")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    row = links.iloc[0]
    # start=60 must belong to window [60,90) (index 2), not [30,60) (index 1)
    assert int(row["window_index"]) == 2
    assert row["event_destiny"] == "linked"
    # that event produced exactly one link row
    assert len(links) == 1
    # the window containing it is positive; the window before it is negative
    pos = idx[idx["window_index"] == 2].iloc[0]
    neg = idx[idx["window_index"] == 1].iloc[0]
    assert pos["label_status"] == "positive" and int(pos["binary_event_label"]) == 1
    assert neg["label_status"] == "negative" and int(neg["binary_event_label"]) == 0


# ---------------------------------------------------------------------------
# Case 2: cross-boundary event counted once; duration/end retained
# ---------------------------------------------------------------------------

def test_case2_cross_boundary_no_double_count(tmp_path: Path) -> None:
    p = _patient_spec(
        patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0),
        events=[{"source_event_index": 0, "start": 25.0, "dur": 41.0, "end": 66.0}],
    )
    cfg, summary, mdir = _run(tmp_path, [p], "iso-stage4-case2")
    links = pd.read_parquet(mdir / "window_event_links.parquet")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    row = links.iloc[0]
    assert int(row["window_index"]) == 0  # start 25 in [0,30)
    assert int(row["event_duration_seconds"]) == 41
    assert int(row["event_end_relative_to_record_start"]) == 66
    # window 0 positive (1 retained), window 1 ([30,60)) negative (NOT positive)
    w0 = idx[idx["window_index"] == 0].iloc[0]
    w1 = idx[idx["window_index"] == 1].iloc[0]
    assert w0["label_status"] == "positive"
    assert w1["label_status"] == "negative"
    assert int(idx["binary_event_label"].fillna(-1).sum() > -1)  # has positives


# ---------------------------------------------------------------------------
# Case 3: 0% / partial / 100% awake overlap and the exclusion threshold
# ---------------------------------------------------------------------------

def test_case3_pure_awake_overlap_thresholds() -> None:
    # 0% overlap
    assert intervals_mod.awake_overlap_fraction(0, 30, [(100, 200)]) == 0.0
    # partial
    frac = intervals_mod.awake_overlap_fraction(0, 30, [(15, 25)])
    assert 0.0 < frac < 1.0
    # 100%
    assert intervals_mod.awake_overlap_fraction(0, 30, [(0, 30)]) == 1.0
    # threshold semantics: at threshold (0.0) NOT excluded; above it excluded
    assert intervals_mod.awake_overlap_is_excluded(0.0, 0.0) is False
    assert intervals_mod.awake_overlap_is_excluded(0.000001, 0.0) is True


def test_case3_runner_excludes_awake_overlap(tmp_path: Path) -> None:
    p = _patient_spec(
        patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0),
        awake=[(45.0, 75.0)],
        events=[{"source_event_index": 0, "start": 5.0, "dur": 10.0, "end": 15.0}],
    )
    cfg, summary, mdir = _run(tmp_path, [p], "iso-stage4-case3")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    # windows [30,60) and [60,90) overlap awake -> excluded; [0,30) and [90,120) eligible
    exc = idx[idx["label_status"] == "excluded"]
    elig = idx[idx["label_status"] != "excluded"]
    assert set(exc["window_index"]) == {1, 2}
    assert (exc["exclusion_reason"] == "excluded_awake_overlap").all()
    assert (exc["binary_event_label"].isna()).all()
    assert set(elig["window_index"]) == {0, 3}


# ---------------------------------------------------------------------------
# Case 4: HR/SpO2 intersection; airflow optional; insufficient core coverage
# ---------------------------------------------------------------------------

def test_case4_intersection_airflow_optional_coverage(tmp_path: Path) -> None:
    pA = _patient_spec(
        patient_id="01",
        hr=(0.0, 200.0), spo2=(50.0, 150.0),  # intersection [50,150]
        airflow=None,
    )
    pB = _patient_spec(
        patient_id="02",
        hr=(0.0, 200.0), spo2=(50.0, 150.0),
        airflow=(60.0, 120.0),
    )
    cfg, summary, mdir = _run(tmp_path, [pA, pB], "iso-stage4-case4")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    a = idx[idx["patient_id"] == "01"]
    b = idx[idx["patient_id"] == "02"]
    # patient 01: tail window [140,170) exceeds spo2 end 150 -> incomplete coverage
    tail = a[a["window_index"] == a["window_index"].max()].iloc[0]
    assert tail["core_coverage_complete"] == False
    assert tail["label_status"] == "excluded"
    assert tail["exclusion_reason"] == "excluded_incomplete_core_signal_coverage"
    # patient 01 has no airflow
    assert (a["has_airflow_coverage"] == False).all()
    # patient 02 has airflow coverage on at least some windows
    assert bool((b["has_airflow_coverage"] == True).any())
    # the candidate domain starts at the intersection lo (50), not 0
    assert float(a["start_relative_to_record_start"].min()) == 50.0


# ---------------------------------------------------------------------------
# Case 5: cross-midnight coordinates; multiple patients never mix
# ---------------------------------------------------------------------------

def test_case5_cross_midnight_coords_and_no_patient_mixing(tmp_path: Path) -> None:
    p1 = _patient_spec(
        patient_id="01", record_start_cumulative=80000.0,
        hr=(100.0, 220.0), spo2=(100.0, 220.0),
    )
    p2 = _patient_spec(
        patient_id="02", record_start_cumulative=90000.0,
        hr=(200.0, 320.0), spo2=(200.0, 320.0),
    )
    cfg, summary, mdir = _run(tmp_path, [p1, p2], "iso-stage4-case5")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    # absolute cumulative == record_start_cumulative + start_rel, per patient
    for pid, rs in (("01", 80000.0), ("02", 90000.0)):
        sub = idx[idx["patient_id"] == pid]
        for _, r in sub.iterrows():
            assert abs(float(r["start_absolute_cumulative_seconds"]) - (rs + float(r["start_relative_to_record_start"]))) < 1e-6
    # patient sets are disjoint; no patient has the other's record_start offset
    assert set(idx["patient_id"]) == {"01", "02"}
    p1_rows = idx[idx["patient_id"] == "01"]
    assert (p1_rows["start_absolute_cumulative_seconds"] >= 80000.0).all()
    assert (p1_rows["start_absolute_cumulative_seconds"] < 90000.0).all()


# ---------------------------------------------------------------------------
# Case 6: positive / negative / excluded mutually exclusive; excluded never neg
# ---------------------------------------------------------------------------

def test_case6_labels_mutually_exclusive_excluded_never_negative(tmp_path: Path) -> None:
    p = _patient_spec(
        patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0),
        awake=[(35.0, 55.0)],  # forces some excluded windows
        events=[
            {"source_event_index": 0, "start": 5.0, "dur": 10.0, "end": 15.0},
            {"source_event_index": 1, "start": 95.0, "dur": 10.0, "end": 105.0},
        ],
    )
    cfg, summary, mdir = _run(tmp_path, [p], "iso-stage4-case6")
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    assert set(idx["label_status"]).issubset({"positive", "negative", "excluded"})

    def _bin(r):
        v = r["binary_event_label"]
        if pd.isna(v):
            return None
        return int(v)

    for _, r in idx.iterrows():
        ls = r["label_status"]
        b = _bin(r)
        if ls == "positive":
            assert b == 1
        elif ls == "negative":
            assert b == 0
        else:  # excluded
            assert b is None, "excluded window must never carry a binary label"
            assert r["exclusion_reason"] != ""
    # at least one of each kind present
    assert "positive" in set(idx["label_status"])
    assert "negative" in set(idx["label_status"])
    assert "excluded" in set(idx["label_status"])
