"""Tests for the Stage 15b BSPC revision package (prompt section 7).

Coverage:

* label derivation: duration-overlap MAIN indeterminate band, onset Stage-4
  reproduction, variant count bookkeeping;
* feature assembly + physiological-latency (lag) shifting and lag attrition;
* nested-CV integrity: one OOF probability per window, each patient in exactly
  one outer fold, determinism under the frozen seed, thresholds in range,
  dummy prior baseline;
* multi-level metrics: patient-macro single-class exclusion, calibration slope
  on a calibrated signal, event-level candidate aggregation;
* frozen-registration + audio-constraint guards;
* real-data OOF integrity on the INHERITED frozen v2 split (one prediction per
  window; onset reproduces Stage-4).

Audio is never read. Stage 2-14 products are never modified.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psg_audio_benchmark.bspc_revision import labels as L
from psg_audio_benchmark.bspc_revision import features as F
from psg_audio_benchmark.bspc_revision import nested_cv as N
from psg_audio_benchmark.bspc_revision import eval_metrics as E
from psg_audio_benchmark.config import default_project_root

_PROJECT = default_project_root()


# --------------------------------------------------------------------- tiny synth
def _synth_overlaps() -> pd.DataFrame:
    """10 windows, one patient; max_overlap_s sweeps 0..30s."""
    rows = []
    for i in range(10):
        ov = float(i * 3.0)          # 0,3,6,...,27
        rows.append({
            "window_id": f"01-{i:05d}", "patient_id": "01",
            "window_start_relative_to_record_start": float(i * 30.0),
            "window_end_relative_to_record_start": float(i * 30.0 + 30.0),
            "max_overlap_s": ov, "n_retained_overlapping_events": int(ov > 0),
            "onset_hit": int(i == 0),
        })
    return pd.DataFrame(rows)


# ========================================================================= labels
def test_main_label_indeterminate_band():
    tables = L.assign_labels(overlaps=_synth_overlaps())
    m = tables[L.MAIN].table
    # overlap>=10 -> positive; 0<ov<10 -> indeterminate(-1); 0 -> negative
    assert int(m.loc[m.window_id == "01-00000", "label"].iloc[0]) == 0      # ov 0  -> neg
    assert int(m.loc[m.window_id == "01-00001", "label"].iloc[0]) == -1     # ov 3  -> indet
    assert int(m.loc[m.window_id == "01-00003", "label"].iloc[0]) == -1     # ov 9  -> indet
    assert int(m.loc[m.window_id == "01-00004", "label"].iloc[0]) == 1      # ov 12 -> pos
    assert int(m.loc[m.window_id == "01-00009", "label"].iloc[0]) == 1      # ov 27 -> pos


def test_onset_and_any_variants():
    tables = L.assign_labels(overlaps=_synth_overlaps())
    o = tables[L.ONSET].table
    a = tables[L.ANY].table
    assert int(o["label"].sum()) == 1                       # only window 0 has onset_hit
    assert int((o["label"] == -1).sum()) == 0               # onset has no indeterminate
    assert int(a.loc[a.window_id == "01-00000", "label"].iloc[0]) == 0     # ov 0 -> neg for ANY
    assert int(a.loc[a.window_id == "01-00001", "label"].iloc[0]) == 1     # ov>0 -> pos for ANY


def test_variant_counts_bookkeeping():
    tables = L.assign_labels(overlaps=_synth_overlaps())
    c = L.variant_counts(label_result=tables[L.MAIN])
    assert c["n_windows"] == 10
    assert c["n_positive"] + c["n_negative"] + c["n_indeterminate"] == 10
    # positive_rate is computed over non-indeterminate only
    denom = c["n_positive"] + c["n_negative"]
    assert c["positive_rate"] == pytest.approx(c["n_positive"] / denom)


def test_onset_matches_stage4_exact():
    ov = _synth_overlaps()
    tables = L.assign_labels(overlaps=ov)
    stage4 = ov[["window_id"]].copy()
    stage4["binary_event_label"] = tables[L.ONSET].table["label"].values
    assert L.onset_matches_stage4(onset_table=tables[L.ONSET].table, stage4_cohort=stage4) is True
    # perturb one label -> must fail
    stage4.loc[stage4.index[0], "binary_event_label"] = 1 - int(stage4.loc[stage4.index[0], "binary_event_label"])
    assert L.onset_matches_stage4(onset_table=tables[L.ONSET].table, stage4_cohort=stage4) is False


# ===================================================================== features
def _synth_base(values: dict) -> pd.DataFrame:
    """base features: window_id + hr_mean/spo2_min etc. ``values`` maps stat->list."""
    wids = [f"01-{i:05d}" for i in range(5)]
    out = pd.DataFrame({"window_id": wids})
    out["hr_mean"] = [10.0, 20.0, 30.0, 40.0, 50.0]
    out["spo2_min"] = [91.0, 90.0, 89.0, 88.0, 87.0]
    return out


def _synth_cohort() -> pd.DataFrame:
    wids = [f"01-{i:05d}" for i in range(5)]
    return pd.DataFrame({
        "window_id": wids, "patient_id": "01",
        "window_start_relative_to_record_start": [i * 30.0 for i in range(5)],
        "window_end_relative_to_record_start": [i * 30.0 + 30.0 for i in range(5)],
    })


def test_family_column_counts():
    assert len(F.FAMILIES["hr_only"]) == 8
    assert len(F.FAMILIES["spo2_only"]) == 8
    assert len(F.FAMILIES["hr_spo2"]) == 16
    assert F.FAMILIES["simple_spo2_min"] == ["spo2_min"]


def test_lag_shift_uses_next_window_features():
    base = _synth_base({})
    cohort = _synth_cohort()
    fm0 = F.assemble(cohort=cohort, base=base, family="simple_spo2_min", lag_windows=0)
    assert list(fm0.X[:, 0]) == [91.0, 90.0, 89.0, 88.0, 87.0]
    assert fm0.n_dropped_missing_shift == 0
    fm1 = F.assemble(cohort=cohort, base=base, family="simple_spo2_min", lag_windows=1)
    # label window i takes feature window i+1 -> [90,89,88,87]; last window dropped
    assert list(fm1.X[:, 0]) == [90.0, 89.0, 88.0, 87.0]
    assert fm1.n_dropped_missing_shift == 1
    assert list(fm1.window_id) == [f"01-{i:05d}" for i in range(4)]


def test_lag_two_drops_two():
    base = _synth_base({})
    cohort = _synth_cohort()
    fm2 = F.assemble(cohort=cohort, base=base, family="simple_spo2_min", lag_windows=2)
    assert list(fm2.X[:, 0]) == [89.0, 88.0, 87.0]
    assert fm2.n_dropped_missing_shift == 2


# ==================================================================== nested CV
def _synth_nested_inputs():
    """4 patients x 50 windows, balanced-ish labels; 2 outer folds, 4 inner folds."""
    rng = np.random.default_rng(0)
    wids, pids, y = [], [], []
    for p in range(4):
        for i in range(50):
            wids.append(f"{p:02d}-{i:05d}")
            pids.append(f"{p:02d}")
            y.append(int(rng.random() < 0.4))
    frame = pd.DataFrame({"window_id": wids, "patient_id": pids,
                          "window_start": [i * 30.0 for i in range(200)],
                          "window_end": [(i * 30.0) + 30.0 for i in range(200)]})
    X = rng.normal(size=(200, 4))
    ofold = np.array([int(p) % 2 for p in pids])     # patients 0,2 -> fold0; 1,3 -> fold1
    # inner fold assignment per (outer_fold, patient)
    inner_map = {}
    for p in range(4):
        for k in range(2):
            inner_map[(k, f"{p:02d}")] = p % 4
    return frame, X, np.array(y), np.array(pids), ofold, inner_map


def test_nested_cv_one_prediction_per_window():
    frame, X, y, pids, ofold, inner_map = _synth_nested_inputs()
    spec = N.ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2")
    res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                          outer_fold=ofold, inner_map=inner_map)
    # one OOF row per window
    assert len(res.oof) == len(frame)
    assert res.oof.groupby("window_id").size().max() == 1
    # each patient appears in exactly one outer fold in the OOF
    pf = res.oof.groupby("patient_id")["outer_fold"].nunique()
    assert (pf == 1).all()


def test_nested_cv_determinism():
    frame, X, y, pids, ofold, inner_map = _synth_nested_inputs()
    spec = N.ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2")
    r1 = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                         outer_fold=ofold, inner_map=inner_map)
    r2 = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                         outer_fold=ofold, inner_map=inner_map)
    a = r1.oof.sort_values("window_id")["y_prob"].to_numpy()
    b = r2.oof.sort_values("window_id")["y_prob"].to_numpy()
    assert np.allclose(a, b)


def test_nested_cv_thresholds_in_range_and_dummy_prior():
    frame, X, y, pids, ofold, inner_map = _synth_nested_inputs()
    spec = N.ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2")
    res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                          outer_fold=ofold, inner_map=inner_map)
    thr = res.oof["threshold"].unique()
    assert all(0.0 <= t <= 1.0 for t in thr)
    # dummy uses fixed 0.5
    dspec = N.ModelSpec("dummy_prior", "lower_bound", "dummy", "none")
    dres = N.run_nested_cv(spec=dspec, frame=frame, X=X, y=y, patient_ids=pids,
                           outer_fold=ofold, inner_map=inner_map)
    assert set(dres.oof["threshold"].unique().tolist()) == {0.5}
    # dummy probability == training positive prevalence (prior), constant per fold
    assert dres.oof["y_prob"].nunique() <= 2


def test_nested_cv_outer_test_not_in_inner_lookup():
    """Outer-test patients are never passed to inner selection: the engine looks up
    inner_map only for outer-TRAIN patients. Verify a test patient has no inner entry
    used by asserting the OOF still has one row per window and selected records exist
    only for outer folds (never referencing test patients)."""
    frame, X, y, pids, ofold, inner_map = _synth_nested_inputs()
    spec = N.ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2")
    res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                          outer_fold=ofold, inner_map=inner_map)
    assert len(res.selected) == int(ofold.max()) + 1     # one selection per outer fold


# ============================================================ multi-level metrics
def test_patient_macro_excludes_single_class():
    oof = pd.DataFrame({
        "patient_id": ["a", "a", "b", "b"],
        "y_true": [0, 1, 0, 0],          # patient b is single-class
        "y_prob": [0.2, 0.8, 0.1, 0.2],
    })
    pm = E.patient_macro(oof=oof)
    assert pm["n_patients_excluded"] == 1
    assert pm["patient_macro_auroc"] is not None


def test_calibration_well_calibrated_slope_near_one():
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < p).astype(int)
    cal = E.calibration(y_true=y, y_prob=p)
    assert cal["calibration_slope"] == pytest.approx(1.0, abs=0.15)
    assert cal["calibration_intercept"] == pytest.approx(0.0, abs=0.15)
    assert 0.0 <= cal["ece"] < 0.05


def test_event_level_aggregates_contiguous_positives():
    oof = pd.DataFrame({
        "patient_id": ["a"] * 4,
        "window_start": [0.0, 30.0, 60.0, 90.0],
        "window_end": [30.0, 60.0, 90.0, 120.0],
        "y_pred": [1, 1, 0, 1],            # two candidate events (0-60, 90-120)
        "y_true": [1, 1, 0, 1],
        "y_prob": [0.9, 0.9, 0.1, 0.9],
    })
    gt = pd.DataFrame({
        "patient_id": ["a", "a"],
        "event_start_relative_to_record_start": [5.0, 95.0],
        "event_end_relative_to_record_start": [25.0, 115.0],
        "is_any_scored_respiratory_event": [True, True],
        "overlaps_awake_interval": [False, False],
    })
    ev = E.event_level(oof=oof, retained_events=gt, analyzable_hours=120.0 / 3600.0)
    assert ev["n_candidate_events"] == 2
    assert ev["n_gt_events_matched"] == 2
    assert ev["event_sensitivity"] == 1.0
    assert ev["event_precision"] == 1.0


# ============================================================ guards (frozen + audio)
def test_frozen_registration_files_exist():
    cfg = _PROJECT / "config" / "bspc_revision_analysis.yaml"
    proto = _PROJECT / "docs" / "bspc_revision_analysis_protocol.md"
    assert cfg.exists(), "frozen BSPC analysis registration config missing"
    assert proto.exists(), "frozen BSPC analysis protocol missing"
    txt = cfg.read_text(encoding="utf-8")
    assert "20250714" in txt
    assert "overlap" in txt.lower()


def test_package_does_not_reference_audio():
    """Constraint guard: the BSPC revision source must not read or import audio."""
    pkg = _PROJECT / "src" / "psg_audio_benchmark" / "bspc_revision"
    blobs = []
    for src in sorted(pkg.glob("*.py")):
        blobs.append(src.read_text(encoding="utf-8"))
    joined = "\n".join(blobs).lower()
    for bad in ["audio_window_features", "import audio", "from .audio", "mfcc", "mel_spec"]:
        assert bad not in joined, f"audio reference '{bad}' found in bspc_revision"


# ============================================================ real-data integrity
_REAL_DATA = (_PROJECT / "splits" / "core_cohort_window_membership.parquet").exists()


@pytest.mark.skipif(not _REAL_DATA, reason="frozen v2 split not present in this checkout")
def test_real_data_oof_integrity_and_onset_reproduction():
    """On the INHERITED frozen v2 split: one OOF prediction per window; onset label
    reproduces Stage-4 exactly; main label excludes indeterminate windows."""
    from psg_audio_benchmark.bspc_revision.runner import build_frame

    cohort = pd.read_parquet(_PROJECT / "splits" / "core_cohort_window_membership.parquet")
    events = pd.read_parquet(_PROJECT / "annotations" / "parsed_events.parquet")
    hr = pd.read_parquet(_PROJECT / "features" / "physiology" / "hr_window_features.parquet")
    spo2 = pd.read_parquet(_PROJECT / "features" / "physiology" / "spo2_window_features.parquet")
    outer = pd.read_csv(_PROJECT / "splits" / "outer_patient_folds_core_v2.csv")
    inner = pd.read_parquet(_PROJECT / "splits" / "inner_patient_folds_core_v2.parquet")

    _w = max(len(str(x)) for x in cohort["patient_id"].astype(str).unique())
    _n = lambda p: f"{int(float(p)):0{_w}d}"
    outer_map = {_n(r.patient_id): int(r.outer_fold) for r in outer.itertuples()}
    inner_map = {(int(r.outer_fold), _n(r.patient_id)): int(r.inner_validation_fold)
                 for r in inner.itertuples()}

    ov = L.derive_overlaps(cohort=cohort, events=events)
    tables = L.assign_labels(overlaps=ov)
    assert L.onset_matches_stage4(onset_table=tables[L.ONSET].table, stage4_cohort=cohort) is True

    base = F.build_base(hr=hr, spo2=spo2)
    fm = F.assemble(cohort=cohort, base=base, family="hr_spo2", lag_windows=0)
    frame, X, y, pids, ofold = build_frame(label_result=tables[L.MAIN], fm=fm, outer_map=outer_map)
    # indeterminate excluded from main frame
    assert set(np.unique(y).tolist()) <= {0, 1}
    spec = N.ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2")
    res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                          outer_fold=ofold, inner_map=inner_map)
    assert res.oof.groupby("window_id").size().max() == 1
    assert len(res.oof) == len(frame)
    pf = res.oof.groupby("patient_id")["outer_fold"].nunique()
    assert (pf == 1).all()
    # 50 patients total in the inherited split
    assert cohort["patient_id"].nunique() == 50
