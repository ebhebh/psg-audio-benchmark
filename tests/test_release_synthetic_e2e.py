"""Release synthetic end-to-end test: runs the FULL Stage-15b analysis runner on a
tiny SYNTHETIC cohort built in a temp project root.

This exercises the real analysis code path end to end — frozen-style parquet
inputs (cohort membership, parsed events, HR/SpO2/airflow window features,
outer/inner patient folds) -> label derivation -> nested patient-level CV ->
multi-level metrics -> bootstrap CIs -> airflow secondary -> output tables.

It is a SYNTHETIC smoke/integration test only:

* the numbers it produces are meaningless (no scientific interpretation);
* passing this test does NOT reproduce or validate any published real-data result;
* real-data verification requires the official dataset and the pipeline stages.

No real patient data is read; nothing outside the tmp project root is written.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from psg_audio_benchmark.bspc_revision import runner as R

RELEASE_ROOT = Path(__file__).resolve().parents[1]

N_PATIENTS = 20
WINDOWS_PER_PATIENT = 24
WINDOW_S = 30.0
RNG = np.random.default_rng(20250714)


def _pid(i: int) -> str:
    return f"{i:02d}"


def _build_synthetic_root(root: Path) -> None:
    (root / "splits").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    (root / "features" / "physiology").mkdir(parents=True)
    (root / "config").mkdir(parents=True)
    shutil.copy(RELEASE_ROOT / "config" / "bspc_revision_analysis.yaml",
                root / "config" / "bspc_revision_analysis.yaml")

    # ---- outer 5 x inner 4 patient-level split (mirrors the real v2 structure) ----
    outer_rows, inner_rows = [], []
    for i in range(N_PATIENTS):
        pid, of = _pid(i), i % 5
        outer_rows.append({"patient_id": int(pid), "outer_fold": of})
        for k in range(5):
            if k != of:
                inner_rows.append({"outer_fold": k, "patient_id": pid,
                                   "inner_validation_fold": (i + k) % 4})
    pd.DataFrame(outer_rows).to_csv(root / "splits" / "outer_patient_folds_core_v2.csv",
                                    index=False)
    pd.DataFrame(inner_rows).to_parquet(root / "splits" / "inner_patient_folds_core_v2.parquet",
                                        index=False)

    # ---- windows + synthetic events + features ----
    cohort_rows, event_rows = [], []
    hr_rows, spo2_rows, af_rows = [], [], []
    stats = ["mean", "median", "std_ddof1", "min", "max", "range", "iqr", "slope_per_second"]
    af_extra = ["rms", "zero_crossing_count", "zero_crossing_rate"]
    airflow_patients = {_pid(i) for i in range(0, N_PATIENTS, 2)}  # half the cohort

    for i in range(N_PATIENTS):
        pid = _pid(i)
        latent = RNG.normal(0.0, 1.0, WINDOWS_PER_PATIENT)          # latent severity
        for w in range(WINDOWS_PER_PATIENT):
            ws, we = w * WINDOW_S, (w + 1) * WINDOW_S
            wid = f"{pid}-{w:05d}"
            # event every other window with jittered duration (mix of overlap bands)
            has_event = (w % 2 == 0) and (RNG.random() < 0.6)
            if has_event:
                dur = float(RNG.uniform(5.0, 28.0))
                es = ws + float(RNG.uniform(0.0, WINDOW_S - dur))
                event_rows.append({
                    "patient_id": pid,
                    "event_start_relative_to_record_start": es,
                    "event_end_relative_to_record_start": es + dur,
                    "is_any_scored_respiratory_event": True,
                    "overlaps_awake_interval": False,
                })
            # onset label rule (must match the Stage-4 start-point definition)
            onset = int(any(
                r["patient_id"] == pid
                and r["event_start_relative_to_record_start"] >= ws
                and r["event_start_relative_to_record_start"] < we
                for r in event_rows if r["patient_id"] == pid))
            sev = latent[w] + (0.9 if onset else 0.0)
            cohort_rows.append({
                "window_id": wid, "patient_id": pid,
                "window_start_relative_to_record_start": ws,
                "window_end_relative_to_record_start": we,
                "binary_event_label": onset,
                "cohort_airflow": pid in airflow_patients,
            })
            # raw per-modality feature parquets store UNPREFIXED stat columns;
            # bspc_revision.features.build_base adds the hr_/spo2_/airflow_ prefixes
            hr = {s: float(sev * c + RNG.normal(0, 0.05))
                  for s, c in zip(stats, [0.4, 0.4, 0.1, -0.5, 0.5, 0.3, 0.2, 0.3])}
            sp = {s: float(-sev * c + RNG.normal(0, 0.05))
                  for s, c in zip(stats, [0.5, 0.5, 0.1, -1.0, 0.2, 0.4, 0.3, 0.2])}
            hr_rows.append({"window_id": wid, **hr})
            spo2_rows.append({"window_id": wid, **sp})
            if pid in airflow_patients:
                af = {s: float(sev * c + RNG.normal(0, 0.05))
                      for s, c in zip(stats + af_extra,
                                      [0.3] * 8 + [0.6, 0.2, 0.1])}
                af_rows.append({"window_id": wid, **af})

    pd.DataFrame(cohort_rows).to_parquet(root / "splits" / "core_cohort_window_membership.parquet",
                                         index=False)
    # a few awake-overlapping / non-respiratory events to exercise the filters
    event_rows += [
        {"patient_id": _pid(0), "event_start_relative_to_record_start": 15.0,
         "event_end_relative_to_record_start": 45.0,
         "is_any_scored_respiratory_event": True, "overlaps_awake_interval": True},
        {"patient_id": _pid(1), "event_start_relative_to_record_start": 100.0,
         "event_end_relative_to_record_start": 130.0,
         "is_any_scored_respiratory_event": False, "overlaps_awake_interval": False},
    ]
    pd.DataFrame(event_rows).to_parquet(root / "annotations" / "parsed_events.parquet", index=False)
    pd.DataFrame(hr_rows).to_parquet(root / "features" / "physiology" / "hr_window_features.parquet",
                                     index=False)
    pd.DataFrame(spo2_rows).to_parquet(root / "features" / "physiology" / "spo2_window_features.parquet",
                                       index=False)
    pd.DataFrame(af_rows).to_parquet(root / "features" / "physiology" / "airflow_window_features.parquet",
                                     index=False)


def test_synthetic_end_to_end_stage15b_runner(tmp_path):
    root = tmp_path / "synthroot"
    _build_synthetic_root(root)
    run_id = "synthetic-e2e"
    result = R.run(project_root=root, run_id=run_id)

    res_dir = root / "results" / "runs" / run_id
    expected = ["summary_metrics.csv", "patient_macro_metrics.csv", "event_level_metrics.csv",
                "calibration_metrics.csv", "lag_sensitivity.csv", "bootstrap_ci_primary.csv",
                "paired_bootstrap_ci.csv", "paired_bootstrap_hgb_vs_lr.csv",
                "airflow_paired_summary.csv", "airflow_paired_ci.csv", "label_versions.csv",
                "label_onset_vs_main_confusion.csv", "selected_hyperparameters.csv",
                "inner_cv_candidate_scores.csv", "oof_predictions.parquet",
                "model_input_signature.json"]
    for name in expected:
        assert (res_dir / name).is_file(), f"missing output {name}"

    # onset variant reproduces the Stage-4-style start-point label on synthetic data
    assert result["onset_matches_stage4"] is True

    sm = pd.read_csv(res_dir / "summary_metrics.csv")
    prim = sm[(sm.task == "primary_battery") & (sm.model == "hr_spo2_lr")
              & (sm.lag_label == "lag_0")]
    assert len(prim) == 1
    row = prim.iloc[0]
    assert row["variant"] == "main" and row["variant_name"] == "overlap_ge_10s"
    assert 0.0 <= row["auroc"] <= 1.0
    assert int(row["n_patients"]) == N_PATIENTS
    # synthetic signal is informative by construction; require better-than-chance pooled AUROC
    assert row["auroc"] > 0.6

    # OOF integrity: exactly one prediction per (model, window) in the primary battery
    oof = pd.read_parquet(res_dir / "oof_predictions.parquet")
    prim_oof = oof[oof.task == "primary_battery"]
    for model in ["dummy_prior", "simple_spo2_min_lr", "hr_only_lr", "spo2_only_lr",
                  "hr_spo2_lr", "hr_spo2_hgb"]:
        d = prim_oof[(prim_oof.model == model) & (prim_oof.variant == "main")
                     & (prim_oof.lag_label == "lag_0")]
        assert d["window_id"].is_unique, f"duplicate OOF rows for {model}"
        assert len(d) == int(row["n_windows"])

    # airflow secondary ran on the synthetic airflow subcohort
    af = pd.read_csv(res_dir / "airflow_paired_summary.csv")
    assert set(af.feature_set) == {"core_restricted", "airflow_enhanced"}
    assert int(af.iloc[0]["n_patients"]) == len({_pid(i) for i in range(0, N_PATIENTS, 2)})

    # signature bookkeeping
    sig = result["sig"]
    assert sig["seed"] == 20250714
    assert sig["audio_blocked"] is True and sig["audio_eligible_windows"] == 0
    assert sig["n_cohort_windows"] == N_PATIENTS * WINDOWS_PER_PATIENT

    # determinism: same inputs + frozen seed -> identical primary AUROC
    result2 = R.run(project_root=root, run_id="synthetic-e2e-rerun")
    sm2 = pd.read_csv(root / "results" / "runs" / "synthetic-e2e-rerun" / "summary_metrics.csv")
    prim2 = sm2[(sm2.task == "primary_battery") & (sm2.model == "hr_spo2_lr")
                & (sm2.lag_label == "lag_0")].iloc[0]
    assert prim2["auroc"] == row["auroc"]
    assert result2["onset_matches_stage4"] is True
