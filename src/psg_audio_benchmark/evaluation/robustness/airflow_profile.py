"""Stage-9 2.6 -- transparent description of the 34-patient airflow sub-cohort.

This is purely **descriptive**: it profiles how the airflow sub-cohort was
selected (patients with usable airflow coverage), its size / window / time /
class-balance structure and the airflow feature quality -- so the selective
nature of the Stage 8 increment is explicit. It NEVER extrapolates the sub-cohort
to the 50-patient main cohort and never fits a model.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ...config import Config
from ...modeling.data_gate import _make_window_id
from . import schema as S
from .config import ResolvedRobustnessConfig
from .signature import RobustnessInput


def _window_index(cfg: Config) -> pd.DataFrame:
    idx = pd.read_parquet(cfg.path("data_manifests") / "csv_window_index.parquet")
    idx["window_id"] = [
        _make_window_id(p, i)
        for p, i in zip(idx["patient_id"], idx["window_index"])
    ]
    return idx


def airflow_profile(
    *, cfg: Config, rin: RobustnessInput, run_id: str, mc: ResolvedRobustnessConfig,
) -> Dict[str, pd.DataFrame]:
    air = rin.airflow_gate.airflow_cohort_windows.copy()
    core = rin.core_gate.cohort_windows.copy()
    air_pids = set(air["patient_id"].astype(str))
    core_pids = set(core["patient_id"].astype(str))

    # ---- cohort summary -----------------------------------------------------
    def _sum(df: pd.DataFrame) -> Dict[str, Any]:
        n = int(len(df)); npos = int((df["binary_event_label"] == 1).sum())
        return {
            "n_windows": n, "n_patients": int(df["patient_id"].nunique()),
            "n_positive": npos, "n_negative": n - npos,
            "positive_rate": (npos / n) if n else None,
        }
    s_air = _sum(air); s_core = _sum(core)
    summary = pd.DataFrame([{
        "run_id": run_id, "analysis": S.A_AIRFLOW_PROFILE,
        "analysis_role": S.ROLE_SENSITIVITY, "block": "cohort_summary",
        "subcohort": S.COHORT_AIRFLOW,
        "subcohort_patients": s_air["n_patients"],
        "subcohort_windows": s_air["n_windows"],
        "subcohort_positive": s_air["n_positive"],
        "subcohort_negative": s_air["n_negative"],
        "subcohort_positive_rate": s_air["positive_rate"],
        "main_cohort": S.COHORT_CORE,
        "main_cohort_patients": s_core["n_patients"],
        "main_cohort_windows": s_core["n_windows"],
        "main_cohort_positive_rate": s_core["positive_rate"],
        "overlap_patients_in_main": len(air_pids & core_pids),
        "patients_in_subcohort_not_main": len(air_pids - core_pids),
        "selection_basis": "core_coverage_complete AND has_airflow_coverage",
        "do_not_extrapolate_to_main_cohort": True,
    }])

    # ---- per-patient profile (join window index for time + coverage) --------
    idx = _window_index(cfg)
    cols = ["window_id", "start_relative_to_record_start",
            "end_relative_to_record_start", "core_coverage_complete",
            "has_airflow_coverage"]
    joined = air[["window_id", "patient_id", "binary_event_label"]].merge(
        idx[cols], on="window_id", how="left")
    per_rows = []
    for pid, g in joined.groupby("patient_id", sort=True):
        n = int(len(g)); npos = int(g["binary_event_label"].sum())
        span = float(g["end_relative_to_record_start"].max()
                     - g["start_relative_to_record_start"].min())
        per_rows.append({
            "run_id": run_id, "analysis": S.A_AIRFLOW_PROFILE,
            "analysis_role": S.ROLE_SENSITIVITY, "block": "per_patient",
            "patient_id": pid, "n_windows": n, "n_positive": npos,
            "n_negative": n - npos,
            "positive_rate": (npos / n) if n else None,
            "record_start_seconds": float(g["start_relative_to_record_start"].min()),
            "record_end_seconds": float(g["end_relative_to_record_start"].max()),
            "record_span_seconds": span,
            "frac_core_coverage_complete": float(g["core_coverage_complete"].mean()),
            "frac_has_airflow_coverage": float(g["has_airflow_coverage"].mean()),
        })
    per_patient = pd.DataFrame(per_rows)

    # ---- airflow feature quality -------------------------------------------
    af = pd.read_parquet(cfg.path("features_physiology") / "airflow_window_features.parquet")
    afj = air[["window_id"]].merge(
        af[["window_id", "coverage_fraction", "missing_fraction", "quality_status"]],
        on="window_id", how="left")
    has_q = "quality_status" in afj.columns and afj["quality_status"].notna().any()
    fq = pd.DataFrame([{
        "run_id": run_id, "analysis": S.A_AIRFLOW_PROFILE,
        "analysis_role": S.ROLE_SENSITIVITY, "block": "feature_quality",
        "n_windows_joined": int(len(afj)),
        "mean_coverage_fraction": float(afj["coverage_fraction"].mean()),
        "mean_missing_fraction": float(afj["missing_fraction"].mean()),
        "frac_full_coverage": float((afj["coverage_fraction"] >= 1.0).mean()),
        "frac_quality_ok": (float((afj["quality_status"] == "ok").mean())
                            if has_q else None),
    }])
    return {"cohort_summary": summary, "per_patient": per_patient,
            "feature_quality": fq}


__all__ = ["airflow_profile"]
