"""Stage-6 cohort assembly: stable-key join over Stage-4 labels + Stage-5 availability.

Builds the audited CSV-physiology research cohorts WITHOUT forming any model
matrix. Membership rows carry only keys + the binary label + availability flags
+ cohort membership; NO Stage-5 numeric feature value ever enters a Stage-6
output (enforced by the column deny-list in :mod:`.schema`).

Cohorts:

* **Core** = Stage-4 positive/negative candidate windows whose Stage-5
  ``core_hr_spo2_available`` is ``True`` (HR AND SpO2 available).
* **Airflow** = core cohort further restricted to ``airflow_available == True``
  (strict: HR + SpO2 + airflow all available). Airflow patients are a strict
  subset of core patients.

Stage-4 ``excluded`` windows, HR/SpO2 low-coverage windows and airflow-missing
windows are recorded in ``cohort_exclusions.csv`` with a reason; no non-selected
window is ever turned into a negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .schema import (
    ALLOWED_LABEL_COLUMN,
    AUDIO_FEATURES_PRESENT,
    CANDIDATE_STATUSES,
    CohortExclusion,
    CohortMembershipRow,
    LABEL_EXCLUDED,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE,
    SCOPE_HR_QUALITY_INSUFFICIENT,
    SCOPE_SPO2_QUALITY_INSUFFICIENT,
    SCOPE_STAGE4_EXCLUDED,
    STAGE6_FORBIDDEN_COLUMNS,
)

_TIME_TOL = 1e-6


def make_window_id(patient_id: Any, window_index: Any) -> str:
    """Stable window id ``<patient_id>-<window_index:05d>`` (matches Stage 5)."""
    return f"{str(patient_id)}-{int(window_index):05d}"


@dataclass
class CohortResult:
    """Everything the split + report stages need from the cohort assembly."""
    core_rows: List[CohortMembershipRow] = field(default_factory=list)
    airflow_rows: List[CohortMembershipRow] = field(default_factory=list)
    exclusions: List[CohortExclusion] = field(default_factory=list)
    # core cohort windows for the splitter: window_id, patient_id, binary_event_label
    core_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    airflow_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    patient_burden: Dict[str, Dict[str, float]] = field(default_factory=dict)
    airflow_patient_burden: Dict[str, Dict[str, float]] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)


def _assert_no_forbidden_columns(df: pd.DataFrame, where: str) -> None:
    """A Stage-6 frame must never carry a numeric feature-value column."""
    bad = [c for c in STAGE6_FORBIDDEN_COLUMNS if c in df.columns]
    if bad:
        raise AssertionError(
            f"{where}: forbidden feature-value column(s) present: {bad}. "
            f"Stage-6 outputs must not form a model matrix."
        )


def _burden(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-patient {n_windows, n_positive, n_negative, pos_rate} over a cohort df."""
    out: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return out
    for pid, sub in df.groupby("patient_id"):
        n = int(len(sub))
        pos = int((sub["binary_event_label"] == 1).sum())
        neg = int((sub["binary_event_label"] == 0).sum())
        out[str(pid)] = {
            "n_windows": float(n),
            "n_positive": float(pos),
            "n_negative": float(neg),
            "pos_rate": float(pos / n) if n else 0.0,
        }
    return out


def build_cohorts(
    *,
    stage4_window_index_path: Path,
    stage5_availability_path: Path,
    run_id: str,
) -> CohortResult:
    """Join Stage-4 labels + Stage-5 availability on stable keys; build cohorts."""
    res = CohortResult()

    try:
        s4 = pd.read_parquet(stage4_window_index_path)
    except Exception as exc:  # pragma: no cover - defensive
        res.anomalies.append(f"stage4_window_index_unreadable:{type(exc).__name__}")
        return res
    try:
        s5 = pd.read_parquet(stage5_availability_path)
    except Exception as exc:  # pragma: no cover - defensive
        res.anomalies.append(f"stage5_availability_unreadable:{type(exc).__name__}")
        return res

    # ---- coerce patient_id to str on BOTH sides before constructing window_id ----
    s4 = s4.copy()
    s5 = s5.copy()
    s4["patient_id"] = s4["patient_id"].astype(str)
    s5["patient_id"] = s5["patient_id"].astype(str)
    s4["window_index"] = s4["window_index"].astype(int)

    # Stage-4 excluded windows -> exclusions (never candidates, never negatives).
    for _, w in s4[s4["label_status"] == LABEL_EXCLUDED].iterrows():
        pid = str(w["patient_id"])
        widx = int(w["window_index"])
        res.exclusions.append(CohortExclusion(
            run_id=run_id, window_id=make_window_id(pid, widx),
            patient_id=pid, window_index=widx, scope=SCOPE_STAGE4_EXCLUDED,
            reason=str(w.get("exclusion_reason", "stage4_excluded") or "stage4_excluded"),
            detail="Stage-4 excluded window is not a cohort candidate; not faked as negative.",
        ))

    # ---- candidates only (positive/negative) ----
    cand = s4[s4["label_status"].isin(list(CANDIDATE_STATUSES))].copy()
    cand["window_id"] = cand.apply(
        lambda r: make_window_id(r["patient_id"], r["window_index"]), axis=1
    )
    if "window_id" not in s5.columns:
        # Stage-5 availability lacks window_id only if the file is malformed.
        s5["window_id"] = s5.apply(
            lambda r: make_window_id(r["patient_id"], r["window_index"]), axis=1
        )
    else:
        s5["window_id"] = s5["window_id"].astype(str)

    # ---- duplicate-key guard BEFORE merge (prevents silent m:1 / m:m collapse) ----
    for side, name in ((cand, "stage4_candidate"), (s5, "stage5_availability")):
        dups = int(side["window_id"].duplicated().sum())
        if dups:
            res.anomalies.append(f"{name}_duplicate_window_id:{dups}")

    cand_ids = set(cand["window_id"])
    s5_ids = set(s5["window_id"])
    intersect = cand_ids & s5_ids
    if len(intersect) != len(cand):
        res.anomalies.append(
            f"candidate_availability_window_id_mismatch:intersect={len(intersect)}"
            f":candidates={len(cand)}"
        )

    # ---- inner join on window_id (stable key) ----
    keep_s5 = [
        "window_id", "patient_id", "hr_available", "spo2_available",
        "airflow_available", "core_hr_spo2_available", "audio_features_present",
    ]
    keep_s5 = [c for c in keep_s5 if c in s5.columns]
    merged = cand.merge(s5[keep_s5], on="window_id", how="inner", suffixes=("", "_s5"))

    # cross-patient guard: the window_id embeds patient_id; both sides must agree.
    if "patient_id_s5" in merged.columns:
        bad_patient = int((merged["patient_id"] != merged["patient_id_s5"]).sum())
        if bad_patient:
            res.anomalies.append(f"cross_patient_window_id:{bad_patient}")
        merged = merged.drop(columns=["patient_id_s5"])

    # time-bound equality (Stage-4 vs Stage-5 clock must agree -> leakage signal if not)
    for s4col, s5col in (
        ("start_relative_to_record_start", "window_start_relative_to_record_start"),
        ("end_relative_to_record_start", "window_end_relative_to_record_start"),
    ):
        if s5col in merged.columns and s4col in merged.columns:
            diff = (pd.to_numeric(merged[s4col], errors="coerce")
                    - pd.to_numeric(merged[s5col], errors="coerce")).abs()
            nbad = int((diff > _TIME_TOL).sum())
            if nbad:
                res.anomalies.append(f"time_bounds_mismatch:{s4col}:{nbad}")

    # Stage-4 excluded must be ABSENT from the merged candidate set.
    if "label_status" in merged.columns:
        unknown = set(merged["label_status"].dropna().astype(str).unique()) - set(CANDIDATE_STATUSES)
        if unknown:
            res.anomalies.append(f"non_candidate_in_merged:{sorted(unknown)}")

    # ---- build membership rows ----
    core_rows: List[CohortMembershipRow] = []
    airflow_rows: List[CohortMembershipRow] = []
    for _, m in merged.iterrows():
        pid = str(m["patient_id"])
        widx = int(m["window_index"])
        wid = make_window_id(pid, widx)
        status = str(m["label_status"])
        label = 1 if status == LABEL_POSITIVE else 0
        hr_av = bool(m.get("hr_available", False))
        sp_av = bool(m.get("spo2_available", False))
        af_av = bool(m.get("airflow_available", False))
        core_av = bool(m.get("core_hr_spo2_available", hr_av and sp_av))
        is_core = core_av and (status in CANDIDATE_STATUSES)
        is_airflow = is_core and af_av

        if is_core:
            row = CohortMembershipRow(
                run_id=run_id, window_id=wid, patient_id=pid, window_index=widx,
                binary_event_label=label, label_status=status,
                window_start_relative_to_record_start=float(m["start_relative_to_record_start"]),
                window_end_relative_to_record_start=float(m["end_relative_to_record_start"]),
                duration_seconds=float(m["duration_seconds"]),
                hr_available=hr_av, spo2_available=sp_av, airflow_available=af_av,
                core_hr_spo2_available=core_av, cohort_core=True, cohort_airflow=is_airflow,
                audio_features_present=AUDIO_FEATURES_PRESENT,
            )
            core_rows.append(row)
            if is_airflow:
                airflow_rows.append(row)

            # exclusions for non-core core candidates (HR / SpO2 quality) and
            # airflow optional-missing (core windows lacking airflow).
            if not hr_av:
                res.exclusions.append(CohortExclusion(
                    run_id=run_id, window_id=wid, patient_id=pid, window_index=widx,
                    scope=SCOPE_HR_QUALITY_INSUFFICIENT,
                    reason="hr_feature_quality_insufficient", detail="hr_available=False",
                ))
            if not sp_av:
                res.exclusions.append(CohortExclusion(
                    run_id=run_id, window_id=wid, patient_id=pid, window_index=widx,
                    scope=SCOPE_SPO2_QUALITY_INSUFFICIENT,
                    reason="spo2_feature_quality_insufficient", detail="spo2_available=False",
                ))
            if not af_av:
                res.exclusions.append(CohortExclusion(
                    run_id=run_id, window_id=wid, patient_id=pid, window_index=widx,
                    scope=SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE,
                    reason="airflow_optional_not_available",
                    detail="airflow unavailable; window stays in core cohort only",
                ))
        else:
            # candidate window not in core cohort -> HR and/or SpO2 quality insufficient.
            why = []
            if not hr_av:
                why.append("hr_available=False")
            if not sp_av:
                why.append("spo2_available=False")
            res.exclusions.append(CohortExclusion(
                run_id=run_id, window_id=wid, patient_id=pid, window_index=widx,
                scope=SCOPE_HR_QUALITY_INSUFFICIENT if not hr_av else SCOPE_SPO2_QUALITY_INSUFFICIENT,
                reason="core_hr_spo2_feature_quality_insufficient",
                detail=";".join(why) or "core_unavailable",
            ))

    res.core_rows = core_rows
    res.airflow_rows = airflow_rows

    # ---- dataframes for the splitter (keys + label ONLY) ----
    if core_rows:
        cdf = pd.DataFrame([
            {"window_id": r.window_id, "patient_id": r.patient_id,
             "binary_event_label": r.binary_event_label}
            for r in core_rows
        ])
        # deterministic order: sort by patient_id then window_id
        cdf = cdf.sort_values(["patient_id", "window_id"]).reset_index(drop=True)
    else:
        cdf = pd.DataFrame(columns=["window_id", "patient_id", "binary_event_label"])
    if airflow_rows:
        adf = pd.DataFrame([
            {"window_id": r.window_id, "patient_id": r.patient_id,
             "binary_event_label": r.binary_event_label}
            for r in airflow_rows
        ])
        adf = adf.sort_values(["patient_id", "window_id"]).reset_index(drop=True)
    else:
        adf = pd.DataFrame(columns=["window_id", "patient_id", "binary_event_label"])
    _assert_no_forbidden_columns(cdf, "core splitter frame")
    _assert_no_forbidden_columns(adf, "airflow splitter frame")
    res.core_df = cdf
    res.airflow_df = adf

    res.patient_burden = _burden(cdf)
    res.airflow_patient_burden = _burden(adf)

    # ---- airflow patient subset-of-core assertion (inheritance precondition) ----
    af_patients = set(res.airflow_patient_burden.keys())
    core_patients = set(res.patient_burden.keys())
    if af_patients and not af_patients.issubset(core_patients):
        res.anomalies.append(
            f"airflow_patients_not_subset_of_core:{sorted(af_patients - core_patients)}"
        )

    return res


def summary_fillers(res: CohortResult, *, s4: pd.DataFrame, s5: pd.DataFrame) -> Dict[str, Any]:
    """Compute the SplitSummary count fields (incl. the airflow discrepancy numbers)."""
    core = res.core_rows
    af = res.airflow_rows
    burden = list(res.patient_burden.values())
    rates = [b["pos_rate"] for b in burden] if burden else [0.0]
    wins = [b["n_windows"] for b in burden] if burden else [0.0]

    # airflow "available-alone" reference (matches the prompt's 28,409 baseline):
    # among ALL candidate rows, windows/patients with airflow_available=True.
    af_avail_mask = s5["airflow_available"].astype(bool) if "airflow_available" in s5.columns else pd.Series([False] * len(s5))
    af_avail_windows = int(af_avail_mask.sum())
    af_avail_patients = int(s5.loc[af_avail_mask, "patient_id"].nunique()) if af_avail_windows else 0

    exclusions_by_scope: Dict[str, int] = {}
    for e in res.exclusions:
        exclusions_by_scope[e.scope] = exclusions_by_scope.get(e.scope, 0) + 1

    return {
        "total_stage4_windows": int(len(s4)),
        "total_candidate_windows": int((s4["label_status"].isin(list(CANDIDATE_STATUSES))).sum()) if "label_status" in s4.columns else 0,
        "total_excluded_stage4_windows": int((s4["label_status"] == LABEL_EXCLUDED).sum()) if "label_status" in s4.columns else 0,
        "core_patients": len(res.patient_burden),
        "core_windows": len(core),
        "core_positive": sum(1 for r in core if r.binary_event_label == 1),
        "core_negative": sum(1 for r in core if r.binary_event_label == 0),
        "airflow_patients": len(res.airflow_patient_burden),
        "airflow_windows": len(af),
        "airflow_positive": sum(1 for r in af if r.binary_event_label == 1),
        "airflow_negative": sum(1 for r in af if r.binary_event_label == 0),
        "airflow_available_alone_windows": af_avail_windows,
        "airflow_available_alone_patients": af_avail_patients,
        "core_windows_per_patient_min": int(min(wins)) if wins else 0,
        "core_windows_per_patient_median": float(np.median(wins)) if wins else 0.0,
        "core_windows_per_patient_max": int(max(wins)) if wins else 0,
        "core_pos_rate_per_patient_min": float(min(rates)) if rates else 0.0,
        "core_pos_rate_per_patient_median": float(np.median(rates)) if rates else 0.0,
        "core_pos_rate_per_patient_max": float(max(rates)) if rates else 0.0,
        "exclusions_by_scope": exclusions_by_scope,
    }


__all__ = [
    "CohortResult",
    "make_window_id",
    "build_cohorts",
    "summary_fillers",
]
