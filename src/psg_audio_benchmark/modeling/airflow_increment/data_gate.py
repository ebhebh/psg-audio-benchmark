"""Stage-8 input data gate: the frozen approved v2 split + the airflow sub-cohort.

This is the single chokepoint that decides whether the airflow-incremental
comparison may proceed. It runs in two layers and refuses with a loud
:class:`AirflowGateError` (BLOCKED) on any failure -- **never a silent fallback**
to v1, to a self-built split, or to the Stage-7 main cohort:

Layer 1 -- re-use the Stage-7 :func:`modeling.data_gate.verify_gate` so the full
v2 signature / cohort / fold / lineage verification happens unchanged (license
PASSED; LATEST_RUN -> v2; approval_status; SHA-256/rows/schema; lineage run ids;
runtime core-cohort re-derivation; assignment + patient-set hashes recomputed;
each patient one outer fold; inner folds exclude outer-test patients).

Layer 2 -- Stage-8 specific:

1. the airflow feature file ``airflow_window_features.parquet`` exists and
   carries all 11 allowed airflow statistics;
2. the airflow-eligible cohort is **re-derived at runtime** from the Stage-4
   labels + Stage-5 availability (HR & SpO2 & airflow ALL available,
   positive/negative windows) -- counts are reported but **never hard-coded**;
3. the airflow patients are a subset of the v2 core patients and INHERIT their
   v2 outer-fold assignment (0 mismatch); no independent / random / re-balanced
   split is built;
4. within each outer fold, the inner folds of the airflow outer-train patients
   contain no outer-test airflow patient (verified independently of the adapter).

The gate reads NO raw data and NO ``*.wav`` (audio BLOCKED).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...config import Config
from . import schema as S
from ..data_gate import (
    Check,
    DataGateError,
    DataGateResult,
    _canonical_pid,
    _make_window_id,
    verify_gate,
)


class AirflowGateError(RuntimeError):
    """Raised when the Stage-8 input gate refuses (BLOCKED)."""


@dataclass
class AirflowGateResult:
    approved: bool
    split_run_id: str
    signature: Dict[str, Any]
    outer_folds: pd.DataFrame
    inner_folds: pd.DataFrame
    core_cohort_windows: pd.DataFrame      # the 50-patient v2 core cohort
    airflow_cohort_windows: pd.DataFrame   # the 34-patient airflow sub-cohort
    airflow_cohort: Dict[str, Any]
    checks: List[Check] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# airflow cohort re-derivation
# ---------------------------------------------------------------------------

def _rederive_airflow_cohort(
    *, stage4_index: pd.DataFrame, stage5_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Re-derive the airflow cohort: HR & SpO2 & airflow ALL available, label in
    {positive, negative}. Returns window_id, patient_id, binary_event_label."""
    idx = stage4_index[["patient_id", "window_index", "label_status",
                        "binary_event_label"]].copy()
    idx["window_id"] = [
        _make_window_id(p, i) for p, i in zip(idx["patient_id"], idx["window_index"])
    ]
    av = stage5_availability[
        ["window_id", "core_hr_spo2_available", "airflow_available"]
    ].copy()
    merged = idx.merge(av, on="window_id", how="left")
    for c in ("core_hr_spo2_available", "airflow_available"):
        col = merged[c]
        merged[c] = np.where(col.isna(), False, col).astype(bool)
    air = merged[
        merged["label_status"].isin(("positive", "negative"))
        & merged["core_hr_spo2_available"]
        & merged["airflow_available"]
    ].copy()
    air["binary_event_label"] = air["binary_event_label"].astype(int)
    air["patient_id"] = air["patient_id"].map(_canonical_pid)
    return air[["window_id", "patient_id", "binary_event_label"]].reset_index(drop=True)


def _cohort_fingerprint(cohort_windows: pd.DataFrame) -> Dict[str, Any]:
    n_windows = int(len(cohort_windows))
    n_patients = int(cohort_windows["patient_id"].nunique())
    n_positive = int((cohort_windows["binary_event_label"] == 1).sum())
    n_negative = int((cohort_windows["binary_event_label"] == 0).sum())
    return {
        "n_windows": n_windows,
        "n_patients": n_patients,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "overall_positive_rate": (n_positive / n_windows) if n_windows else None,
        "patient_set": sorted(str(p) for p in cohort_windows["patient_id"].unique()),
    }


# ---------------------------------------------------------------------------
# main gate
# ---------------------------------------------------------------------------

def verify_airflow_gate(
    *, cfg: Config, config, split_run_id_override: Optional[str] = None,
) -> AirflowGateResult:
    """Run the Stage-8 input gate. Raises :class:`AirflowGateError` if refused."""
    checks: List[Check] = []

    # ---- Layer 1: re-use the full Stage-7 v2 gate -----------------------
    try:
        core_gate: DataGateResult = verify_gate(
            cfg=cfg, config=config, split_run_id_override=split_run_id_override,
        )
    except DataGateError as exc:
        # surface the v2 failure as a Stage-8 BLOCKED
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED at the v2 layer:\n" + str(exc)
        ) from exc
    # carry the v2 checks forward (all passed here)
    checks.extend(core_gate.checks)

    split_run_id = core_gate.split_run_id
    signature = core_gate.signature
    outer_folds = core_gate.outer_folds
    inner_folds = core_gate.inner_folds
    core_cohort = core_gate.cohort_windows

    # ---- Layer 2: Stage-8 airflow checks --------------------------------
    # 1. airflow feature file exists + carries the 11 statistics
    af_path = cfg.path("features_physiology") / "airflow_window_features.parquet"
    if not af_path.is_file():
        raise AirflowGateError(
            f"STAGE-8 INPUT GATE BLOCKED: airflow feature file missing: {af_path}"
        )
    af_df = pd.read_parquet(af_path)
    missing_air_stats = [c for c in S.AIRFLOW_STATISTICS if c not in af_df.columns]
    checks.append(Check(
        "airflow_feature_file_present",
        not missing_air_stats,
        f"{af_path.name}: present=True; missing statistics={missing_air_stats}",
    ))
    if missing_air_stats:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: airflow feature file is missing allowed "
            f"statistics {missing_air_stats}."
        )

    # 2. re-derive the airflow cohort at runtime (never hard-coded)
    stage4_path = cfg.path("data_manifests") / "csv_window_index.parquet"
    av_path = cfg.path("features_physiology") / "physiology_feature_availability.parquet"
    if not stage4_path.is_file() or not av_path.is_file():
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: Stage-4 window index / Stage-5 "
            "availability input missing."
        )
    stage4_index = pd.read_parquet(stage4_path)
    stage5_av = pd.read_parquet(av_path)
    airflow_cohort = _rederive_airflow_cohort(
        stage4_index=stage4_index, stage5_availability=stage5_av,
    )
    fp = _cohort_fingerprint(airflow_cohort)
    checks.append(Check(
        "airflow_cohort_rederived_nonempty",
        fp["n_windows"] > 0 and fp["n_patients"] > 0,
        f"runtime airflow cohort: {fp['n_windows']} windows / "
        f"{fp['n_patients']} patients / pos={fp['n_positive']} "
        f"neg={fp['n_negative']} (re-derived, NOT hard-coded)",
    ))
    if fp["n_windows"] == 0 or fp["n_patients"] == 0:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: re-derived airflow cohort is empty."
        )

    # 3. airflow patients are a SUBSET of the v2 core patients
    core_pids = set(str(p) for p in core_cohort["patient_id"].unique())
    air_pids = set(fp["patient_set"])
    not_in_core = sorted(air_pids - core_pids)
    checks.append(Check(
        "airflow_patients_subset_of_core_v2",
        not not_in_core,
        f"airflow patients not in v2 core: {not_in_core}",
    ))
    if not_in_core:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: airflow patients not in the v2 core "
            f"cohort: {not_in_core}."
        )

    # 4. airflow patients INHERIT their v2 outer-fold assignment (0 mismatch)
    outer_pid_fold = dict(zip(outer_folds["patient_id"], outer_folds["outer_fold"]))
    air_pid_fold = {p: outer_pid_fold.get(p) for p in air_pids}
    missing_outer = sorted(p for p, f in air_pid_fold.items() if f is None)
    checks.append(Check(
        "airflow_inherits_v2_outer_folds",
        not missing_outer,
        f"airflow patients without a v2 outer fold: {missing_outer}",
    ))
    if missing_outer:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: airflow patients with no inherited v2 "
            f"outer fold: {missing_outer}."
        )
    # airflow patients per outer fold (report only; counts NOT validated vs a
    # hard-coded number)
    per_fold = {
        int(k): sorted(p for p, f in air_pid_fold.items() if int(f) == int(k))
        for k in sorted(set(outer_pid_fold.values()))
    }
    checks.append(Check(
        "airflow_patients_per_outer_fold",
        True,
        "; ".join(f"fold{k}={len(v)}" for k, v in sorted(per_fold.items())),
    ))

    # 5. inner folds of the airflow outer-train exclude the outer-test airflow
    #    patients (independent of the fold adapter).
    pid_to_outer = {str(p): int(f) for p, f in zip(outer_folds["patient_id"],
                                                   outer_folds["outer_fold"])}
    inner_folds_chk = inner_folds.copy()
    inner_folds_chk["patient_id"] = inner_folds_chk["patient_id"].map(_canonical_pid)
    leaks: List[str] = []
    for outer_k in sorted(pid_to_outer.values()):
        air_test = set(p for p, f in pid_to_outer.items()
                       if f == outer_k and p in air_pids)
        rows = inner_folds_chk[
            (inner_folds_chk["outer_fold"] == outer_k)
            & (inner_folds_chk["patient_id"].isin(air_pids))
        ]
        bad = sorted(set(rows["patient_id"]) & air_test)
        if bad:
            leaks.append(f"outer_fold={outer_k}: {len(bad)} airflow outer-test "
                         f"patient(s) leaked into inner folds")
    checks.append(Check(
        "airflow_inner_folds_exclude_outer_test",
        not leaks,
        "; ".join(leaks) if leaks else "no airflow outer-test patient in any "
        "airflow inner fold",
    ))
    if leaks:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED: airflow inner-fold leakage: " + "; ".join(leaks)
        )

    # ---- decision ----
    reasons = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    approved = all(c.passed for c in checks)
    if not approved:
        raise AirflowGateError(
            "STAGE-8 INPUT GATE BLOCKED. Failures:\n  - " + "\n  - ".join(reasons)
        )
    return AirflowGateResult(
        approved=True,
        split_run_id=split_run_id,
        signature=signature,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        core_cohort_windows=core_cohort,
        airflow_cohort_windows=airflow_cohort,
        airflow_cohort=fp,
        checks=checks,
        reasons=[],
    )


__all__ = [
    "AirflowGateError",
    "AirflowGateResult",
    "verify_airflow_gate",
]
