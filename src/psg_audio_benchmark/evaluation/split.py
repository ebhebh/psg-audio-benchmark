"""Stage-6 patient-level nested cross-validation assignment.

Patient-level, leakage-safe, deterministic:

* **Outer** : ``StratifiedGroupKFold`` (``n_splits=outer``, ``shuffle=True``,
  fixed seed) over core-cohort *windows*, stratifying on each window's
  ``binary_event_label`` while grouping by ``patient_id``. Patients are sorted by
  ``str(patient_id)`` ascending before every ``split()`` call so the assignment
  is deterministic regardless of parquet row order. Every window of a patient
  lands in exactly one outer test fold.
* **Inner** : within each outer fold, a second ``StratifiedGroupKFold``
  (``n_splits=inner``, fixed seed) over the **outer-train** patients only. Each
  outer-train patient gets exactly one inner validation fold. **Outer-test
  patients never appear in any inner fold** (asserted).
* **Airflow** : each airflow patient inherits its core outer fold; there is no
  independent airflow split.

Balance is reported honestly: per-fold train/test patient & window counts,
positive/negative counts and positive rate, with a warning when the test-fold
positive-rate spread exceeds the configured threshold. Patients are never moved
between folds to improve ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "scikit-learn is required for Stage-6 patient splits. "
        "Install the cpu_science extra: python -m pip install "
        "\"scikit-learn>=1.3\""
    ) from _exc

from .schema import (
    AirflowInheritanceRow,
    InnerPatientFold,
    OuterPatientFold,
    ResolvedSplitsConfig,
)


@dataclass
class SplitResult:
    outer_rows: List[OuterPatientFold] = field(default_factory=list)
    inner_rows: List[InnerPatientFold] = field(default_factory=list)
    airflow_rows: List[AirflowInheritanceRow] = field(default_factory=list)
    patient_to_outer: Dict[str, int] = field(default_factory=dict)
    outer_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    inner_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    airflow_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    balance_warnings: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)


def _patient_window_stats(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Per-patient {n, pos, neg} over a (window_id, patient_id, label) frame."""
    out: Dict[str, Dict[str, int]] = {}
    if df.empty:
        return out
    for pid, sub in df.groupby("patient_id"):
        n = int(len(sub))
        pos = int((sub["binary_event_label"] == 1).sum())
        out[str(pid)] = {"n": n, "pos": pos, "neg": n - pos}
    return out


def _aggregate(patients: Set[str], stats: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    n = sum(stats[p]["n"] for p in patients if p in stats)
    pos = sum(stats[p]["pos"] for p in patients if p in stats)
    return {"n_windows": n, "n_positive": pos, "n_negative": n - pos,
            "positive_rate": float(pos / n) if n else 0.0}


def _stratified_group_folds(
    df: pd.DataFrame, *, n_splits: int, seed: int
) -> Dict[str, int]:
    """Return patient_id -> test-fold index via StratifiedGroupKFold.

    ``df`` must already be sorted (patient_id ascending) for determinism.
    """
    if df.empty:
        return {}
    n_patients = int(df["patient_id"].nunique())
    if n_patients < n_splits:
        raise ValueError(
            f"Cannot build {n_splits} folds from {n_patients} patient(s); "
            f"need >= {n_splits} groups."
        )
    X = np.zeros(len(df), dtype=np.int8)
    y = df["binary_event_label"].to_numpy(dtype=np.int8)
    groups = df["patient_id"].to_numpy(dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_to_fold: Dict[str, int] = {}
    for fold_i, (_train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        test_groups = {str(g) for g in groups[test_idx]}
        for g in test_groups:
            # each group appears in exactly one test fold (SGKFold guarantee)
            patient_to_fold[g] = fold_i
    return patient_to_fold


def assign_patient_splits(
    *,
    core_df: pd.DataFrame,
    airflow_df: pd.DataFrame,
    patient_burden: Dict[str, Dict[str, float]],
    config: ResolvedSplitsConfig,
    run_id: str,
) -> SplitResult:
    """Assign patient-level outer + inner folds (core) and airflow inheritance."""
    res = SplitResult()

    # core_df is already sorted by (patient_id, window_id) in cohort.py; re-sort
    # defensively to guarantee deterministic group order for SGKFold.
    core = core_df.sort_values(["patient_id", "window_id"]).reset_index(drop=True) if not core_df.empty else core_df

    core_stats = _patient_window_stats(core)
    all_patients: Set[str] = set(core_stats.keys())

    # ---- OUTER: patient -> outer test fold ----
    try:
        patient_to_outer = _stratified_group_folds(
            core, n_splits=config.outer_n_folds, seed=config.seed
        )
    except ValueError as exc:
        res.anomalies.append(f"outer_split_failed:{exc}")
        return res
    res.patient_to_outer = patient_to_outer

    # invariant: every core patient in exactly one outer fold
    missing = all_patients - set(patient_to_outer.keys())
    extra = set(patient_to_outer.keys()) - all_patients
    if missing or extra:
        res.anomalies.append(
            f"outer_fold_assignment_incomplete:missing={sorted(missing)}:extra={sorted(extra)}"
        )

    outer_rows: List[OuterPatientFold] = []
    for pid in sorted(all_patients):
        st = core_stats[pid]
        n, pos = st["n"], st["pos"]
        outer_rows.append(OuterPatientFold(
            run_id=run_id, patient_id=pid, outer_fold=int(patient_to_outer[pid]),
            n_windows=n, n_positive=pos, n_negative=n - pos,
            positive_rate=float(pos / n) if n else 0.0,
        ))
    res.outer_rows = outer_rows

    # ---- per-outer-fold train/test stats ----
    test_pos_rates: List[float] = []
    outer_fold_stats: List[Dict[str, Any]] = []
    for f in range(config.outer_n_folds):
        test_p = {p for p in all_patients if patient_to_outer.get(p) == f}
        train_p = all_patients - test_p
        test_agg = _aggregate(test_p, core_stats)
        train_agg = _aggregate(train_p, core_stats)
        test_pos_rates.append(test_agg["positive_rate"])
        # disjointness + completeness
        if test_p & train_p:
            res.anomalies.append(f"outer_fold_{f}_train_test_patient_overlap")
        outer_fold_stats.append({
            "outer_fold": f,
            "n_train_patients": len(train_p), "n_test_patients": len(test_p),
            "n_train_windows": train_agg["n_windows"], "n_test_windows": test_agg["n_windows"],
            "train_positive": train_agg["n_positive"], "train_negative": train_agg["n_negative"],
            "train_positive_rate": train_agg["positive_rate"],
            "test_positive": test_agg["n_positive"], "test_negative": test_agg["n_negative"],
            "test_positive_rate": test_agg["positive_rate"],
        })
    res.outer_fold_stats = outer_fold_stats

    # ---- INNER: per outer fold, SGKFold over outer-TRAIN patients only ----
    inner_rows: List[InnerPatientFold] = []
    inner_fold_stats: List[Dict[str, Any]] = []
    for f in range(config.outer_n_folds):
        outer_test_p = {p for p in all_patients if patient_to_outer.get(p) == f}
        outer_train_p = all_patients - outer_test_p
        train_core = core[core["patient_id"].isin(outer_train_p)].sort_values(
            ["patient_id", "window_id"]
        ).reset_index(drop=True)
        try:
            inner_assign = _stratified_group_folds(
                train_core, n_splits=config.inner_n_folds, seed=config.inner_seed + f
            )
        except ValueError as exc:
            res.anomalies.append(f"inner_split_failed_outer{f}:{exc}")
            continue

        inner_patients = set(inner_assign.keys())
        # load-bearing leakage guards
        if inner_patients != outer_train_p:
            res.anomalies.append(
                f"inner_patient_set_mismatch_outer{f}:"
                f"only_inner={sorted(inner_patients - outer_train_p)}:"
                f"only_outer_train={sorted(outer_train_p - inner_patients)}"
            )
        if outer_test_p & inner_patients:
            res.anomalies.append(
                f"outer_test_in_inner_outer{f}:{sorted(outer_test_p & inner_patients)}"
            )

        for pid in sorted(outer_train_p):
            st = core_stats[pid]
            n, pos = st["n"], st["pos"]
            inner_rows.append(InnerPatientFold(
                run_id=run_id, outer_fold=f, patient_id=pid,
                inner_validation_fold=int(inner_assign.get(pid, -1)),
                n_windows=n, n_positive=pos, n_negative=n - pos,
                positive_rate=float(pos / n) if n else 0.0,
            ))

        train_core_stats = _patient_window_stats(train_core)
        for i in range(config.inner_n_folds):
            val_p = {p for p in outer_train_p if inner_assign.get(p) == i}
            inner_train_p = outer_train_p - val_p
            val_agg = _aggregate(val_p, train_core_stats)
            itrain_agg = _aggregate(inner_train_p, train_core_stats)
            if val_p & inner_train_p:
                res.anomalies.append(f"inner_outer{f}_fold{i}_train_val_overlap")
            inner_fold_stats.append({
                "outer_fold": f, "inner_fold": i,
                "n_inner_train_patients": len(inner_train_p),
                "n_inner_val_patients": len(val_p),
                "n_inner_train_windows": itrain_agg["n_windows"],
                "n_inner_val_windows": val_agg["n_windows"],
                "inner_train_positive": itrain_agg["n_positive"],
                "inner_train_negative": itrain_agg["n_negative"],
                "inner_train_positive_rate": itrain_agg["positive_rate"],
                "inner_val_positive": val_agg["n_positive"],
                "inner_val_negative": val_agg["n_negative"],
                "inner_val_positive_rate": val_agg["positive_rate"],
            })
    res.inner_rows = inner_rows
    res.inner_fold_stats = inner_fold_stats

    # ---- AIRFLOW inheritance (no independent split) ----
    airflow = airflow_df.sort_values(["patient_id", "window_id"]).reset_index(drop=True) if not airflow_df.empty else airflow_df
    af_stats = _patient_window_stats(airflow)
    airflow_rows: List[AirflowInheritanceRow] = []
    airflow_fold_stats: List[Dict[str, Any]] = []
    for pid in sorted(af_stats.keys()):
        if pid not in patient_to_outer:
            res.anomalies.append(f"airflow_patient_missing_outer_fold:{pid}")
            continue
        st = af_stats[pid]
        n, pos = st["n"], st["pos"]
        airflow_rows.append(AirflowInheritanceRow(
            run_id=run_id, patient_id=pid, outer_fold=int(patient_to_outer[pid]),
            n_airflow_windows=n, n_airflow_positive=pos, n_airflow_negative=n - pos,
            airflow_positive_rate=float(pos / n) if n else 0.0,
        ))
    res.airflow_rows = airflow_rows
    # per-fold airflow availability (inherited outer fold)
    for f in range(config.outer_n_folds):
        fold_af_p = {p for p in af_stats.keys() if patient_to_outer.get(p) == f}
        agg = _aggregate(fold_af_p, af_stats)
        airflow_fold_stats.append({
            "outer_fold": f,
            "n_airflow_patients": len(fold_af_p),
            "n_airflow_windows": agg["n_windows"],
            "airflow_positive": agg["n_positive"],
            "airflow_negative": agg["n_negative"],
            "airflow_positive_rate": agg["positive_rate"],
        })
    res.airflow_fold_stats = airflow_fold_stats

    # ---- balance warnings (honest; never fake balance) ----
    if test_pos_rates:
        spread = max(test_pos_rates) - min(test_pos_rates)
        if spread > config.warn_pos_rate_spread_threshold:
            res.balance_warnings.append(
                f"outer_test_pos_rate_spread={spread:.4f} exceeds threshold "
                f"{config.warn_pos_rate_spread_threshold:.4f} "
                f"(per-fold test pos_rate={[round(r,4) for r in test_pos_rates]}); "
                f"patient isolation preserved, imbalance reported not corrected."
            )
        for stt in outer_fold_stats:
            if stt["n_test_patients"] == 0:
                res.balance_warnings.append(f"outer_fold_{stt['outer_fold']}_has_no_test_patients")
            if stt["n_test_windows"] and (stt["test_positive"] == 0 or stt["test_negative"] == 0):
                res.balance_warnings.append(
                    f"outer_fold_{stt['outer_fold']}_test_single_class"
                    f"(pos={stt['test_positive']},neg={stt['test_negative']})"
                )
    if len(all_patients) < config.outer_n_folds * 2:
        res.balance_warnings.append(
            f"small_cohort:{len(all_patients)}_patients_for_{config.outer_n_folds}_folds; "
            f"stratification is approximate, not forced."
        )

    return res


__all__ = ["SplitResult", "assign_patient_splits"]
