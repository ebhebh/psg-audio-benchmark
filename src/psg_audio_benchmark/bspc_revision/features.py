"""Stage 15b — feature-family assembly and physiological-latency (lag) shifting.

Families (pre-registered): ``hr_only``, ``spo2_only``, ``hr_spo2`` (primary),
``simple_spo2_min`` (univariate), and the airflow-enhanced family for the shared-source
secondary analysis.

Latency: ``lag_k`` shifts the FEATURE context +k windows (k*30 s) relative to the label
window, i.e. it uses the signal from [start+k*30, start+(k+1)*30) while keeping the label of
the original window. lag_0 is causal/online (MAIN); lag_1 (+30 s) and lag_2 (+60 s) use future
signal -> OFFLINE / context sensitivity. A label window whose shifted feature window does not
exist in the feature grid is dropped for that branch and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

STATS = ["mean", "median", "std_ddof1", "min", "max", "range", "iqr", "slope_per_second"]
AIRFLOW_EXTRA = ["rms", "zero_crossing_count", "zero_crossing_rate"]

FAMILIES = {
    "hr_only": [f"hr_{s}" for s in STATS],
    "spo2_only": [f"spo2_{s}" for s in STATS],
    "hr_spo2": [f"hr_{s}" for s in STATS] + [f"spo2_{s}" for s in STATS],
    "simple_spo2_min": ["spo2_min"],
}
AIRFLOW_FAMILY = [f"airflow_{s}" for s in STATS] + [f"airflow_{s}" for s in AIRFLOW_EXTRA]


def _prefixed(df: pd.DataFrame, modality: str, stats: List[str]) -> pd.DataFrame:
    out = df[["window_id"]].copy()
    for s in stats:
        if s in df.columns:
            out[f"{modality}_{s}"] = df[s]
    return out


def build_base(
    *, hr: pd.DataFrame, spo2: pd.DataFrame, airflow: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge prefixed HR + SpO2 (+ optional airflow) features on window_id (ALL candidate
    windows, not just cohort)."""
    base = _prefixed(hr, "hr", STATS).merge(_prefixed(spo2, "spo2", STATS), on="window_id", how="outer")
    if airflow is not None:
        base = base.merge(_prefixed(airflow, "airflow", STATS + AIRFLOW_EXTRA), on="window_id", how="outer")
    return base


@dataclass(frozen=True)
class FeatureMatrix:
    window_id: np.ndarray        # (n,) cohort window ids in cohort order
    patient_id: np.ndarray       # (n,)
    columns: List[str]
    X: np.ndarray                # (n, p) float; NaN where missing (imputer handles fold-internally)
    n_dropped_missing_shift: int # lag attrition (0 for lag_0)


def _index(window_id: str) -> int:
    return int(str(window_id).split("-")[1])


def _prefix(window_id: str) -> str:
    return str(window_id).split("-")[0]


def assemble(
    *, cohort: pd.DataFrame, base: pd.DataFrame, family: str, lag_windows: int = 0,
) -> FeatureMatrix:
    """Assemble one feature family at one lag for the cohort (in cohort row order)."""
    cols = FAMILIES[family]
    # build pid -> {idx: window_id} over ALL base feature windows (robust to cohort gaps)
    base_idx = base["window_id"].astype(str).to_numpy()
    idx_of = {_prefix(w): {} for w in base_idx}
    wid_to_row: Dict[str, int] = {}
    for r, w in enumerate(base_idx):
        idx_of[_prefix(w)][_index(w)] = w
        wid_to_row[w] = r

    coh_wid = cohort["window_id"].astype(str).to_numpy()
    coh_pid = cohort["patient_id"].astype(str).to_numpy()
    keep_mask = np.ones(len(cohort), dtype=bool)
    src_wid = np.empty(len(cohort), dtype=object)
    for i, w in enumerate(coh_wid):
        if lag_windows == 0:
            src_wid[i] = w
            continue
        pid = _prefix(w)
        shifted = _index(w) + lag_windows
        mmap = idx_of.get(pid, {})
        if shifted in mmap:
            src_wid[i] = mmap[shifted]
        else:
            keep_mask[i] = False
            src_wid[i] = w  # placeholder, dropped below

    keep_idx = np.nonzero(keep_mask)[0]
    src_lookup = np.array([wid_to_row.get(w, -1) for w in src_wid[keep_idx]], dtype=int)
    valid = src_lookup >= 0
    keep_idx = keep_idx[valid]
    src_lookup = src_lookup[valid]
    base_arr = base[cols].to_numpy(dtype=float)
    X = base_arr[src_lookup]
    n_dropped = int(len(cohort) - len(keep_idx))
    return FeatureMatrix(
        window_id=coh_wid[keep_idx],
        patient_id=coh_pid[keep_idx],
        columns=list(cols),
        X=X,
        n_dropped_missing_shift=n_dropped,
    )


def assemble_airflow_enhanced(
    *, cohort_airflow: pd.DataFrame, base: pd.DataFrame, lag_windows: int = 0,
) -> FeatureMatrix:
    """Airflow-enhanced family (hr_spo2 + airflow) for the shared-source secondary analysis."""
    cols = FAMILIES["hr_spo2"] + AIRFLOW_FAMILY
    return _assemble_cols(cohort_airflow, base, cols, lag_windows)


def _assemble_cols(cohort: pd.DataFrame, base: pd.DataFrame, cols: List[str], lag_windows: int) -> FeatureMatrix:
    base_idx = base["window_id"].astype(str).to_numpy()
    idx_of = {_prefix(w): {} for w in base_idx}
    wid_to_row: Dict[str, int] = {}
    for r, w in enumerate(base_idx):
        idx_of[_prefix(w)][_index(w)] = w
        wid_to_row[w] = r
    coh_wid = cohort["window_id"].astype(str).to_numpy()
    coh_pid = cohort["patient_id"].astype(str).to_numpy()
    keep_mask = np.ones(len(cohort), dtype=bool)
    src_wid = np.empty(len(cohort), dtype=object)
    for i, w in enumerate(coh_wid):
        if lag_windows == 0:
            src_wid[i] = w
            continue
        pid = _prefix(w)
        shifted = _index(w) + lag_windows
        mmap = idx_of.get(pid, {})
        src_wid[i] = mmap[shifted] if shifted in mmap else (keep_mask.__setitem__(i, False) or w)
    keep_idx = np.nonzero(keep_mask)[0]
    src_lookup = np.array([wid_to_row.get(w, -1) for w in src_wid[keep_idx]], dtype=int)
    valid = src_lookup >= 0
    keep_idx = keep_idx[valid]
    src_lookup = src_lookup[valid]
    X = base[cols].to_numpy(dtype=float)[src_lookup]
    return FeatureMatrix(
        window_id=coh_wid[keep_idx], patient_id=coh_pid[keep_idx], columns=list(cols),
        X=X, n_dropped_missing_shift=int(len(cohort) - len(keep_idx)),
    )


__all__ = [
    "STATS", "AIRFLOW_EXTRA", "FAMILIES", "AIRFLOW_FAMILY", "build_base",
    "FeatureMatrix", "assemble", "assemble_airflow_enhanced",
]
