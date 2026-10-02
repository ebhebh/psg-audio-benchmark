"""Stage 15b — duration-overlap label derivation.

Re-derives window labels from the FROZEN Stage-3 parsed events + canonical awake
intervals and the Stage-4 window/time contract. The historical Stage-4 start-point
label is **never overwritten**: this module emits new label tables isolated under the
Stage-15b run dir.

Label variants (pre-registered, fixed before any re-run):

* ``main``  -> ``overlap_ge_10s``: positive if overlap >= 10 s; **indeterminate** if
  0 < overlap < 10 s (excluded from main train/eval); negative if overlap == 0.
* ``sensitivity_onset`` -> ``event_start_point_half_open``: positive iff a retained
  event start lies in [window_start, window_end). Must reproduce Stage-4 exactly.
* ``sensitivity_any`` -> ``any_overlap``: positive if overlap > 0.
* ``sensitivity_cov50`` -> ``coverage_ge_50pct``: positive if overlap >= 15 s
  (>= 50% of a 30-s window); indeterminate if 0 < overlap < 15 s.

Retained events = scored respiratory events that do NOT overlap an awake interval.
``overlap(window, event) = max(0, min(w_end, e_end) - max(w_start, e_start))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

MAIN = "main"
ONSET = "sensitivity_onset"
ANY = "sensitivity_any"
COV50 = "sensitivity_cov50"

VARIANT_ORDER = [MAIN, ONSET, ANY, COV50]
VARIANT_NAMES = {
    MAIN: "overlap_ge_10s",
    ONSET: "event_start_point_half_open",
    ANY: "any_overlap",
    COV50: "coverage_ge_50pct",
}
MAIN_OVERLAP_S = 10.0
COV50_FRACTION = 0.5
WINDOW_LENGTH_S = 30.0
INDETERMINATE = -1   # internal sentinel for "excluded from main train/eval"


@dataclass(frozen=True)
class LabelResult:
    variant: str
    name: str
    table: pd.DataFrame   # columns: window_id, patient_id, label (1/0/-1 indeterminate), max_overlap_s, n_retained_overlapping_events, onset_hit


def _parse_window_index(window_id: str) -> str:
    """Return the patient prefix of a ``{pid}-{index:05d}`` window id."""
    return str(window_id).split("-")[0]


def derive_overlaps(
    *, cohort: pd.DataFrame, events: pd.DataFrame,
) -> pd.DataFrame:
    """Attach per-window ``max_overlap_s``, ``n_retained_overlapping_events`` and
    ``onset_hit`` to the cohort windows using retained (non-awake scored) events.

    Vectorized per patient by broadcasting window x event intervals.
    """
    retained = events[
        events["is_any_scored_respiratory_event"].fillna(False)
        & (~events["overlaps_awake_interval"].fillna(False).astype(bool))
    ].copy()

    ws = cohort["window_start_relative_to_record_start"].to_numpy(dtype=float)
    we = cohort["window_end_relative_to_record_start"].to_numpy(dtype=float)
    pids = cohort["patient_id"].astype(str).to_numpy()
    max_overlap = np.zeros(len(cohort), dtype=float)
    n_overlapping = np.zeros(len(cohort), dtype=int)
    onset_hit = np.zeros(len(cohort), dtype=int)

    for pid in sorted(set(pids.tolist())):
        wmask = pids == pid
        idxs = np.nonzero(wmask)[0]
        if idxs.size == 0:
            continue
        ev = retained[retained["patient_id"].astype(str) == pid]
        if ev.empty:
            continue
        es = ev["event_start_relative_to_record_start"].to_numpy(dtype=float)
        ee = ev["event_end_relative_to_record_start"].to_numpy(dtype=float)
        w_s = ws[idxs][:, None]   # (n_w, 1)
        w_e = we[idxs][:, None]
        e_s = es[None, :]         # (1, n_e)
        e_e = ee[None, :]
        inter = np.minimum(w_e, e_e) - np.maximum(w_s, e_s)
        ov = np.where(inter > 0, inter, 0.0)            # (n_w, n_e)
        max_overlap[idxs] = ov.max(axis=1)
        n_overlapping[idxs] = (ov > 0).sum(axis=1)
        # onset: event start within [w_start, w_end)
        onset = ((e_s >= w_s) & (e_s < w_e))            # (n_w, n_e)
        onset_hit[idxs] = onset.any(axis=1).astype(int)

    out = cohort[["window_id", "patient_id", "window_start_relative_to_record_start",
                  "window_end_relative_to_record_start"]].copy()
    out["max_overlap_s"] = max_overlap
    out["n_retained_overlapping_events"] = n_overlapping
    out["onset_hit"] = onset_hit
    return out


def assign_labels(*, overlaps: pd.DataFrame) -> Dict[str, LabelResult]:
    """Assign the four pre-registered label variants from the overlap table."""
    cov50_s = COV50_FRACTION * WINDOW_LENGTH_S
    tables: Dict[str, LabelResult] = {}

    # MAIN: >= 10 s positive; (0,10) indeterminate; 0 negative
    m = overlaps.copy()
    mlab = np.where(m["max_overlap_s"] >= MAIN_OVERLAP_S, 1,
                    np.where(m["max_overlap_s"] > 0, INDETERMINATE, 0))
    m["label"] = mlab.astype(int)
    tables[MAIN] = LabelResult(MAIN, VARIANT_NAMES[MAIN], m)

    # ONSET (must match Stage-4 start-point label)
    o = overlaps.copy()
    o["label"] = o["onset_hit"].astype(int)
    tables[ONSET] = LabelResult(ONSET, VARIANT_NAMES[ONSET], o)

    # ANY overlap
    a = overlaps.copy()
    a["label"] = (a["max_overlap_s"] > 0).astype(int)
    tables[ANY] = LabelResult(ANY, VARIANT_NAMES[ANY], a)

    # COV50: >= 15 s positive; (0,15) indeterminate; 0 negative
    c = overlaps.copy()
    clab = np.where(c["max_overlap_s"] >= cov50_s, 1,
                    np.where(c["max_overlap_s"] > 0, INDETERMINATE, 0))
    c["label"] = clab.astype(int)
    tables[COV50] = LabelResult(COV50, VARIANT_NAMES[COV50], c)

    return tables


def variant_counts(*, label_result: LabelResult) -> Dict[str, int]:
    """Positive / negative / indeterminate window counts for a variant."""
    lab = label_result.table["label"]
    return {
        "n_windows": int(len(lab)),
        "n_positive": int((lab == 1).sum()),
        "n_negative": int((lab == 0).sum()),
        "n_indeterminate": int((lab == INDETERMINATE).sum()),
        "positive_rate": float((lab == 1).sum() / max(int((lab != INDETERMINATE).sum()), 1)),
    }


def onset_matches_stage4(*, onset_table: pd.DataFrame, stage4_cohort: pd.DataFrame) -> bool:
    """Sanity: the onset variant must reproduce the historical Stage-4 label exactly."""
    a = onset_table[["window_id", "label"]].rename(columns={"label": "onset_label"})
    b = stage4_cohort[["window_id", "binary_event_label"]].copy()
    j = a.merge(b, on="window_id", how="inner")
    if j.empty:
        return False
    mism = int((j["onset_label"].astype(int) != j["binary_event_label"].astype(int)).sum())
    return mism == 0


__all__ = [
    "MAIN", "ONSET", "ANY", "COV50", "VARIANT_ORDER", "VARIANT_NAMES",
    "MAIN_OVERLAP_S", "COV50_FRACTION", "WINDOW_LENGTH_S", "INDETERMINATE",
    "LabelResult", "derive_overlaps", "assign_labels", "variant_counts",
    "onset_matches_stage4",
]
