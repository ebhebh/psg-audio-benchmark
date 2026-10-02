"""Deterministic patient-level fold-balance optimizer (Stage 6B core).

Re-partitions a fixed set of patients into ``n_folds`` of **exactly**
``per_fold`` patients each, minimizing a pre-registered, patient-level-only
objective built from per-patient window/positive/negative counts:

    aggregate = w1 * positive_rate_spread + w2 * window_count_cv

where ``positive_rate_spread = max_f rate_f - min_f rate_f`` (rate_f =
sum(pos in f)/sum(win in f)) and ``window_count_cv = std(ddof=0)/mean`` of the
per-fold window counts.

The search is deterministic and reproducible:

* multi-start: each candidate derives its own seed from ``(base_seed, k)``;
* per candidate: a balanced **greedy seed** (patients inserted in a burden-ordered
  or seed-permuted order, each placed into the still-roomy fold that least worsens
  the incremental objective), then **pairwise local swap** between two folds
  (which preserves the exactly-``per_fold`` invariant by construction);
* selection is a fixed deterministic rule over the aggregate objective (ties broken
  by spread, then window CV, then lexicographically-minimum assignment).

It uses NO feature values, NO model output, drops/relables NO patient or window,
and performs NO window-random split. The cohort (patient set + per-patient
burden) is an immutable input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .split_balance_config import ResolvedSplitBalanceConfig


# ---------------------------------------------------------------------------
# Patient burden
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatientBurden:
    """One patient's pre-model label-burden summary (no feature values)."""

    patient_id: str
    n_windows: int
    n_positive: int
    n_negative: int
    positive_rate: float


def burden_from_outer_folds_df(df: Any) -> List[PatientBurden]:
    """Build sorted ``PatientBurden`` list from a v1 ``outer_patient_folds_core.csv``.

    The v1 product carries exactly the patient-level aggregates this optimizer is
    allowed to use (patient_id, n_windows, n_positive, n_negative). Sorting by
    ``str(patient_id)`` ascending makes the optimizer's internal indexing
    deterministic regardless of CSV row order.
    """
    import pandas as pd  # local import keeps the module importable without pandas at parse time

    if isinstance(df, (str, Path)):
        df = pd.read_csv(df, dtype={"patient_id": str})
    sub = df[["patient_id", "n_windows", "n_positive", "n_negative"]].copy()
    sub["patient_id"] = sub["patient_id"].astype(str)
    sub = sub.sort_values("patient_id").reset_index(drop=True)
    out: List[PatientBurden] = []
    for _, r in sub.iterrows():
        n = int(r["n_windows"]); pos = int(r["n_positive"])
        out.append(PatientBurden(
            patient_id=str(r["patient_id"]), n_windows=n, n_positive=pos,
            n_negative=int(r["n_negative"]),
            positive_rate=float(pos / n) if n else 0.0,
        ))
    return out


def burden_from_membership_df(df: Any) -> List[PatientBurden]:
    """Build sorted ``PatientBurden`` list from a ``(patient_id, binary_event_label)``
    window-membership frame (per-patient groupby). Used for the airflow subcohort."""
    import pandas as pd

    if isinstance(df, str):
        df = pd.read_parquet(df)
    sub = df[["patient_id", "binary_event_label"]].copy()
    sub["patient_id"] = sub["patient_id"].astype(str)
    rows = []
    for pid, g in sub.groupby("patient_id"):
        n = int(len(g)); pos = int((g["binary_event_label"] == 1).sum())
        rows.append(PatientBurden(
            patient_id=str(pid), n_windows=n, n_positive=pos, n_negative=n - pos,
            positive_rate=float(pos / n) if n else 0.0,
        ))
    rows.sort(key=lambda b: b.patient_id)
    return rows


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def fold_aggregates(assign: np.ndarray, W: np.ndarray, P: np.ndarray, nf: int):
    """Return per-fold (window_count, positive_count) integer arrays."""
    fw = np.zeros(nf, dtype=np.int64)
    fp = np.zeros(nf, dtype=np.int64)
    np.add.at(fw, assign, W)
    np.add.at(fp, assign, P)
    return fw, fp


def objective(fw: np.ndarray, fp: np.ndarray) -> Dict[str, float]:
    """``{positive_rate_spread, window_count_cv, aggregate}`` for fold sums.

    CV uses population std (ddof=0), matching the contract. Rates are guarded
    against zero-window folds (cannot occur under per_fold>=1 with n_windows>=1,
    but defended for generality).
    """
    nf = len(fw)
    rates = fp / np.maximum(fw.astype(float), 1.0)
    spread = float(rates.max() - rates.min()) if nf > 1 else 0.0
    mean_w = float(fw.mean())
    wcv = float(fw.std(ddof=0) / mean_w) if mean_w > 0 else 0.0
    return {"positive_rate_spread": spread, "window_count_cv": wcv}


def per_fold_stats(assign: np.ndarray, W: np.ndarray, P: np.ndarray, nf: int) -> List[Dict[str, Any]]:
    fw, fp = fold_aggregates(assign, W, P, nf)
    counts = np.bincount(assign, minlength=nf)
    out: List[Dict[str, Any]] = []
    for f in range(nf):
        n = int(fw[f]); pos = int(fp[f])
        out.append({
            "fold": f, "n_patients": int(counts[f]),
            "n_windows": n, "n_positive": pos, "n_negative": n - pos,
            "positive_rate": float(pos / n) if n else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CandidateMetrics:
    candidate_index: int
    seed: int
    positive_rate_spread: float
    window_count_cv: float
    aggregate_objective: float
    per_fold: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationResult:
    patient_ids: List[str]
    patient_to_fold: Dict[str, int]
    best: CandidateMetrics
    all_candidates: List[CandidateMetrics]
    selection_rule: str
    anomalies: List[str] = field(default_factory=list)

    @property
    def positive_rate_spread(self) -> float:
        return self.best.positive_rate_spread

    @property
    def window_count_cv(self) -> float:
        return self.best.window_count_cv


# ---------------------------------------------------------------------------
# Greedy seed + local swap
# ---------------------------------------------------------------------------

def _candidate_seed(base_seed: int, k: int) -> int:
    """Deterministic, stable per-candidate seed (independent of numpy version)."""
    return int((int(base_seed) * 1_000_003 + int(k)) & 0x7FFFFFFF)


def _greedy_seed(
    order_idx: np.ndarray, W: np.ndarray, P: np.ndarray, nf: int, per_fold: int,
    w1: float, w2: float, rng: np.random.Generator,
) -> np.ndarray:
    """Balanced greedy placement: insert each patient (in ``order_idx``) into the
    still-roomy fold that least worsens the incremental objective (positive-rate
    spread over started folds + relative window-count L1 to the equal-share target).
    The window proxy is a surrogate; the local swap then optimizes the true CV."""
    n = len(order_idx)
    assign = np.full(n, -1, dtype=np.int64)
    fold_w = np.zeros(nf, dtype=np.float64)
    fold_p = np.zeros(nf, dtype=np.float64)
    fold_cnt = np.zeros(nf, dtype=np.int64)
    for idx in order_idx:
        room = [f for f in range(nf) if fold_cnt[f] < per_fold]
        best_f = -1
        best_key: Any = None
        total_w = fold_w.sum() + W[idx]
        target = total_w / nf
        for f in room:
            tw = fold_w[f] + W[idx]; tp = fold_p[f] + P[idx]
            started = fold_w.copy(); started[f] = tw
            started_p = fold_p.copy(); started_p[f] = tp
            mask = started > 0
            rates = started_p[mask] / started[mask]
            spread = float(rates.max() - rates.min()) if mask.sum() > 1 else 0.0
            wimb = float(np.abs(started - target).sum() / max(target, 1.0))
            obj = w1 * spread + w2 * wimb
            key = (obj, -float(fold_cnt[f]), float(rng.random()))  # lower obj, fuller fold, then random
            if best_key is None or key < best_key:
                best_key = key; best_f = f
        assign[idx] = best_f
        fold_w[best_f] += W[idx]; fold_p[best_f] += P[idx]; fold_cnt[best_f] += 1
    return assign


def _local_swap(
    assign: np.ndarray, W: np.ndarray, P: np.ndarray, nf: int,
    rng: np.random.Generator, iters: int, w1: float, w2: float,
) -> tuple:
    """Pairwise fold swap minimizing the TRUE aggregate objective.

    A swap moves patient ``i`` (fold f1) and patient ``j`` (fold f2) to each
    other's fold, preserving per-fold patient counts exactly. Accepts any move
    that does not worsen the aggregate (``<=``), so the walk is monotone-ish and
    deterministic for a fixed RNG. Per-fold sums are maintained incrementally.
    """
    assign = assign.copy()
    fold_w = np.zeros(nf, dtype=np.float64)
    fold_p = np.zeros(nf, dtype=np.float64)
    members: List[List[int]] = [[] for _ in range(nf)]
    for i, f in enumerate(assign):
        fold_w[f] += W[i]; fold_p[f] += P[i]; members[int(f)].append(i)

    def _obj(fw: np.ndarray, fp: np.ndarray) -> tuple:
        rates = fp / np.maximum(fw, 1.0)
        spread = float(rates.max() - rates.min()) if nf > 1 else 0.0
        mean_w = float(fw.mean())
        wcv = float(fw.std(ddof=0) / mean_w) if mean_w > 0 else 0.0
        return spread, wcv, w1 * spread + w2 * wcv

    cur_sp, cur_wcv, cur_obj = _obj(fold_w, fold_p)
    for _ in range(iters):
        f1 = int(rng.integers(nf)); f2 = int(rng.integers(nf))
        if f1 == f2:
            continue
        m1 = members[f1]; m2 = members[f2]
        if not m1 or not m2:
            continue
        i = m1[int(rng.integers(len(m1)))]; j = m2[int(rng.integers(len(m2)))]
        # trial fold sums
        dw1 = W[j] - W[i]; dw2 = W[i] - W[j]
        fw = fold_w.copy(); fp = fold_p.copy()
        fw[f1] += dw1; fw[f2] += dw2
        fp[f1] += P[j] - P[i]; fp[f2] += P[i] - P[j]
        sp, wcv, o = _obj(fw, fp)
        if o <= cur_obj + 1e-15:
            assign[i] = f2; assign[j] = f1
            m1.remove(i); m1.append(j)
            m2.remove(j); m2.append(i)
            fold_w = fw; fold_p = fp; cur_obj = o; cur_sp = sp; cur_wcv = wcv
    return assign, cur_sp, cur_wcv


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def optimize_patient_folds(
    burden: List[PatientBurden],
    *,
    config: ResolvedSplitBalanceConfig,
    n_folds: int | None = None,
    per_fold: int | None = None,
    n_candidate_seeds: int | None = None,
    local_swap_iterations: int | None = None,
) -> OptimizationResult:
    """Optimize a patient->fold assignment for balance (deterministic, reproducible).

    Parameters mirror the resolved config; ``n_folds``/``per_fold`` default to the
    outer values and may be overridden (e.g. for inner CV with 4 folds).
    ``n_candidate_seeds``/``local_swap_iterations`` optionally override the search
    budget (used by tests and the soft inner CV) without touching the pre-registered
    config defaults.
    """
    nf = int(n_folds if n_folds is not None else config.outer_n_folds)
    ppf = int(per_fold if per_fold is not None else config.patients_per_outer_fold)
    w1 = float(config.weight_positive_rate_spread)
    w2 = float(config.weight_window_count_cv)
    base_seed = int(config.base_seed)
    n_cand = int(config.n_candidate_seeds if n_candidate_seeds is None else n_candidate_seeds)
    iters = int(config.local_swap_iterations if local_swap_iterations is None else local_swap_iterations)

    anomalies: List[str] = []
    n = len(burden)
    if n == 0:
        anomalies.append("empty_burden")
        return OptimizationResult({}, {}, _empty_candidate(-1, 0, nf), [], config.selection_rule, anomalies)
    if n != nf * ppf:
        anomalies.append(
            f"cardinality_mismatch:{n}_patients_not_{nf}_x_{ppf}={nf*ppf}; "
            f"exact-per-fold constraint cannot be satisfied."
        )
        # Still attempt a best-effort partition so downstream can report; the gate
        # will reject on the hard-constraint check.
        ppf = n // nf if n >= nf else 1

    # deterministic internal indexing: patients already sorted ascending by caller
    pids = [b.patient_id for b in burden]
    W = np.array([b.n_windows for b in burden], dtype=np.int64)
    P = np.array([b.n_positive for b in burden], dtype=np.int64)

    # burden-descending insertion order for the canonical (k=0) greedy seed
    burden_desc = np.argsort(-(P.astype(np.float64) / np.maximum(W, 1)), kind="stable")

    candidates: List[CandidateMetrics] = []
    best_assign: np.ndarray | None = None
    best_key: Any = None
    best_metrics: CandidateMetrics | None = None
    best_patient_order: np.ndarray | None = None  # for lexmin tiebreak via assign vector

    for k in range(max(n_cand, 1)):
        seed = _candidate_seed(base_seed, k)
        rng = np.random.default_rng(seed)
        if k == 0:
            order = burden_desc
        else:
            order = rng.permutation(n)
        greedy = _greedy_seed(order, W, P, nf, ppf, w1, w2, rng)
        refined, sp, wcv = _local_swap(greedy, W, P, nf, rng, iters, w1, w2)
        agg = w1 * sp + w2 * wcv
        pfs = per_fold_stats(refined, W, P, nf)
        candidates.append(CandidateMetrics(
            candidate_index=k, seed=seed, positive_rate_spread=float(sp),
            window_count_cv=float(wcv), aggregate_objective=float(agg), per_fold=pfs,
        ))
        # deterministic selection key: (aggregate, spread, wcv, lexmin assign vector)
        sel_key = (float(agg), float(sp), float(wcv), refined.tolist())
        if best_key is None or sel_key < best_key:
            best_key = sel_key
            best_assign = refined.copy()
            best_metrics = candidates[-1]
            best_patient_order = refined.tolist()

    assert best_assign is not None and best_metrics is not None  # at least one candidate

    patient_to_fold = {pids[i]: int(best_assign[i]) for i in range(n)}

    # ---- hard-constraint verification (report; gate enforces) ----
    counts = np.bincount(best_assign, minlength=nf)
    if list(counts) != [ppf] * nf:
        anomalies.append(f"per_fold_violation:counts={counts.tolist()}_expected_{ppf}")
    if len(set(patient_to_fold.values())) < nf:
        anomalies.append(f"not_all_folds_used:{set(patient_to_fold.values())}")

    return OptimizationResult(
        patient_ids=pids, patient_to_fold=patient_to_fold,
        best=best_metrics, all_candidates=candidates,
        selection_rule=config.selection_rule, anomalies=anomalies,
    )


def _empty_candidate(k: int, seed: int, nf: int) -> CandidateMetrics:
    return CandidateMetrics(
        candidate_index=k, seed=seed, positive_rate_spread=0.0,
        window_count_cv=0.0, aggregate_objective=0.0,
        per_fold=[{"fold": f, "n_patients": 0, "n_windows": 0, "n_positive": 0,
                   "n_negative": 0, "positive_rate": 0.0} for f in range(nf)],
    )


__all__ = [
    "PatientBurden",
    "burden_from_outer_folds_df",
    "burden_from_membership_df",
    "fold_aggregates",
    "objective",
    "per_fold_stats",
    "CandidateMetrics",
    "OptimizationResult",
    "optimize_patient_folds",
]
