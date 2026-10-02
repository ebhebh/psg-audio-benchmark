"""Inner patient-level CV for the v2 split (Stage 6B).

Built ONLY after the v2 outer split is approved. For each outer fold, the 40
outer-TRAIN patients are re-partitioned into ``inner_n_folds`` (4) patient-level
inner folds of exactly ``patients_per_inner_fold`` (10) patients, reusing the same
deterministic optimizer. Patient isolation has priority over ratio balance; any
balance shortfall is recorded as a non-fatal inner warning.

Leakage guards (load-bearing):

* each outer fold's inner patient set == that outer fold's outer-train patients;
* an outer fold's outer-TEST patients never appear in that outer fold's inner folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .balance_optimizer import (
    OptimizationResult,
    PatientBurden,
    optimize_patient_folds,
    per_fold_stats,
)
from .split_balance_config import ResolvedSplitBalanceConfig


@dataclass
class InnerSplitResult:
    """All inner folds across all outer folds + leakage diagnostics."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    per_outer: Dict[int, OptimizationResult] = field(default_factory=dict)


def build_inner_folds(
    *,
    burden: List[PatientBurden],
    outer_patient_to_fold: Dict[str, int],
    config: ResolvedSplitBalanceConfig,
    run_id: str,
    inner_balance_warn_threshold: float = 0.10,
) -> InnerSplitResult:
    """Build per-outer-fold 4-fold inner patient CV over outer-TRAIN patients only."""
    res = InnerSplitResult()
    nf_outer = config.outer_n_folds
    nf_inner = config.inner_n_folds
    ppf_inner = config.patients_per_inner_fold

    all_pids = {b.patient_id for b in burden}
    burden_by_pid = {b.patient_id: b for b in burden}

    for f in range(nf_outer):
        outer_test_pids = {p for p, fo in outer_patient_to_fold.items() if fo == f}
        outer_train_pids = all_pids - outer_test_pids
        if len(outer_train_pids) < nf_inner:
            res.anomalies.append(
                f"outer{f}_inner_train_too_few:{len(outer_train_pids)}_for_{nf_inner}_folds"
            )
            continue

        train_burden = [burden_by_pid[p] for p in sorted(outer_train_pids)]
        # soft inner balance: smaller budget is fine (isolation >> balance); reuse
        # the same deterministic optimizer with inner fold counts.
        opt = optimize_patient_folds(
            train_burden, config=config,
            n_folds=nf_inner, per_fold=ppf_inner,
            n_candidate_seeds=max(4, config.n_candidate_seeds // 3),
            local_swap_iterations=max(2000, config.local_swap_iterations // 4),
        )
        res.per_outer[f] = opt
        res.anomalies.extend(a if a.startswith(f"outer{f}_") else f"outer{f}_{a}" for a in opt.anomalies)

        inner_map = opt.patient_to_fold
        inner_pids = set(inner_map.keys())

        # load-bearing leakage guards
        if inner_pids != outer_train_pids:
            res.anomalies.append(
                f"outer{f}_inner_patient_set_mismatch:"
                f"only_inner={sorted(inner_pids - outer_train_pids)}:"
                f"only_outer_train={sorted(outer_train_pids - inner_pids)}"
            )
        if outer_test_pids & inner_pids:
            res.anomalies.append(
                f"outer{f}_outer_test_in_inner:{sorted(outer_test_pids & inner_pids)}"
            )

        # inner rows (one per outer-train patient -> its inner validation fold)
        for pid in sorted(outer_train_pids):
            b = burden_by_pid[pid]
            res.rows.append({
                "run_id": run_id, "outer_fold": f, "patient_id": pid,
                "inner_validation_fold": int(inner_map.get(pid, -1)),
                "n_windows": b.n_windows, "n_positive": b.n_positive,
                "n_negative": b.n_negative, "positive_rate": b.positive_rate,
            })

        # inner per-fold stats (validation fold perspective), computed directly
        # from the inner_map + burden (no feature values used).
        for i in range(nf_inner):
            val_pids = {p for p, fo in inner_map.items() if fo == i}
            inner_train_pids = outer_train_pids - val_pids
            vw = sum(burden_by_pid[p].n_windows for p in val_pids)
            vp = sum(burden_by_pid[p].n_positive for p in val_pids)
            tw = sum(burden_by_pid[p].n_windows for p in inner_train_pids)
            tp = sum(burden_by_pid[p].n_positive for p in inner_train_pids)
            if val_pids & inner_train_pids:
                res.anomalies.append(f"outer{f}_inner{i}_train_val_overlap")
            res.fold_stats.append({
                "outer_fold": f, "inner_fold": i,
                "n_inner_train_patients": len(inner_train_pids),
                "n_inner_val_patients": len(val_pids),
                "n_inner_train_windows": tw, "inner_train_positive": tp,
                "inner_train_negative": tw - tp,
                "inner_train_positive_rate": float(tp / tw) if tw else 0.0,
                "n_inner_val_windows": vw, "inner_val_positive": vp,
                "inner_val_negative": vw - vp,
                "inner_val_positive_rate": float(vp / vw) if vw else 0.0,
            })

        # soft balance warning (informational; isolation already guaranteed)
        if opt.positive_rate_spread > inner_balance_warn_threshold:
            res.warnings.append(
                f"outer{f}_inner_pos_rate_spread={opt.positive_rate_spread:.4f} "
                f"exceeds {inner_balance_warn_threshold:.4f}; patient isolation preserved."
            )

    return res


__all__ = ["InnerSplitResult", "build_inner_folds"]
