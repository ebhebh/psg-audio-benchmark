"""Adoption gate for the v2 split (Stage 6B).

Pre-registered, non-cherry-pickable decision: v2 is ``approved_for_modeling``
only if ALL of (hard constraints, positive-rate spread <= 0.10, spread strictly
below v1, window-count CV not worse than v1 within the contract ceiling, and
inner/airflow leakage checks) pass simultaneously. Otherwise the status is
``not_approved_balance_target_not_met`` and LATEST is never updated.

The gate NEVER relaxes a threshold, drops a patient/window, or relabels to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .candidate_comparison import FoldMetrics
from .split_balance_config import ResolvedSplitBalanceConfig

APPROVED = "approved_for_modeling"
NOT_APPROVED = "not_approved_balance_target_not_met"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class GateResult:
    approved: bool
    status: str
    checks: List[Check] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.approved


def verify_hard_constraints(
    *, patient_to_fold: Dict[str, int], burden_by_pid: Dict[str, Any],
    config: ResolvedSplitBalanceConfig,
) -> List[Check]:
    """Verify the section-2.1 hard constraints over a patient->fold assignment."""
    nf = config.outer_n_folds
    ppf = config.patients_per_outer_fold
    checks: List[Check] = []

    all_pids = set(burden_by_pid.keys())
    assigned = set(patient_to_fold.keys())

    # 1. exactly per_fold patients in every fold
    counts: Dict[int, int] = {}
    for fold in patient_to_fold.values():
        counts[int(fold)] = counts.get(int(fold), 0) + 1
    exact = all(counts.get(f, 0) == ppf for f in range(nf)) and len(counts) == nf
    checks.append(Check(
        "exactly_patients_per_outer_fold", exact,
        f"per-fold counts={[counts.get(f,0) for f in range(nf)]}, expected {ppf} x {nf}",
    ))

    # 2. each patient assigned exactly once (no overlap is implied by a dict)
    no_dup = len(patient_to_fold) == len(assigned)
    checks.append(Check("each_patient_assigned_once", no_dup,
                        f"n_assigned={len(patient_to_fold)} unique={len(assigned)}"))

    # 3. union == all core patients (no drop, no extra)
    union_ok = assigned == all_pids
    checks.append(Check("patient_union_equals_core_cohort", union_ok,
                        f"missing={sorted(all_pids - assigned)} extra={sorted(assigned - all_pids)}"))

    # 4. all folds present
    folds_used = set(patient_to_fold.values())
    all_folds = folds_used == set(range(nf))
    checks.append(Check("all_outer_folds_used", all_folds, f"folds={sorted(folds_used)}"))

    # 5. cohort identity (counts) matches the approved v1 cohort
    total_w = sum(b.n_windows for b in burden_by_pid.values())
    total_p = sum(b.n_positive for b in burden_by_pid.values())
    ident = (len(all_pids) == config.expected_core_patients
             and total_w == config.expected_core_windows
             and total_p == config.expected_core_positive)
    checks.append(Check("cohort_identity_matches_v1", ident,
                        f"patients={len(all_pids)}/{config.expected_core_patients} "
                        f"windows={total_w}/{config.expected_core_windows} "
                        f"positive={total_p}/{config.expected_core_positive}"))

    return checks


def evaluate_gate(
    *, v2: FoldMetrics, v1: FoldMetrics, hard_checks: List[Check],
    leakage_checks: List[Check], config: ResolvedSplitBalanceConfig,
) -> GateResult:
    """Evaluate the section-2.3 adoption criteria. Returns a structured decision."""
    checks: List[Check] = []

    # 1. hard constraints
    hard_ok = all(c.passed for c in hard_checks)
    checks.append(Check("all_hard_constraints_pass", hard_ok,
                        "; ".join(f"{c.name}={'ok' if c.passed else 'FAIL'}" for c in hard_checks)))

    # 2. spread <= 0.10
    spread_ok = v2.positive_rate_spread <= config.gate_relative_positive_rate_spread_max + 1e-12
    checks.append(Check("positive_rate_spread_le_threshold", spread_ok,
                        f"v2_spread={v2.positive_rate_spread:.6f} <= "
                        f"{config.gate_relative_positive_rate_spread_max:.6f}"))

    # 3. spread strictly below v1
    below_v1_ok = v2.positive_rate_spread < v1.positive_rate_spread - 1e-12
    checks.append(Check("v2_spread_strictly_below_v1", below_v1_ok,
                        f"v2={v2.positive_rate_spread:.6f} < v1={v1.positive_rate_spread:.6f}"))

    # 4. window-count CV not worse than v1 within ceiling (criterion 4)
    wcv_ceiling = v1.window_count_cv * config.gate_v2_wcv_ceiling_factor
    wcv_ok = v2.window_count_cv <= wcv_ceiling + 1e-12
    checks.append(Check("v2_window_cv_not_worse_than_v1_ceiling", wcv_ok,
                        f"v2_wcv={v2.window_count_cv:.6f} <= "
                        f"v1_wcv*{config.gate_v2_wcv_ceiling_factor}={wcv_ceiling:.6f}"))

    # 5. inner + airflow leakage checks
    leak_ok = all(c.passed for c in leakage_checks)
    checks.append(Check("inner_and_airflow_leakage_checks_pass", leak_ok,
                        "; ".join(f"{c.name}={'ok' if c.passed else 'FAIL'}" for c in leakage_checks)))

    approved = all(c.passed for c in checks)
    reasons = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    status = APPROVED if approved else NOT_APPROVED
    return GateResult(approved=approved, status=status, checks=checks, reasons=reasons)


__all__ = [
    "APPROVED",
    "NOT_APPROVED",
    "Check",
    "GateResult",
    "verify_hard_constraints",
    "evaluate_gate",
]
