"""(2,3) Adoption gate: v1 comparator = 0.1756, and the approved/not-approved
branches. ``evaluate_gate`` is driven directly with crafted FoldMetrics so every
branch is exercised deterministically (independent of the optimizer).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _stage6b_fixtures import REAL_V1_RUN_ID, load_6b_config
from psg_audio_benchmark.evaluation.candidate_comparison import (
    FoldMetrics,
    compute_v1_metrics_from_outer_csv,
)
from psg_audio_benchmark.evaluation.gate import (
    APPROVED,
    NOT_APPROVED,
    Check,
    evaluate_gate,
    verify_hard_constraints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _perfold(rates, wins=None, base_w=1000):
    wins = wins or [base_w] * len(rates)
    return [{"fold": f, "n_patients": 10, "n_windows": w, "n_positive": int(round(r * w)),
             "n_negative": w - int(round(r * w)), "positive_rate": r}
            for f, (r, w) in enumerate(zip(rates, wins))]


def _metrics(rates, overall=None, wins=None) -> FoldMetrics:
    pf = _perfold(rates, wins)
    import numpy as np
    r = np.array(rates, dtype=float)
    w = np.array([p["n_windows"] for p in pf], dtype=float)
    overall = overall if overall is not None else float(r.mean())
    return FoldMetrics(
        per_fold=pf,
        positive_rate_spread=float(r.max() - r.min()),
        window_count_cv=float(w.std(ddof=0) / w.mean()),
        overall_positive_rate=overall,
        max_abs_deviation=float(np.max(np.abs(r - overall))),
        n_patients_total=sum(p["n_patients"] for p in pf),
        n_windows_total=int(w.sum()),
        n_positive_total=sum(p["n_positive"] for p in pf),
    )


def _hard_ok_checks() -> list:
    return [Check("dummy_hard", True, "ok")]


def _leak_ok_checks() -> list:
    return [Check("dummy_leak", True, "ok")]


@pytest.fixture(scope="module")
def cfg():
    return load_6b_config(PROJECT_ROOT)


# ---- (2) v1 comparator is exactly the pre-registered 0.1756 ----

def test_v1_comparator_spread_is_0_1756():
    v1_csv = (PROJECT_ROOT / "splits" / "runs" / REAL_V1_RUN_ID
              / "outer_patient_folds_core.csv")
    m = compute_v1_metrics_from_outer_csv(v1_csv, n_folds=5)
    # the prompt's comparator is 0.1756; allow more precision
    assert round(m.positive_rate_spread, 4) == 0.1756
    assert abs(m.positive_rate_spread - 0.17556619642728147) < 1e-9
    # v1 per-fold patient counts are NOT exactly 10 (the reason 6B exists)
    counts = [p["n_patients"] for p in m.per_fold]
    assert counts != [10, 10, 10, 10, 10]
    assert sum(counts) == 50


# ---- (3) approved branch: all criteria pass ----

def test_gate_approves_when_all_criteria_pass(cfg):
    v1 = _metrics([0.10, 0.27, 0.30, 0.40])      # spread 0.30, wcv 0
    v2 = _metrics([0.275, 0.280, 0.282, 0.285])  # spread 0.01, wcv 0
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=_hard_ok_checks(),
                         leakage_checks=_leak_ok_checks(), config=cfg)
    assert gate.approved is True
    assert gate.status == APPROVED
    assert gate.reasons == []


def test_gate_rejects_when_spread_above_threshold(cfg):
    v1 = _metrics([0.10, 0.30])
    v2 = _metrics([0.20, 0.32])  # spread 0.12 > 0.10
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=_hard_ok_checks(),
                         leakage_checks=_leak_ok_checks(), config=cfg)
    assert gate.approved is False
    assert gate.status == NOT_APPROVED
    assert any("le_threshold" in c.name and not c.passed for c in gate.checks)


def test_gate_rejects_when_not_strictly_below_v1(cfg):
    v1 = _metrics([0.10, 0.12])  # v1 spread 0.02
    v2 = _metrics([0.10, 0.119])  # spread 0.019 < 0.10 but NOT < v1 (0.02)? it is < 0.02 -> passes below_v1
    # craft v2 spread equal-ish to v1 to trigger the strictly-below failure
    v2 = _metrics([0.10, 0.12])  # spread == v1 -> strictly-below fails
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=_hard_ok_checks(),
                         leakage_checks=_leak_ok_checks(), config=cfg)
    assert gate.approved is False
    assert any("strictly_below_v1" in c.name and not c.passed for c in gate.checks)


def test_gate_rejects_when_window_cv_worse_than_v1(cfg):
    v1 = _metrics([0.10, 0.30], wins=[1000, 1000])  # wcv 0
    v2 = _metrics([0.27, 0.29], wins=[100, 1900])   # spread ok, wcv huge
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=_hard_ok_checks(),
                         leakage_checks=_leak_ok_checks(), config=cfg)
    assert gate.approved is False
    assert any("window_cv" in c.name and not c.passed for c in gate.checks)


def test_gate_rejects_on_hard_constraint_failure(cfg):
    v1 = _metrics([0.10, 0.30])
    v2 = _metrics([0.27, 0.29])
    bad_hard = [Check("exactly_patients_per_outer_fold", False, "9 in fold 0")]
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=bad_hard,
                         leakage_checks=_leak_ok_checks(), config=cfg)
    assert gate.approved is False
    assert any("hard_constraints" in c.name and not c.passed for c in gate.checks)


def test_gate_rejects_on_leakage_failure(cfg):
    v1 = _metrics([0.10, 0.30])
    v2 = _metrics([0.27, 0.29])
    bad_leak = [Check("airflow_inherits_v2_outer_fold_zero_mismatch", False, "mismatch=2")]
    gate = evaluate_gate(v2=v2, v1=v1, hard_checks=_hard_ok_checks(),
                         leakage_checks=bad_leak, config=cfg)
    assert gate.approved is False
    assert any("leakage" in c.name and not c.passed for c in gate.checks)


def test_hard_constraint_verifier_detects_wrong_per_fold(cfg):
    from psg_audio_benchmark.evaluation.balance_optimizer import PatientBurden
    burden = [PatientBurden(f"{i:02d}", 100, 30, 70, 0.3) for i in range(50)]
    by_pid = {b.patient_id: b for b in burden}
    # assign 9 to fold 0, 11 to fold 1 (violates exactly-10)
    p2f = {}
    for i, b in enumerate(burden):
        p2f[b.patient_id] = 0 if i < 9 else (1 if i < 20 else (i % 3) + 2)
    checks = verify_hard_constraints(patient_to_fold=p2f, burden_by_pid=by_pid, config=cfg)
    exact = next(c for c in checks if c.name == "exactly_patients_per_outer_fold")
    assert exact.passed is False
