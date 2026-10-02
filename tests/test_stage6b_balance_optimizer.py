"""(1,4,5) Balance optimizer: hard constraints, determinism, no-drop, objective.

Synthetic 50-patient burden (fabricated, varied positive rates + window counts).
The optimizer must: place exactly 10 patients per outer fold; assign each patient
to exactly one fold; drop/relable NO patient or window; be bit-identical under a
fixed seed; and report the correct positive-rate spread + window-count CV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _stage6b_fixtures import (
    fast_config,
    load_6b_config,
    synthetic_balanceable_burden,
)
from psg_audio_benchmark.evaluation.balance_optimizer import (
    fold_aggregates,
    objective,
    optimize_patient_folds,
    per_fold_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfg():
    return fast_config(load_6b_config(PROJECT_ROOT))


@pytest.fixture(scope="module")
def burden():
    return synthetic_balanceable_burden(50)


@pytest.fixture(scope="module")
def result(cfg, burden):
    return optimize_patient_folds(burden, config=cfg)


def test_exactly_10_patients_per_outer_fold(result):
    counts = [s["n_patients"] for s in result.best.per_fold]
    assert counts == [10, 10, 10, 10, 10]
    assert result.anomalies == []


def test_each_patient_assigned_to_exactly_one_fold(result, burden):
    pids = {b.patient_id for b in burden}
    assigned = result.patient_to_fold
    assert set(assigned.keys()) == pids            # no drop, no extra
    assert len(assigned) == len(pids)               # each exactly once (dict)
    assert set(assigned.values()) == {0, 1, 2, 3, 4}  # all folds used


def test_no_patients_or_windows_dropped_or_relabeled(result, burden):
    # union of fold patients == all input patients; window + pos totals preserved
    by_pid = {b.patient_id: b for b in burden}
    total_win = sum(s["n_windows"] for s in result.best.per_fold)
    total_pos = sum(s["n_positive"] for s in result.best.per_fold)
    total_neg = sum(s["n_negative"] for s in result.best.per_fold)
    assert total_win == sum(b.n_windows for b in burden)
    assert total_pos == sum(b.n_positive for b in burden)
    assert total_neg == sum(b.n_negative for b in burden)
    # per-fold n_negative == n_windows - n_positive (no relabel)
    for s in result.best.per_fold:
        assert s["n_negative"] == s["n_windows"] - s["n_positive"]


def test_fixed_seed_is_reproducible(cfg, burden):
    r1 = optimize_patient_folds(burden, config=cfg)
    r2 = optimize_patient_folds(burden, config=cfg)
    assert r1.patient_to_fold == r2.patient_to_fold
    assert r1.positive_rate_spread == r2.positive_rate_spread
    assert r1.window_count_cv == r2.window_count_cv


def test_candidate_selection_is_order_independent(cfg, burden):
    """The deterministic selection rule must not depend on candidate evaluation
    order: two independent runs pick the identical best assignment, and the
    candidates list is indexed by candidate_index (0..n-1)."""
    r1 = optimize_patient_folds(burden, config=cfg)
    r2 = optimize_patient_folds(burden, config=cfg)
    assert [c.candidate_index for c in r1.all_candidates] == list(range(len(r1.all_candidates)))
    assert r1.best.candidate_index == r2.best.candidate_index
    # best is the min-aggregate candidate (ties -> spread -> wcv -> lexmin)
    aggs = [c.aggregate_objective for c in r1.all_candidates]
    assert r1.best.aggregate_objective == pytest.approx(min(aggs), abs=1e-12)


def test_objective_components_are_correct(result, burden):
    # recompute spread + wcv independently from the assignment and compare
    pids = [b.patient_id for b in burden]
    W = np.array([b.n_windows for b in burden], dtype=np.int64)
    P = np.array([b.n_positive for b in burden], dtype=np.int64)
    assign = np.array([result.patient_to_fold[p] for p in pids], dtype=np.int64)
    fw, fp = fold_aggregates(assign, W, P, 5)
    obj = objective(fw, fp)
    assert obj["positive_rate_spread"] == pytest.approx(result.positive_rate_spread, abs=1e-12)
    assert obj["window_count_cv"] == pytest.approx(result.window_count_cv, abs=1e-12)
    # per_fold_stats agree with fold_aggregates
    pfs = per_fold_stats(assign, W, P, 5)
    for a, b in zip(pfs, result.best.per_fold):
        assert a["n_windows"] == b["n_windows"]
        assert a["n_positive"] == b["n_positive"]


def test_balanced_result_beats_v1_spread_threshold(result):
    # synthetic balanceable cohort must clear the 0.10 gate and beat v1's 0.1756
    assert result.positive_rate_spread <= 0.10
    assert result.positive_rate_spread < 0.1756
