"""(6) Balance is reported honestly, never faked.

The default world spans positive rates 0.0 (patient 09) to 0.80 (patient 01), so
the across-fold test positive-rate spread exceeds the 0.10 threshold -> a balance
warning is emitted. A smaller cohort (< 10 patients) additionally triggers the
``small_cohort`` warning. In both cases patient isolation is preserved (the
warning is reported, not corrected) and the status is not blocked.
"""

from __future__ import annotations

from _stage6_fixtures import (
    build_stage6_world,
    default_windows,
    make_isolated_cfg,
    make_windows,
    run_stage6,
)


def test_high_pos_rate_spread_emits_warning(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)

    assert summary.overall_status == "PASS", summary.anomalies
    assert summary.anomalies == []                 # imbalance is a warning, not an anomaly
    assert any("pos_rate_spread" in w for w in summary.balance_warnings), summary.balance_warnings
    # the cohort is large enough that the small_cohort warning does NOT fire here
    assert not any("small_cohort" in w for w in summary.balance_warnings)


def test_small_cohort_emits_warning_and_keeps_isolation(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    # 8 patients (< outer_n_folds*2=10) -> small_cohort warning; still enough for
    # 5-fold outer and 4-fold inner (>=4 outer-train per fold).
    windows = make_windows([
        ("01", 3, 1, 0, True),
        ("02", 2, 2, 0, False),
        ("03", 1, 3, 0, True),
        ("04", 2, 2, 0, False),
        ("05", 1, 3, 0, False),
        ("06", 0, 4, 0, False),   # all-negative
        ("07", 3, 1, 0, True),
        ("08", 2, 2, 0, False),
    ])
    ids = build_stage6_world(cfg, windows=windows)
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)

    assert summary.overall_status == "PASS", summary.anomalies
    assert summary.anomalies == []
    assert any("small_cohort" in w for w in summary.balance_warnings), summary.balance_warnings
    # core cohort still has 8 patients, each in exactly one fold
    assert summary.core_patients == 8
