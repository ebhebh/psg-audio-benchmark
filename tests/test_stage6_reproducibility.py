"""(5) Reproducibility: fixed seed + sort order -> identical fold assignment.

* same seed + same sklearn -> identical patient->outer-fold mapping;
* a different seed -> a different mapping, BUT patient isolation still holds
  (every patient in exactly one outer fold; no anomalies).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6


def _outer_map(out_dir: Path) -> dict:
    outer = pd.read_csv(out_dir / "outer_patient_folds_core.csv", dtype={"patient_id": str})
    return dict(zip(outer["patient_id"], outer["outer_fold"]))


def _alt_seed_config(tmp_path: Path, cfg, new_seed: int) -> Path:
    src = (cfg.path("config") / "splits.yaml").read_text(encoding="utf-8")
    src = src.replace("outer_fixed_seed: 20250714", f"outer_fixed_seed: {new_seed}")
    src = src.replace("inner_fixed_seed: 20250714", f"inner_fixed_seed: {new_seed}")
    alt = tmp_path / f"splits_alt_{new_seed}.yaml"
    alt.write_text(src, encoding="utf-8")
    return alt


def test_same_seed_gives_identical_folds(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())

    s1, _r1, _o1 = run_stage6(cfg, tmp_path, ids=ids, run_id="stage6-synth-repro1",
                              output_root=tmp_path / "a")
    s2, _r2, _o2 = run_stage6(cfg, tmp_path, ids=ids, run_id="stage6-synth-repro2",
                              output_root=tmp_path / "b")
    assert s1.overall_status == "PASS" and s2.overall_status == "PASS"

    a = _outer_map(tmp_path / "a" / "splits" / "runs" / "stage6-synth-repro1")
    b = _outer_map(tmp_path / "b" / "splits" / "runs" / "stage6-synth-repro2")
    assert a == b


def test_different_seed_gives_different_but_isolated_folds(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())

    s1, _r1, _o1 = run_stage6(cfg, tmp_path, ids=ids, run_id="stage6-synth-reproA",
                              output_root=tmp_path / "a")
    alt = _alt_seed_config(tmp_path, cfg, 99999)
    s2, _r2, _o2 = run_stage6(cfg, tmp_path, ids=ids, run_id="stage6-synth-reproB",
                              output_root=tmp_path / "b", config_path=alt)
    assert s1.overall_status == "PASS" and s2.overall_status == "PASS"
    assert s2.anomalies == []                      # isolation preserved under new seed

    a = _outer_map(tmp_path / "a" / "splits" / "runs" / "stage6-synth-reproA")
    b = _outer_map(tmp_path / "b" / "splits" / "runs" / "stage6-synth-reproB")
    assert a != b                                  # different seed -> different assignment
    # ... but the partition is still a valid patient-level 5-fold split
    assert set(a.keys()) == set(b.keys())
    assert set(a.values()) == {0, 1, 2, 3, 4}
    assert sorted(a.values()) == sorted(b.values()) == sorted(a.values())
