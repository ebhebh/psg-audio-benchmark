"""(10) Boundary cases: dry-run, output isolation, path purity, the
contamination guard, and raw-data immutability.

* dry-run runs the gates but writes NO products and does not snapshot raw;
* a real isolated run writes strictly under ``output_root``; production fixed
  paths (``splits/``, ``reports/evaluation/``) are never touched;
* no product string value is absolute or carries a test-path token;
* the contamination guard refuses test-shaped production runs and isolated runs
  that target a production path;
* ``data/raw`` is provably unchanged after a full run.
"""

from __future__ import annotations

import csv as csvlib
from pathlib import Path

import pandas as pd
import pytest

from _stage6_fixtures import (
    build_stage6_world,
    default_windows,
    make_isolated_cfg,
    run_stage6,
    splits_out,
)
from psg_audio_benchmark.evaluation.qc import (
    EvaluationContaminationError,
    assert_not_contaminating,
    assert_paths_clean,
    snapshot_raw,
)

_RUN = "stage6-synth-20260808T000000Z-abcd1234"


def _world(cfg, tmp_path):
    return build_stage6_world(cfg, windows=default_windows())


def test_dry_run_writes_no_products_and_does_not_snapshot_raw(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids, dry_run=True)

    assert summary.overall_status == "PASS WITH WARNINGS"
    assert summary.license_gate_passed is True
    assert summary.raw_scanned is False and summary.raw_modified is False
    assert not splits_out(tmp_path).exists()


def test_isolated_run_writes_only_under_output_root(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids, run_id=_RUN)
    assert summary.overall_status == "PASS", summary.anomalies

    # products exist under the isolated output_root ...
    assert splits_out(tmp_path).is_dir()
    # ... and NOT under the production fixed paths.
    assert not (cfg.path("splits") / "runs" / _RUN).exists()
    assert not (cfg.path("reports_evaluation") / "runs" / _RUN).exists()
    assert not (cfg.path("reports_evaluation") / "LATEST_RUN.txt").exists()


def test_raw_data_is_unchanged_after_a_full_run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    before = snapshot_raw(cfg.path("data_raw"), tmp_path)
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    after = snapshot_raw(cfg.path("data_raw"), tmp_path)

    assert before == after                       # independent snapshot equality
    assert summary.raw_modified is False         # runner's own verdict


def _collect_product_strings(tmp_path) -> list:
    """String values from MACHINE-READABLE products (parquet/csv/yaml/json under
    splits_out) -- markdown reports are human prose and may legitimately mention
    guard tokens descriptively, so they are excluded (same scoping as Stage 4/5)."""
    values: list = []
    root = splits_out(tmp_path)
    if not root.is_dir():
        return values
    for p in root.iterdir():
        if p.suffix == ".csv":
            with open(p, "r", encoding="utf-8", newline="") as h:
                for row in csvlib.DictReader(h):
                    values.extend(v for v in row.values() if isinstance(v, str))
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
            for col in df.columns:
                if df[col].dtype == object:
                    values.extend(str(v) for v in df[col].dropna().tolist())
        elif p.suffix in (".yaml", ".yml", ".json"):
            values.extend(p.read_text(encoding="utf-8").splitlines())
    return values


def test_all_product_string_values_are_path_clean(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    assert_paths_clean(_collect_product_strings(tmp_path))
    yml = (splits_out(tmp_path) / "splits_resolved.yaml").read_text(encoding="utf-8")
    assert "C:\\" not in yml and "D:\\" not in yml


def test_contamination_guard_refuses_test_shaped_production_run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    with pytest.raises(EvaluationContaminationError):
        assert_not_contaminating(
            cfg=cfg, run_id="stage6-patient-splits-TEST-1234",
            config_hash="a" * 64, output_root=None,
        )


def test_contamination_guard_refuses_isolated_run_targeting_production_path(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    with pytest.raises(EvaluationContaminationError):
        assert_not_contaminating(
            cfg=cfg, run_id="stage6-patient-splits-20260808T000000Z",
            config_hash="a" * 64, output_root=cfg.path("splits"),
        )
