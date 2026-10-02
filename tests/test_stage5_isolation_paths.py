"""Stage 5 boundary cases: dry-run, output isolation, path purity, the
contamination guard, and raw-data immutability.

(9) Dry-run runs the gates but writes NO products and does not snapshot raw.
    A real isolated run writes strictly under output_root; production fixed
    paths are never touched.
(10) No product string value is absolute or carries a test-path token; the
     contamination guard refuses test-shaped production runs and isolated runs
     that target a production path; data/raw is provably unchanged.
"""

from __future__ import annotations

import csv as csvlib
from pathlib import Path

import pandas as pd
import pytest

from _stage5_fixtures import (
    build_stage5_world,
    features_out,
    make_isolated_cfg,
    reports_out,
    run_stage5,
)
from psg_audio_benchmark.feature_extraction.qc import (
    FeatureContaminationError,
    assert_not_contaminating,
    assert_paths_clean,
    snapshot_raw,
)

_SYNTH_RUN = "stage5-synth-20260808T000000Z-abcd1234"


def _world(cfg, tmp_path):
    patients = [{"patient_id": "01", "hr": [60.0] * 6, "spo2": [97.0] * 6}]
    windows = [{"patient_id": "01", "window_index": 0, "start_rel": 0.0, "end_rel": 3.0, "label_status": "positive"}]
    return build_stage5_world(cfg, patients=patients, windows=windows)


def test_dry_run_writes_no_products_and_does_not_snapshot_raw(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage5(cfg, tmp_path, dry_run=True, **ids)

    assert summary.overall_status == "PASS WITH WARNINGS"
    assert summary.license_gate_passed is True
    assert summary.raw_scanned is False and summary.raw_modified is False
    # No product dirs created at all.
    assert not features_out(tmp_path).exists()
    assert not reports_out(tmp_path).exists()


def test_isolated_run_writes_only_under_output_root(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)

    # Products exist under the isolated output_root ...
    assert features_out(tmp_path).is_dir() and reports_out(tmp_path).is_dir()
    # ... and NOT under the production fixed paths.
    assert not (cfg.path("features_physiology") / "runs" / _SYNTH_RUN).exists()
    assert not (cfg.path("reports_feature_extraction") / "runs" / _SYNTH_RUN).exists()
    assert not (cfg.path("reports_feature_extraction") / "LATEST_RUN.txt").exists()


def test_raw_data_is_unchanged_after_a_full_run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    before = snapshot_raw(cfg.path("data_raw"), tmp_path)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)
    after = snapshot_raw(cfg.path("data_raw"), tmp_path)

    assert before == after                       # independent snapshot equality
    assert summary.raw_modified is False         # runner's own verdict


def _collect_product_strings(tmp_path) -> list:
    """Collect string values from MACHINE-READABLE products only (parquet/csv/yaml
    under features_out) -- mirroring the runner's own purity-guard scope. Markdown
    reports are human prose and may legitimately mention the guard's tokens
    descriptively, so they are intentionally excluded (same as Stage 4)."""
    values: list = []
    root = features_out(tmp_path)
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
        elif p.suffix in (".yaml", ".yml"):
            values.extend(p.read_text(encoding="utf-8").splitlines())
    return values


def test_all_product_string_values_are_path_clean(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = _world(cfg, tmp_path)
    summary, _r, _o = run_stage5(cfg, tmp_path, **ids)
    # Must not raise: no absolute paths, no pytest/AppData/Temp/tmp tokens in any
    # machine-readable product.
    assert_paths_clean(_collect_product_strings(tmp_path))
    # The resolved config is project-relative (no absolute raw path leaked).
    yml = (features_out(tmp_path) / "physiology_features_resolved.yaml").read_text(encoding="utf-8")
    assert "C:\\" not in yml and "D:\\" not in yml


def test_contamination_guard_refuses_test_shaped_production_run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    with pytest.raises(FeatureContaminationError):
        assert_not_contaminating(
            cfg=cfg, run_id="stage5-physiology-TEST-1234",
            config_hash="a" * 64, output_root=None,
        )


def test_contamination_guard_refuses_isolated_run_targeting_production_path(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    with pytest.raises(FeatureContaminationError):
        assert_not_contaminating(
            cfg=cfg, run_id="stage5-physiology-features-20260808T000000Z",
            config_hash="a" * 64, output_root=cfg.path("features_physiology"),
        )
