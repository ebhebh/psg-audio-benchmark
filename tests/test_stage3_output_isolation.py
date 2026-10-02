"""Stage-3 boundary case 9: output isolation + contamination guards.

Guarantees (prompt section 6.9):
  * a fixture Stage-3 run writes ALL artifacts strictly under ``tmp_path``;
  * production mode refuses a test run_id / test config_hash / tmp raw;
  * isolated mode refuses an output_root that is (or is inside) a production
    path;
  * a passing fixture run emits the expected eight tables + four reports.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from psg_audio_benchmark.annotation_parser import (
    Stage3ContaminationError,
    Stage3Options,
    Stage3Runner,
)
from psg_audio_benchmark.config import Config, default_project_root

from _stage3_fixtures import build_stage3_patient, make_isolated_cfg

_EXPECTED_TABLES = (
    "parsed_events.parquet",
    "event_parse_rejections.csv",
    "awake_intervals_raw.parquet",
    "awake_intervals_canonical.parquet",
    "record_time_anchors.parquet",
    "signal_time_ranges.parquet",
    "audio_time_alignment_inventory.csv",
    "synchronization_warnings.csv",
)
_EXPECTED_REPORTS = (
    "annotation_parse_report.md",
    "time_synchronization_report.md",
    "event_type_mapping_report.md",
    "phase_03_completion_report.md",
)


def _make_runner(tmp_path: Path, *, run_id: str, options: Stage3Options) -> Stage3Runner:
    cfg = make_isolated_cfg(tmp_path)
    build_stage3_patient(cfg.path("data_raw"), "01", cross_midnight_csv=True,
                         short_audio=True)
    return Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": run_id, "config_hash": "realhash1234"},
        options=options,
    )


def test_fixture_artifacts_strictly_under_tmp(tmp_path: Path) -> None:
    runner = _make_runner(
        tmp_path, run_id="iso-stage3-iso-1",
        options=Stage3Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    adir = tmp_path / "outputs" / "annotations" / "runs" / "iso-stage3-iso-1"
    for name in _EXPECTED_TABLES:
        assert (adir / name).is_file(), f"missing product {name}"
    rdir = tmp_path / "outputs" / "reports" / "annotations" / "runs" / "iso-stage3-iso-1"
    for name in _EXPECTED_REPORTS:
        assert (rdir / name).is_file(), f"missing report {name}"
    # nothing landed under the real production dirs
    for prod in ("annotations/runs", "reports/annotations/runs"):
        assert not (default_project_root() / prod / "iso-stage3-iso-1").exists()


def test_parsed_events_standardized_types(tmp_path: Path) -> None:
    runner = _make_runner(
        tmp_path, run_id="iso-stage3-iso-2",
        options=Stage3Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    df = pd.read_parquet(tmp_path / "outputs" / "annotations" / "runs" /
                         "iso-stage3-iso-2" / "parsed_events.parquet")
    assert not df.empty
    assert set(df["event_type_standardized"]).issubset({"hypopnea", "apnea_unspecified"})
    assert set(df["is_any_scored_respiratory_event"]) == {True}
    assert summary.n_events_standardized == len(df)


# ---------------------------------------------------------------------------
# Production-mode contamination guards
# ---------------------------------------------------------------------------

def test_production_mode_rejects_test_run_id(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    build_stage3_patient(cfg.path("data_raw"), "01")
    # production mode but raw under tmp_path -> also outside project root, but the
    # run_id token is checked first.
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "stage3-test-run", "config_hash": "realhash1234"},
        options=Stage3Options(),  # no output_root => production
    )
    with pytest.raises(Stage3ContaminationError):
        runner.run()


def test_production_mode_rejects_test_config_hash(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())  # real raw (inside project)
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260807T120000Z-abcdef12", "config_hash": "test"},
        options=Stage3Options(),
    )
    with pytest.raises(Stage3ContaminationError):
        runner.run()


def test_production_mode_rejects_tmp_raw(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())
    cfg._resolved_paths["data_raw"] = tmp_path / "raw"  # outside project root
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260807T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage3Options(),
    )
    with pytest.raises(Stage3ContaminationError):
        runner.run()


# ---------------------------------------------------------------------------
# Isolated-mode output_root targeting production
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_key", ["annotations", "reports_annotations", "data_manifests", "reports_data_audit"])
def test_isolated_mode_rejects_production_output_root(target_key: str) -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260807T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage3Options(output_root=cfg.path(target_key)),
    )
    with pytest.raises(Stage3ContaminationError):
        runner.run()
