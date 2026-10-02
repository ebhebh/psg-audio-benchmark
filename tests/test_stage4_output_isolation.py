"""Stage-4 boundary case 10: output isolation, contamination guards, path purity.

Guarantees (prompt section 6.10):
  * a fixture Stage-4 run writes ALL artifacts strictly under ``tmp_path`` and
    nothing under the real production run dirs;
  * production mode refuses a test run_id / test config_hash / tmp raw;
  * isolated mode refuses an output_root that is (or is inside) a production
    path (data_manifests / reports_windowing / annotations / reports_annotations
    / reports_data_audit / reports_data_download);
  * every string in every machine-readable product is relative + clean (no
    absolute / pytest / AppData / Temp tokens).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from psg_audio_benchmark.config import default_project_root
from psg_audio_benchmark.windowing import (
    Stage4Options,
    Stage4Runner,
    WindowingContaminationError,
)
from psg_audio_benchmark.windowing.qc import assert_paths_clean

from _stage4_fixtures import (
    DEFAULT_INPUT_RUN_ID,
    _patient_spec,
    build_input_run,
    make_isolated_cfg,
)

_EXPECTED_TABLES = (
    "csv_window_index.parquet",
    "window_event_links.parquet",
    "window_exclusions.csv",
    "patient_window_summary.csv",
    "windowing_config_resolved.yaml",
)
_EXPECTED_REPORTS = (
    "window_label_report.md",
    "windowing_qc_report.md",
    "phase_04_completion_report.md",
)
_BAD_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "/tmp/")


def _make_runner(tmp_path: Path, *, run_id: str, options: Stage4Options) -> Stage4Runner:
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    return Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": run_id, "config_hash": "realhash1234"},
        options=options,
    )


def test_case10_artifacts_strictly_under_tmp(tmp_path: Path) -> None:
    runner = _make_runner(
        tmp_path, run_id="iso-stage4-iso-1",
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    mdir = tmp_path / "outputs" / "data" / "manifests" / "runs" / "iso-stage4-iso-1"
    rdir = tmp_path / "outputs" / "reports" / "windowing" / "runs" / "iso-stage4-iso-1"
    for name in _EXPECTED_TABLES:
        assert (mdir / name).is_file(), f"missing product {name}"
    for name in _EXPECTED_REPORTS:
        assert (rdir / name).is_file(), f"missing report {name}"
    # nothing landed under the real production dirs
    root = default_project_root()
    for prod in ("data/manifests/runs", "reports/windowing/runs"):
        assert not (root / prod / "iso-stage4-iso-1").exists()


def _all_string_values(mdir: Path) -> list:
    values = []
    for p in sorted(mdir.iterdir()):
        if p.suffix == ".csv":
            with open(p, "r", encoding="utf-8", newline="") as h:
                for row in csv.DictReader(h):
                    values.extend(v for v in row.values() if isinstance(v, str))
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
            for col in df.columns:
                if df[col].dtype == object:
                    values.extend(str(v) for v in df[col].dropna().tolist())
    return values


def test_case10_products_carry_no_absolute_or_test_tokens(tmp_path: Path) -> None:
    runner = _make_runner(
        tmp_path, run_id="iso-stage4-clean-1",
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    runner.run()
    mdir = tmp_path / "outputs" / "data" / "manifests" / "runs" / "iso-stage4-clean-1"
    values = _all_string_values(mdir)
    assert values, "expected some string values to scan"
    for v in values:
        as_posix = v.replace("\\", "/")
        assert not Path(v).is_absolute(), f"absolute path in product: {v!r}"
        assert not as_posix.startswith("/"), f"absolute-ish path in product: {v!r}"
        low = v.lower()
        for tok in _BAD_TOKENS:
            assert tok not in low, f"test-path token {tok!r} in product: {v!r}"


def test_case10_empty_exclusions_csv_has_header(tmp_path: Path) -> None:
    # A patient whose whole domain is awake -> all windows awake-excluded, but
    # exclusions.csv still has a header. Use a domain fully inside awake.
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 90.0), spo2=(0.0, 90.0),
                       awake=[(0.0, 90.0)])],
    )
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage4-clean-2", "config_hash": "realhash1234"},
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    runner.run()
    mdir = tmp_path / "outputs" / "data" / "manifests" / "runs" / "iso-stage4-clean-2"
    p = mdir / "window_exclusions.csv"
    assert p.is_file() and p.stat().st_size > 0
    with open(p, "r", encoding="utf-8", newline="") as h:
        reader = csv.DictReader(h)
        assert "exclusion_reason" in (reader.fieldnames or [])
        list(reader)


# ---------------------------------------------------------------------------
# Production-mode contamination guards
# ---------------------------------------------------------------------------

def test_case10_production_rejects_test_run_id(tmp_path: Path) -> None:
    runner = _make_runner(
        tmp_path, run_id="stage4-test-run",
        options=Stage4Options(input_run_id=DEFAULT_INPUT_RUN_ID),  # production (no output_root)
    )
    with pytest.raises(WindowingContaminationError):
        runner.run()


def test_case10_production_rejects_test_config_hash(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260808T120000Z-abcdef12", "config_hash": "test"},
        options=Stage4Options(input_run_id=DEFAULT_INPUT_RUN_ID),
    )
    with pytest.raises(WindowingContaminationError):
        runner.run()


def test_case10_production_rejects_tmp_raw(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260808T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage4Options(input_run_id=DEFAULT_INPUT_RUN_ID),
    )
    with pytest.raises(WindowingContaminationError):
        runner.run()


# ---------------------------------------------------------------------------
# Isolated-mode output_root targeting a production path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "target_key",
    ["data_manifests", "reports_windowing", "annotations", "reports_annotations",
     "reports_data_audit", "reports_data_download"],
)
def test_case10_isolated_rejects_production_output_root(
    target_key: str, tmp_path: Path
) -> None:
    # placeholder config under the pytest-managed tmp dir (portable: a literal
    # "/tmp" maps to the drive root on Windows and may be unwritable); only
    # cfg.path() is used — the rejection under test comes from output_root.
    cfg = make_isolated_cfg(tmp_path)
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260808T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage4Options(output_root=cfg.path(target_key)),
    )
    with pytest.raises(WindowingContaminationError):
        runner.run()


# ---------------------------------------------------------------------------
# The purity guard itself must not be weakened
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "/abs/path/x",
    "C:\\Users\\someone\\AppData\\Local\\x",
    "x/pytest_tmp/y",
    "c:/temp/secret",
])
def test_case10_assert_paths_clean_rejects_bad_tokens(bad: str) -> None:
    with pytest.raises(AssertionError):
        assert_paths_clean([bad])


def test_case10_assert_paths_clean_accepts_clean_relative() -> None:
    assert_paths_clean(["data/raw/V5/Data/01/01_HR.csv", "runs/x/y.parquet"])
