"""Smoke test: setup check builds JSON + Markdown even with optional deps missing.

The current environment is missing torch/librosa/soundfile/lightgbm/xgboost and
ffmpeg. The setup check must NOT crash on these and must still emit a structured
report with the PASS/WARNING/FAIL/NOT_APPLICABLE vocabulary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from psg_audio_benchmark.config import default_project_root


def _load_setup_check_module():
    """Load scripts/00_setup_check.py by path (its name starts with a digit)."""
    root = default_project_root()
    script = root / "scripts" / "00_setup_check.py"
    assert script.is_file(), f"setup check script missing: {script}"
    spec = importlib.util.spec_from_file_location("setup_check_mod", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, root


@pytest.fixture(scope="module")
def setup_check():
    return _load_setup_check_module()


def test_build_report_returns_well_formed_dict(setup_check) -> None:
    mod, root = setup_check
    report = mod.build_report(root, argv=["prog"])
    assert "overall_status" in report
    assert report["overall_status"] in {
        mod.PASS,
        mod.WARNING,
        mod.FAIL,
        mod.NA,
    }
    assert "run_metadata" in report
    assert "checks" in report
    for key in (
        "os",
        "python",
        "hardware",
        "disk",
        "git",
        "ffmpeg",
        "packages",
        "gpu",
        "directories",
        "config",
        "risks",
    ):
        assert key in report["checks"], f"missing check {key}"


def test_setup_check_survives_missing_optional_packages(setup_check) -> None:
    """torch/librosa/etc may be missing; the report must still build."""
    mod, root = setup_check
    report = mod.build_report(root)
    pk = report["checks"]["packages"]
    # at least one optional package is missing in this environment
    missing = pk.get("missing", [])
    assert isinstance(missing, list)
    gpu = report["checks"]["gpu"]
    # GPU status must never be FAIL (GPU is optional -> WARNING at most)
    assert gpu["status"] != mod.FAIL
    # the report must build regardless
    assert report["overall_status"] in {mod.PASS, mod.WARNING, mod.FAIL}


def test_write_reports_creates_json_and_markdown(setup_check, tmp_path: Path) -> None:
    mod, root = setup_check
    report = mod.build_report(root)
    json_path, md_path = mod.write_reports(report, tmp_path, run_id_override="smoke")
    assert json_path.is_file()
    assert md_path.is_file()
    # JSON parses and carries the run id we supplied
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["run_metadata"]["run_id"] == "smoke"
    assert "overall_status" in data
    # Markdown mentions the status vocabulary and is non-trivial
    md = md_path.read_text(encoding="utf-8")
    assert "总体状态" in md
    assert "PASS" in md or "WARNING" in md or "FAIL" in md


def test_render_markdown_contains_key_sections(setup_check) -> None:
    mod, root = setup_check
    report = mod.build_report(root)
    md = mod.render_markdown(report)
    for section in (
        "GPU / CUDA / PyTorch",
        "Python 关键包",
        "目录可读写性",
        "配置加载与校验",
        "环境风险汇总",
    ):
        assert section in md


def test_setup_check_does_not_touch_raw_data(setup_check, tmp_path: Path) -> None:
    """build_report must not write into data/raw."""
    mod, root = setup_check
    raw_dir = root / "data" / "raw"
    before = {p.name for p in raw_dir.iterdir()} if raw_dir.is_dir() else set()
    mod.build_report(root)
    after = {p.name for p in raw_dir.iterdir()} if raw_dir.is_dir() else set()
    assert before == after, "setup_check wrote into data/raw (forbidden)"
