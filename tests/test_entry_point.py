"""Smoke test for the Phase-0 leftover console entry point (prompt section 0.1).

``pyproject.toml`` declares ``psgb-setup-check = "psg_audio_benchmark.__main__:main"``
but Phase 0 never created ``__main__.py``. Phase 1 fixes this with a minimal,
testable entry. This test verifies the module + ``main`` exist and run, and
that the declared entry-point string actually resolves.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from psg_audio_benchmark import __version__


def test_main_module_importable() -> None:
    mod = importlib.import_module("psg_audio_benchmark.__main__")
    assert hasattr(mod, "main")
    assert callable(mod.main)


def test_main_version_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from psg_audio_benchmark.__main__ import main

    rc = main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == __version__


def test_main_info_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from psg_audio_benchmark.__main__ import main

    rc = main(["--info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "PSG_Audio_Benchmark" in out
    assert "benchmark" in out.lower()


def test_main_no_args_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    from psg_audio_benchmark.__main__ import main

    assert main([]) == 0
    assert "PSG_Audio_Benchmark" in capsys.readouterr().out


def test_pyproject_entry_point_resolves(project_root_fixture: Path) -> None:
    """The [project.scripts] entry must point to an importable module:attr."""
    pyproject = project_root_fixture / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'psgb-setup-check\s*=\s*"([^:]+):([^"]+)"', text)
    assert m, "psgb-setup-check entry not found in pyproject.toml"
    module_path, attr = m.group(1), m.group(2)
    mod = importlib.import_module(module_path)
    assert hasattr(mod, attr), f"{module_path} has no attribute {attr}"
    assert callable(getattr(mod, attr))


@pytest.fixture(scope="module")
def project_root_fixture() -> Path:
    from psg_audio_benchmark.config import default_project_root

    return default_project_root()
