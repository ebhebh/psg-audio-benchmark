"""Tests for path configuration and project-root resolution.

Covers: all configured paths resolve inside the project root; the raw-read-only
declaration is present and true; no absolute path values exist in paths.yaml;
project root resolves to the directory that actually contains config/ and src/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from psg_audio_benchmark.config import (
    Config,
    ConfigError,
    default_project_root,
    validate_relative_path,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    root = default_project_root()
    assert root.is_dir(), f"project root not found: {root}"
    return root


@pytest.fixture(scope="module")
def cfg(project_root: Path) -> Config:
    return Config(project_root=project_root)


# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------

def test_project_root_contains_config_and_src(project_root: Path) -> None:
    assert (project_root / "config").is_dir()
    assert (project_root / "src").is_dir()
    assert (project_root / "config" / "paths.yaml").is_file()
    assert (project_root / "config" / "config.yaml").is_file()


def test_default_project_root_matches_package_location() -> None:
    root = default_project_root()
    # config.py lives at <root>/src/psg_audio_benchmark/config.py
    assert (root / "src" / "psg_audio_benchmark" / "config.py").is_file()


# ---------------------------------------------------------------------------
# Raw read-only declaration
# ---------------------------------------------------------------------------

def test_raw_data_read_only_true(cfg: Config) -> None:
    assert cfg.paths_data["raw_data_read_only"] is True


# ---------------------------------------------------------------------------
# No absolute paths; all paths inside root
# ---------------------------------------------------------------------------

def test_no_absolute_path_values(cfg: Config) -> None:
    assert cfg.has_absolute_path_values() is False


def test_all_paths_resolve_inside_root(cfg: Config) -> None:
    assert cfg.all_paths_inside_root() is True
    root = cfg.project_root
    for key, resolved in cfg.paths.items():
        # each resolved path must be the project root or a descendant
        assert resolved == root or root in resolved.parents, (
            f"path '{key}' escapes root: {resolved}"
        )


def test_expected_path_keys_present(cfg: Config) -> None:
    expected = {
        "data_raw", "data_external", "data_interim", "data_processed",
        "data_manifests", "annotations", "features", "features_audio",
        "features_physiology", "features_multimodal", "splits", "models",
        "results", "figures", "tables", "reports", "logs",
    }
    assert expected.issubset(set(cfg.paths.keys()))


def test_known_paths_have_expected_names(cfg: Config) -> None:
    assert cfg.path("data_raw").name == "raw"
    assert cfg.path("data_manifests").name == "manifests"
    assert cfg.path("features_audio").name == "audio"


# ---------------------------------------------------------------------------
# validate_relative_path rejects bad input
# ---------------------------------------------------------------------------

def test_validate_relative_path_rejects_absolute(project_root: Path) -> None:
    with pytest.raises(ConfigError):
        validate_relative_path("/etc/passwd", project_root, "test")
    # Windows-style absolute
    with pytest.raises(ConfigError):
        validate_relative_path("D:\\some\\where", project_root, "test")


def test_validate_relative_path_rejects_escape(project_root: Path) -> None:
    with pytest.raises(ConfigError):
        validate_relative_path("../../../../etc", project_root, "test")


def test_validate_relative_path_accepts_valid(project_root: Path) -> None:
    resolved = validate_relative_path("data/raw", project_root, "test")
    assert resolved == (project_root / "data" / "raw").resolve()
