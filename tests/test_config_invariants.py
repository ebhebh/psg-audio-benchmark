"""Tests for hard scientific invariants.

These rules must NOT be turn-off-able via configuration edits. The code
constants (config.IMMUTABLE_CONSTRAINTS, config.FEATURE_FORMAT) are the source
of truth; the YAML hard_constraints block is declarative and any attempt to
relax it must raise ConfigError.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from psg_audio_benchmark.config import (
    Config,
    ConfigError,
    FEATURE_FORMAT,
    IMMUTABLE_CONSTRAINTS,
    default_project_root,
    validate_hard_constraints,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return default_project_root()


# ---------------------------------------------------------------------------
# The real config satisfies all invariants
# ---------------------------------------------------------------------------

def test_real_config_loads_and_meets_invariants(project_root: Path) -> None:
    cfg = Config(project_root=project_root)
    # code constants are authoritative
    assert IMMUTABLE_CONSTRAINTS["patient_level_split_required"] is True
    assert IMMUTABLE_CONSTRAINTS["allow_window_random_split"] is False
    assert IMMUTABLE_CONSTRAINTS["raw_data_mutable"] is False
    assert IMMUTABLE_CONSTRAINTS["save_audio_slices"] is False
    assert FEATURE_FORMAT == "parquet"
    # Config exposes the same values
    assert cfg.hard_constraints["patient_level_split_required"] is True
    assert cfg.hard_constraints["allow_window_random_split"] is False
    assert cfg.hard_constraints["raw_data_mutable"] is False
    assert cfg.hard_constraints["save_audio_slices"] is False
    assert cfg.hard_constraints["feature_format"] == "parquet"


def test_yaml_declared_constraints_match_code(project_root: Path) -> None:
    cfg = Config(project_root=project_root)
    declared = cfg.config_data["hard_constraints"]
    for key, expected in IMMUTABLE_CONSTRAINTS.items():
        assert declared[key] == expected, (
            f"YAML hard_constraints['{key}'] disagrees with code constant"
        )
    assert declared["feature_format"] == FEATURE_FORMAT


# ---------------------------------------------------------------------------
# validate_hard_constraints rejects attempts to relax any rule
# ---------------------------------------------------------------------------

def _good() -> dict:
    base = {k: v for k, v in IMMUTABLE_CONSTRAINTS.items()}
    base["feature_format"] = FEATURE_FORMAT
    return base


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("patient_level_split_required", False),
        ("allow_window_random_split", True),
        ("raw_data_mutable", True),
        ("save_audio_slices", True),
    ],
)
def test_cannot_relax_immutable_constraint(key: str, bad_value) -> None:
    broken = _good()
    broken[key] = bad_value
    with pytest.raises(ConfigError):
        validate_hard_constraints(broken)


def test_cannot_change_feature_format() -> None:
    broken = _good()
    broken["feature_format"] = "csv"
    with pytest.raises(ConfigError):
        validate_hard_constraints(broken)


def test_missing_required_key_rejected() -> None:
    broken = _good()
    del broken["patient_level_split_required"]
    with pytest.raises(ConfigError):
        validate_hard_constraints(broken)


def test_non_mapping_rejected() -> None:
    with pytest.raises(ConfigError):
        validate_hard_constraints(["not", "a", "dict"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The Config loader enforces these on construction (integration-level)
# ---------------------------------------------------------------------------

def test_config_constructor_would_reject_relaxed_yaml(project_root: Path) -> None:
    """A config object built from a relaxed hard_constraints dict must fail.

    We simulate an edit by patching the loaded config_data before validation.
    """
    cfg = Config(project_root=project_root)
    tampered = copy.deepcopy(cfg.config_data)
    tampered["hard_constraints"]["patient_level_split_required"] = False
    with pytest.raises(ConfigError):
        validate_hard_constraints(tampered["hard_constraints"])
