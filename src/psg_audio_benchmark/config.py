"""Configuration loading, project-root resolution and path/constraint checks.

Design rules (enforced here so they cannot be relaxed by editing YAML):

* ``project_root`` is resolved from the location of this package on disk, or
  from the ``PSG_BENCHMARK_ROOT`` environment variable. It is NEVER taken from
  a hardcoded developer absolute path.
* Every path declared in ``config/paths.yaml`` must be relative and must
  resolve *inside* the project root. Absolute paths and directory-escape
  (``..`` above root) are rejected.
* ``raw_data_read_only`` must be ``true``.
* The hard scientific constraints (patient-level split, no window random
  split, raw immutability, no saved audio slices, parquet features) are defined
  as Python constants in :data:`IMMUTABLE_CONSTRAINTS` and
  :data:`FEATURE_FORMAT`. The YAML ``hard_constraints`` block is declarative;
  if it disagrees with these constants the loader raises
  :class:`ConfigError` instead of silently relaxing the rule.

Importing this module does NOT create directories, write files, or read data.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml  # PyYAML
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required to read configuration. "
        "Install it with: python -m pip install pyyaml"
    ) from _exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Environment variable that overrides the auto-detected project root.
ENV_PROJECT_ROOT = "PSG_BENCHMARK_ROOT"

#: Subdirectory holding the YAML files.
CONFIG_DIRNAME = "config"

#: Hard scientific constraints. Source of truth lives in CODE, not YAML, so a
#: careless config edit cannot turn a constraint off.
IMMUTABLE_CONSTRAINTS: Dict[str, Any] = {
    "patient_level_split_required": True,
    "allow_window_random_split": False,
    "raw_data_mutable": False,
    "save_audio_slices": False,
}

#: Required feature storage format.
FEATURE_FORMAT = "parquet"

#: YAML key -> relative path (under data/) that is strictly read-only.
RAW_READ_ONLY_FLAG = "raw_data_read_only"


class ConfigError(Exception):
    """Raised when configuration violates a hard rule."""


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def default_project_root() -> Path:
    """Return the project root.

    Resolution order:
      1. ``$PSG_BENCHMARK_ROOT`` if set.
      2. Derived from this file's location:
         ``<root>/src/psg_audio_benchmark/config.py`` -> ``<root>``.
    """
    env = os.environ.get(ENV_PROJECT_ROOT)
    if env:
        return Path(env).expanduser().resolve()
    # config.py lives at <root>/src/psg_audio_benchmark/config.py
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} did not parse to a mapping.")
    return data


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------

def validate_relative_path(
    raw: str, project_root: Path, source: str
) -> Path:
    """Validate that ``raw`` is a relative path contained inside ``project_root``.

    Returns the resolved absolute path. Raises :class:`ConfigError` if the path
    is absolute or escapes the project root.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"{source}: empty or non-string path value.")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ConfigError(
            f"{source}: absolute paths are forbidden (found '{raw}'). "
            f"Use a path relative to the project root."
        )
    root = project_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ConfigError(
            f"{source}: path escapes the project root "
            f"('{raw}' -> '{resolved}')."
        )
    return resolved


# ---------------------------------------------------------------------------
# Hard-constraint validation
# ---------------------------------------------------------------------------

def validate_hard_constraints(hard_constraints: Dict[str, Any]) -> None:
    """Ensure the declarative YAML constraints agree with the code constants."""
    if not isinstance(hard_constraints, dict):
        raise ConfigError("'hard_constraints' must be a mapping.")
    # every immutable constraint must be present and equal to the code constant
    for key, expected in IMMUTABLE_CONSTRAINTS.items():
        if key not in hard_constraints:
            raise ConfigError(
                f"hard_constraints is missing required key '{key}'."
            )
        actual = hard_constraints[key]
        if actual != expected:
            raise ConfigError(
                f"hard_constraints['{key}'] = {actual!r} is not allowed; "
                f"the immutable rule requires {expected!r}. This constraint "
                f"cannot be turned off via configuration."
            )
    # feature format
    fmt = hard_constraints.get("feature_format")
    if fmt != FEATURE_FORMAT:
        raise ConfigError(
            f"hard_constraints['feature_format'] = {fmt!r}; only "
            f"{FEATURE_FORMAT!r} is permitted."
        )


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------

class Config:
    """Loaded configuration with validated, resolved paths."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_dirname: str = CONFIG_DIRNAME,
    ) -> None:
        self.project_root: Path = (project_root or default_project_root()).resolve()
        if not self.project_root.is_dir():
            raise ConfigError(
                f"Resolved project root does not exist or is not a directory: "
                f"{self.project_root}"
            )
        self.config_dir: Path = (self.project_root / config_dirname).resolve()

        self.paths_yaml: Path = self.config_dir / "paths.yaml"
        self.config_yaml: Path = self.config_dir / "config.yaml"
        self.experiments_yaml: Path = self.config_dir / "experiments.yaml"

        self.paths_data: Dict[str, Any] = _load_yaml(self.paths_yaml)
        self.config_data: Dict[str, Any] = _load_yaml(self.config_yaml)

        # Validate the raw read-only declaration first.
        if not bool(self.paths_data.get(RAW_READ_ONLY_FLAG, False)):
            raise ConfigError(
                f"{self.paths_yaml.name}: '{RAW_READ_ONLY_FLAG}' must be true."
            )

        # Resolve and validate every declared path.
        self._resolved_paths: Dict[str, Path] = self._resolve_paths()

        # Enforce hard constraints (code constants vs declarative YAML).
        hard = self.config_data.get("hard_constraints", {})
        validate_hard_constraints(hard)
        self.hard_constraints: Dict[str, Any] = dict(IMMUTABLE_CONSTRAINTS)
        self.hard_constraints["feature_format"] = FEATURE_FORMAT

    # -- paths -------------------------------------------------------------
    def _resolve_paths(self) -> Dict[str, Path]:
        paths_block = self.paths_data.get("paths")
        if not isinstance(paths_block, dict):
            raise ConfigError(
                f"{self.paths_yaml.name}: a 'paths:' mapping is required."
            )
        resolved: Dict[str, Path] = {}
        for key, raw in paths_block.items():
            resolved[key] = validate_relative_path(
                raw, self.project_root, source=f"paths.{key}"
            )
        return resolved

    @property
    def paths(self) -> Dict[str, Path]:
        """Mapping of path key -> resolved absolute path inside project root."""
        return dict(self._resolved_paths)

    def path(self, key: str) -> Path:
        """Return the resolved absolute path for ``key`` (e.g. 'data_raw')."""
        if key not in self._resolved_paths:
            raise ConfigError(f"Unknown path key: {key!r}")
        return self._resolved_paths[key]

    def relative_path(self, key: str) -> str:
        """Return the configured relative path string for ``key``."""
        block = self.paths_data.get("paths", {})
        if key not in block:
            raise ConfigError(f"Unknown path key: {key!r}")
        return str(block[key])

    def all_paths_inside_root(self) -> bool:
        """True iff every resolved path stays inside the project root."""
        root = self.project_root
        for resolved in self._resolved_paths.values():
            try:
                resolved.relative_to(root)
            except ValueError:
                return False
        return True

    def has_absolute_path_values(self) -> bool:
        """True iff any configured path value is absolute (should be False)."""
        for raw in self.paths_data.get("paths", {}).values():
            if isinstance(raw, str) and Path(raw).is_absolute():
                return True
        return False

    # -- experiments (optional, lazy) --------------------------------------
    def load_experiments(self) -> Dict[str, Any]:
        """Load experiments.yaml if present (no results expected there)."""
        return _load_yaml(self.experiments_yaml)

    # -- summary -----------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "config_dir": str(self.config_dir),
            "raw_data_read_only": bool(
                self.paths_data.get(RAW_READ_ONLY_FLAG, False)
            ),
            "n_path_keys": len(self._resolved_paths),
            "all_paths_inside_root": self.all_paths_inside_root(),
            "has_absolute_path_values": self.has_absolute_path_values(),
            "hard_constraints": dict(self.hard_constraints),
        }


def load_config(project_root: Optional[Path] = None) -> Config:
    """Convenience constructor used by scripts and tests."""
    return Config(project_root=project_root)


__all__ = [
    "Config",
    "ConfigError",
    "IMMUTABLE_CONSTRAINTS",
    "FEATURE_FORMAT",
    "ENV_PROJECT_ROOT",
    "default_project_root",
    "validate_relative_path",
    "validate_hard_constraints",
    "load_config",
]
