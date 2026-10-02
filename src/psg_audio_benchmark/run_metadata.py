"""Reproducible run metadata.

Generates a run id and captures the non-sensitive context needed to reproduce a
run: UTC/local timestamps, timezone, a deterministic hash of the configuration,
git information, Python/platform info and the resolved package versions.

Privacy rules:
  * No patient identifiers or absolute raw-data filenames are recorded.
  * Only relative configured paths are stored; absolute raw file paths are
    never written here.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

#: Package versions worth capturing (all optional; missing -> "not_installed").
TRACKED_PACKAGES: Iterable[str] = (
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "soundfile",
    "librosa",
    "matplotlib",
    "pyyaml",
    "pytest",
    "torch",
    "lightgbm",
    "xgboost",
)


class RunMetadataError(Exception):
    """Raised for non-fatal reproducibility problems (kept rare)."""


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_now_iso() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _compact_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    """Full SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(*config_texts: str) -> str:
    """Deterministic SHA-256 over the concatenated canonical config text.

    Given identical config file contents this always returns the same digest,
    which makes run metadata reproducible and testable.
    """
    joined = "\n---config-separator---\n".join(t for t in config_texts)
    return sha256_text(joined)


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def git_info(project_root: Path) -> Dict[str, Any]:
    """Capture git state without raising if git is absent or uninitialized."""
    info: Dict[str, Any] = {
        "git_available": False,
        "initialized": False,
        "commit": None,
        "branch": None,
        "dirty": None,
        "error": None,
    }
    git_bin = shutil.which("git")
    if not git_bin:
        info["error"] = "git binary not found on PATH"
        return info
    info["git_available"] = True

    def _run(args: list) -> tuple:
        try:
            proc = subprocess.run(
                [git_bin, *args],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=8,
            )
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except Exception as exc:  # pragma: no cover - environment dependent
            return None, "", str(exc)

    rc, out, _err = _run(["rev-parse", "--is-inside-work-tree"])
    if rc != 0 or out.strip() != "true":
        info["error"] = "repository not initialized (no valid .git work tree)"
        return info

    info["initialized"] = True
    rc, commit, _ = _run(["rev-parse", "HEAD"])
    info["commit"] = commit if rc == 0 and commit else None
    rc, branch, _ = _run(["rev-parse", "--abbrev-ref", "HEAD"])
    info["branch"] = branch if rc == 0 and branch else None
    rc, status, _ = _run(["status", "--porcelain"])
    info["dirty"] = bool(status.strip()) if rc == 0 else None
    return info


# ---------------------------------------------------------------------------
# Python / platform / packages
# ---------------------------------------------------------------------------

def python_info() -> Dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version,
        "version_info": list(sys.version_info[:5]),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


#: Map of distribution name -> import name, for packages whose two differ.
_IMPORT_ALIASES: Dict[str, str] = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
}


def package_versions(packages: Iterable[str] = TRACKED_PACKAGES) -> Dict[str, str]:
    """Return installed package versions; missing packages -> 'not_installed'.

    Resolution order per package:
      1. ``importlib.metadata.version(name)`` — keyed by the *distribution*
         name, so it is correct even when the pip name differs from the import
         name (e.g. ``pyyaml``/``yaml``, ``scikit-learn``/``sklearn``).
      2. Fall back to importing the module (using the alias if any) and reading
         ``__version__``.
    This avoids falsely reporting an installed package as missing.
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, version as _md_version

    versions: Dict[str, str] = {}
    for name in packages:
        # 1) distribution metadata (handles pip-name != import-name)
        try:
            versions[name] = str(_md_version(name))
            continue
        except PackageNotFoundError:
            pass
        # 2) import the module directly
        imp = _IMPORT_ALIASES.get(name, name)
        try:
            mod = importlib.import_module(imp)
        except Exception:
            versions[name] = "not_installed"
            continue
        ver = getattr(mod, "__version__", None) or "unknown"
        versions[name] = str(ver)
    return versions


# ---------------------------------------------------------------------------
# Run metadata assembly
# ---------------------------------------------------------------------------

def generate_run_id(config_digest: str) -> str:
    """Build a human-readable, sortable run id from UTC time + config digest."""
    return f"{_compact_utc()}-{config_digest[:8]}"


def build_run_metadata(
    project_root: Path,
    config_texts: Dict[str, str],
    argv: Optional[list] = None,
    packages: Iterable[str] = TRACKED_PACKAGES,
) -> Dict[str, Any]:
    """Assemble the full run metadata dict.

    ``config_texts`` maps a short label (e.g. 'config.yaml') to that file's raw
    text. Only the *hash* is stored, never raw file contents, to keep the
    metadata compact and to avoid leaking absolute raw paths that might appear
    inside free-text fields.
    """
    ordered = [config_texts[k] for k in sorted(config_texts)]
    digest = config_hash(*ordered)
    run_id = generate_run_id(digest)

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "utc_time": utc_now_iso(),
        "local_time": local_now_iso(),
        "timezone": _local_timezone_name(),
        "config_hash": digest,
        "config_files": sorted(config_texts.keys()),
        "project_root_relative_note": (
            "project paths are stored relative to the project root; "
            "no absolute raw-data filenames are recorded"
        ),
        "git": git_info(project_root),
        "python": python_info(),
        "packages": package_versions(packages),
        "argv": list(argv if argv is not None else sys.argv),
    }
    return metadata


def _local_timezone_name() -> str:
    try:
        return datetime.now().astimezone().tzinfo.tzname(None) or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def assert_no_absolute_raw_path(metadata: Dict[str, Any]) -> None:
    """Sanity guard: ensure no value looks like an absolute raw-data path.

    This checks the *metadata structure* only; it does not inspect patient data.
    It rejects any string value that is an absolute path containing a raw-data
    segment (e.g. '/data/raw/...' or 'D:\\...\\data\\raw\\...').
    """
    raw_markers = ("data/raw", "data\\raw", "data/external", "data\\external")

    def _walk(obj: Any) -> None:
        if isinstance(obj, str):
            as_path = obj.replace("\\", "/")
            if (Path(obj).is_absolute() or as_path.startswith("/")) and any(
                m.replace("\\", "/") in as_path for m in raw_markers
            ):
                raise RunMetadataError(
                    f"Refusing to record an absolute raw-data path in run "
                    f"metadata: {obj!r}"
                )
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)

    _walk(metadata)


__all__ = [
    "TRACKED_PACKAGES",
    "RunMetadataError",
    "utc_now_iso",
    "local_now_iso",
    "sha256_text",
    "config_hash",
    "git_info",
    "python_info",
    "package_versions",
    "generate_run_id",
    "build_run_metadata",
    "assert_no_absolute_raw_path",
]
