"""Output isolation, contamination guard and path-purity helpers (Stage 3).

Mirrors the Stage-1/Stage-2 isolation discipline:

* **Production** (``output_root`` unset): artifacts live under per-run dirs
  ``annotations/runs/<run_id>/`` and ``reports/annotations/runs/<run_id>/``.
  Fixed paths are regenerated only by the CLI after a successful production run.
* **Isolated / test** (``output_root`` set): every artifact strictly under
  ``<output_root>``; production dirs never touched.
* A production run that *looks* like a test fixture (test run_id / config_hash /
  tmp raw / output_root pointed at a production dir) is refused.

Also: lightweight ``data/raw`` snapshots (before/after) and a path-purity check
that rejects absolute / pytest / AppData / Temp tokens in machine-readable
products (prompt section 6.10).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import Config

_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "\\tmp\\", "/tmp/")
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

#: Production output roots that an isolated output_root must never target.
_PROD_OUTPUT_PATH_KEYS = (
    "annotations",
    "reports_annotations",
    "data_manifests",
    "reports_data_audit",
    "reports_data_download",
)

_IGNORED_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}


class Stage3ContaminationError(RuntimeError):
    """Raised when a run would write test output into production paths, or when
    a production run looks like a test fixture."""


def assert_not_contaminating(
    *,
    cfg: Config,
    run_id: str,
    config_hash: str,
    output_root: Optional[Path],
) -> None:
    """Refuse test-shaped production runs and production-targeted isolated runs."""
    run_id_low = (run_id or "").lower()
    if output_root is None:
        hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
        if hit:
            raise Stage3ContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}."
            )
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise Stage3ContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value."
            )
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise Stage3ContaminationError(
                f"Production-mode run refused: data_raw {raw} is outside the "
                f"project root."
            ) from exc
    else:
        root = Path(output_root).resolve()
        for key in _PROD_OUTPUT_PATH_KEYS:
            forbidden = cfg.path(key).resolve()
            try:
                root.relative_to(forbidden)
                inside = True
            except ValueError:
                inside = root == forbidden
            if inside:
                raise Stage3ContaminationError(
                    f"Isolated output_root {root} must not be (or be inside) the "
                    f"production path {forbidden} ({key})."
                )


def relativize(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` as a forward-slashed string.

    Refuses paths that cannot be relativized to ``base`` (contamination guard).
    """
    try:
        rel = Path(path).resolve().relative_to(Path(base).resolve())
    except ValueError as exc:
        raise Stage3ContaminationError(
            f"Refusing to record a non-relativizable path {path!s} (base={base})."
        ) from exc
    return str(rel).replace("\\", "/")


def scan_raw_tree(root: Path) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for fname in filenames:
            if fname in _IGNORED_NAMES:
                continue
            out.append(Path(dirpath) / fname)
    return out


def snapshot_raw(raw_root: Path, base: Path) -> Dict[str, str]:
    """Lightweight ``size:mtime_ns`` snapshot of ``data/raw`` (read-only)."""
    snap: Dict[str, str] = {}
    for path in scan_raw_tree(raw_root):
        try:
            rel = relativize(path, base)
        except Stage3ContaminationError:
            rel = path.name
        try:
            st = path.stat()
            snap[rel] = f"{st.st_size}:{st.st_mtime_ns}"
        except OSError as exc:
            snap[rel] = f"-1:stat_failed:{type(exc).__name__}"
    return snap


def assert_paths_clean(values: Iterable[Any]) -> None:
    """Reject any string value that is absolute or carries a test-path token."""
    for v in values:
        if not isinstance(v, str) or not v:
            continue
        low = v.lower()
        as_posix = v.replace("\\", "/")
        if Path(v).is_absolute() or as_posix.startswith("/"):
            raise AssertionError(f"absolute path in output: {v!r}")
        for tok in _TEST_PATH_TOKENS:
            if tok in low:
                raise AssertionError(f"test-path token {tok!r} in output: {v!r}")


__all__ = [
    "Stage3ContaminationError",
    "assert_not_contaminating",
    "relativize",
    "scan_raw_tree",
    "snapshot_raw",
    "assert_paths_clean",
]
