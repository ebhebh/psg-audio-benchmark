"""Output isolation, contamination guard, raw-snapshot and path-purity for Stage 5.

Mirrors the Stage 1-4 read-only, run-dir-isolated discipline, adapted to the
Stage-5 output roots (``features/physiology/runs/<run_id>`` and
``reports/feature_extraction/runs/<run_id>``):

* **Production** (``output_root`` unset): artifacts live under per-run dirs and
  the fixed mirror paths are regenerated only by the CLI after a successful run.
* **Isolated / test** (``output_root`` set): every artifact strictly under
  ``<output_root>``; production dirs never touched.
* A production run that looks like a test fixture (test run_id / config_hash /
  tmp raw) is refused; an isolated run whose output_root targets a production
  path is refused.

Stage 5 reads raw physiological CSVs (read-only) but NEVER raw audio; ``data/raw``
is snapshotted (size:mtime) before/after to PROVE immutability. Stage 1-4
historical run dirs and fixed products are never rewritten.

The generic helpers (``relativize``, ``snapshot_raw``, ``assert_paths_clean``)
are reused verbatim from Stage 4 so there is a single path-purity implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import Config
from ..windowing.qc import (  # reuse the single, generic implementation
    assert_paths_clean,
    relativize,
    snapshot_raw,
)

_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "\\tmp\\", "/tmp/")
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

#: Production output roots that an isolated output_root must never target. Stage
#: 5 writes into ``features_physiology`` and ``reports_feature_extraction``; the
#: others are prior-stage / sibling production dirs that must stay untouched.
_PROD_OUTPUT_PATH_KEYS = (
    "features_physiology",
    "reports_feature_extraction",
    "features_audio",
    "features_multimodal",
    "data_manifests",
    "reports_windowing",
    "annotations",
    "reports_annotations",
    "reports_data_audit",
    "reports_data_download",
)


class FeatureContaminationError(RuntimeError):
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
            raise FeatureContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}."
            )
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise FeatureContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value."
            )
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise FeatureContaminationError(
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
                raise FeatureContaminationError(
                    f"Isolated output_root {root} must not be (or be inside) the "
                    f"production path {forbidden} ({key})."
                )


__all__ = [
    "FeatureContaminationError",
    "assert_not_contaminating",
    "relativize",
    "snapshot_raw",
    "assert_paths_clean",
]
