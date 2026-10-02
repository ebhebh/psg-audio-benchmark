"""Output isolation, contamination guard, raw-snapshot and path-purity for Stage 6.

Mirrors the Stage 1-5 read-only, run-dir-isolated discipline, adapted to the
Stage-6 output roots (``splits/runs/<run_id>`` and
``reports/evaluation/runs/<run_id>``):

* **Production** (``output_root`` unset): artifacts live under per-run dirs and
  the fixed mirror paths are regenerated only by the CLI after a successful run.
* **Isolated / test** (``output_root`` set): every artifact strictly under
  ``<output_root>``; production dirs never touched.
* A production run that looks like a test fixture (test run_id / config_hash /
  tmp raw) is refused; an isolated run whose output_root targets a production
  path is refused.

Stage 6 reads NO raw data and NO audio (``*.wav`` is never opened); ``data/raw``
is snapshotted (size:mtime) before/after to PROVE immutability. Stage 1-5
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
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

#: Production output roots that an isolated output_root must never target. Stage
#: 6 writes into ``splits`` and ``reports_evaluation``; the rest are prior-stage
#: / sibling production dirs that must stay untouched.
_PROD_OUTPUT_PATH_KEYS = (
    "splits",
    "reports_evaluation",
    "features_physiology",
    "features_audio",
    "features_multimodal",
    "data_manifests",
    "reports_windowing",
    "reports_feature_extraction",
    "annotations",
    "reports_annotations",
    "reports_data_audit",
    "reports_data_download",
)


class EvaluationContaminationError(RuntimeError):
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
            raise EvaluationContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}."
            )
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise EvaluationContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value."
            )
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise EvaluationContaminationError(
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
                raise EvaluationContaminationError(
                    f"Isolated output_root {root} must not be (or be inside) the "
                    f"production path {forbidden} ({key})."
                )


__all__ = [
    "EvaluationContaminationError",
    "assert_not_contaminating",
    "relativize",
    "snapshot_raw",
    "assert_paths_clean",
]
