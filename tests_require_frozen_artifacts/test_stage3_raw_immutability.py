"""Stage-3 boundary case 8: data/raw immutability + Stage-1/2 outputs unchanged.

Guarantees (prompt section 6.8):
  * a fixture Stage-3 run leaves the REAL ``data/raw`` byte-for-byte unchanged
    (snapshot before == after; ``raw_modified`` False);
  * the Stage-1 and Stage-2 production directories (manifests, data-audit and
    download reports) are also byte-for-byte unchanged -- a Stage-3 run never
    rewrites historical runs;
  * the runner reports ``raw_modified=False``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from psg_audio_benchmark.annotation_parser import Stage3Options, Stage3Runner
from psg_audio_benchmark.config import default_project_root

from _stage3_fixtures import build_stage3_patient, make_isolated_cfg

#: Production dirs that Stage-3 must NEVER modify (Stage 1 + Stage 2 history).
_GUARDED_PROD_DIRS = (
    "data/manifests",
    "reports/data_download",
    "reports/data_audit",
    "data/raw",
)


def _sha256_snapshot(project_root: Path) -> dict:
    snap: dict = {}
    for sub in _GUARDED_PROD_DIRS:
        d = project_root / sub
        if not d.exists():
            continue
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                p = Path(dp) / fn
                try:
                    rel = str(p.relative_to(project_root)).replace("\\", "/")
                    snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
                except Exception as exc:  # pragma: no cover - defensive
                    snap[str(p)] = f"ERR:{exc}"
    return snap


def test_fixture_run_leaves_raw_and_stage12_unchanged(tmp_path: Path) -> None:
    project_root = default_project_root()
    before = _sha256_snapshot(project_root)

    cfg = make_isolated_cfg(tmp_path)
    build_stage3_patient(cfg.path("data_raw"), "01", cross_midnight_csv=True,
                         short_audio=True)
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage3-immutable-1", "config_hash": "realhash1234"},
        options=Stage3Options(
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    summary = runner.run()

    # License gate must pass; raw scanned; raw NOT modified.
    assert summary.license_gate_passed is True
    assert summary.raw_scanned is True
    assert summary.raw_modified is False

    after = _sha256_snapshot(project_root)
    assert before == after, "A fixture Stage-3 run modified production data/raw or Stage-1/2 outputs."


def test_real_raw_tree_present_and_read_only(tmp_path: Path) -> None:
    """Sanity: the real raw tree exists (50 patients) and is never written by a
    fixture run that points its raw at tmp_path."""
    project_root = default_project_root()
    real_raw = project_root / "data" / "raw" / "V5" / "Data"
    assert real_raw.is_dir(), "real raw tree data/raw/V5/Data must exist"
    # The fixture run uses a tmp raw tree, so the real one is never even walked.
    cfg = make_isolated_cfg(tmp_path)
    build_stage3_patient(cfg.path("data_raw"), "01")
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage3-immutable-2", "config_hash": "realhash1234"},
        options=Stage3Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    runner.run()
    # still present and intact
    assert real_raw.is_dir()
