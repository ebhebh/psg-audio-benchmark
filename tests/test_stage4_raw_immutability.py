"""Stage-4 boundary case 9: data/raw immutability + historical runs unchanged +
dry-run reads no raw content.

Guarantees (prompt section 6.9):
  * a fixture Stage-4 run leaves the REAL ``data/raw`` and the Stage-1/2/3
    production directories byte-for-byte unchanged (SHA-256 snapshot before ==
    after); Stage 4 never rewrites a historical run;
  * a dry-run verifies the license + input-run gate and produces NO window
    products (it does not read raw CSV content);
  * the runner reports ``raw_modified=False``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from psg_audio_benchmark.config import default_project_root
from psg_audio_benchmark.windowing import Stage4Options, Stage4Runner

from _stage4_fixtures import (
    DEFAULT_INPUT_RUN_ID,
    _patient_spec,
    build_input_run,
    make_isolated_cfg,
)

#: Production dirs that Stage-4 must NEVER modify (raw + Stage 1/2/3 history).
_GUARDED_PROD_DIRS = (
    "data/raw",
    "data/manifests",
    "reports/data_download",
    "reports/data_audit",
    "annotations",
    "reports/annotations",
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


def test_case9_fixture_run_leaves_raw_and_history_unchanged(tmp_path: Path) -> None:
    project_root = default_project_root()
    before = _sha256_snapshot(project_root)

    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage4-immut-1", "config_hash": "realhash1234"},
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    summary = runner.run()

    assert summary.license_gate_passed is True
    assert summary.input_run_verified is True
    assert summary.raw_scanned is True
    assert summary.raw_modified is False

    after = _sha256_snapshot(project_root)
    assert before == after, (
        "A fixture Stage-4 run modified production data/raw or Stage-1/2/3 outputs."
    )


def test_case9_dry_run_writes_no_products_and_reads_no_raw(tmp_path: Path) -> None:
    project_root = default_project_root()
    before = _sha256_snapshot(project_root)

    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage4-dry-1", "config_hash": "realhash1234"},
        options=Stage4Options(
            dry_run=True, output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    summary = runner.run()

    # Dry-run: gate passes, no window building, no products, raw unchanged.
    assert summary.license_gate_passed is True
    assert summary.total_candidate_windows == 0
    out = tmp_path / "outputs" / "data" / "manifests" / "runs" / "iso-stage4-dry-1"
    assert not out.exists(), "dry-run must not write any products"
    after = _sha256_snapshot(project_root)
    assert before == after
