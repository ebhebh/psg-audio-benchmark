"""Raw-immutability tests for Stage 2 (prompt section 2.4 & 7.7).

Stage 2 only reads raw files (audio header, CSV header/sample, JSON structure),
which never changes size/mtime. The before/after lightweight snapshot must be
identical -- both on the real ``data/raw`` and on a synthetic fixture across a
full runner pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_audit.runner import Stage2Options, Stage2Runner

from _stage2_fixtures import build_synthetic_raw, write_complete_license_evidence


def _lightweight_snapshot(raw: Path) -> dict:
    snap: dict = {}
    if not raw.exists():
        return snap
    for dp, _dn, fns in os.walk(raw):
        for fn in fns:
            p = Path(dp) / fn
            rel = str(p.relative_to(raw)).replace("\\", "/")
            st = p.stat()
            snap[rel] = f"{st.st_size}:{st.st_mtime_ns}"
    return snap


def test_real_raw_lightweight_snapshot_is_stable() -> None:
    cfg = Config(project_root=default_project_root())
    raw = cfg.path("data_raw")
    if not raw.exists() or not any(raw.rglob("*")):
        pytest.skip("real data/raw is absent/empty on this machine")
    first = _lightweight_snapshot(raw)
    second = _lightweight_snapshot(raw)
    assert first == second, "real data/raw changed between two read-only snapshots"


def test_synthetic_raw_unchanged_across_full_run(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    build_synthetic_raw(raw)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    write_complete_license_evidence(docs)
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["docs"] = docs

    before = _lightweight_snapshot(raw)

    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-immut-1", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    assert summary.raw_modified is False
    assert summary.raw_before == summary.raw_after

    after = _lightweight_snapshot(raw)
    assert before == after, "Stage-2 read-only audit modified the synthetic raw tree"


def test_no_absolute_paths_in_outputs(tmp_path: Path) -> None:
    """No output manifest/receipt path field is absolute or pytest-temp."""
    import csv

    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    build_synthetic_raw(raw)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    write_complete_license_evidence(docs)
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["docs"] = docs

    Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-immut-2", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    ).run()

    mdir = tmp_path / "outputs" / "manifests"
    for name in ("patient_manifest", "file_manifest_enriched",
                 "annotation_structure_inventory", "event_distribution_preliminary"):
        with open(mdir / f"{name}.csv", encoding="utf-8") as h:
            for row in csv.DictReader(h):
                for key, val in row.items():
                    if "path" in key.lower() or "dir" in key.lower():
                        if val:
                            assert not Path(val).is_absolute(), (
                                f"absolute path in {name}.{key}: {val}"
                            )
                            low = val.lower()
                            for tok in ("pytest", "appdata", "/temp/", "\\temp\\"):
                                assert tok not in low, f"bad token {tok!r} in {val}"
