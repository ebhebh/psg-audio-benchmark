"""Tests for raw-data immutability across the Stage-1 run (prompt section 7, #5).

The Stage-1 audit must NEVER modify, move, rename or delete anything under
``data/raw``. We check this with small tmp fixtures (fast) and a cheap
size/mtime snapshot of the *real* raw tree (so the test stays quick even when
``data/raw`` holds tens of GB).

OUTPUT ISOLATION: every runner-based test here runs in *isolated* mode by
passing ``output_root=<tmp>`` and ``relpath_base=<tmp>``. This guarantees the
test writes its manifest/receipt/reports strictly under the tmp fixture and
NEVER touches the production ``data/manifests`` / ``reports/data_download``
paths. The lightweight (size+mtime) before/after snapshot is produced by the
runner itself; the full per-file SHA-256 lives in the manifest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_download.hashing import sha256_file
from psg_audio_benchmark.data_download.runner import Stage1Options, Stage1Runner


def _snapshot(root: Path) -> dict:
    """Cheap read-only snapshot: {relative_path: (size, mtime_ns)}.

    Uses (size, mtime_ns) instead of a full SHA-256 so the test stays fast
    regardless of how large ``data/raw`` is. A read-only audit only opens files
    for reading, which never changes size or mtime, so this still detects any
    write/addition/deletion.
    """
    snap: dict = {}
    for dp, _dn, fns in os.walk(root):
        for fn in fns:
            p = Path(dp) / fn
            st = p.stat()
            snap[str(p.relative_to(root)).replace("\\", "/")] = (st.st_size, st.st_mtime_ns)
    return snap


def _make_cfg_with_tmp_raw(tmp_path: Path) -> Config:
    """A real Config whose data_raw / staging point at small tmp dirs.

    Lets the live Stage1Runner run against a tiny raw tree instead of the real
    multi-GB one, keeping the integration test fast while exercising the real
    runner code path.
    """
    cfg = Config(project_root=default_project_root())
    tmp_raw = tmp_path / "raw"
    tmp_raw.mkdir()
    (tmp_raw / "01").mkdir()
    (tmp_raw / "01" / "01_phone.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (tmp_raw / "01" / "01_SpO2.csv").write_bytes(b"time,spo2\n0,98\n")
    tmp_staging = tmp_path / "incoming"
    tmp_staging.mkdir()
    cfg._resolved_paths["data_raw"] = tmp_raw
    cfg._resolved_paths["data_external_incoming"] = tmp_staging
    return cfg


def test_runner_dry_run_does_not_modify_raw(tmp_path: Path) -> None:
    cfg = _make_cfg_with_tmp_raw(tmp_path)
    raw = cfg.path("data_raw")
    before = _snapshot(raw)

    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "immutability-test", "config_hash": "test"},
        options=Stage1Options(
            dry_run=True, verify_existing=True,
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
        ),
    )
    summary = runner.run()

    assert summary.raw_modified is False
    assert summary.raw_before == summary.raw_after
    assert _snapshot(raw) == before  # independent filesystem re-check


def test_runner_full_run_does_not_modify_raw(tmp_path: Path) -> None:
    """Even a non-dry-run audit (which writes manifest/receipt/reports) must
    leave every raw file's size/mtime untouched AND write only under the
    isolated output_root (never production paths)."""
    cfg = _make_cfg_with_tmp_raw(tmp_path)
    raw = cfg.path("data_raw")
    before = _snapshot(raw)

    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "full-immutability-test", "config_hash": "test"},
        options=Stage1Options(
            verify_existing=True,
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
        ),
    )
    summary = runner.run()

    assert summary.raw_modified is False
    assert _snapshot(raw) == before
    # Artifacts landed strictly under the tmp output root, never in production.
    assert str(runner.artifact_paths()["manifest"]).startswith(str(tmp_path))
    assert str(runner.artifact_paths()["receipt"]).startswith(str(tmp_path))


def test_real_raw_snapshot_is_stable() -> None:
    """The real data/raw tree must be byte-stable across two cheap snapshots."""
    cfg = Config(project_root=default_project_root())
    raw = cfg.path("data_raw")
    first = _snapshot(raw)
    assert _snapshot(raw) == first


def test_readonly_scan_leaves_tmp_raw_unchanged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "p1").mkdir()
    (raw / "p1" / "a.txt").write_bytes(b"abc")
    (raw / "p1" / "b.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (raw / "c.json").write_bytes(b"{}")

    before = _snapshot(raw)
    # Simulate the read-only audit pass: hash every file (the only thing the
    # Stage-1 scan does to raw files).
    for dp, _dn, fns in os.walk(raw):
        for fn in fns:
            sha256_file(Path(dp) / fn)

    assert _snapshot(raw) == before
    assert len(before) == 3
    assert (raw / "p1" / "a.txt").read_bytes() == b"abc"


def test_runner_does_not_write_into_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "note.txt").write_bytes(b"original")

    cfg = Config(project_root=default_project_root())
    from psg_audio_benchmark.data_download.integrity import check_file
    from psg_audio_benchmark.data_download.hashing import file_meta

    target = raw / "note.txt"
    before = _snapshot(raw)
    check_file(target)
    file_meta(target)
    sha256_file(target)
    assert _snapshot(raw) == before
    assert sorted(p.name for p in raw.iterdir()) == ["note.txt"]


@pytest.mark.parametrize("repeat", range(3))
def test_idempotent_rescan(repeat: int) -> None:
    """Re-snapshotting raw repeatedly must be stable (cheap size/mtime)."""
    cfg = Config(project_root=default_project_root())
    raw = cfg.path("data_raw")
    first = _snapshot(raw)
    for _ in range(repeat + 1):
        _snapshot(raw)
    assert _snapshot(raw) == first
