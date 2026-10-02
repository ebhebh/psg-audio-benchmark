"""Tests for archive safety (prompt section 7, test #3).

Covers: archives containing absolute paths or ``..`` traversal members are
detected and the dangerous members are NOT extracted; a corrupt archive is
reported (``corrupt_archive``) without crashing the audit.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from psg_audio_benchmark.data_download.archives import (
    ARCHIVE_CORRUPT,
    KIND_TAR,
    KIND_UNKNOWN,
    KIND_ZIP,
    detect_archive_kind,
    inspect_archive,
    safe_extract,
)


def _make_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ok/file.txt", "safe")
        zf.writestr("../escape.txt", "should-be-blocked")
        zf.writestr("/absolute_evil.txt", "should-be-blocked")


def test_detect_zip_kind(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    _make_zip(z)
    assert detect_archive_kind(z) == KIND_ZIP


def test_zip_dangerous_members_detected(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    _make_zip(z)
    insp = inspect_archive(z)
    assert insp.kind == KIND_ZIP
    assert insp.has_dangerous_members
    assert insp.n_absolute >= 1
    assert insp.n_path_escape >= 1
    assert not insp.corrupt
    assert insp.test_status == "ok"


def test_safe_extract_skips_dangerous_and_writes_only_safe(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    _make_zip(z)
    dest = tmp_path / "out"
    insp, skipped, status = safe_extract(
        z, dest, min_free_bytes=0, forbidden_prefix=tmp_path / "raw"
    )
    assert status == "extracted"
    assert set(skipped) == {"../escape.txt", "/absolute_evil.txt"}
    # safe member extracted under dest
    assert (dest / "ok" / "file.txt").read_text(encoding="utf-8") == "safe"
    # nothing written above dest
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_refuses_data_raw(tmp_path: Path) -> None:
    z = tmp_path / "a.zip"
    _make_zip(z)
    # dest resolves inside the forbidden raw prefix -> refused
    raw = tmp_path / "raw"
    raw.mkdir()
    dest = raw / "extracted"
    _insp, skipped, status = safe_extract(
        z, dest, min_free_bytes=0, forbidden_prefix=raw
    )
    assert status == "skipped_forbidden"
    assert not dest.exists()


def test_corrupt_zip_reported_without_crash(tmp_path: Path) -> None:
    z = tmp_path / "good.zip"
    _make_zip(z)
    data = z.read_bytes()
    # truncate to simulate a corrupted/truncated archive
    truncated = tmp_path / "broken.zip"
    truncated.write_bytes(data[: len(data) // 2])
    insp = inspect_archive(truncated)
    # must not raise; must be flagged corrupt
    assert insp.test_status == ARCHIVE_CORRUPT
    assert insp.corrupt


def test_corrupt_zip_extraction_skipped(tmp_path: Path) -> None:
    z = tmp_path / "good.zip"
    _make_zip(z)
    data = z.read_bytes()
    broken = tmp_path / "broken.zip"
    broken.write_bytes(data[: len(data) // 2])
    dest = tmp_path / "out"
    _insp, _skipped, status = safe_extract(broken, dest, min_free_bytes=0)
    assert status == "skipped_corrupt"
    assert not dest.exists() or not any(dest.iterdir())


def test_tar_traversal_member_detected(tmp_path: Path) -> None:
    t = tmp_path / "evil.tar"
    # build a tar with a traversal entry manually
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name="../escape_tar.txt")
        data = b"x"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        good = tarfile.TarInfo(name="inside.txt")
        good.size = 1
        tf.addfile(good, io.BytesIO(b"y"))
    t.write_bytes(buf.getvalue())
    insp = inspect_archive(t)
    assert insp.kind == KIND_TAR
    assert insp.has_dangerous_members
    assert insp.n_path_escape >= 1


def test_unknown_kind_not_crash(tmp_path: Path) -> None:
    f = tmp_path / "not_an_archive.bin"
    f.write_bytes(b"\x00\x01\x02not an archive")
    insp = inspect_archive(f)
    assert insp.kind == KIND_UNKNOWN
    assert not insp.corrupt
