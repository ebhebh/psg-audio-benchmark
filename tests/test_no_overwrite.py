"""Tests for the non-destructive staging rule (prompt section 7, test #2).

Same content + same path -> ``already_present`` (no overwrite).
Different content + same path -> ``conflict`` (original left byte-for-byte
unchanged; never replaced).
Different path + same content -> recorded as ``duplicate_content``.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.data_download.downloader import (
    STATUS_ALREADY_PRESENT,
    STATUS_CONFLICT,
    STATUS_DOWNLOADED,
    stage_bytes,
)
from psg_audio_benchmark.data_download.manifest import (
    STATUS_DUPLICATE_CONTENT,
    ManifestRow,
    decide_target_action,
    mark_duplicates,
)


def test_decide_target_action_absent_ok() -> None:
    assert decide_target_action(False, "", "abc") == "ok"


def test_decide_target_action_same_hash_already_present() -> None:
    assert decide_target_action(True, "abc", "abc") == STATUS_ALREADY_PRESENT


def test_decide_target_action_diff_hash_conflict() -> None:
    assert decide_target_action(True, "abc", "xyz") == STATUS_CONFLICT


def test_stage_same_bytes_is_already_present(tmp_path: Path) -> None:
    staging = tmp_path / "incoming"
    data = b"payload-A"
    r1 = stage_bytes(staging, "f.bin", data)
    assert r1.status == STATUS_DOWNLOADED
    assert (staging / "f.bin").exists()

    r2 = stage_bytes(staging, "f.bin", data)
    assert r2.status == STATUS_ALREADY_PRESENT
    # file untouched, still the original bytes
    assert (staging / "f.bin").read_bytes() == data


def test_stage_diff_bytes_is_conflict_and_original_unchanged(tmp_path: Path) -> None:
    staging = tmp_path / "incoming"
    original = b"ORIGINAL-DATA"
    replacement = b"DIFFERENT-DATA"

    stage_bytes(staging, "f.bin", original)
    target = staging / "f.bin"
    before_hash_target = target.read_bytes()

    r = stage_bytes(staging, "f.bin", replacement)
    assert r.status == STATUS_CONFLICT

    # The original file must be byte-for-byte unchanged.
    after = target.read_bytes()
    assert after == before_hash_target
    assert after == original
    assert after != replacement


def test_no_temp_part_left_behind(tmp_path: Path) -> None:
    staging = tmp_path / "incoming"
    stage_bytes(staging, "f.bin", b"x")
    stage_bytes(staging, "f.bin", b"y")  # conflict
    parts = list(staging.glob("**/*.part"))
    assert parts == []


def test_mark_duplicates_flags_different_paths_same_hash() -> None:
    # Realistic case: distinct relative paths sharing one hash are duplicates.
    rows = [
        ManifestRow(relative_path="data/raw/p1/a.wav", sha256="deadbeef"),
        ManifestRow(relative_path="data/raw/p2/b.wav", sha256="deadbeef"),  # dup of a
        ManifestRow(relative_path="data/raw/p3/c.wav", sha256="cafef00d"),  # unique
    ]
    mark_duplicates(rows)
    flagged = [r for r in rows if r.error_code == STATUS_DUPLICATE_CONTENT]
    # the two distinct-path rows sharing deadbeef are flagged
    assert len(flagged) == 2
    flagged_paths = {r.relative_path for r in flagged}
    assert flagged_paths == {"data/raw/p1/a.wav", "data/raw/p2/b.wav"}


def test_mark_duplicates_same_path_not_flagged_as_dup() -> None:
    # Same path repeated (a re-scan artifact) is not cross-path duplicate.
    rows = [
        ManifestRow(relative_path="data/raw/p1/a.wav", sha256="deadbeef"),
        ManifestRow(relative_path="data/raw/p1/a.wav", sha256="deadbeef"),
    ]
    mark_duplicates(rows)
    assert all(r.error_code == "" for r in rows)


def test_download_http_existing_target_is_already_present_no_network(tmp_path: Path) -> None:
    """If the target already exists, download_http short-circuits to
    already_present WITHOUT contacting the network (idempotent re-run)."""
    from psg_audio_benchmark.data_download.downloader import (
        STATUS_ALREADY_PRESENT,
        download_http,
    )

    staging = tmp_path / "incoming"
    staging.mkdir()
    target = staging / "V5" / "Data" / "01" / "x.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already-here")
    before = target.read_bytes()

    # an invalid URL that MUST NOT be contacted (short-circuit happens first)
    res = download_http(
        "http://0.0.0.0:1/never-contacted.csv",
        staging,
        filename="V5/Data/01/x.csv",
        allow_auto_download=True,
    )
    assert res.status == STATUS_ALREADY_PRESENT
    assert target.read_bytes() == before  # untouched


def test_download_http_existing_target_conflict_when_expected_hash_mismatch(
    tmp_path: Path,
) -> None:
    from psg_audio_benchmark.data_download.downloader import (
        STATUS_CONFLICT,
        download_http,
    )

    staging = tmp_path / "incoming"
    staging.mkdir()
    target = staging / "y.csv"
    target.write_bytes(b"on-disk-content")
    before = target.read_bytes()

    res = download_http(
        "http://0.0.0.0:1/y.csv",
        staging,
        filename="y.csv",
        allow_auto_download=True,
        expected_sha256="0" * 64,  # deliberately wrong
    )
    assert res.status == STATUS_CONFLICT
    assert target.read_bytes() == before  # original untouched
