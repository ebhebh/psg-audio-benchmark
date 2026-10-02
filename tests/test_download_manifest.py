"""Tests for the Stage-1 file manifest (prompt section 7, test #1).

Covers: the manifest schema has every required field; every path field is
relative to the project root (never absolute); and SHA-256 is reproducible
(recomputing a file's hash matches the recorded value).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from psg_audio_benchmark.data_download.hashing import HASH_ALGORITHM, sha256_file
from psg_audio_benchmark.data_download.manifest import (
    MANIFEST_FIELDS,
    ManifestRow,
    dataclass_field_names,
    manifest_field_set,
    read_manifest_csv,
    write_manifest_csv,
)

#: Fields mandated by prompt section 4.2.
REQUIRED_FIELDS = {
    "manifest_schema_version",
    "run_id",
    "dataset_role",
    "dataset_name",
    "paper_doi",
    "data_doi",
    "dataset_version",
    "access_date",
    "source_url",
    "source_archive_name",
    "relative_path",
    "staging_or_raw",
    "file_name",
    "extension",
    "size_bytes",
    "mtime_utc",
    "sha256",
    "hash_algorithm",
    "is_archive",
    "archive_integrity_status",
    "readability_status",
    "download_status",
    "overwrite_detected",
    "error_code",
    "error_message",
}


def test_manifest_schema_has_all_required_fields() -> None:
    assert manifest_field_set() == REQUIRED_FIELDS
    assert len(MANIFEST_FIELDS) == 25
    # dataclass field order must equal the canonical schema order
    assert dataclass_field_names() == MANIFEST_FIELDS


def test_manifest_row_roundtrip(tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "sub" / "b.json"
    f2.parent.mkdir(parents=True)
    f1.write_text("hello stage1", encoding="utf-8")
    f2.write_text("{}\n", encoding="utf-8")

    rows = []
    for p in (f1, f2):
        digest, _err = sha256_file(p)
        st = p.stat()
        rows.append(
            ManifestRow(
                run_id="test-run",
                dataset_role="primary",
                dataset_name="T",
                paper_doi="10.1038/s41597-025-05583-8",
                data_doi="10.57760/sciencedb.19070",
                relative_path=str(p.relative_to(tmp_path)).replace("\\", "/"),
                staging_or_raw="staging",
                file_name=p.name,
                extension=p.suffix.lower().lstrip("."),
                size_bytes=st.st_size,
                sha256=digest,
                hash_algorithm=HASH_ALGORITHM,
                readability_status="readable",
            )
        )

    out = tmp_path / "file_manifest_stage1.csv"
    n, path = write_manifest_csv(rows, out)
    assert n == 2
    assert path == out
    assert out.exists()

    read_back = read_manifest_csv(out)
    assert len(read_back) == 2
    with open(out, encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == list(MANIFEST_FIELDS)


def test_relative_paths_are_not_absolute(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("data", encoding="utf-8")
    digest, _ = sha256_file(f)
    row = ManifestRow(
        relative_path=str(f.relative_to(tmp_path)).replace("\\", "/"),
        sha256=digest,
    )
    rel = row.relative_path
    from pathlib import PurePath

    assert not PurePath(rel).is_absolute()
    assert not rel.startswith(("/", "C:\\", "D:\\", "d:\\", "c:\\"))


def test_sha256_is_reproducible(tmp_path: Path) -> None:
    """Recomputing the hash of a file equals the value recorded in the row."""
    f = tmp_path / "repro.txt"
    payload = b"stage-1 reproducibility payload \x00\x01\x02"
    f.write_bytes(payload)
    digest_row, _ = sha256_file(f)
    digest_again, _ = sha256_file(f)
    assert digest_row == digest_again
    assert len(digest_row) == 64
    row = ManifestRow(sha256=digest_row, hash_algorithm=HASH_ALGORITHM)
    assert row.sha256 == digest_again
    assert row.hash_algorithm == "sha256"
