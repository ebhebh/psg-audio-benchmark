"""Stage-1 file manifest: schema, row construction, CSV writing and audit.

The manifest is the auditable record of every file seen in ``data/raw`` and in
the external staging area. Its schema is fixed (prompt section 4.2) and every
path field is stored **relative to the project root** — never as a machine
absolute path.

Key safety rules encoded here (prompt section 4.3):
  * SHA-256 is required; a read failure yields an empty hash plus a reason.
  * Same relative path + same hash  -> ``already_present`` (no overwrite).
  * Same relative path + diff hash  -> ``conflict`` (highlight; never replace).
  * Different path  + same hash     -> ``duplicate_content`` (record; never
    delete or merge).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .hashing import HASH_ALGORITHM

#: Canonical, ordered field list (prompt section 4.2). Do not reorder.
MANIFEST_FIELDS: Tuple[str, ...] = (
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
)

#: Status vocabulary used in the manifest.
STATUS_ALREADY_PRESENT = "already_present"
STATUS_CONFLICT = "conflict"
STATUS_DUPLICATE_CONTENT = "duplicate_content"
STATUS_OK = "ok"
STATUS_MANUAL_REQUIRED = "manual_required"
STATUS_VERIFY_EXISTING = "verify_existing"
STATUS_NOT_CHECKED = "not_checked_due_to_dependency"
STATUS_CORRUPT_ARCHIVE = "corrupt_archive"


@dataclass
class ManifestRow:
    """One row of ``file_manifest_stage1.csv``."""

    manifest_schema_version: str = "1.0"
    run_id: str = ""
    dataset_role: str = ""
    dataset_name: str = ""
    paper_doi: str = ""
    data_doi: str = ""
    dataset_version: str = "TBD_AFTER_DOWNLOAD_AUDIT"
    access_date: str = ""
    source_url: str = ""
    source_archive_name: str = ""
    relative_path: str = ""
    staging_or_raw: str = ""
    file_name: str = ""
    extension: str = ""
    size_bytes: int = -1
    mtime_utc: str = ""
    sha256: str = ""
    hash_algorithm: str = HASH_ALGORITHM
    is_archive: bool = False
    archive_integrity_status: str = ""
    readability_status: str = ""
    download_status: str = ""
    overwrite_detected: bool = False
    error_code: str = ""
    error_message: str = ""

    def as_ordered_dict(self) -> Dict[str, object]:
        """Return the row as a dict keyed by the canonical field order."""
        data = asdict(self)
        return {name: data[name] for name in MANIFEST_FIELDS}


# ---------------------------------------------------------------------------
# Conflict / duplicate resolution
# ---------------------------------------------------------------------------

def decide_target_action(
    target_exists: bool, existing_sha256: str, new_sha256: str
) -> str:
    """Decide what to do when a target path already exists.

    Returns ``ok`` (target absent, safe to write), ``already_present`` (identical
    content, no action) or ``conflict`` (different content — never overwrite).
    """
    if not target_exists:
        return STATUS_OK
    if not existing_sha256 or not new_sha256:
        # Cannot prove equality -> treat as conflict rather than risk overwrite.
        return STATUS_CONFLICT
    return STATUS_ALREADY_PRESENT if existing_sha256 == new_sha256 else STATUS_CONFLICT


def mark_duplicates(rows: List[ManifestRow]) -> List[ManifestRow]:
    """Annotate rows that share a SHA-256 under a *different* relative path.

    The rows are mutated in place (and returned) by setting
    ``error_code='duplicate_content'`` and an explanatory message. Files are
    never deleted or merged — only flagged.
    """
    by_hash: Dict[str, List[ManifestRow]] = {}
    for row in rows:
        if row.sha256:
            by_hash.setdefault(row.sha256, []).append(row)
    for sha, group in by_hash.items():
        if len(group) < 2:
            continue
        paths = sorted({r.relative_path for r in group})
        if len(paths) < 2:
            continue  # same path repeated, not a content duplicate
        msg = (
            f"duplicate_content: sha256 {sha[:12]}... appears under "
            f"{len(paths)} distinct paths; not deleted or merged"
        )
        for row in group:
            if not row.error_code:
                row.error_code = STATUS_DUPLICATE_CONTENT
                row.error_message = msg
            elif STATUS_DUPLICATE_CONTENT not in (row.error_message or ""):
                row.error_message = (row.error_message or "") + " | " + msg
    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def write_manifest_csv(
    rows: Iterable[ManifestRow], out_path: Path
) -> Tuple[int, Path]:
    """Write rows to ``out_path`` as CSV with the canonical header.

    Returns ``(n_rows_written, out_path)``. The parent directory is created
    (manifests are project products, never patient data). Writes are atomic at
    the filesystem level via a temp file + replace.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    rows_list = list(rows)
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row.as_ordered_dict())
    tmp.replace(out_path)
    return len(rows_list), out_path


def read_manifest_csv(in_path: Path) -> List[Dict[str, str]]:
    """Read a manifest CSV back into a list of plain dicts (strings)."""
    in_path = Path(in_path)
    with open(in_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header_ok = list(reader.fieldnames or []) == list(MANIFEST_FIELDS)
        if not header_ok:
            raise ValueError(
                f"manifest {in_path} header does not match the Stage-1 schema"
            )
        return [dict(row) for row in reader]


def manifest_field_set() -> set:
    """Set of canonical field names (used by tests)."""
    return set(MANIFEST_FIELDS)


def dataclass_field_names() -> Tuple[str, ...]:
    """Order of the :class:`ManifestRow` dataclass fields (== schema order)."""
    return tuple(f.name for f in fields(ManifestRow))


__all__ = [
    "MANIFEST_FIELDS",
    "STATUS_ALREADY_PRESENT",
    "STATUS_CONFLICT",
    "STATUS_DUPLICATE_CONTENT",
    "STATUS_OK",
    "STATUS_MANUAL_REQUIRED",
    "STATUS_VERIFY_EXISTING",
    "STATUS_NOT_CHECKED",
    "STATUS_CORRUPT_ARCHIVE",
    "ManifestRow",
    "decide_target_action",
    "mark_duplicates",
    "write_manifest_csv",
    "read_manifest_csv",
    "manifest_field_set",
    "dataclass_field_names",
]
