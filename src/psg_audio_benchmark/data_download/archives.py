"""Archive safety: member audit, dangerous-member rejection, safe extraction.

Stage 1, prompt section 5.1. Uses only the standard library (``zipfile`` /
``tarfile``) for ZIP / TAR / TAR.GZ. 7Z is supported *only if* ``py7zr`` is
importable; otherwise it is reported as ``not_checked_due_to_dependency``
rather than silently skipped.

Safety contract:
  * Members with absolute paths, ``..`` traversal, or symlinks/hardlinks that
    escape the destination are **rejected (quarantined)**, never extracted.
  * Corrupt / password-protected / truncated archives are flagged
    ``corrupt_archive`` and the audit continues with the remaining files.
  * Extraction NEVER targets ``data/raw``; it only writes under the configured
    external extraction dir, and only after a free-space estimate.
  * No partial, unrecorded extraction: if free space is insufficient the whole
    archive is skipped with a clear reason.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .hashing import free_bytes

#: Archive kinds this module can reason about.
KIND_ZIP = "zip"
KIND_TAR = "tar"        # covers plain tar and tar.gz / tar.bz2 (tarfile auto)
KIND_7Z = "7z"
KIND_UNKNOWN = "unknown"

ARCHIVE_TEST_OK = "ok"
ARCHIVE_CORRUPT = "corrupt_archive"
ARCHIVE_PASSWORD = "password_protected"
ARCHIVE_TRUNCATED = "truncated"
ARCHIVE_NOT_CHECKED = "not_checked_due_to_dependency"


# ---------------------------------------------------------------------------
# Magic-byte detection
# ---------------------------------------------------------------------------

def _read_head(path: Path, n: int = 264) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(n)
    except OSError:
        return b""


def detect_archive_kind(path: Path) -> str:
    """Identify an archive by magic bytes (not by extension)."""
    head = _read_head(path)
    if len(head) >= 4 and head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return KIND_ZIP
    if len(head) >= 2 and head[:2] == b"\x1f\x8b":
        return KIND_TAR          # gzip -> tarfile auto-detects inner tar
    if len(head) >= 6 and head[:6] == b"7z\xbc\xaf\x27\x1c":
        return KIND_7Z
    # tar "ustar" magic lives at offset 257
    if len(head) >= 263 and head[257:262] == b"ustar":
        return KIND_TAR
    return KIND_UNKNOWN


def is_archive(path: Path) -> bool:
    return detect_archive_kind(path) != KIND_UNKNOWN


# ---------------------------------------------------------------------------
# Member model
# ---------------------------------------------------------------------------

@dataclass
class ArchiveMember:
    name: str
    size: int = 0
    is_dir: bool = False
    is_absolute: bool = False
    escapes_root: bool = False
    is_symlink: bool = False
    is_hardlink: bool = False
    link_target: str = ""
    dangerous: bool = False
    danger_reason: str = ""


@dataclass
class ArchiveInspection:
    path: str
    kind: str
    member_count: int = 0
    members: List[ArchiveMember] = field(default_factory=list)
    total_uncompressed_size: int = 0
    n_path_escape: int = 0
    n_absolute: int = 0
    n_duplicates: int = 0
    has_dangerous_links: bool = False
    test_status: str = ARCHIVE_TEST_OK
    error: str = ""
    dependency_missing: bool = False

    @property
    def corrupt(self) -> bool:
        return self.test_status in (
            ARCHIVE_CORRUPT, ARCHIVE_TRUNCATED, ARCHIVE_PASSWORD
        )

    @property
    def has_dangerous_members(self) -> bool:
        return any(m.dangerous for m in self.members)


def _member_is_absolute(name: str) -> bool:
    if not name:
        return False
    if name.startswith("/") or name.startswith("\\"):
        return True
    # Windows drive letter, e.g. C:\ or C:/
    if len(name) >= 2 and name[1] == ":":
        return True
    return False


def _member_escapes_root(dest: Path, name: str) -> bool:
    """True if ``name`` resolves above ``dest`` after normalization."""
    # normalize separators
    rel = name.replace("\\", "/")
    try:
        resolved = (dest / rel).resolve()
    except (OSError, ValueError):
        return True
    try:
        resolved.relative_to(dest.resolve())
    except ValueError:
        return True
    return False


# ---------------------------------------------------------------------------
# ZIP inspection
# ---------------------------------------------------------------------------

def _inspect_zip(path: Path) -> ArchiveInspection:
    insp = ArchiveInspection(path=str(path), kind=KIND_ZIP)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            # CRC / compression test -> name of first bad file, or None
            bad = zf.testzip()
            if bad is not None:
                insp.test_status = ARCHIVE_CORRUPT
                insp.error = f"testzip bad member: {bad!r}"
            seen_names: dict = {}
            for info in zf.infolist():
                member = _zip_member_safety(info)
                insp.members.append(member)
                insp.member_count += 1
                insp.total_uncompressed_size += max(0, info.file_size)
                if member.is_absolute:
                    insp.n_absolute += 1
                if member.escapes_root:
                    insp.n_path_escape += 1
                if member.is_symlink or member.is_hardlink:
                    insp.has_dangerous_links = True
                seen_names[info.filename] = seen_names.get(info.filename, 0) + 1
            insp.n_duplicates = sum(1 for c in seen_names.values() if c > 1)
    except zipfile.BadZipFile as exc:
        insp.test_status = ARCHIVE_CORRUPT
        insp.error = f"BadZipFile: {exc}"
    except RuntimeError as exc:  # e.g. password-protected
        insp.test_status = ARCHIVE_PASSWORD
        insp.error = f"RuntimeError: {exc}"
    except OSError as exc:        # truncated / unreadable
        insp.test_status = ARCHIVE_TRUNCATED
        insp.error = f"OSError: {exc}"
    return insp


_UNIX_SLINK = 0o120000   # symlink file type bits (external_attr, unix mode)
_UNIX_HLINK = 0o100000


def _zip_member_safety(info: zipfile.ZipInfo) -> ArchiveMember:
    name = info.filename
    member = ArchiveMember(
        name=name,
        size=int(getattr(info, "file_size", 0) or 0),
        is_dir=bool(getattr(info, "is_dir", lambda: False)()),
    )
    # We do not have a dest here; mark absolute/escape on a nominal root check.
    if _member_is_absolute(name):
        member.is_absolute = True
        member.dangerous = True
        member.danger_reason = "absolute member path"
    if ".." in Path(name).parts:
        member.escapes_root = True
        member.dangerous = True
        member.danger_reason = (
            member.danger_reason or "member contains '..' traversal"
        )
    # unix symlink via external_attr (high 16 bits = mode)
    ext = getattr(info, "external_attr", 0) or 0
    mode = (ext >> 16) & 0xFFFF
    if (mode & 0o170000) == _UNIX_SLINK:
        member.is_symlink = True
        member.dangerous = True
        member.danger_reason = (
            member.danger_reason or "zip member is a symlink"
        )
    return member


# ---------------------------------------------------------------------------
# TAR inspection
# ---------------------------------------------------------------------------

def _inspect_tar(path: Path) -> ArchiveInspection:
    import tarfile

    insp = ArchiveInspection(path=str(path), kind=KIND_TAR)
    try:
        with tarfile.open(path, "r:*") as tf:
            seen_names: dict = {}
            members = tf.getmembers()
            for ti in members:
                member = ArchiveMember(
                    name=ti.name,
                    size=int(getattr(ti, "size", 0) or 0),
                    is_dir=ti.isdir(),
                    is_symlink=ti.issym(),
                    is_hardlink=ti.islnk(),
                )
                if ti.issym() or ti.islnk():
                    member.link_target = str(getattr(ti, "linkname", "") or "")
                if _member_is_absolute(ti.name):
                    member.is_absolute = True
                    member.dangerous = True
                    member.danger_reason = "absolute member path"
                if ".." in Path(ti.name).parts:
                    member.escapes_root = True
                    member.dangerous = True
                    member.danger_reason = (
                        member.danger_reason or "member contains '..' traversal"
                    )
                if member.is_symlink or member.is_hardlink:
                    member.dangerous = True
                    member.danger_reason = (
                        member.danger_reason or "tar member is a link"
                    )
                insp.members.append(member)
                insp.member_count += 1
                if not member.is_dir:
                    insp.total_uncompressed_size += max(0, ti.size)
                if member.is_absolute:
                    insp.n_absolute += 1
                if member.escapes_root:
                    insp.n_path_escape += 1
                if member.is_symlink or member.is_hardlink:
                    insp.has_dangerous_links = True
                seen_names[ti.name] = seen_names.get(ti.name, 0) + 1
            insp.n_duplicates = sum(1 for c in seen_names.values() if c > 1)
    except tarfile.ReadError as exc:
        insp.test_status = ARCHIVE_CORRUPT
        insp.error = f"tarfile.ReadError: {exc}"
    except OSError as exc:
        insp.test_status = ARCHIVE_TRUNCATED
        insp.error = f"OSError: {exc}"
    except EOFError as exc:
        insp.test_status = ARCHIVE_TRUNCATED
        insp.error = f"EOFError: {exc}"
    return insp


# ---------------------------------------------------------------------------
# 7Z inspection (optional dependency)
# ---------------------------------------------------------------------------

def _py7zr_available() -> bool:
    try:
        import py7zr  # noqa: F401
        return True
    except Exception:
        return False


def _inspect_7z(path: Path) -> ArchiveInspection:
    insp = ArchiveInspection(path=str(path), kind=KIND_7Z)
    if not _py7zr_available():
        insp.test_status = ARCHIVE_NOT_CHECKED
        insp.error = "py7zr not installed; 7z archive not inspected"
        insp.dependency_missing = True
        return insp
    try:
        import py7zr

        with py7zr.SevenZipFile(path, mode="r") as zf:
            info_list = zf.list()
            seen: dict = {}
            for entry in info_list:
                name = getattr(entry, "filename", str(entry))
                member = ArchiveMember(
                    name=name,
                    size=int(getattr(entry, "uncompressed", 0) or 0),
                )
                if _member_is_absolute(name) or ".." in Path(name).parts:
                    member.is_absolute = _member_is_absolute(name)
                    member.escapes_root = ".." in Path(name).parts
                    member.dangerous = True
                    member.danger_reason = "absolute or traversal member path"
                insp.members.append(member)
                insp.member_count += 1
                insp.total_uncompressed_size += max(0, member.size)
                seen[name] = seen.get(name, 0) + 1
            insp.n_duplicates = sum(1 for c in seen.values() if c > 1)
    except Exception as exc:
        insp.test_status = ARCHIVE_CORRUPT
        insp.error = f"py7zr error: {type(exc).__name__}: {exc}"
    return insp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect_archive(path: Path) -> ArchiveInspection:
    """Inspect an archive without extracting it. Never raises on corruption."""
    path = Path(path)
    kind = detect_archive_kind(path)
    if kind == KIND_ZIP:
        return _inspect_zip(path)
    if kind == KIND_TAR:
        return _inspect_tar(path)
    if kind == KIND_7Z:
        return _inspect_7z(path)
    insp = ArchiveInspection(path=str(path), kind=KIND_UNKNOWN)
    insp.test_status = ARCHIVE_NOT_CHECKED
    insp.error = "unknown archive kind (magic bytes unrecognized)"
    insp.dependency_missing = False
    return insp


def safe_extract(
    archive_path: Path,
    dest_dir: Path,
    min_free_bytes: int,
    forbidden_prefix: Optional[Path] = None,
) -> Tuple[ArchiveInspection, List[str], str]:
    """Safely extract ``archive_path`` into ``dest_dir``.

    Guarantees (prompt section 5.1):
      * ``dest_dir`` is created (under the external extraction dir).
      * If ``dest_dir`` resolves inside ``forbidden_prefix`` (default
        ``data/raw``) extraction is refused.
      * Dangerous members are skipped and listed; only safe members extract.
      * Free space is estimated first; insufficient space -> no extraction.
      * The original archive is never modified or deleted.

    Returns ``(inspection, skipped_members, status)`` where ``status`` is one of
    ``extracted`` / ``skipped_no_space`` / ``skipped_forbidden`` /
    ``skipped_corrupt`` / ``skipped_dependency`` / ``nothing_to_extract``.
    """
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    forbidden = (forbidden_prefix or Path("data/raw"))

    insp = inspect_archive(archive_path)

    # Refuse to write into data/raw (or any forbidden prefix).
    try:
        dest_resolved = dest_dir.resolve()
        forb_resolved = forbidden.resolve()
        try:
            dest_resolved.relative_to(forb_resolved)
            inside_forbidden = True
        except ValueError:
            inside_forbidden = False
    except OSError:
        inside_forbidden = False
    if inside_forbidden:
        return insp, [m.name for m in insp.members], "skipped_forbidden"

    # Dependency missing (e.g. 7z without py7zr) -> do not extract.
    if insp.dependency_missing or insp.kind == KIND_UNKNOWN:
        return insp, [], "skipped_dependency"

    # Corrupt -> report, do not extract.
    if insp.corrupt:
        return insp, [], "skipped_corrupt"

    safe_members = [m for m in insp.members if not m.dangerous]
    skipped = [m.name for m in insp.members if m.dangerous]

    if not safe_members:
        return insp, skipped, "nothing_to_extract"

    # Free-space estimate: refuse entirely (no partial extraction) if short.
    avail, err = free_bytes(dest_dir.parent if not dest_dir.exists() else dest_dir)
    needed = insp.total_uncompressed_size
    if err or avail < 0 or avail < max(min_free_bytes, needed):
        return insp, skipped, "skipped_no_space"

    dest_dir.mkdir(parents=True, exist_ok=True)
    kind = insp.kind
    if kind == KIND_ZIP:
        _extract_zip_safe(archive_path, dest_dir, safe_members)
    elif kind == KIND_TAR:
        _extract_tar_safe(archive_path, dest_dir, safe_members)
    else:  # pragma: no cover - 7z path guarded by dependency check above
        return insp, skipped, "skipped_dependency"

    return insp, skipped, "extracted"


def _extract_zip_safe(
    archive_path: Path, dest_dir: Path, safe_members: List[ArchiveMember]
) -> None:
    safe_names = {m.name for m in safe_members}
    with zipfile.ZipFile(archive_path, "r") as zf:
        for info in zf.infolist():
            if info.filename not in safe_names:
                continue
            # final per-member escape check against the real dest
            if _member_is_absolute(info.filename) or _member_escapes_root(
                dest_dir, info.filename
            ):
                continue
            zf.extract(info, path=dest_dir)


def _extract_tar_safe(
    archive_path: Path, dest_dir: Path, safe_members: List[ArchiveMember]
) -> None:
    import tarfile

    safe_names = {m.name for m in safe_members}
    with tarfile.open(archive_path, "r:*") as tf:
        for ti in tf.getmembers():
            if ti.name not in safe_names:
                continue
            if _member_is_absolute(ti.name) or _member_escapes_root(dest_dir, ti.name):
                continue
            # 'data' filter (Py3.12+) strips unsafe metadata; pass if accepted.
            try:
                tf.extract(ti, path=dest_dir, filter="data")
            except TypeError:  # pragma: no cover - older tarfile w/o filter kw
                tf.extract(ti, path=dest_dir)


__all__ = [
    "KIND_ZIP",
    "KIND_TAR",
    "KIND_7Z",
    "KIND_UNKNOWN",
    "ARCHIVE_TEST_OK",
    "ARCHIVE_CORRUPT",
    "ARCHIVE_PASSWORD",
    "ARCHIVE_TRUNCATED",
    "ARCHIVE_NOT_CHECKED",
    "ArchiveMember",
    "ArchiveInspection",
    "detect_archive_kind",
    "is_archive",
    "inspect_archive",
    "safe_extract",
]
