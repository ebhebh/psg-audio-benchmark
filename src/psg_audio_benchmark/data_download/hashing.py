"""SHA-256 hashing and file-metadata helpers for the Stage-1 audit.

Stage 1 only needs the standard library: ``hashlib`` for SHA-256 and ``os`` for
file metadata. No third-party audio/science package is required here, so the
hash audit never blocks on missing ``soundfile``/``librosa``/``torch``.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

#: Read the file in 1 MiB chunks so large archives do not blow up memory.
_CHUNK = 1024 * 1024

#: The hash algorithm mandated by the Stage-1 manifest.
HASH_ALGORITHM = "sha256"

#: Sentinel written when a hash could not be computed (read failure).
HASH_UNAVAILABLE = ""


def sha256_file(path: Path) -> Tuple[str, Optional[str]]:
    """Return ``(sha256_hex, error_message)`` for ``path``.

    The file is read in chunks. On any read error the hash is the empty string
    and a human-readable ``error_message`` is returned (never an exception),
    so the manifest can record *why* the hash is missing instead of crashing.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest(), None
    except OSError as exc:
        return HASH_UNAVAILABLE, f"hash_read_failed: {type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return HASH_UNAVAILABLE, f"hash_unexpected_error: {type(exc).__name__}: {exc}"


def file_meta(path: Path) -> Tuple[int, str, Optional[str]]:
    """Return ``(size_bytes, mtime_utc_iso, error_message)`` for ``path``.

    ``mtime_utc_iso`` is the modification time in UTC ISO-8601 (stable across
    machines, no local-time ambiguity). Read errors are returned, not raised.
    """
    try:
        st = path.stat()
        mtime_iso = (
            datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        return int(st.st_size), mtime_iso, None
    except OSError as exc:
        return -1, "", f"meta_stat_failed: {type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return -1, "", f"meta_unexpected_error: {type(exc).__name__}: {exc}"


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex of an in-memory byte string (used for fixtures/tests)."""
    return hashlib.sha256(data).hexdigest()


def free_bytes(path: Path) -> Tuple[int, Optional[str]]:
    """Return ``(free_bytes, error_message)`` for the filesystem holding ``path``.

    Uses ``shutil.disk_usage`` on the resolved directory (or its nearest
    existing ancestor) so it works even when ``path`` does not exist yet.
    """
    import shutil

    probe = path if path.is_dir() else path.parent
    try:
        return shutil.disk_usage(probe).free, None
    except OSError as exc:
        return -1, f"disk_usage_failed: {type(exc).__name__}: {exc}"


__all__ = [
    "HASH_ALGORITHM",
    "HASH_UNAVAILABLE",
    "sha256_file",
    "file_meta",
    "sha256_bytes",
    "free_bytes",
]
