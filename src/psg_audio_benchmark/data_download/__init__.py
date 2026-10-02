"""Data download / integrity-audit package (Stage 1).

Stage 1 responsibility:
  * Verify the primary dataset DOI identity against hard-coded constants
    (``identity``); never infer identity from a filename.
  * Generate ``file_manifest_stage1.csv`` with mandatory SHA-256 (``manifest``).
  * Audit archive safety (``archives``) and file-level readability
    (``integrity``) without parsing any medical / annotation field.
  * Provide a non-destructive staging downloader (``downloader``) that defaults
    to ``manual_required`` and never overwrites an existing target.

Privacy / safety: this package never modifies, moves or deletes anything under
``data/raw``; it only READS raw files to hash them. Staging lives under
``data/external/incoming`` and is promoted into ``data/raw`` only by a human.
"""

from __future__ import annotations

from . import archives, downloader, hashing, identity, integrity, manifest

__all__ = [
    "archives",
    "downloader",
    "hashing",
    "identity",
    "integrity",
    "manifest",
]
