"""File-level integrity checks (Stage 1, prompt section 5.2).

This is a *container/encoding* audit, deliberately shallow:
  * TXT / CSV / JSON  -> encoding readability + size + truncation heuristic.
    No medical / annotation field is parsed here (that is Stage 2's job).
  * WAV / MP3 / audio -> container-level readability only. If neither
    ``soundfile`` nor an ``ffmpeg`` probe is available, the status is
    ``not_checked_due_to_dependency``; we never claim audio *content* is
    complete.
  * Empty, tiny, or extension-vs-magic-mismatched files are flagged as
    anomalies (recorded, not deleted).

Any sample rate / channel count / duration recorded here is **file metadata**
for traceability, never a Stage-2 patient-level scientific result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

#: Files below this many bytes with a non-trivial extension are flagged tiny.
TINY_FILE_BYTES = 16

READ_OK = "readable"
READ_UNREADABLE = "unreadable"
READ_NOT_CHECKED = "not_checked_due_to_dependency"
READ_EMPTY = "empty"
READ_TOO_SMALL = "too_small"
READ_MAGIC_MISMATCH = "magic_mismatch"
READ_TRUNCATED = "truncated_or_malformed"

#: Extension buckets used for choosing a check strategy.
_TEXT_EXT = {".txt", ".text", ".csv", ".tsv", ".log", ".jsonl"}
_JSON_EXT = {".json"}
_CSV_EXT = {".csv", ".tsv"}
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}


# ---------------------------------------------------------------------------
# Magic detection
# ---------------------------------------------------------------------------

def _read_head(path: Path, n: int = 16) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(n)
    except OSError:
        return b""


def magic_kind(path: Path) -> str:
    """Classify a file by its leading magic bytes (best effort)."""
    head = _read_head(path)
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:3] == b"ID3":
        return "mp3"
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"):
        return "zip"
    if head[:6] == b"7z\xbc\xaf\x27\x1c":
        return "7z"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    return "unknown"


def extension_bucket(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _JSON_EXT:
        return "json"
    if ext in _CSV_EXT:
        return "csv"
    if ext in _TEXT_EXT:
        return "text"
    if ext in _AUDIO_EXT:
        return "audio"
    return "other"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class FileIntegrity:
    path: str
    size: int = -1
    extension_bucket: str = ""
    magic_kind: str = ""
    readability_status: str = READ_OK
    is_archive: bool = False
    audio_metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly: str = ""
    error: str = ""

    @property
    def has_anomaly(self) -> bool:
        return bool(self.anomaly) or self.readability_status not in (
            READ_OK, READ_NOT_CHECKED
        )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _try_decode(data: bytes) -> Optional[str]:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


def check_text_like(path: Path) -> FileIntegrity:
    fi = FileIntegrity(
        path=str(path),
        extension_bucket=extension_bucket(path),
        magic_kind="text",
    )
    try:
        data = path.read_bytes()
    except OSError as exc:
        fi.readability_status = READ_UNREADABLE
        fi.error = f"read_failed: {type(exc).__name__}: {exc}"
        return fi
    fi.size = len(data)
    if len(data) == 0:
        fi.readability_status = READ_EMPTY
        fi.anomaly = "empty_text_file"
        return fi
    if len(data) < TINY_FILE_BYTES:
        fi.anomaly = "very_small_file"
    text = _try_decode(data)
    if text is None:
        fi.readability_status = READ_UNREADABLE
        fi.error = "not decodable as utf-8/utf-8-sig/latin-1"
        return fi
    if fi.extension_bucket == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            # structural truncation / malformed JSON (NO field semantics)
            fi.readability_status = READ_TRUNCATED
            fi.error = f"json_decode_error: {exc}"
    return fi


def _soundfile_available() -> bool:
    try:
        import soundfile  # noqa: F401
        return True
    except Exception:
        return False


def _ffmpeg_available() -> bool:
    import shutil

    return shutil.which("ffprobe") is not None or shutil.which("ffmpeg") is not None


def check_audio(path: Path) -> FileIntegrity:
    fi = FileIntegrity(
        path=str(path),
        extension_bucket="audio",
        magic_kind=magic_kind(path),
    )
    try:
        fi.size = path.stat().st_size
    except OSError as exc:
        fi.readability_status = READ_UNREADABLE
        fi.error = f"stat_failed: {type(exc).__name__}: {exc}"
        return fi
    if fi.size == 0:
        fi.readability_status = READ_EMPTY
        fi.anomaly = "empty_audio_file"
        return fi
    if fi.size < TINY_FILE_BYTES:
        fi.anomaly = "very_small_audio_file"
    # extension vs magic mismatch
    expected = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".ogg": "ogg"}
    exp_magic = expected.get(path.suffix.lower())
    if exp_magic and fi.magic_kind not in (exp_magic, "unknown"):
        fi.readability_status = READ_MAGIC_MISMATCH
        fi.anomaly = (
            f"extension '{path.suffix}' but magic looks like '{fi.magic_kind}'"
        )
        return fi
    # Try a lossless container probe if a dependency is available.
    if _soundfile_available():
        try:
            import soundfile as sf

            info = sf.info(str(path))
            fi.audio_metadata = {
                "sample_rate_hz": int(info.samplerate),
                "channels": int(info.channels),
                "frames": int(info.frames),
                "duration_seconds": float(info.frames) / float(info.samplerate)
                if info.samplerate
                else None,
                "format": getattr(info, "format", None),
                "subtype": getattr(info, "subtype", None),
            }
            fi.readability_status = READ_OK
            return fi
        except Exception as exc:
            # soundfile present but could not probe (e.g. MP3 without a
            # libsndfile MP3 build): fall through to container-level verdict.
            fi.error = f"soundfile_probe_failed: {type(exc).__name__}: {exc}"
    # Container-level verdict from magic only.
    if fi.magic_kind in ("wav", "mp3", "flac", "ogg"):
        fi.readability_status = READ_OK
        fi.audio_metadata = {"probe": "magic_only", "ffmpeg_available": _ffmpeg_available()}
        if not _ffmpeg_available():
            fi.anomaly = "ffmpeg_missing; only container magic verified"
        return fi
    fi.readability_status = READ_NOT_CHECKED
    fi.error = "no audio dependency available (soundfile/ffmpeg) for container probe"
    return fi


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def check_file(path: Path) -> FileIntegrity:
    """Run the right shallow check for ``path`` based on its extension/magic."""
    path = Path(path)
    bucket = extension_bucket(path)
    if bucket in ("text", "csv", "json"):
        return check_text_like(path)
    if bucket == "audio":
        return check_audio(path)
    # Unknown / other (incl. archives handled by archives.py): record size only.
    fi = FileIntegrity(
        path=str(path),
        extension_bucket=bucket,
        magic_kind=magic_kind(path),
        is_archive=magic_kind(path) in ("zip", "7z", "gzip"),
    )
    try:
        fi.size = path.stat().st_size
        if fi.size == 0:
            fi.readability_status = READ_EMPTY
            fi.anomaly = "empty_file"
        else:
            fi.readability_status = READ_OK
    except OSError as exc:
        fi.readability_status = READ_UNREADABLE
        fi.error = f"stat_failed: {type(exc).__name__}: {exc}"
    return fi


__all__ = [
    "TINY_FILE_BYTES",
    "READ_OK",
    "READ_UNREADABLE",
    "READ_NOT_CHECKED",
    "READ_EMPTY",
    "READ_TOO_SMALL",
    "READ_MAGIC_MISMATCH",
    "READ_TRUNCATED",
    "FileIntegrity",
    "magic_kind",
    "extension_bucket",
    "check_text_like",
    "check_audio",
    "check_file",
]
