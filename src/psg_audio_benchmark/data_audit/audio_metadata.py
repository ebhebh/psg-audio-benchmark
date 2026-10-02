"""WAV header metadata probe (Stage 2, prompt 4.1).

Reads only the WAV *header* with the standard library ``wave`` module -- no
``ffmpeg``/``soundfile``/``librosa`` is installed or required. ``wave`` opens
the RIFF/WAVE container and reads the ``fmt`` chunk plus the declared data-chunk
size; it does NOT decode the samples, so the probe never loads whole audio into
memory.

Recorded per file: container, sample rate, channels, sample width / bit depth
(when available), frame count, header-derived duration, readability status and
any error. The header duration is **file metadata for traceability only** -- it
is NOT a statement about clock alignment across devices; that belongs to a later
synchronization stage.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Readability vocabulary for audio probing.
AUDIO_READABLE = "readable"
AUDIO_METADATA_UNAVAILABLE = "metadata_unavailable"
AUDIO_NOT_WAV = "not_wav_container"
AUDIO_TRUNCATED = "truncated_or_malformed"
AUDIO_EMPTY = "empty"


@dataclass
class AudioMetadata:
    """Header-derived metadata for one WAV file (or a failure record)."""

    container: str = ""
    sample_rate_hz: Optional[int] = None
    channels: Optional[int] = None
    sample_width_bits: Optional[int] = None
    frame_count: Optional[int] = None
    header_duration_s: Optional[float] = None
    readability_status: str = AUDIO_METADATA_UNAVAILABLE
    error: str = ""
    #: How the header-derived duration was obtained (always header-only here).
    duration_source: str = "wave_header"

    def as_dict(self) -> dict:
        return {
            "audio_container": self.container,
            "audio_sample_rate_hz": _fmt_opt(self.sample_rate_hz),
            "audio_channels": _fmt_opt(self.channels),
            "audio_sample_width_bits": _fmt_opt(self.sample_width_bits),
            "audio_frame_count": _fmt_opt(self.frame_count),
            "audio_header_duration_s": _fmt_opt(self.header_duration_s),
            "audio_readability_status": self.readability_status,
            "audio_error": self.error,
            "audio_duration_source": self.duration_source,
        }


def _fmt_opt(v) -> str:
    return "" if v is None else str(v)


def _read_head(path: Path, n: int = 12) -> bytes:
    try:
        with open(path, "rb") as handle:
            return handle.read(n)
    except OSError:
        return b""


def probe_wav(path: Path) -> AudioMetadata:
    """Probe a WAV header. Never raises: failures become ``metadata_unavailable``.

    Only the header is read; the samples are never decoded into memory.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return AudioMetadata(
            readability_status=AUDIO_METADATA_UNAVAILABLE,
            error=f"stat_failed: {type(exc).__name__}: {exc}",
        )
    if size == 0:
        return AudioMetadata(
            container="wav",
            readability_status=AUDIO_EMPTY,
            error="empty_file",
        )

    # Quick magic sanity before handing to ``wave``.
    head = _read_head(path, 12)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        # Could still be a non-WAV audio file; record honestly.
        return AudioMetadata(
            container="unknown",
            readability_status=AUDIO_NOT_WAV,
            error="magic_not_riff_wave",
        )

    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            nframes = w.getnframes()
            duration = (nframes / framerate) if framerate else None
            return AudioMetadata(
                container="wav",
                sample_rate_hz=framerate,
                channels=channels,
                sample_width_bits=sampwidth * 8 if sampwidth else None,
                frame_count=nframes,
                header_duration_s=duration,
                readability_status=AUDIO_READABLE,
                duration_source="wave_header",
            )
    except (wave.Error, EOFError, OSError) as exc:
        # Truncated / malformed WAV (e.g. a header without a complete data chunk).
        return AudioMetadata(
            container="wav",
            readability_status=AUDIO_TRUNCATED,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return AudioMetadata(
            container="wav",
            readability_status=AUDIO_METADATA_UNAVAILABLE,
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "AUDIO_READABLE",
    "AUDIO_METADATA_UNAVAILABLE",
    "AUDIO_NOT_WAV",
    "AUDIO_TRUNCATED",
    "AUDIO_EMPTY",
    "AudioMetadata",
    "probe_wav",
]
