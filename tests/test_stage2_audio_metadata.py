"""WAV header metadata tests (prompt section 4.1).

The probe reads only the header via the stdlib ``wave`` module. Failures
(truncated / non-WAV / empty) are recorded, never raised. Header duration is
metadata only -- not a sync statement.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.data_audit import audio_metadata

from _stage2_fixtures import write_wav, write_truncated_wav


def test_readable_wav_header(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "p.wav", channels=1, sampwidth=2, framerate=8000, nframes=8000)
    am = audio_metadata.probe_wav(p)
    assert am.readability_status == audio_metadata.AUDIO_READABLE
    assert am.container == "wav"
    assert am.sample_rate_hz == 8000
    assert am.channels == 1
    assert am.sample_width_bits == 16
    assert am.frame_count == 8000
    assert am.header_duration_s == 1.0
    assert am.duration_source == "wave_header"


def test_truncated_wav_is_flagged_not_raised(tmp_path: Path) -> None:
    p = write_truncated_wav(tmp_path / "trunc.wav")
    am = audio_metadata.probe_wav(p)
    assert am.readability_status == audio_metadata.AUDIO_TRUNCATED
    assert am.error  # non-empty reason
    # fields are None on failure, never an exception bubbling up
    assert am.sample_rate_hz is None


def test_non_wav_file_flagged(tmp_path: Path) -> None:
    p = tmp_path / "not.wav"
    p.write_bytes(b"ID3" + b"\x00" * 200)  # mp3 magic, not RIFF/WAVE
    am = audio_metadata.probe_wav(p)
    assert am.readability_status == audio_metadata.AUDIO_NOT_WAV


def test_empty_wav_flagged(tmp_path: Path) -> None:
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    am = audio_metadata.probe_wav(p)
    assert am.readability_status == audio_metadata.AUDIO_EMPTY


def test_metadata_unavailable_on_missing_file(tmp_path: Path) -> None:
    am = audio_metadata.probe_wav(tmp_path / "does_not_exist.wav")
    assert am.readability_status == audio_metadata.AUDIO_METADATA_UNAVAILABLE
    assert am.error
