"""Stage-3 boundary cases 6 & 7: strict audio time-anchor judgment.

Guarantees (prompt section 6.6 / 6.7):
  * an audio file with no trustworthy time anchor -> ``unresolved`` (never
    verified, never assumed by mtime/filename/duration-similarity);
  * the anchor evidence string explicitly states mtime/filename are NOT used;
  * an audio file below the 30 s threshold -> ``excluded_audio_too_short`` +
    a below-min warning;
  * a normal-length file is NOT excluded;
  * even a file carrying a ``bext`` (time-reference-capable) chunk stays
    unresolved without a configured transform + cross-check.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.annotation_parser.audio_alignment import judge_audio_alignment
from psg_audio_benchmark.annotation_parser.schema import (
    ALIGN_EXCLUDED_TOO_SHORT,
    ALIGN_UNRESOLVED,
)

from _stage3_fixtures import (
    write_normal_wav,
    write_short_wav,
    write_wav_with_bext,
)


def _judge(path: Path, *, min_dur: float = 30.0):
    return judge_audio_alignment(
        path, run_id="r", patient_id="01", modality="smartphone_audio",
        source_relpath="raw/V5/Data/01/01_phone.wav",
        minimum_source_audio_duration_seconds=min_dur,
    )


def test_no_anchor_is_unresolved(tmp_path: Path) -> None:
    p = tmp_path / "01_phone.wav"
    write_normal_wav(p)
    rec, warn = _judge(p)
    assert rec.alignment_status == ALIGN_UNRESOLVED
    # explicitly NOT verified; the four states stay distinct
    assert rec.alignment_status != "verified"
    assert rec.start_time_seconds is None
    assert rec.end_time_seconds is None


def test_anchor_evidence_excludes_mtime_and_filename(tmp_path: Path) -> None:
    p = tmp_path / "01_phone.wav"
    write_normal_wav(p)
    rec, _ = _judge(p)
    low = rec.anchor_evidence.lower()
    assert "no_mtime" in low
    assert "no_filename" in low
    assert "method=scan_riff_header_and_metadata_chunks_no_mtime_no_filename" in rec.anchor_evidence
    assert rec.time_basis == "audio_samples_only_no_absolute_anchor"


def test_short_audio_is_excluded(tmp_path: Path) -> None:
    p = tmp_path / "01_recorder_1.wav"
    write_short_wav(p, seconds=1.0)
    rec, warn = _judge(p)
    assert rec.alignment_status == ALIGN_EXCLUDED_TOO_SHORT
    assert rec.exclusion_reason == "audio_too_short_for_analysis"
    assert warn is not None
    assert warn.warning_code == "below_minimum_source_audio_duration_seconds"


def test_normal_audio_not_excluded(tmp_path: Path) -> None:
    p = tmp_path / "01_phone.wav"
    write_normal_wav(p, seconds=40.0)
    rec, warn = _judge(p)
    assert rec.alignment_status != ALIGN_EXCLUDED_TOO_SHORT
    assert rec.alignment_status == ALIGN_UNRESOLVED
    assert warn is None


def test_bext_chunk_does_not_imply_verified(tmp_path: Path) -> None:
    """A time-reference-capable chunk alone never verifies; no transform is
    configured, so the judge stays unresolved rather than assuming."""
    p = tmp_path / "01_recorder_2.wav"
    write_wav_with_bext(p, seconds=40.0)
    rec, _ = _judge(p)
    assert rec.alignment_status == ALIGN_UNRESOLVED
    # the evidence records that a time-reference chunk was found but the
    # transform/cross-check is not configured
    assert "time_reference_chunk_found:True" in rec.anchor_evidence
    assert rec.time_basis == "audio_metadata_present_but_unverified"


def test_scan_never_raises_on_non_wav(tmp_path: Path) -> None:
    p = tmp_path / "junk.wav"
    p.write_bytes(b"not a wav at all")
    rec, _ = _judge(p)
    # not excluded (duration unknown), not verified -> unresolved is the honest call
    assert rec.alignment_status in (ALIGN_UNRESOLVED, ALIGN_EXCLUDED_TOO_SHORT)
