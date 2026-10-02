"""Shared synthetic fixtures for Stage-3 tests.

Builds on the Stage-2 fixtures (same fabricated content shapes, no real patient
data) and adds Stage-3-specific helpers for the prompt-section-6 boundary cases:

* annotation JSON with a controlled ``event_start`` / ``evnet_start`` combination
  (including the typo, the canonical field, and a deliberate conflict);
* a CSV whose absolute clock column crosses midnight (to exercise the single
  rollover rule) with a matching relative column;
* a WAV shorter than the 30 s input-QC threshold (excluded) and one longer
  (unresolved, not excluded);
* a tiny isolated Config whose raw + docs point under ``tmp_path`` and whose
  license gate passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from _stage2_fixtures import (
    write_complete_license_evidence,
    write_wav,
)

from psg_audio_benchmark.config import Config, default_project_root


# ---------------------------------------------------------------------------
# Annotation JSON with controlled start fields (boundary case 1)
# ---------------------------------------------------------------------------

def write_annotation_json_controlled(
    path: Path,
    *,
    events: list,
    record_start: float = 75454.0,
    awake: Optional[list] = None,
) -> Path:
    """Write an annotation JSON whose event dicts are used verbatim.

    Each event dict may carry ``evnet_start`` (the dataset's typo) and/or
    ``event_start`` (canonical) so the start-field resolver can be exercised for
    typo-tolerance, priority and conflict detection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if awake is None:
        awake = [[75500.0, 75600.0]]
    payload = {"record_start": record_start, "awake_intervals": awake, "events": events}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cross-midnight CSV (boundary case 2)
# ---------------------------------------------------------------------------

def write_cross_midnight_csv(
    path: Path,
    *,
    start_sod: float = 86340.0,
    step: float = 60.0,
    n: int = 4,
    filename: str = "01_SpO2.csv",
) -> Path:
    """A SpO2-style CSV whose absolute clock crosses midnight exactly once.

    The relative column advances monotonically in lock-step so the absolute and
    relative durations match (err == 0), exercising the single-rollover rule and
    proving a legitimate midnight wrap is NOT flagged non-monotonic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),OSat ("%")']
    for i in range(n):
        rel = i * step
        abst = start_sod + rel
        abst_wrapped = abst % 86400.0
        lines.append(f"{_hhmmss(rel)},{_hhmmss(abst_wrapped)},{97}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_monotonic_csv(
    path: Path,
    *,
    start_sod: float = 75454.0,
    step: float = 60.0,
    n: int = 4,
) -> Path:
    """A SpO2-style CSV that stays within one day (no midnight wrap)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),OSat ("%")']
    for i in range(n):
        rel = i * step
        lines.append(f"{_hhmmss(rel)},{_hhmmss(start_sod + rel)},{97}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _hhmmss(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    rem = seconds - h * 3600
    m = int(rem // 60)
    s = rem - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Audio files (boundary cases 6 & 7)
# ---------------------------------------------------------------------------

def write_short_wav(path: Path, *, seconds: float = 1.0, framerate: int = 8000) -> Path:
    """A WAV shorter than the 30 s input-QC threshold -> must be EXCLUDED."""
    nframes = max(1, int(seconds * framerate))
    return write_wav(path, channels=1, sampwidth=2, framerate=framerate, nframes=nframes)


def write_normal_wav(path: Path, *, seconds: float = 40.0, framerate: int = 8000) -> Path:
    """A WAV long enough to clear the 30 s threshold -> unresolved (not excluded)."""
    nframes = int(seconds * framerate)
    return write_wav(path, channels=1, sampwidth=2, framerate=framerate, nframes=nframes)


def write_wav_with_bext(path: Path, *, seconds: float = 40.0) -> Path:
    """A WAV that carries a synthetic ``bext`` (broadcast) chunk.

    Even with a time-reference-capable chunk present, the strict judge must stay
    UNRESOLVED because no documented transform + numeric cross-check is
    configured (it never assumes a chunk implies a calibrated clock).
    """
    write_normal_wav(path, seconds=seconds)
    data = path.read_bytes()
    # Minimal bext chunk (raw_value=0, time_reference=0, 256-byte body zeroed).
    bext_body = b"\x00" * (256 + 2)
    bext_chunk = b"bext" + len(bext_body).to_bytes(4, "little") + bext_body
    # Re-inject bext between "WAVE" and the original "fmt " chunk.
    new = data[:12] + bext_chunk + data[12:]
    path.write_bytes(new)
    return path


# ---------------------------------------------------------------------------
# Isolated config + synthetic raw tree for Stage-3 fixture runs
# ---------------------------------------------------------------------------

def make_isolated_cfg(tmp_path: Path) -> Config:
    """Real Config whose raw + docs point at small tmp dirs under tmp_path."""
    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    write_complete_license_evidence(docs)  # gate passes
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["docs"] = docs
    return cfg


def build_stage3_patient(
    raw_root: Path,
    pid: str = "01",
    *,
    annotation_events: Optional[list] = None,
    awake: Optional[list] = None,
    record_start: float = 75454.0,
    cross_midnight_csv: bool = False,
    short_audio: bool = False,
    bext_audio: bool = False,
    include_phone: bool = True,
) -> Path:
    """Build a single synthetic patient dir under ``raw_root/V5/Data/<pid>``."""
    d = raw_root / "V5" / "Data" / pid
    d.mkdir(parents=True, exist_ok=True)
    if annotation_events is None:
        annotation_events = [
            {"event_type": "hypo", "evnet_start": record_start + 400, "event_duration": 41.0, "sleep_stage": "W"},
            {"event_type": "osa", "evnet_start": record_start + 600, "event_duration": 22.0, "sleep_stage": "N1"},
        ]
    write_annotation_json_controlled(
        d / f"{pid}_annotation.json",
        events=annotation_events,
        record_start=record_start,
        awake=awake,
    )
    if cross_midnight_csv:
        write_cross_midnight_csv(d / f"{pid}_SpO2.csv")
    else:
        write_monotonic_csv(d / f"{pid}_SpO2.csv", start_sod=record_start)
    # sleep_structure CSV intentionally has NO relative column (real shape).
    (d / f"{pid}_sleep_stage.csv").write_text(
        'position (epoch),absolute position (hh:mm:ss.ms),Default Staging Set ("stage")\n'
        f"1,{_hhmmss(record_start)},W\n2,{_hhmmss(record_start + 30)},N1\n",
        encoding="utf-8",
    )
    if include_phone:
        write_normal_wav(d / f"{pid}_phone.wav")
    if short_audio:
        write_short_wav(d / f"{pid}_recorder_1.wav")
    if bext_audio:
        write_wav_with_bext(d / f"{pid}_recorder_2.wav")
    return d
