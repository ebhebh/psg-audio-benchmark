"""Shared synthetic fixtures for Stage-2 tests.

Every fixture here is SMALL and SYNTHETIC -- no real patient data is copied into
tests, Git or reports. The shapes mirror the real dataset's naming (so the
classifier/probes exercise the same code paths) but the contents are fabricated.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Optional


def write_wav(path: Path, *, channels: int = 1, sampwidth: int = 2,
              framerate: int = 8000, nframes: int = 8000) -> Path:
    """Write a tiny valid WAV (1 s of silence at 8 kHz by default)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * nframes)
    return path


def write_truncated_wav(path: Path) -> Path:
    """A RIFF/WAVE header with no complete data chunk -> probe must flag it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    return path


def write_spo2_csv(path: Path, rows: int = 5, missing_every: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        'relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),OSat ("%")'
    ]
    for i in range(rows):
        rel = f"00:00:{i:02d}.000"
        abst = f"22:39:{34+i:02d}.000"
        val = "-" if (i % missing_every == 0) else str(97 - (i % 3))
        lines.append(f"{rel},{abst},{val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_hr_csv(path: Path, rows: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),Heart Rate ("bpm")']
    for i in range(rows):
        lines.append(f"00:00:{i:02d}.000,22:39:{34+i:02d}.000,{60+i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_flow_csv(path: Path, rows: int = 6) -> Path:
    """Airflow at 0.5 s step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["relative position (hh:mm:ss.ms),absolute position (hh:mm:ss.ms),Flow_DR"]
    for i in range(rows):
        t = i * 0.5
        rel = f"00:00:{int(t):02d}.{int((t - int(t)) * 1000):03d}"
        lines.append(f"{rel},22:39:34.000,{-0.5 + i * 0.1:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_sleep_stage_csv(path: Path, epochs: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['position (epoch),absolute position (hh:mm:ss.ms),Default Staging Set ("stage")']
    for i in range(1, epochs + 1):
        abst = f"22:{39 + i:02d}:04.000"
        lines.append(f"{i},{abst},{'W' if i == 1 else 'N1'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_annotation_json(
    path: Path,
    *,
    events: Optional[list] = None,
    record_start: float = 75454.0,
    awake: Optional[list] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if events is None:
        events = [
            {"event_type": "hypo", "evnet_start": 81888.0, "event_duration": 41.0, "sleep_stage": "W"},
            {"event_type": "osa", "evnet_start": 82000.0, "event_duration": 22.0, "sleep_stage": "N1"},
        ]
    if awake is None:
        awake = [[81583.0, 81697.0]]
    payload = {"record_start": record_start, "awake_intervals": awake, "events": events}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_no_time_csv(path: Path) -> Path:
    """A CSV with no time/position column (must NOT get a guessed sampling rate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("label,value\na,1\nb,2\n", encoding="utf-8")
    return path


def build_synthetic_raw(
    root: Path,
    *,
    patients: Optional[dict] = None,
) -> Path:
    """Build a tiny synthetic raw tree under ``root``.

    ``patients`` maps a patient id -> a dict of which files to create. Defaults
    build two patients (one full, one with many recorders + airflow) so the
    classifier/probes/runner exercise the varied recorder-count path.
    """
    if patients is None:
        patients = {
            "01": {"phone": True, "recorders": 2, "flow": False},
            "02": {"phone": True, "recorders": 6, "flow": True},
        }
    base = root / "V5" / "Data"
    for pid, spec in patients.items():
        d = base / pid
        write_spo2_csv(d / f"{pid}_SpO2.csv")
        write_hr_csv(d / f"{pid}_HR.csv")
        write_sleep_stage_csv(d / f"{pid}_sleep_stage.csv")
        write_annotation_json(d / f"{pid}_annotation.json")
        if spec.get("phone"):
            write_wav(d / f"{pid}_phone.wav")
        n = spec.get("recorders", 0)
        if n == 1:
            write_wav(d / f"{pid}_recorder.wav")
        else:
            for k in range(1, n + 1):
                write_wav(d / f"{pid}_recorder_{k}.wav")
        if spec.get("flow"):
            write_flow_csv(d / f"{pid}_Flow_DR.csv")
    return base


COMPLETE_LICENSE_EVIDENCE = """# Primary dataset license confirmation (synthetic, test fixture)

- data_doi: 10.57760/sciencedb.19070
- dataset_version: V5
- access_date: 2026-07-14
- license_terms: CC BY-NC 4.0 per Science Data Bank page terms
- noncommercial_research_basis: permitted for non-commercial research per page terms
- deriv_features_policy: derived features and aggregates allowed; no raw audio redistribution
- redistribute_audio_policy: redistribution of raw identifiable audio not allowed
- evidence_screenshot_relpath: docs/license_evidence/saved_pages/index.html
- confirmer_name: Test Confirmer
- confirmation_date: 2026-07-14
- open_items: none
"""


def write_complete_license_evidence(docs_dir: Path) -> Path:
    """Write a COMPLETE, gate-passing evidence file under ``docs_dir``."""
    ev_dir = docs_dir / "license_evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    p = ev_dir / "primary_dataset_license_confirmation.md"
    p.write_text(COMPLETE_LICENSE_EVIDENCE, encoding="utf-8")
    return p
