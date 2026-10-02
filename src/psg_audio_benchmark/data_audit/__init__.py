"""Data audit package (Stage 2).

Stage 2 responsibility (prompt ``04_阶段2...``):
  * Enforce the license confirmation gate before reading any patient content
    (``license_gate``).
  * Resolve patient directories and detect id issues (``patient_resolution``).
  * Classify every file into a candidate modality from multiple evidence
    sources (``classifier``).
  * Probe WAV headers with the stdlib ``wave`` (``audio_metadata``).
  * Stream-probe CSV schema / time axis (``csv_probe``).
  * Explore JSON/TXT annotation structure, preliminary only
    (``annotation_probe``).
  * Write the six Stage-2 CSV products atomically (``manifests``).
  * Render Stage-2 markdown reports (``reports``).
  * Orchestrate the above with output isolation + a contamination guard
    (``runner``).

Privacy / safety: this package never modifies, moves or deletes anything under
``data/raw``. When the license gate is blocked it does not even scan ``data/raw``.
"""

from __future__ import annotations

from . import (
    annotation_probe,
    audio_metadata,
    classifier,
    csv_probe,
    license_gate,
    manifests,
    patient_resolution,
    reports,
    runner,
)

__all__ = [
    "annotation_probe",
    "audio_metadata",
    "classifier",
    "csv_probe",
    "license_gate",
    "manifests",
    "patient_resolution",
    "reports",
    "runner",
]
