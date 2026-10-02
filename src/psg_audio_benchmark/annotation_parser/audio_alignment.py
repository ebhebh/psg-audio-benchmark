"""Strict, evidence-first audio time-anchor judgment (prompt 4.2).

Audio is held to a STRICTER standard than CSV: a WAV supplies a sample count
(hence a duration) but no wall-clock field by default. This module:

* reads the WAV **container header + non-audio metadata chunks** to look for a
  dataset-provided time reference, recording the exact method and chunks found;
* rejects NTFS ``mtime``, download time, file order and file-name digits as a
  clinical acquisition start, and rejects duration similarity as sync evidence;
* classifies each audio file as ``verified`` / ``unresolved`` / ``excluded`` /
  ``not_applicable`` (never a single boolean);
* marks audio below ``minimum_source_audio_duration_seconds`` as
  ``excluded_audio_too_short_for_analysis`` regardless of container readability.

Findings in the real dataset: every WAV carries only ``fmt `` + ``LIST/INFO``
(``ISFT`` encoder software) + ``data``. No ``bext``, no ``ICRD`` / creation
date, no calibrated clock. The honest outcome is therefore ``unresolved`` for
all audio (and ``excluded`` for the sub-threshold file).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import List, Optional, Tuple

from ..data_audit import audio_metadata
from .schema import (
    ALIGN_EXCLUDED_TOO_SHORT,
    ALIGN_NOT_APPLICABLE,
    ALIGN_UNRESOLVED,
    AudioAlignmentRecord,
    SyncWarning,
    WARN_BELOW_MIN_DURATION,
)

#: Chunks that COULD carry a dataset-provided time reference, if present.
_TIME_REFERENCE_CHUNKS = ("bext", "ID3 ", "iXML", "cue ", "list")
#: INFO sub-fields that could carry a date/time.
_TIME_INFO_IDS = ("ICRD", "ISRF", "ICMT", "IKEY")

#: How many bytes of non-audio chunks to scan (metadata lives near the head).
_SCAN_CAP_BYTES = 16_384


def scan_container_chunks(path: Path) -> Tuple[List[Tuple[str, int]], List[str], bool]:
    """Scan a WAV's RIFF chunks (header + bounded metadata).

    Returns ``(chunk_id_size_pairs, info_field_findings, found_time_reference_chunk)``.
    Never raises; failures -> empty lists.
    """
    chunks: List[Tuple[str, int]] = []
    info_findings: List[str] = []
    found_time_ref = False
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return chunks, info_findings, found_time_ref
            scanned = 12
            while scanned < _SCAN_CAP_BYTES:
                ch = f.read(8)
                if len(ch) < 8:
                    break
                cid = ch[:4].decode("latin-1", "replace")
                try:
                    size = struct.unpack("<I", ch[4:8])[0]
                except struct.error:
                    break
                chunks.append((cid, size))
                if cid in _TIME_REFERENCE_CHUNKS:
                    found_time_ref = True
                if cid == "LIST" and size > 4 and size < _SCAN_CAP_BYTES:
                    body = f.read(size)
                    scanned += 8 + size
                    ltype = body[:4].decode("latin-1", "replace")
                    if ltype == "INFO":
                        info_findings.extend(_parse_info_fields(body[4:]))
                    continue
                if cid == "data":
                    # The payload; do not scan its body.
                    break
                # other metadata chunk: read bounded body (cap)
                body = f.read(min(size, _SCAN_CAP_BYTES))
                scanned += 8 + len(body)
                if size > len(body):
                    f.seek(size - len(body) + (size & 1), 1)
    except OSError:
        pass
    return chunks, info_findings, found_time_ref


def _parse_info_fields(body: bytes) -> List[str]:
    """Parse LIST/INFO sub-fields; keep only non-identifying categorical tags."""
    out: List[str] = []
    i = 0
    while i + 8 <= len(body):
        tag = body[i : i + 4].decode("latin-1", "replace")
        try:
            size = struct.unpack("<I", body[i + 4 : i + 8])[0]
        except struct.error:
            break
        val = body[i + 8 : i + 8 + size].split(b"\x00", 1)[0]
        i += 8 + size + (size & 1)
        try:
            text = val.decode("latin-1", "replace").strip()
        except Exception:  # pragma: no cover - defensive
            text = ""
        # Only echo categorical/software tags; never echo free text that could
        # be identifying. ICRD/ISRF would indicate a (possibly) date-like field.
        if tag in ("ISFT", "ISRF", "ICRD", "IENG", "ITRK"):
            out.append(f"{tag}={text}" if text else tag)
        else:
            out.append(tag)
    return out


def judge_audio_alignment(
    path,
    *,
    run_id: str,
    patient_id: str,
    modality: str,
    source_relpath: str,
    minimum_source_audio_duration_seconds: float,
) -> Tuple[AudioAlignmentRecord, Optional[SyncWarning]]:
    """Produce the strict alignment record for one audio file. Never raises."""
    path = Path(path)
    am = audio_metadata.probe_wav(path)
    duration = am.header_duration_s

    chunks, info_findings, found_time_ref = scan_container_chunks(path)
    chunk_summary = ",".join(f"{c}({s})" for c, s in chunks) or "none"
    info_summary = ",".join(info_findings)

    anchor_evidence = (
        f"riff_chunks:{chunk_summary}; "
        f"info_fields:{info_summary or 'none'}; "
        f"time_reference_chunk_found:{found_time_ref}; "
        f"method=scan_riff_header_and_metadata_chunks_no_mtime_no_filename"
    )

    # 1) Input-QC: below the minimum source duration.
    if duration is not None and duration < minimum_source_audio_duration_seconds:
        rec = AudioAlignmentRecord(
            run_id=run_id,
            patient_id=patient_id,
            modality=modality,
            source_relpath=source_relpath,
            time_basis="audio_samples_only_no_absolute_anchor",
            start_time_seconds=None,
            end_time_seconds=None,
            duration_seconds=duration,
            anchor_evidence=anchor_evidence,
            alignment_status=ALIGN_EXCLUDED_TOO_SHORT,
            alignment_error_seconds=None,
            exclusion_reason="audio_too_short_for_analysis",
            warning_code=WARN_BELOW_MIN_DURATION,
        )
        warn = SyncWarning(
            run_id=run_id,
            patient_id=patient_id,
            scope="audio",
            source_relpath=source_relpath,
            warning_code=WARN_BELOW_MIN_DURATION,
            detail=f"duration_s={duration}<{minimum_source_audio_duration_seconds}",
        )
        return rec, warn

    # 2) A verifiable anchor requires an actual time-reference chunk AND a
    #    transform + cross-check (none exists in this dataset).
    if not found_time_ref:
        rec = AudioAlignmentRecord(
            run_id=run_id,
            patient_id=patient_id,
            modality=modality,
            source_relpath=source_relpath,
            time_basis="audio_samples_only_no_absolute_anchor",
            start_time_seconds=None,
            end_time_seconds=None,
            duration_seconds=duration,
            anchor_evidence=anchor_evidence,
            alignment_status=ALIGN_UNRESOLVED,
            alignment_error_seconds=None,
            exclusion_reason="",
            warning_code="",
        )
        return rec, None

    # 3) If a time-reference chunk WERE found, we still require a documented
    #    transform + numeric cross-check before claiming verified. Absent that
    #    machinery (none is configured), stay unresolved rather than assume.
    rec = AudioAlignmentRecord(
        run_id=run_id,
        patient_id=patient_id,
        modality=modality,
        source_relpath=source_relpath,
        time_basis="audio_metadata_present_but_unverified",
        start_time_seconds=None,
        end_time_seconds=None,
        duration_seconds=duration,
        anchor_evidence=anchor_evidence + "; transform_or_crosscheck_not_configured",
        alignment_status=ALIGN_UNRESOLVED,
        alignment_error_seconds=None,
        exclusion_reason="",
        warning_code="",
    )
    return rec, None


__all__ = [
    "judge_audio_alignment",
    "scan_container_chunks",
    "ALIGN_NOT_APPLICABLE",
]
