"""Multi-evidence file/modality classification (Stage 2, prompt 3.2).

Every file is assigned a *candidate* modality using several independent
evidence sources, never a single filename guess:

* the relative path and a normalised file name,
* the extension,
* the CSV header (column names),
* the JSON top-level keys,
* the audio container kind.

Categories (prompt 3.2)::

    smartphone_audio, recorder_audio, spo2, heart_rate, airflow,
    sleep_structure, annotation_json, annotation_txt, unknown

Each classification carries ``classification_evidence`` (what fired) and a
``classification_confidence`` of ``high`` / ``medium`` / ``low`` / ``unknown``.
When evidence is ambiguous we keep ``unknown`` rather than guess. Recorder
audio is handled per-file so a patient may have 0, 1 or many recorder segments
(prompt: do NOT assume exactly two).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

#: Allowed modality candidates (prompt 3.2).
MODALITY_SMARTPHONE_AUDIO = "smartphone_audio"
MODALITY_RECORDER_AUDIO = "recorder_audio"
MODALITY_SPO2 = "spo2"
MODALITY_HEART_RATE = "heart_rate"
MODALITY_AIRFLOW = "airflow"
MODALITY_SLEEP_STRUCTURE = "sleep_structure"
MODALITY_ANNOTATION_JSON = "annotation_json"
MODALITY_ANNOTATION_TXT = "annotation_txt"
MODALITY_UNKNOWN = "unknown"

ALL_MODALITIES = (
    MODALITY_SMARTPHONE_AUDIO,
    MODALITY_RECORDER_AUDIO,
    MODALITY_SPO2,
    MODALITY_HEART_RATE,
    MODALITY_AIRFLOW,
    MODALITY_SLEEP_STRUCTURE,
    MODALITY_ANNOTATION_JSON,
    MODALITY_ANNOTATION_TXT,
    MODALITY_UNKNOWN,
)

#: Parse a recorder segment index from a file name. ``_recorder_3.wav`` -> 3;
#: a bare ``_recorder.wav`` -> 1 (single segment).
_RECORDER_INDEX_RE = re.compile(r"recorder(?:[_-]?(\d+))?\.(?:wav|mp3|flac)", re.I)

#: Heuristic signal tokens used for CSV header / filename matching.
_SPO2_TOKENS = ("osat", "spo2", "o_sat", "saturation")
_HR_TOKENS = ("heart rate", "heart_rate", "heartrate", " bpm")
_AIRFLOW_TOKENS = ("flow", "airflow", "nasal", "thermistor")
_SLEEP_TOKENS = ("stage", "default staging", "sleep stage", "epoch")


@dataclass
class Classification:
    """Result of classifying one file."""

    modality_candidate: str = MODALITY_UNKNOWN
    classification_evidence: List[str] = field(default_factory=list)
    classification_confidence: str = "unknown"  # high | medium | low | unknown
    #: For recorder_audio only: 1-based segment index (1 for a bare recorder.wav).
    recorder_index: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "modality_candidate": self.modality_candidate,
            "classification_evidence": "; ".join(self.classification_evidence),
            "classification_confidence": self.classification_confidence,
            "recorder_index": "" if self.recorder_index is None else str(self.recorder_index),
        }


def _has_any(text: str, tokens: Sequence[str]) -> Optional[str]:
    """Return the first token found in ``text`` (lowercased), else None."""
    low = text.lower()
    for tok in tokens:
        if tok in low:
            return tok
    return None


def _classify_csv(name: str, header: Optional[Sequence[str]]) -> Classification:
    """Classify a CSV from its header (preferred) and file-name fallback."""
    joined_header = " ".join(header or []).lower()
    # Header is the strong signal.
    if header:
        tok = _has_any(joined_header, _SPO2_TOKENS)
        if tok:
            return Classification(
                MODALITY_SPO2,
                [f"csv_header_token:{tok}"],
                "high",
            )
        tok = _has_any(joined_header, _HR_TOKENS)
        if tok:
            return Classification(
                MODALITY_HEART_RATE,
                [f"csv_header_token:{tok}"],
                "high",
            )
        tok = _has_any(joined_header, _AIRFLOW_TOKENS)
        if tok:
            return Classification(
                MODALITY_AIRFLOW,
                [f"csv_header_token:{tok}"],
                "high",
            )
        tok = _has_any(joined_header, _SLEEP_TOKENS)
        if tok:
            return Classification(
                MODALITY_SLEEP_STRUCTURE,
                [f"csv_header_token:{tok}"],
                "high",
            )
    # Filename fallback (weaker).
    low = name.lower()
    tok = _has_any(low, _SPO2_TOKENS) or _has_any(low.replace("_", " "), _SPO2_TOKENS)
    if tok:
        return Classification(MODALITY_SPO2, [f"filename_token:{tok}"], "medium")
    tok = _has_any(low, _AIRFLOW_TOKENS + ("flow_dr",))
    if tok:
        return Classification(MODALITY_AIRFLOW, [f"filename_token:{tok}"], "medium")
    tok = _has_any(low, _HR_TOKENS) or "hr" in _split_tokens(low)
    if tok:
        return Classification(MODALITY_HEART_RATE, [f"filename_token:{tok or 'hr'}"], "medium")
    tok = _has_any(low, _SLEEP_TOKENS)
    if tok:
        return Classification(MODALITY_SLEEP_STRUCTURE, [f"filename_token:{tok}"], "medium")
    return Classification(MODALITY_UNKNOWN, ["csv:no_matching_header_or_name_token"], "low")


def _split_tokens(name: str) -> List[str]:
    """Split a filename into lowercased alphanumeric tokens for token match."""
    return [t for t in re.split(r"[^a-z0-9]+", name) if t]


def classify_file(
    rel_path: str,
    name: str,
    ext: str,
    csv_header: Optional[Sequence[str]] = None,
    json_keys: Optional[Sequence[str]] = None,
    audio_kind: Optional[str] = None,
) -> Classification:
    """Classify one file into a candidate modality.

    Parameters mirror the evidence sources gathered elsewhere (audio probe /
    CSV probe / JSON probe). All are optional; when a source is missing the
    classifier degrades gracefully rather than guessing.
    """
    ext = ext.lower().lstrip(".")
    low_name = name.lower()

    # --- Audio -----------------------------------------------------------
    if ext in ("wav", "mp3", "flac", "ogg", "m4a"):
        if "phone" in low_name:
            return Classification(
                MODALITY_SMARTPHONE_AUDIO,
                [f"filename_token:phone ext:{ext}"],
                "high",
            )
        if "recorder" in low_name:
            m = _RECORDER_INDEX_RE.search(low_name)
            idx = int(m.group(1)) if (m and m.group(1)) else 1
            return Classification(
                MODALITY_RECORDER_AUDIO,
                [f"filename_token:recorder ext:{ext} segment_index:{idx}"],
                "high",
                recorder_index=idx,
            )
        # Audio container but no device hint -> unknown audio.
        return Classification(
            MODALITY_UNKNOWN,
            [f"audio:{audio_kind or ext}_no_device_token_in_name"],
            "low",
        )

    # --- JSON annotation -------------------------------------------------
    if ext == "json":
        keys = set(k.lower() for k in (json_keys or []))
        strong = {"events", "record_start", "awake_intervals"}
        hits = sorted(keys & strong)
        if hits:
            return Classification(
                MODALITY_ANNOTATION_JSON,
                [f"json_top_keys:{','.join(hits)}"],
                "high",
            )
        return Classification(
            MODALITY_ANNOTATION_JSON,
            ["json:extension_no_known_annotation_keys"],
            "medium",
        )

    # --- TXT annotation --------------------------------------------------
    if ext in ("txt", "text"):
        return Classification(
            MODALITY_ANNOTATION_TXT,
            [f"txt:extension"],
            "medium",
        )

    # --- CSV physiological / sleep structure ----------------------------
    if ext in ("csv", "tsv"):
        return _classify_csv(name, csv_header)

    return Classification(
        MODALITY_UNKNOWN,
        [f"unhandled_extension:{ext or '(none)'}"],
        "low",
    )


def count_recorder_segments(classifications: Sequence[Classification]) -> int:
    """Number of recorder_audio files among ``classifications`` (0, 1 or many)."""
    return sum(1 for c in classifications if c.modality_candidate == MODALITY_RECORDER_AUDIO)


__all__ = [
    "MODALITY_SMARTPHONE_AUDIO",
    "MODALITY_RECORDER_AUDIO",
    "MODALITY_SPO2",
    "MODALITY_HEART_RATE",
    "MODALITY_AIRFLOW",
    "MODALITY_SLEEP_STRUCTURE",
    "MODALITY_ANNOTATION_JSON",
    "MODALITY_ANNOTATION_TXT",
    "MODALITY_UNKNOWN",
    "ALL_MODALITIES",
    "Classification",
    "classify_file",
    "count_recorder_segments",
]
