"""Raw physiological CSV reader + half-open window time selector (Stage 5).

Reads one modality CSV (heart_rate / spo2 / airflow), parses the
``relative position (hh:mm:ss.ms)`` column to seconds (measured from the file's
first sample) and the value column (``"-"``/blank/non-numeric -> NaN), then
selects the samples whose Stage-3-clock-relative time falls in a half-open window
``[start, end)``.

The record-start-relative time of a sample is::

    t_rel = parsed_relative_position_seconds + offset

where ``offset = first_absolute_relative_to_record_start - first_relative_seconds``
(inherited from the Stage-3 verified time axis; equals
``first_absolute_relative_to_record_start`` for every verified file). This
reproduces the Stage-3 per-sample clock exactly without re-deriving the absolute
midnight-rollover, and selection is by TIME, not CSV row index.

This module is the ONLY place that opens raw physiological CSVs. It NEVER opens
``*.wav`` (a hard guard) and never writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class CSVReadError(RuntimeError):
    """Raised when a physiological CSV cannot be parsed as expected."""


class AudioAccessForbidden(RuntimeError):
    """Raised when a ``*.wav`` path is handed to the CSV reader (hard block)."""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_hhmmss_to_seconds(text: str) -> float:
    """Parse ``HH:MM:SS.ms`` (or ``HH:MM:SS``) into seconds (float).

    Tolerant of surrounding whitespace. Raises :class:`CSVReadError` on a value
    that is not a valid ``HH:MM:SS`` clock string.
    """
    s = str(text).strip()
    if not s:
        raise CSVReadError("empty time cell")
    parts = s.split(":")
    if len(parts) != 3:
        raise CSVReadError(f"expected HH:MM:SS.ms, got {text!r}")
    h, m, rest = parts
    if "." in rest:
        sec, frac = rest.split(".", 1)
        fractional = float("0." + frac)
    else:
        sec, fractional = rest, 0.0
    try:
        return int(h) * 3600 + int(m) * 60 + int(sec) + fractional
    except ValueError as exc:  # pragma: no cover - defensive
        raise CSVReadError(f"unparseable time {text!r}: {exc}") from exc


def parse_value_cell(text: str) -> float:
    """Parse a value cell; the dataset's ``"-"`` (and blank/non-numeric) -> NaN."""
    if text is None:
        return float("nan")
    s = str(text).strip()
    if s in ("", "-", "nan", "NaN", "None", "NA"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Column discovery
# ---------------------------------------------------------------------------

def _find_columns(columns) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(relative_col, absolute_col, value_col)`` by name.

    ``relative_col`` is the first column whose lower-cased name contains both a
    time hint and ``relative``; ``absolute_col`` similarly for ``absolute``;
    ``value_col`` is the remaining non-time column.
    """
    rel = abs_ = None
    for c in columns:
        low = str(c).lower()
        if rel is None and "relative" in low and "position" in low:
            rel = c
        elif abs_ is None and "absolute" in low and "position" in low:
            abs_ = c
    value = None
    for c in columns:
        if c in (rel, abs_):
            continue
        # skip obvious non-value time columns
        low = str(c).lower()
        if "position" in low or "epoch" in low:
            continue
        value = c
        break
    return rel, abs_, value


# ---------------------------------------------------------------------------
# Loaded signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalitySignal:
    """One patient's one modality CSV loaded into memory for windowed access."""

    patient_id: str
    modality: str
    source_relpath: str
    rel_seconds: np.ndarray            # parsed relative-position seconds
    values: np.ndarray                 # float, NaN for missing
    offset: float                      # Stage-3 clock offset (rel -> record_start)
    time_monotonic: bool               # whether rel_seconds is non-decreasing

    def select(self, start_rel: float, end_rel: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(t_win, v_win)`` for samples with ``start <= t < end``.

        ``t = rel_seconds + offset``. Half-open: a boundary sample belongs to
        exactly one window. Uses ``searchsorted`` on the monotonic time axis
        (the normal verified case) and falls back to a boolean mask otherwise.
        """
        t = self.rel_seconds + self.offset
        if self.time_monotonic:
            lo = int(np.searchsorted(t, start_rel, side="left"))
            hi = int(np.searchsorted(t, end_rel, side="left"))
            return t[lo:hi], self.values[lo:hi]
        mask = (t >= start_rel) & (t < end_rel)
        return t[mask], self.values[mask]


def _assert_not_wav(path: Path) -> None:
    name = path.name.lower()
    if name.endswith(".wav") or ".wav" in name:
        raise AudioAccessForbidden(
            f"Refusing to open an audio file {path.name!r}; Stage 5 reads only "
            f"physiological CSVs (audio_features_present is hard False)."
        )


def read_modality_csv(
    path: Path,
    *,
    patient_id: str,
    modality: str,
    source_relpath: str,
    offset: float,
) -> ModalitySignal:
    """Read a raw modality CSV into a :class:`ModalitySignal`.

    The CSV is read as raw strings (``dtype=str``) so the dataset's ``"-"``
    missing marker is handled explicitly rather than coerced. Raises
    :class:`AudioAccessForbidden` for any ``*.wav`` path and :class:`CSVReadError`
    when the expected relative-position / value columns are absent.
    """
    _assert_not_wav(path)
    if not path.is_file():
        raise CSVReadError(f"modality CSV not found: {path}")
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as exc:  # pragma: no cover - defensive
        raise CSVReadError(f"failed to read CSV {path}: {exc}") from exc

    rel_col, abs_col, val_col = _find_columns(list(df.columns))
    if rel_col is None:
        raise CSVReadError(
            f"{path}: no 'relative position' time column found (header="
            f"{list(df.columns)!r}); this modality cannot be windowed."
        )
    if val_col is None:
        raise CSVReadError(
            f"{path}: no value column found (header={list(df.columns)!r})."
        )

    rel_seconds = np.asarray(
        [parse_hhmmss_to_seconds(x) for x in df[rel_col].tolist()], dtype=float
    )
    values = np.asarray([parse_value_cell(x) for x in df[val_col].tolist()], dtype=float)
    time_monotonic = bool(rel_seconds.shape[0] < 2 or np.all(np.diff(rel_seconds) >= 0))

    return ModalitySignal(
        patient_id=patient_id,
        modality=modality,
        source_relpath=source_relpath,
        rel_seconds=rel_seconds,
        values=values,
        offset=float(offset),
        time_monotonic=time_monotonic,
    )


__all__ = [
    "CSVReadError",
    "AudioAccessForbidden",
    "parse_hhmmss_to_seconds",
    "parse_value_cell",
    "ModalitySignal",
    "read_modality_csv",
]
