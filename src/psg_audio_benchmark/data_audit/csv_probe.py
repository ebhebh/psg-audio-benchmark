"""CSV structure & time-column exploration (Stage 2, prompt 4.2).

A *streaming*, deliberately shallow probe of physiological / sleep-structure
CSV files. It reads the header, a bounded sample of leading/trailing rows, and
the statistics needed to characterise the time axis -- it never loads the whole
file, never imputes, never models.

Per file it records: column names, an inferred role, the row count (streaming
tally), encoding/delimiter, candidate timestamp columns, the first & last
parseable time, the time-delta distribution / sampling-rate estimate, basic
cell missingness (``-`` or empty only), and parse anomalies. Files with no time
column or an unparseable time format get a structure *problem*, not a guessed
sampling rate.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

#: How many leading + trailing rows to keep in memory for time/missingness stats.
_SAMPLE_HEAD = 200
_SAMPLE_TAIL = 200

#: Columns whose name suggests a time / position / epoch axis.
_TIME_NAME_TOKENS = ("position", "time", "timestamp", "epoch", "clock")

#: Missing-value markers used by this dataset (basic cell missingness only).
_MISSING_MARKERS = ("-", "", "nan", "none")

#: Match ``HH:MM:SS`` with optional ``.ms`` fraction (possibly >24h on the hour).
_HHMMSS_RE = re.compile(r"^\s*(\d{1,3}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?\s*$")

#: Max rows we will ever iterate (safety valve against pathological files).
_MAX_ROWS_ITER = 5_000_000


@dataclass
class CsvProbe:
    """Structure + time-axis summary for one CSV file."""

    columns: List[str] = field(default_factory=list)
    inferred_role: str = "unknown"
    row_count: int = 0
    delimiter: str = ","
    encoding: str = "utf-8"
    candidate_timestamp_cols: List[str] = field(default_factory=list)
    time_column_used: str = ""
    first_parseable_time: str = ""
    last_parseable_time: str = ""
    time_delta_min_s: Optional[float] = None
    time_delta_median_s: Optional[float] = None
    time_delta_max_s: Optional[float] = None
    time_delta_mode_s: Optional[float] = None
    sample_rate_estimate_hz: Optional[float] = None
    basic_cell_missingness_ratio: Optional[float] = None
    parse_anomalies: List[str] = field(default_factory=list)
    time_axis_kind: str = "none"  # hhmmss | epoch | none
    has_time_column: bool = False

    def as_dict(self) -> dict:
        return {
            "csv_columns": "; ".join(self.columns),
            "csv_inferred_role": self.inferred_role,
            "csv_row_count": self.row_count,
            "csv_delimiter": self.delimiter,
            "csv_encoding": self.encoding,
            "csv_candidate_timestamp_cols": "; ".join(self.candidate_timestamp_cols),
            "csv_time_column_used": self.time_column_used,
            "csv_first_parseable_time": self.first_parseable_time,
            "csv_last_parseable_time": self.last_parseable_time,
            "csv_time_axis_kind": self.time_axis_kind,
            "csv_time_delta_min_s": _opt(self.time_delta_min_s),
            "csv_time_delta_median_s": _opt(self.time_delta_median_s),
            "csv_time_delta_max_s": _opt(self.time_delta_max_s),
            "csv_time_delta_mode_s": _opt(self.time_delta_mode_s),
            "csv_sample_rate_estimate_hz": _opt(self.sample_rate_estimate_hz),
            "csv_basic_cell_missingness_ratio": _opt(self.basic_cell_missingness_ratio),
            "csv_parse_anomalies": "; ".join(self.parse_anomalies),
            "csv_has_time_column": self.has_time_column,
        }


def _opt(v) -> str:
    return "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------------------
# Encoding / delimiter detection
# ---------------------------------------------------------------------------

def _detect_encoding(path: Path) -> str:
    """Best-effort encoding detection from the first bytes."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(4096)
    except OSError:
        return "utf-8"
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "latin-1"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _detect_delimiter(sample_line: str) -> str:
    """Pick the delimiter present most often on the header line."""
    counts = {d: sample_line.count(d) for d in (",", ";", "\t", "|")}
    delim, n = max(counts.items(), key=lambda kv: kv[1])
    return delim if n > 0 else ","


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def _parse_hhmmss_seconds(value: str) -> Optional[float]:
    m = _HHMMSS_RE.match(value or "")
    if not m:
        return None
    h, mi, s, ms = m.groups()
    frac = 0.0
    if ms:
        frac = float("0." + ms)
    return int(h) * 3600 + int(mi) * 60 + int(s) + frac


def _parse_epoch(value: str) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _is_time_named(col: str) -> bool:
    low = col.lower()
    return any(tok in low for tok in _TIME_NAME_TOKENS)


def _choose_time_column(columns: List[str], sample_rows: List[List[str]]) -> Tuple[str, str]:
    """Return (column_name, axis_kind) for the best time column, or ('', 'none').

    Prefers a 'relative position' style column parseable as HH:MM:SS; falls back
    to an integer epoch column. Never guesses when nothing parses.
    """
    rel_candidates = [
        c for c in columns if "relative" in c.lower() and _is_time_named(c)
    ]
    abs_candidates = [
        c for c in columns if "absolute" in c.lower() and _is_time_named(c)
    ]
    epoch_candidates = [c for c in columns if "epoch" in c.lower()]
    generic_time = [c for c in columns if _is_time_named(c)]

    def _col_idx(name: str) -> Optional[int]:
        try:
            return columns.index(name)
        except ValueError:
            return None

    def _first_parseable(cands: List[str], parser) -> Optional[str]:
        for name in cands:
            idx = _col_idx(name)
            if idx is None:
                continue
            ok = sum(
                1 for r in sample_rows
                if len(r) > idx and parser(r[idx]) is not None
            )
            if ok > 0:
                return name
        return None

    # 1) relative HH:MM:SS, 2) absolute HH:MM:SS, 3) any time-named HH:MM:SS.
    for cands, kind in (
        (rel_candidates, "hhmmss"),
        (abs_candidates, "hhmmss"),
        (generic_time, "hhmmss"),
    ):
        name = _first_parseable(cands, _parse_hhmmss_seconds)
        if name:
            return name, kind
    # 4) integer epoch.
    name = _first_parseable(epoch_candidates, _parse_epoch)
    if name:
        return name, "epoch"
    return "", "none"


# ---------------------------------------------------------------------------
# Inferred role (header-only hint; the classifier is the authority)
# ---------------------------------------------------------------------------

def _infer_role(columns: List[str]) -> str:
    joined = " ".join(columns).lower()
    if "osat" in joined or "spo2" in joined or "saturation" in joined:
        return "spo2"
    if "heart rate" in joined or "bpm" in joined:
        return "heart_rate"
    if "flow" in joined:
        return "airflow"
    if "stage" in joined or "staging" in joined:
        return "sleep_structure"
    return "unknown"


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def probe_csv(path: Path) -> CsvProbe:
    """Stream-probe a CSV. Never raises: problems are recorded as anomalies."""
    path = Path(path)
    probe = CsvProbe()

    if not path.is_file():
        probe.parse_anomalies.append("file_not_found")
        return probe
    try:
        if path.stat().st_size == 0:
            probe.parse_anomalies.append("empty_file")
            return probe
    except OSError as exc:
        probe.parse_anomalies.append(f"stat_failed:{type(exc).__name__}")
        return probe

    encoding = _detect_encoding(path)

    try:
        handle = open(path, "r", encoding=encoding, newline="", errors="replace")
    except OSError as exc:
        probe.parse_anomalies.append(f"open_failed:{type(exc).__name__}:{exc}")
        return probe

    with handle:
        # Peek the first line to detect the delimiter.
        first_line = handle.readline()
        delimiter = _detect_delimiter(first_line)
        probe.delimiter = delimiter
        probe.encoding = encoding
        handle.seek(0)

        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            probe.parse_anomalies.append("empty_no_header")
            return probe
        except csv.Error as exc:
            probe.parse_anomalies.append(f"header_parse_error:{exc}")
            return probe

        probe.columns = [c.strip() for c in header]
        probe.inferred_role = _infer_role(probe.columns)

        # Stream rows: keep a bounded head/tail sample, count the rest.
        head: List[List[str]] = []
        tail: List[List[str]] = []
        n_rows = 0
        n_cells = 0
        n_missing = 0
        try:
            for row in reader:
                n_rows += 1
                # basic cell missingness across all cells of sampled rows only.
                if n_rows <= _SAMPLE_HEAD:
                    head.append(row)
                    for cell in row:
                        n_cells += 1
                        if (cell or "").strip().lower() in _MISSING_MARKERS:
                            n_missing += 1
                else:
                    tail.append(row)
                    if len(tail) > _SAMPLE_TAIL:
                        tail.pop(0)
                if n_rows > _MAX_ROWS_ITER:
                    probe.parse_anomalies.append(
                        f"row_iter_capped_at_{_MAX_ROWS_ITER}"
                    )
                    break
        except csv.Error as exc:
            probe.parse_anomalies.append(f"row_parse_error:{exc}")

        probe.row_count = n_rows
        sample = head + tail[-_SAMPLE_TAIL:]
        if n_cells:
            probe.basic_cell_missingness_ratio = round(n_missing / n_cells, 6)

        # Time-axis characterisation.
        time_col, axis_kind = _choose_time_column(probe.columns, sample)
        probe.time_axis_kind = axis_kind
        probe.has_time_column = bool(time_col)
        probe.candidate_timestamp_cols = [
            c for c in probe.columns if _is_time_named(c)
        ]
        if not time_col:
            if probe.candidate_timestamp_cols:
                probe.parse_anomalies.append(
                    "time_named_columns_present_but_unparseable"
                )
            else:
                probe.parse_anomalies.append("no_time_column")
            return probe  # do NOT guess a sampling rate

        probe.time_column_used = time_col
        idx = probe.columns.index(time_col)
        parser = _parse_hhmmss_seconds if axis_kind == "hhmmss" else _parse_epoch

        times: List[float] = []
        first_raw, last_raw = "", ""
        for row in sample:
            if len(row) <= idx:
                continue
            raw = row[idx]
            val = parser(raw)
            if val is None:
                continue
            times.append(float(val))
            if not first_raw:
                first_raw = raw
            last_raw = raw
        probe.first_parseable_time = first_raw
        probe.last_parseable_time = last_raw

        if len(times) >= 2:
            deltas = [b - a for a, b in zip(times, times[1:]) if b >= a]
            valid = [d for d in deltas if d > 0]
            if valid:
                probe.time_delta_min_s = round(min(valid), 6)
                probe.time_delta_max_s = round(max(valid), 6)
                probe.time_delta_median_s = round(median(valid), 6)
                # mode: most common delta rounded to 3 decimals.
                rounded = Counter(round(d, 3) for d in valid)
                probe.time_delta_mode_s = round(rounded.most_common(1)[0][0], 6)
                med = probe.time_delta_median_s
                if med and med > 0:
                    probe.sample_rate_estimate_hz = round(1.0 / med, 6)
            if any(b < a for a, b in zip(times, times[1:])):
                probe.parse_anomalies.append("non_monotonic_time_observed")

    return probe


__all__ = [
    "CsvProbe",
    "probe_csv",
]
