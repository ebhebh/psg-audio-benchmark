"""Dataclasses and status vocabularies for Stage 3 annotation parsing.

Holds the typed records produced by the parser/normalizer and the categorical
status strings used across the synchronization inventory. Status values are kept
as named constants (never ad-hoc booleans) so the four alignment states
(verified / unresolved / excluded / not_applicable) stay distinct (prompt 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Alignment / synchronization status vocabulary (prompt 4.3, 4.2)
# ---------------------------------------------------------------------------

ALIGN_VERIFIED = "verified"
ALIGN_UNRESOLVED = "unresolved_no_trustworthy_audio_time_anchor"
ALIGN_EXCLUDED_TOO_SHORT = "excluded_audio_too_short_for_analysis"
ALIGN_NOT_APPLICABLE = "not_applicable"

#: All distinct alignment states (no boolean masking).
ALL_ALIGNMENT_STATES = (
    ALIGN_VERIFIED,
    ALIGN_UNRESOLVED,
    ALIGN_EXCLUDED_TOO_SHORT,
    ALIGN_NOT_APPLICABLE,
)

#: Warning codes used in synchronization_warnings.csv / inventory.
WARN_NON_MONOTONIC_CSV = "non_monotonic_csv_absolute_time"
WARN_ABS_REL_DURATION_MISMATCH = "csv_absolute_relative_duration_mismatch"
WARN_NO_ABSOLUTE_COLUMN = "csv_no_absolute_time_column"
WARN_NO_RELATIVE_COLUMN = "csv_no_relative_time_column"
WARN_EVENT_BEFORE_RECORD_START = "event_start_before_record_start"
WARN_SIGNAL_STARTS_BEFORE_RECORD_START = "signal_starts_before_record_start"
WARN_AWAKE_BEFORE_RECORD_START = "awake_interval_before_record_start"
WARN_BELOW_MIN_DURATION = "below_minimum_source_audio_duration_seconds"
WARN_UNKNOWN = "unknown_warning"

# ---------------------------------------------------------------------------
# Rejection severity
# ---------------------------------------------------------------------------

SEV_ERROR = "error"      # event dropped from parsed_events
SEV_WARNING = "warning"  # event still usable but flagged


@dataclass
class RejectionRecord:
    """One rejected/flagged annotation event (prompt 3.1)."""

    run_id: str
    source_annotation_relpath: str
    patient_id: str
    source_event_index: int
    reason: str
    severity: str = SEV_ERROR
    raw_field_hint: str = ""  # safe, non-identifying hint (e.g. field name)


@dataclass
class ParsedEvent:
    """One standardized, on-timeline event (prompt 3.1/3.2)."""

    run_id: str
    patient_id: str
    source_annotation_relpath: str
    source_event_index: int
    event_type_raw: str
    event_type_standardized: str
    mapping_status: str
    is_any_scored_respiratory_event: bool
    event_start_raw_seconds: float
    event_start_seconds_of_day: float
    event_start_day_offset: int
    event_start_relative_to_record_start: float
    event_duration_seconds: float
    event_end_relative_to_record_start: float
    overlaps_awake_interval: bool
    awake_overlap_seconds: float
    awake_overlap_fraction: float
    sleep_stage_raw: str = ""
    conflicting_event_start_fields: bool = False
    # event <-> per-signal availability (filled by signals step)
    overlaps_signals: str = ""  # ";".join(modality) that the event falls within
    n_signals_covering: int = 0


@dataclass
class AwakeRaw:
    """One awake interval exactly as stored (prompt 3.3)."""

    run_id: str
    patient_id: str
    source_annotation_relpath: str
    raw_order_index: int
    start_raw_seconds: float
    end_raw_seconds: float


@dataclass
class AwakeCanonical:
    """One canonical awake interval after dedup/merge, with provenance."""

    run_id: str
    patient_id: str
    start_seconds_of_day: float
    start_day_offset: int
    start_relative_to_record_start: float
    end_seconds_of_day: float
    end_day_offset: int
    end_relative_to_record_start: float
    duration_seconds: float
    cross_midnight: bool
    #: raw_order_index values merged into this canonical interval.
    source_raw_indices: str
    n_source_intervals: int


@dataclass
class SignalTimeRange:
    """Time range of one CSV signal for one patient (prompt 4.1)."""

    run_id: str
    patient_id: str
    modality: str
    source_relpath: str
    time_axis_kind: str  # hhmmss | epoch | none
    has_absolute_column: bool
    has_relative_column: bool
    first_absolute_cumulative_seconds: Optional[float]
    last_absolute_cumulative_seconds: Optional[float]
    first_absolute_relative_to_record_start: Optional[float]
    last_absolute_relative_to_record_start: Optional[float]
    first_relative_seconds: Optional[float]
    last_relative_seconds: Optional[float]
    sample_count: int
    absolute_duration_seconds: Optional[float]
    relative_duration_seconds: Optional[float]
    cross_midnight: bool
    first_day_offset: int
    last_day_offset: int
    verification_status: str
    data_quality_status: str
    alignment_error_seconds: Optional[float]
    warnings: str


@dataclass
class RecordTimeAnchor:
    """The per-patient master record_start anchor (prompt 4.1)."""

    run_id: str
    patient_id: str
    source_annotation_relpath: str
    record_start_present: bool
    record_start_raw_seconds: Optional[float]
    record_start_seconds_of_day: Optional[float]
    record_start_day_offset: Optional[int]
    record_start_cumulative_seconds: Optional[float]
    time_basis: str


@dataclass
class AudioAlignmentRecord:
    """Per-audio-file strict anchor judgment (prompt 4.2/4.3)."""

    run_id: str
    patient_id: str
    modality: str
    source_relpath: str
    time_basis: str
    start_time_seconds: Optional[float]
    end_time_seconds: Optional[float]
    duration_seconds: Optional[float]
    anchor_evidence: str
    alignment_status: str
    alignment_error_seconds: Optional[float]
    exclusion_reason: str
    warning_code: str


@dataclass
class SyncWarning:
    """One row of synchronization_warnings.csv."""

    run_id: str
    patient_id: str
    scope: str  # annotation | csv | audio | awake
    source_relpath: str
    warning_code: str
    detail: str


@dataclass
class Stage3Summary:
    """Aggregate outcome of a Stage-3 run (used by the report renderers)."""

    run_id: str
    overall_status: str = "BLOCKED"
    access_date: str = ""
    license_gate_passed: bool = False
    license_status: str = ""
    n_patients: int = 0
    n_patients_parsed_ok: int = 0
    n_patients_with_parse_failures: int = 0
    n_events_raw: int = 0
    n_events_standardized: int = 0
    n_events_rejected: int = 0
    raw_event_type_counts: Dict[str, int] = field(default_factory=dict)
    standardized_type_counts: Dict[str, int] = field(default_factory=dict)
    mapping_version: str = ""
    rejection_reason_counts: Dict[str, int] = field(default_factory=dict)
    awake_raw_total: int = 0
    awake_canonical_total: int = 0
    awake_exact_duplicates_removed: int = 0
    awake_overlaps_or_adjacent_merged: int = 0
    awake_cross_midnight_intervals: int = 0
    patients_cross_midnight: int = 0
    events_overlapping_awake: int = 0
    # signals
    signal_modality_counts: Dict[str, int] = field(default_factory=dict)
    signal_verified_counts: Dict[str, int] = field(default_factory=dict)
    # audio
    audio_total: int = 0
    audio_status_counts: Dict[str, int] = field(default_factory=dict)
    n_audio_excluded_too_short: int = 0
    short_audio_files: List[str] = field(default_factory=list)
    # immutability / isolation
    raw_modified: bool = False
    raw_scanned: bool = False
    warnings_total: int = 0
    anomalies: List[str] = field(default_factory=list)
    product_paths: Dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    stage2_production_run_id: str = ""


__all__ = [
    # status
    "ALIGN_VERIFIED",
    "ALIGN_UNRESOLVED",
    "ALIGN_EXCLUDED_TOO_SHORT",
    "ALIGN_NOT_APPLICABLE",
    "ALL_ALIGNMENT_STATES",
    # warnings
    "WARN_NON_MONOTONIC_CSV",
    "WARN_ABS_REL_DURATION_MISMATCH",
    "WARN_NO_ABSOLUTE_COLUMN",
    "WARN_NO_RELATIVE_COLUMN",
    "WARN_EVENT_BEFORE_RECORD_START",
    "WARN_SIGNAL_STARTS_BEFORE_RECORD_START",
    "WARN_AWAKE_BEFORE_RECORD_START",
    "WARN_BELOW_MIN_DURATION",
    "WARN_UNKNOWN",
    "SEV_ERROR",
    "SEV_WARNING",
    # records
    "RejectionRecord",
    "ParsedEvent",
    "AwakeRaw",
    "AwakeCanonical",
    "SignalTimeRange",
    "RecordTimeAnchor",
    "AudioAlignmentRecord",
    "SyncWarning",
    "Stage3Summary",
]
