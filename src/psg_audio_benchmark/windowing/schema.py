"""Dataclasses and status vocabularies for Stage-4 CSV-side windowing.

The label/exclusion/destiny states are kept as named constants (never ad-hoc
booleans) so the three label states (positive / negative / excluded) stay
mutually exclusive and an excluded window can never be mistaken for a negative.

All records are CSV/annotation-side only: no audio slice, no audio feature, no
raw sample array is ever stored -- only indices, time coordinates, coverage
booleans, relative provenance paths and research labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Label / exclusion vocabulary (mutually exclusive)
# ---------------------------------------------------------------------------

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_EXCLUDED = "excluded"

#: All distinct label states. An excluded window NEVER carries a binary label.
ALL_LABEL_STATUSES = (LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDED)

#: Binary research label values (None means "not labeled / excluded").
BINARY_POSITIVE = 1
BINARY_NEGATIVE = 0

# Exclusion reasons (one per excluded window; never silently dropped).
EXC_INCOMPLETE_CORE = "excluded_incomplete_core_signal_coverage"
EXC_AWAKE_OVERLAP = "excluded_awake_overlap"
EXC_NO_CORE_DOMAIN = "excluded_no_core_signal_domain"

# ---------------------------------------------------------------------------
# Event destiny vocabulary (every event accounted; none hidden)
# ---------------------------------------------------------------------------

EVT_LINKED = "linked"                       # event start in a label-eligible window
EVT_AWAKE = "awake_overlap_event"           # event itself overlaps awake -> set aside
EVT_OUT_OF_DOMAIN = "event_start_outside_core_domain"  # no candidate window contains start
EVT_NO_VALID_WINDOW = "event_in_excluded_or_absent_window"  # start in excluded window

ALL_EVENT_DESTINIES = (EVT_LINKED, EVT_AWAKE, EVT_OUT_OF_DOMAIN, EVT_NO_VALID_WINDOW)

# Link detail codes that explain WHY an event took its destiny.
LINK_START_IN_LABEL_WINDOW = "event_start_in_label_window"
LINK_EVENT_OVERLAPS_AWAKE = "event_overlaps_awake_interval"
LINK_START_OUTSIDE_GRID = "event_start_outside_candidate_grid"
LINK_IN_WINDOW_EXCLUDED_COVERAGE = "event_in_window_excluded_incomplete_coverage"
LINK_IN_WINDOW_EXCLUDED_AWAKE = "event_in_window_excluded_awake_overlap"

# ---------------------------------------------------------------------------
# Audio block (hard, never relaxable)
# ---------------------------------------------------------------------------

AUDIO_WINDOW_ELIGIBLE: bool = False
AUDIO_BLOCK_REASON = "no_trustworthy_audio_time_anchor_unresolved"


# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedWindowingConfig:
    """Fully-resolved, immutable windowing configuration for one run."""

    window_length_s: float
    hop_step_s: float
    boundary: str
    coordinate: str
    candidate_time_domain: str
    core_signal_modalities: tuple
    optional_signal_modalities: tuple
    sleep_structure_role: str
    require_complete_core_coverage: bool
    max_awake_overlap_fraction: float
    event_link_rule: str
    require_event_not_awake_overlap: bool
    binary_event_label_field: str
    non_clinical_label_note: str
    audio_window_eligible: bool
    audio_block_reason: str
    allow_substitute_anchors: bool
    perform_random_window_split: bool
    emit_train_val_test_split: bool

    def as_dict(self) -> Dict[str, Any]:
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            d[k] = list(v) if isinstance(v, tuple) else v
        return d


# ---------------------------------------------------------------------------
# Per-patient signal coverage (computed from Stage-3 signal_time_ranges)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatientSignalCoverage:
    """Resolved per-patient core/optional signal extents (relative to record_start)."""

    patient_id: str
    record_start_cumulative_seconds: float
    hr_verified: bool
    hr_first_rel: Optional[float]
    hr_last_rel: Optional[float]
    hr_source_relpath: str
    spo2_verified: bool
    spo2_first_rel: Optional[float]
    spo2_last_rel: Optional[float]
    spo2_source_relpath: str
    airflow_verified: bool
    airflow_first_rel: Optional[float]
    airflow_last_rel: Optional[float]
    airflow_source_relpath: str
    sleep_structure_status: str  # observational only; never used to label

    @property
    def has_core_domain(self) -> bool:
        """True iff both core modalities are verified with a positive extent."""
        if not (self.hr_verified and self.spo2_verified):
            return False
        return (
            self.hr_first_rel is not None
            and self.hr_last_rel is not None
            and self.spo2_first_rel is not None
            and self.spo2_last_rel is not None
            and self.hr_last_rel > self.hr_first_rel
            and self.spo2_last_rel > self.spo2_first_rel
        )

    @property
    def core_domain(self) -> Optional[tuple]:
        """Intersection ``(lo, hi)`` of the verified core extents, or None."""
        if not self.has_core_domain:
            return None
        lo = max(self.hr_first_rel, self.spo2_first_rel)  # type: ignore[arg-type]
        hi = min(self.hr_last_rel, self.spo2_last_rel)  # type: ignore[arg-type]
        if hi <= lo:
            return None
        return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Window record (one row of csv_window_index.parquet)
# ---------------------------------------------------------------------------

@dataclass
class WindowRecord:
    run_id: str
    patient_id: str
    window_index: int
    start_absolute_cumulative_seconds: float
    end_absolute_cumulative_seconds: float
    start_relative_to_record_start: float
    end_relative_to_record_start: float
    duration_seconds: float
    core_heart_rate_coverage: bool
    core_spo2_coverage: bool
    has_airflow_coverage: bool
    core_coverage_complete: bool
    awake_overlap_fraction: float
    audio_window_eligible: bool            # always False
    audio_block_reason: str
    label_status: str                      # positive | negative | excluded
    binary_event_label: Optional[int]      # 1 | 0 | None
    exclusion_reason: str                  # empty for non-excluded
    n_linked_retained_events: int
    sleep_structure_observational_status: str


@dataclass
class WindowEventLink:
    """One row of window_event_links.parquet -- every event accounted, none hidden."""

    run_id: str
    patient_id: str
    window_index: int                      # -1 when no candidate window contains the start
    source_event_index: int
    event_type_standardized: str
    event_start_relative_to_record_start: float
    event_duration_seconds: float
    event_end_relative_to_record_start: float
    event_overlaps_awake_interval: bool
    event_destiny: str                     # linked | awake | out_of_domain | no_valid_window
    link_reason: str


@dataclass
class WindowExclusion:
    """One row of window_exclusions.csv (only excluded windows)."""

    run_id: str
    patient_id: str
    window_index: int
    start_relative_to_record_start: float
    end_relative_to_record_start: float
    exclusion_reason: str
    detail: str


@dataclass
class PatientWindowSummary:
    """One row of patient_window_summary.csv."""

    run_id: str
    patient_id: str
    n_candidate_windows: int
    n_core_complete_windows: int
    n_awake_excluded_windows: int
    n_coverage_excluded_windows: int
    n_excluded_windows: int
    n_positive_windows: int
    n_negative_windows: int
    n_usable_windows: int                  # positive + negative
    n_linked_events: int
    has_airflow_coverage: bool
    sleep_structure_observational_status: str
    zero_usable_windows: bool
    zero_reason: str


@dataclass
class WindowingSummary:
    """Aggregate outcome of a Stage-4 run (used by the report renderers)."""

    run_id: str
    overall_status: str = "BLOCKED"
    access_date: str = ""
    license_gate_passed: bool = False
    license_status: str = ""
    config_hash: str = ""
    input_run_id: str = ""
    input_run_config_hash: str = ""
    input_run_verified: bool = False
    # counts
    n_patients: int = 0
    n_patients_with_core_domain: int = 0
    n_patients_zero_usable: int = 0
    total_candidate_windows: int = 0
    total_core_complete_windows: int = 0
    total_awake_excluded_windows: int = 0
    total_coverage_excluded_windows: int = 0
    total_excluded_windows: int = 0
    total_positive_windows: int = 0
    total_negative_windows: int = 0
    total_usable_windows: int = 0
    # per-patient window-count distribution
    per_patient_window_min: int = 0
    per_patient_window_median: float = 0.0
    per_patient_window_max: int = 0
    zero_usable_patients: List[str] = field(default_factory=list)
    # event destiny
    total_events: int = 0
    event_destiny_counts: Dict[str, int] = field(default_factory=dict)
    total_linked_events: int = 0
    # modality coverage (window counts)
    hr_window_coverage: int = 0
    spo2_window_coverage: int = 0
    airflow_window_coverage: int = 0
    sleep_structure_status_counts: Dict[str, int] = field(default_factory=dict)
    # audio
    audio_window_eligible_count: int = 0
    audio_total_files: int = 0
    audio_status_counts: Dict[str, int] = field(default_factory=dict)
    # immutability / isolation
    raw_modified: bool = False
    raw_scanned: bool = False
    anomalies: List[str] = field(default_factory=list)
    product_paths: Dict[str, str] = field(default_factory=dict)
    stage3_production_run_id: str = ""


__all__ = [
    # label / exclusion
    "LABEL_POSITIVE",
    "LABEL_NEGATIVE",
    "LABEL_EXCLUDED",
    "ALL_LABEL_STATUSES",
    "BINARY_POSITIVE",
    "BINARY_NEGATIVE",
    "EXC_INCOMPLETE_CORE",
    "EXC_AWAKE_OVERLAP",
    "EXC_NO_CORE_DOMAIN",
    # event destiny
    "EVT_LINKED",
    "EVT_AWAKE",
    "EVT_OUT_OF_DOMAIN",
    "EVT_NO_VALID_WINDOW",
    "ALL_EVENT_DESTINIES",
    "LINK_START_IN_LABEL_WINDOW",
    "LINK_EVENT_OVERLAPS_AWAKE",
    "LINK_START_OUTSIDE_GRID",
    "LINK_IN_WINDOW_EXCLUDED_COVERAGE",
    "LINK_IN_WINDOW_EXCLUDED_AWAKE",
    # audio
    "AUDIO_WINDOW_ELIGIBLE",
    "AUDIO_BLOCK_REASON",
    # records
    "ResolvedWindowingConfig",
    "PatientSignalCoverage",
    "WindowRecord",
    "WindowEventLink",
    "WindowExclusion",
    "PatientWindowSummary",
    "WindowingSummary",
]
