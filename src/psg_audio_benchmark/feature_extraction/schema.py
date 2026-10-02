"""Dataclasses, vocabularies and versions for Stage-5 physiology feature extraction.

Stage 5 reads raw physiological CSV samples (heart_rate, spo2, airflow) inside
each Stage-4 positive/negative window and emits small, interpretable,
input-side feature tables plus a per-window quality/availability audit.

Hard rules encoded here (mirror the Stage 4-5 discipline):

* Features are derived ONLY from raw CSV signal samples and window time bounds.
* Label / annotation side columns are on a hard deny-list and never become
  features (see :data:`FORBIDDEN_FEATURE_INPUTS`).
* ``audio_features_present`` is hard-coded False; no ``*.wav`` is ever read.
* Low-coverage modalities are NULL + a quality code, never zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Modalities
# ---------------------------------------------------------------------------

MOD_HEART_RATE = "heart_rate"
MOD_SPO2 = "spo2"
MOD_AIRFLOW = "airflow"

CORE_MODALITIES: Tuple[str, ...] = (MOD_HEART_RATE, MOD_SPO2)
OPTIONAL_MODALITIES: Tuple[str, ...] = (MOD_AIRFLOW,)
ALL_FEATURE_MODALITIES: Tuple[str, ...] = (MOD_HEART_RATE, MOD_SPO2, MOD_AIRFLOW)

MODALITY_VALUE_UNIT: Dict[str, str] = {
    MOD_HEART_RATE: "bpm",
    MOD_SPO2: "%",
    MOD_AIRFLOW: "arbitrary_units",
}

# Feature / version stamps (written into every row + resolved config).
FEATURE_SET_VERSION = "physiology_baseline_v1"
MODALITY_FEATURE_VERSION: Dict[str, str] = {
    MOD_HEART_RATE: "hr_v1",
    MOD_SPO2: "spo2_v1",
    MOD_AIRFLOW: "airflow_v1",
}

# ---------------------------------------------------------------------------
# Quality vocabulary (mutually exclusive primary verdict per modality/window)
# ---------------------------------------------------------------------------

Q_AVAILABLE = "available"
Q_LOW_COVERAGE = "low_coverage"
Q_ALL_MISSING = "all_missing"
Q_INSUFFICIENT_SAMPLES = "insufficient_samples"
Q_MODALITY_NOT_AVAILABLE_FOR_PATIENT = "modality_not_available_for_patient"
Q_TIME_AXIS_UNAVAILABLE = "time_axis_unavailable"

ALL_QUALITY_STATUSES = (
    Q_AVAILABLE,
    Q_LOW_COVERAGE,
    Q_ALL_MISSING,
    Q_INSUFFICIENT_SAMPLES,
    Q_MODALITY_NOT_AVAILABLE_FOR_PATIENT,
    Q_TIME_AXIS_UNAVAILABLE,
)

# Non-blocking warnings (joined into quality_warnings by ';'); never delete/clip.
WARN_NON_FINITE = "non_finite_present"
WARN_CONSTANT_SIGNAL = "constant_signal"
WARN_RANGE = "physiological_range_warning"
WARN_ABNORMAL_TIME = "abnormal_time"

# Audio block (hard, never relaxable).
AUDIO_FEATURES_PRESENT: bool = False
AUDIO_BLOCK_REASON = "no_trustworthy_audio_time_anchor_unresolved"

# Label/annotation columns that MUST NEVER enter a feature computation.
FORBIDDEN_FEATURE_INPUTS: Tuple[str, ...] = (
    "binary_event_label",
    "label_status",
    "exclusion_reason",
    "n_linked_retained_events",
    "event_type_standardized",
    "awake_overlap_fraction",
    "sleep_structure_observational_status",
    "annotation",
    "any_audio_field",
)

# Stage-4 label statuses that ARE feature candidates (excluded windows are not).
FEATURE_CANDIDATE_STATUSES: Tuple[str, ...] = ("positive", "negative")


# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedPhysiologyFeatureConfig:
    """Fully-resolved, immutable Stage-5 feature configuration for one run."""

    stage4_input_run_id: str
    stage3_input_run_id: str
    feature_only_for_label_statuses: Tuple[str, ...]
    boundary: str
    coordinate: str
    cross_midnight_clock: str
    time_mapping: str
    forbidden_feature_inputs: Tuple[str, ...]
    min_coverage: Dict[str, float]
    min_samples_for_statistics: int
    min_samples_for_slope: int
    min_samples_for_zcr: int
    coverage_definition: str
    missing_definition: str
    expected_samples_from: str
    nominal_sampling_rate_hz: Dict[str, float]
    std_ddof: int
    slope_definition: str
    slope_unit: str
    airflow_rms_definition: str
    airflow_zcr_definition: str
    airflow_zcr_rate_unit: str
    spo2_desaturation_enabled: bool
    spo2_desaturation_threshold_pct: float
    spo2_desaturation_role: str
    physiological_range_warnings_enabled: bool
    physiological_ranges: Dict[str, Tuple[float, float]]
    audio_features_present: bool
    audio_block_reason: str
    read_wav_forbidden: bool
    no_model_matrix: bool
    no_imputation: bool
    no_normalization: bool
    no_split: bool
    feature_set_version: str
    modality_feature_version: Dict[str, str]
    non_clinical_note: str

    def min_coverage_for(self, modality: str) -> float:
        return float(self.min_coverage.get(modality, 0.80))

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                dd = {}
                for kk, vv in v.items():
                    dd[str(kk)] = list(vv) if isinstance(vv, tuple) else vv
                d[k] = dd
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Per-modality extent (one verified modality of one patient, from Stage 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModalityExtent:
    """Stage-3-verified extent of one modality for one patient."""

    patient_id: str
    modality: str
    source_relpath: str
    first_absolute_relative_to_record_start: Optional[float]
    last_absolute_relative_to_record_start: Optional[float]
    first_relative_seconds: Optional[float]
    last_relative_seconds: Optional[float]
    sample_count: Optional[int]
    relative_duration_seconds: Optional[float]
    verification_status: str
    data_quality_status: str

    @property
    def offset(self) -> Optional[float]:
        """Clock offset: record-start-relative = relative_position + offset.

        Equals ``first_absolute_relative_to_record_start`` when the relative
        column starts at 0 (the verified case). Inherits the Stage-3 clock.
        """
        f = self.first_absolute_relative_to_record_start
        r = self.first_relative_seconds
        if f is None or r is None:
            return None
        return float(f) - float(r)

    @property
    def sampling_rate_hz(self) -> Optional[float]:
        """Inferred per-patient rate (sample_count / relative_duration_seconds)."""
        if (
            self.sample_count is None
            or self.relative_duration_seconds is None
            or self.relative_duration_seconds <= 0
        ):
            return None
        return float(self.sample_count) / float(self.relative_duration_seconds)


# ---------------------------------------------------------------------------
# Window identity (the ONLY window fields allowed near feature computation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowIdentity:
    """The stable, label-free identity of one Stage-4 window.

    These are the only window fields the feature functions ever see; passing a
    label/annotation field here is a programming error.
    """

    window_id: str
    patient_id: str
    window_index: int
    start_relative_to_record_start: float
    end_relative_to_record_start: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# Feature row (one modality of one window)
# ---------------------------------------------------------------------------

@dataclass
class FeatureRow:
    # identity / time / version
    run_id: str
    window_id: str
    patient_id: str
    window_index: int
    modality: str
    window_start_relative_to_record_start: float
    window_end_relative_to_record_start: float
    duration_seconds: float
    modality_feature_version: str
    audio_features_present: bool
    # coverage / quality
    n_expected: int
    n_observed: int
    n_finite: int
    coverage_fraction: Optional[float]
    missing_fraction: Optional[float]
    quality_status: str
    quality_warnings: str
    # statistics
    mean: Optional[float]
    median: Optional[float]
    std_ddof1: Optional[float]
    min: Optional[float]
    max: Optional[float]
    range: Optional[float]
    iqr: Optional[float]
    slope_per_second: Optional[float]
    value_unit: str
    # airflow-only extras (None for HR/SpO2)
    rms: Optional[float] = None
    zero_crossing_count: Optional[int] = None
    zero_crossing_rate: Optional[float] = None
    # SpO2 optional descriptive desaturation (None unless enabled)
    n_below_desaturation_threshold: Optional[int] = None
    desaturation_threshold_pct: Optional[float] = None


@dataclass
class AvailabilityRow:
    run_id: str
    window_id: str
    patient_id: str
    window_index: int
    window_start_relative_to_record_start: float
    window_end_relative_to_record_start: float
    duration_seconds: float
    hr_available: bool
    hr_coverage_fraction: Optional[float]
    hr_quality_status: str
    spo2_available: bool
    spo2_coverage_fraction: Optional[float]
    spo2_quality_status: str
    airflow_modality_present_for_patient: bool
    airflow_available: bool
    airflow_coverage_fraction: Optional[float]
    airflow_quality_status: str
    core_hr_spo2_available: bool
    audio_features_present: bool


@dataclass
class FeatureExclusion:
    run_id: str
    window_id: str
    patient_id: str
    window_index: int
    scope: str
    modality: str
    reason: str
    detail: str


# Exclusion scope codes.
SCOPE_STAGE4_WINDOW_NOT_CANDIDATE = "stage4_window_not_candidate"
SCOPE_CORE_HR_SPO2_FEATURE_QUALITY = "core_hr_spo2_feature_quality"
SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE = "airflow_optional_not_available"


# ---------------------------------------------------------------------------
# Aggregate summary (drives the reports + figures)
# ---------------------------------------------------------------------------

@dataclass
class FeatureSummary:
    run_id: str
    overall_status: str = "BLOCKED"
    access_date: str = ""
    license_gate_passed: bool = False
    license_status: str = ""
    config_hash: str = ""
    input_stage4_run_id: str = ""
    input_stage4_verified: bool = False
    input_stage4_config_hash: str = ""
    input_stage3_run_id: str = ""
    # window accounting
    n_patients: int = 0
    total_stage4_windows: int = 0
    total_candidate_windows: int = 0           # positive + negative
    total_excluded_stage4_windows: int = 0
    positive_windows: int = 0
    negative_windows: int = 0
    # per-modality feature extraction
    hr_rows_extracted: int = 0
    spo2_rows_extracted: int = 0
    airflow_rows_extracted: int = 0
    hr_low_coverage: int = 0
    spo2_low_coverage: int = 0
    airflow_low_coverage: int = 0
    airflow_patients_with_modality: int = 0
    airflow_patients_without_modality: int = 0
    # availability
    core_hr_spo2_available_windows: int = 0
    core_hr_spo2_unavailable_windows: int = 0
    # quality-code counts per modality
    quality_status_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # tests / guards
    audio_features_present: bool = False
    audio_window_eligible_any: bool = False
    raw_modified: bool = False
    raw_scanned: bool = False
    anomalies: List[str] = field(default_factory=list)
    product_paths: Dict[str, str] = field(default_factory=dict)


__all__ = [
    # modalities / versions
    "MOD_HEART_RATE", "MOD_SPO2", "MOD_AIRFLOW",
    "CORE_MODALITIES", "OPTIONAL_MODALITIES", "ALL_FEATURE_MODALITIES",
    "MODALITY_VALUE_UNIT", "FEATURE_SET_VERSION", "MODALITY_FEATURE_VERSION",
    # quality
    "Q_AVAILABLE", "Q_LOW_COVERAGE", "Q_ALL_MISSING", "Q_INSUFFICIENT_SAMPLES",
    "Q_MODALITY_NOT_AVAILABLE_FOR_PATIENT", "Q_TIME_AXIS_UNAVAILABLE",
    "ALL_QUALITY_STATUSES",
    "WARN_NON_FINITE", "WARN_CONSTANT_SIGNAL", "WARN_RANGE", "WARN_ABNORMAL_TIME",
    "AUDIO_FEATURES_PRESENT", "AUDIO_BLOCK_REASON",
    "FORBIDDEN_FEATURE_INPUTS", "FEATURE_CANDIDATE_STATUSES",
    # exclusion scopes
    "SCOPE_STAGE4_WINDOW_NOT_CANDIDATE", "SCOPE_CORE_HR_SPO2_FEATURE_QUALITY",
    "SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE",
    # dataclasses
    "ResolvedPhysiologyFeatureConfig", "ModalityExtent", "WindowIdentity",
    "FeatureRow", "AvailabilityRow", "FeatureExclusion", "FeatureSummary",
]
