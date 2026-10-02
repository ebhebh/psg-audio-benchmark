"""Dataclasses, vocabularies and versions for Stage-6 patient-level splits.

Stage 6 builds the audited CSV-physiology research cohort and the patient-level
nested (outer x inner) cross-validation split, plus a version signature and audit
reports. It reads only Stage-4 (labels) and Stage-5 (availability) parquet
outputs, joins on stable keys, and assigns patients to folds.

Hard rules encoded here (mirror the Stage 4-6 discipline):

* Splitting is **patient-level**; window-random splitting is forbidden.
* Balance uses **only** pre-model patient-level summaries; never model output.
* ``audio`` is BLOCKED: ``audio_cohort_count`` is hard-coded 0; no ``*.wav`` is
  ever read.
* Stage 6 computes **no** model matrix / imputation / normalization / feature
  selection / resampling / threshold / metrics. The column deny-list
  :data:`STAGE6_FORBIDDEN_COLUMNS` keeps every numeric Stage-5 feature value out
  of all Stage-6 outputs (membership / folds / exclusions / summary).
* ``binary_event_label`` is the stratification target ``y`` and is allowed
  **only** in the cohort membership tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Versions / audio block
# ---------------------------------------------------------------------------

PATIENT_SPLIT_VERSION = "patient_splits_v1"
AUDIO_FEATURES_PRESENT: bool = False
AUDIO_COHORT_COUNT: int = 0
AUDIO_BLOCK_REASON = "no_trustworthy_audio_time_anchor_unresolved"

# ---------------------------------------------------------------------------
# Label statuses & cohorts
# ---------------------------------------------------------------------------

LABEL_POSITIVE = "positive"
LABEL_NEGATIVE = "negative"
LABEL_EXCLUDED = "excluded"
CANDIDATE_STATUSES: Tuple[str, ...] = (LABEL_POSITIVE, LABEL_NEGATIVE)
ALL_LABEL_STATUSES: Tuple[str, ...] = (LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_EXCLUDED)

COHORT_CORE = "core_hr_spo2"
COHORT_AIRFLOW = "airflow_enhanced"
AIRFLOW_MEMBERSHIP_RULE = "core_hr_spo2_AND_airflow_available"

# ---------------------------------------------------------------------------
# Exclusion scope codes (recorded in cohort_exclusions.csv)
# ---------------------------------------------------------------------------

SCOPE_STAGE4_EXCLUDED = "stage4_window_excluded"
SCOPE_HR_QUALITY_INSUFFICIENT = "hr_feature_quality_insufficient"
SCOPE_SPO2_QUALITY_INSUFFICIENT = "spo2_feature_quality_insufficient"
SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE = "airflow_optional_not_available"

# ---------------------------------------------------------------------------
# No-model-matrix guard: the Stage-5 numeric feature columns that MUST NEVER
# appear in any Stage-6 output (membership / folds / exclusions / summary).
# These are derived statistic / coverage columns, never keys or availability
# flags. ``binary_event_label`` is allowed only in the membership tables.
# ---------------------------------------------------------------------------

STAGE6_FORBIDDEN_COLUMNS: Tuple[str, ...] = (
    "mean", "median", "std_ddof1", "std", "min", "max", "range", "iqr",
    "slope_per_second", "rms", "zero_crossing_count", "zero_crossing_rate",
    "n_below_desaturation_threshold", "desaturation_threshold_pct",
    "coverage_fraction", "missing_fraction",
    "n_expected", "n_observed", "n_finite",
    "hr_coverage_fraction", "spo2_coverage_fraction", "airflow_coverage_fraction",
    "value_unit",
)
#: The stratification label; allowed ONLY in the cohort membership tables.
ALLOWED_LABEL_COLUMN = "binary_event_label"


# ---------------------------------------------------------------------------
# Resolved configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedSplitsConfig:
    """Fully-resolved, immutable Stage-6 split configuration for one run."""

    stage4_input_run_id: str
    stage5_input_run_id: str
    stage3_input_run_id: str
    # outer / inner CV
    outer_n_folds: int
    inner_n_folds: int
    seed: int
    inner_seed: int
    stratifier: str
    group_column: str
    label_for_stratification: str
    sort_patients_by: str
    # airflow subcohort
    airflow_membership_rule: str
    airflow_inherits_outer_fold_from: str
    airflow_independent_split_forbidden: bool
    # balance
    warn_pos_rate_spread_threshold: float
    # hard leakage rules
    patient_level_split_required: bool
    allow_window_random_split: bool
    split_uses_only_pre_model_summaries: bool
    # audio block
    audio_cohort_count: int
    audio_features_present: bool
    audio_block_reason: str
    read_wav_forbidden: bool
    # nothing downstream
    no_model_matrix: bool
    no_imputation: bool
    no_normalization: bool
    no_feature_selection: bool
    no_class_resampling: bool
    no_threshold_optimization: bool
    no_performance_metrics: bool
    no_audio_access: bool
    # versions / notes
    patient_split_version: str
    non_clinical_note: str

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, tuple):
                d[k] = list(v)
            else:
                d[k] = v
        return d


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CohortMembershipRow:
    """One candidate window's membership + availability (no feature values)."""
    run_id: str
    window_id: str
    patient_id: str
    window_index: int
    binary_event_label: int            # 0/1; stratification target, allowed here
    label_status: str
    window_start_relative_to_record_start: float
    window_end_relative_to_record_start: float
    duration_seconds: float
    hr_available: bool
    spo2_available: bool
    airflow_available: bool
    core_hr_spo2_available: bool
    cohort_core: bool
    cohort_airflow: bool
    audio_features_present: bool


@dataclass
class OuterPatientFold:
    """One core patient -> its single outer test fold (+ label burden)."""
    run_id: str
    patient_id: str
    outer_fold: int
    n_windows: int
    n_positive: int
    n_negative: int
    positive_rate: float


@dataclass
class InnerPatientFold:
    """One outer-train patient -> its single inner validation fold (within an outer fold)."""
    run_id: str
    outer_fold: int
    patient_id: str
    inner_validation_fold: int
    n_windows: int
    n_positive: int
    n_negative: int
    positive_rate: float


@dataclass
class AirflowInheritanceRow:
    """One airflow patient -> inherited core outer fold (+ airflow-window burden)."""
    run_id: str
    patient_id: str
    outer_fold: int
    n_airflow_windows: int
    n_airflow_positive: int
    n_airflow_negative: int
    airflow_positive_rate: float


@dataclass
class CohortExclusion:
    """A window excluded from a cohort, with reason (never faked as negative)."""
    run_id: str
    window_id: str
    patient_id: str
    window_index: int
    scope: str
    reason: str
    detail: str


# ---------------------------------------------------------------------------
# Aggregate summary (drives the reports)
# ---------------------------------------------------------------------------

@dataclass
class SplitSummary:
    run_id: str
    overall_status: str = "BLOCKED"
    access_date: str = ""
    license_gate_passed: bool = False
    license_status: str = ""
    config_hash: str = ""
    # inputs
    input_stage4_run_id: str = ""
    input_stage4_verified: bool = False
    input_stage4_config_hash: str = ""
    input_stage5_run_id: str = ""
    input_stage5_verified: bool = False
    input_stage5_config_hash: str = ""
    input_stage3_run_id: str = ""
    # window accounting
    total_stage4_windows: int = 0
    total_candidate_windows: int = 0
    total_excluded_stage4_windows: int = 0
    # core cohort
    core_patients: int = 0
    core_windows: int = 0
    core_positive: int = 0
    core_negative: int = 0
    # airflow subcohort (strict: core AND airflow available)
    airflow_patients: int = 0
    airflow_windows: int = 0
    airflow_positive: int = 0
    airflow_negative: int = 0
    # airflow "available-alone" reference (matches the prompt's 28,409 baseline;
    # recorded so the strict-vs-available discrepancy is explicit/auditable).
    airflow_available_alone_windows: int = 0
    airflow_available_alone_patients: int = 0
    # per-patient burden (core)
    core_windows_per_patient_min: int = 0
    core_windows_per_patient_median: float = 0.0
    core_windows_per_patient_max: int = 0
    core_pos_rate_per_patient_min: float = 0.0
    core_pos_rate_per_patient_median: float = 0.0
    core_pos_rate_per_patient_max: float = 0.0
    # exclusions by scope
    exclusions_by_scope: Dict[str, int] = field(default_factory=dict)
    # fold-level stats (list of dicts; one per outer fold / per inner fold)
    outer_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    inner_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    airflow_fold_stats: List[Dict[str, Any]] = field(default_factory=list)
    balance_warnings: List[str] = field(default_factory=list)
    # hard flags
    audio_cohort_count: int = AUDIO_COHORT_COUNT
    audio_block_reason: str = AUDIO_BLOCK_REASON
    no_model_matrix: bool = True
    no_imputation: bool = True
    no_normalization: bool = True
    no_feature_selection: bool = True
    no_class_resampling: bool = True
    no_threshold_optimization: bool = True
    no_performance_metrics: bool = True
    # tests / guards
    raw_modified: bool = False
    raw_scanned: bool = False
    anomalies: List[str] = field(default_factory=list)
    product_paths: Dict[str, str] = field(default_factory=dict)


__all__ = [
    # versions / audio
    "PATIENT_SPLIT_VERSION", "AUDIO_FEATURES_PRESENT", "AUDIO_COHORT_COUNT",
    "AUDIO_BLOCK_REASON",
    # labels / cohorts
    "LABEL_POSITIVE", "LABEL_NEGATIVE", "LABEL_EXCLUDED",
    "CANDIDATE_STATUSES", "ALL_LABEL_STATUSES",
    "COHORT_CORE", "COHORT_AIRFLOW", "AIRFLOW_MEMBERSHIP_RULE",
    # exclusion scopes
    "SCOPE_STAGE4_EXCLUDED", "SCOPE_HR_QUALITY_INSUFFICIENT",
    "SCOPE_SPO2_QUALITY_INSUFFICIENT", "SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE",
    # no-model-matrix guard
    "STAGE6_FORBIDDEN_COLUMNS", "ALLOWED_LABEL_COLUMN",
    # dataclasses
    "ResolvedSplitsConfig", "CohortMembershipRow", "OuterPatientFold",
    "InnerPatientFold", "AirflowInheritanceRow", "CohortExclusion", "SplitSummary",
]
