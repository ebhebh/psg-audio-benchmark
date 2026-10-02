"""Constants, vocabularies and versions for Stage-8 airflow-incremental modeling.

This is the single in-code authority for the Stage-8 *paired, within-cohort*
airflow-gain comparison. It re-uses the Stage-7 estimator vocabulary and fold
discipline, and ONLY relaxes one thing relative to the core model: airflow
window statistics are an ALLOWED feature set for the ``airflow_enhanced`` matrix.

Everything that must NEVER be silently relaxed by editing YAML lives here:

* the two feature sets (``core_restricted`` = the 16 HR+SpO2 statistics;
  ``airflow_enhanced`` = those 16 + the 11 airflow statistics);
* the airflow allow-list (the 11 allowed airflow statistics) and the forbidden
  category tokens (everything Stage-7 forbids EXCEPT ``airflow``);
* the two pre-specified model families and their fixed roles (LR primary, HGB
  exploratory);
* the inherited v2 nested-CV structure (5 x 4, patient unit);
* the audio block (audio cohort hard-coded 0; ``*.wav`` never read);
* the non-clinical, internal-only, not-generalizable-to-50 framing.

The module mirrors ``config/modeling_airflow_incremental.yaml``; the loader
(:mod:`airflow_increment.config`) cross-checks the two so a careless edit
cannot relax a rule.
"""

from __future__ import annotations

from typing import Tuple

# Re-use the Stage-7 estimator + fold vocabulary (identical model families,
# roles, selection rule, threshold rule, bootstrap contract). Stage 8 only adds
# the airflow feature set and the paired within-cohort comparison.
from ..schema import (  # noqa: F401  (re-exported below)
    MODEL_HIST_GRADIENT_BOOSTING,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_ROLE as _STAGE7_MODEL_ROLE,
    SELECTION_PRIMARY_SCORE,
    SELECTION_TIE_BREAK,
    THRESHOLD_DUMMY_VALUE,
    THRESHOLD_FIT_FROM,
    THRESHOLD_RULE_DATA_DRIVEN,
    THRESHOLD_RULE_DUMMY,
)

# ---------------------------------------------------------------------------
# Versions / task framing / cohort / audio block
# ---------------------------------------------------------------------------

MODELING_STAGE: str = "stage8"
MODELING_VERSION: str = "airflow_incremental_v1"
TASK_NAME: str = (
    "window-level dataset-scored respiratory-event airflow-incremental "
    "paired within-cohort comparison"
)
COHORT: str = "airflow_eligible_internal"
SPLIT_BALANCE_VERSION: str = "split_balance_v2"

APPROVAL_REQUIRED: str = "approved_for_modeling"

AUDIO_BLOCK_REASON: str = "no_trustworthy_audio_time_anchor_unresolved"
AUDIO_COHORT_COUNT: int = 0
AUDIO_FEATURES_PRESENT: bool = False
READ_WAV_FORBIDDEN: bool = True

NON_CLINICAL_NOTE: str = (
    "Single public dataset, patient-level INTERNAL cross-validation only on the "
    "34-patient airflow-eligible sub-cohort. This is a retrospective, "
    "PSG-anchored CSV benchmark of the airflow feature INCREMENT over HR+SpO2 "
    "WITHIN the same patients/windows -- NOT a clinical/diagnostic/deployment "
    "claim, NOT a gain over the Stage-7 50-patient main cohort (different "
    "population), and NOT generalizable to audio. HGB is exploratory only and "
    "is never declared the winner by outer-test score. Audio remains BLOCKED."
)

# ---------------------------------------------------------------------------
# Nested CV structure (INHERITED from the frozen v2 split; never rebuilt).
# The 34 airflow-eligible patients keep their v2 outer/inner assignments.
# ---------------------------------------------------------------------------

OUTER_N_FOLDS: int = 5
INNER_N_FOLDS: int = 4
CV_UNIT: str = "patient"
INHERITS_SPLIT: bool = True
REBUILDS_SPLIT: bool = False
WINDOW_RANDOM_SPLIT: bool = False

# ---------------------------------------------------------------------------
# Feature sets.
#
# core_restricted   = the exact Stage-7 16-column HR+SpO2 allow-list.
# airflow_enhanced  = those 16 + the 11 allowed airflow statistics (27 columns).
# Both matrices are built on the IDENTICAL airflow-eligible window set, so the
# core/enhanced OOF predictions are one-to-one pairable per window.
# ---------------------------------------------------------------------------

#: The 11 allowed airflow window statistics (prefixed ``airflow_``). IDs, time,
#: quality/coverage, availability, label, annotation, awake, sleep stage and
#: audio are NEVER airflow features (enforced by the exact allow-list + the
#: deny-list below).
AIRFLOW_PREFIX: str = "airflow_"
AIRFLOW_STATISTICS: Tuple[str, ...] = (
    "mean", "median", "std_ddof1", "min", "max", "range", "iqr",
    "slope_per_second", "rms", "zero_crossing_count", "zero_crossing_rate",
)
AIRFLOW_FEATURE_WHITELIST: Tuple[str, ...] = tuple(
    f"{AIRFLOW_PREFIX}{s}" for s in AIRFLOW_STATISTICS
)

#: core_restricted = Stage-7's exact HR+SpO2 allow-list (16 columns, fixed order).
from ..schema import FEATURE_WHITELIST as _CORE_WHITELIST  # noqa: E402

CORE_FEATURE_WHITELIST: Tuple[str, ...] = tuple(_CORE_WHITELIST)

#: airflow_enhanced = core 16 + airflow 11 (fixed order).
ENHANCED_FEATURE_WHITELIST: Tuple[str, ...] = (
    CORE_FEATURE_WHITELIST + AIRFLOW_FEATURE_WHITELIST
)

#: The two compared feature sets -> their exact allow-lists.
FEATURE_SET_CORE: str = "core_restricted"
FEATURE_SET_ENHANCED: str = "airflow_enhanced"
FEATURE_SETS: Tuple[str, ...] = (FEATURE_SET_CORE, FEATURE_SET_ENHANCED)
FEATURE_SET_WHITELIST: dict = {
    FEATURE_SET_CORE: CORE_FEATURE_WHITELIST,
    FEATURE_SET_ENHANCED: ENHANCED_FEATURE_WHITELIST,
}
FEATURE_SET_ROLE: dict = {
    FEATURE_SET_CORE: "comparator_baseline",
    FEATURE_SET_ENHANCED: "increment_under_test",
}

FEATURE_JOIN_KEY: str = "window_id"
LABEL_COLUMN: str = "binary_event_label"

# ---------------------------------------------------------------------------
# Forbidden feature category tokens for Stage 8. Everything Stage-7 forbids
# EXCEPT ``airflow`` (airflow statistics are admissible in the enhanced set).
# The allow-list is the primary guard; this deny-list is defence in depth.
# ---------------------------------------------------------------------------

from ..schema import FORBIDDEN_CATEGORY_TOKENS as _STAGE7_FORBIDDEN  # noqa: E402

FORBIDDEN_CATEGORY_TOKENS: Tuple[str, ...] = tuple(
    t for t in _STAGE7_FORBIDDEN if t != "airflow"
)

#: Explicit forbidden column names (airflow availability / quality / coverage
#: flags + IDs / time / label) that must never be estimator features even though
#: they share no substring with the allowed airflow statistic names.
FORBIDDEN_COLUMNS: Tuple[str, ...] = (
    FEATURE_JOIN_KEY, "patient_id", "window_index", "run_id", "modality",
    "modality_feature_version", "audio_features_present",
    "n_expected", "n_observed", "n_finite",
    "coverage_fraction", "missing_fraction",
    "quality_status", "quality_warnings", "value_unit",
    "has_airflow_coverage", "airflow_available",
    "airflow_coverage_fraction", "airflow_quality_status",
    "airflow_modality_present_for_patient",
    "core_heart_rate_coverage", "core_spo2_coverage", "core_coverage_complete",
    "core_hr_spo2_available", "hr_available", "spo2_available",
    "hr_coverage_fraction", "spo2_coverage_fraction",
    "hr_quality_status", "spo2_quality_status",
)

#: Audit/identity keys retained on the frame for traceability + paired patient
#: bootstrap ONLY; stripped before any estimator sees the matrix.
AUDIT_KEYS: Tuple[str, ...] = (
    "window_id", "patient_id", "outer_fold", "feature_set",
)

# ---------------------------------------------------------------------------
# Pre-specified model families and their pre-registered roles (Stage 8 subset).
# ---------------------------------------------------------------------------

MODEL_FAMILIES: Tuple[str, ...] = (
    MODEL_LOGISTIC_REGRESSION, MODEL_HIST_GRADIENT_BOOSTING,
)
PRIMARY_MODEL: str = MODEL_LOGISTIC_REGRESSION
EXPLORATORY_MODELS: Tuple[str, ...] = (MODEL_HIST_GRADIENT_BOOSTING,)

#: A model may NOT be declared the winner by outer-test score. The main
#: comparison is FIXED: LR airflow_enhanced - LR core_restricted.
MODEL_ROLE: dict = {
    MODEL_LOGISTIC_REGRESSION: "primary",
    MODEL_HIST_GRADIENT_BOOSTING: "exploratory",
}

#: The pre-registered primary paired comparison.
PRIMARY_COMPARISON: Tuple[str, str, str] = (
    MODEL_LOGISTIC_REGRESSION, FEATURE_SET_CORE, FEATURE_SET_ENHANCED,
)

# ---------------------------------------------------------------------------
# Selection + threshold rules (identical to Stage 7).
# ---------------------------------------------------------------------------

SELECTION_PRIMARY_SCORE  # re-exported
SELECTION_TIE_BREAK      # re-exported
THRESHOLD_RULE_DATA_DRIVEN  # re-exported
THRESHOLD_RULE_DUMMY         # re-exported
THRESHOLD_DUMMY_VALUE        # re-exported
THRESHOLD_FIT_FROM           # re-exported

# ---------------------------------------------------------------------------
# Paired bootstrap contract.
# ---------------------------------------------------------------------------

BOOTSTRAP_UNIT_PATIENT: str = "patient"
BOOTSTRAP_DEFAULT_N: int = 1000
BOOTSTRAP_DEFAULT_SEED: int = 20250714
BOOTSTRAP_CI_LEVEL: float = 0.95
BOOTSTRAP_FORBID_WINDOW_IID: bool = True
BOOTSTRAP_FORBID_CROSS_COHORT_PAIRING: bool = True

#: Metrics whose paired increment (enhanced - core) is bootstrapped.
BOOTSTRAP_THRESHOLD_FREE_METRICS: Tuple[str, ...] = ("auroc", "auprc", "brier")
BOOTSTRAP_THRESHOLD_BASED_METRICS: Tuple[str, ...] = (
    "sensitivity", "specificity", "f1", "balanced_accuracy",
)

__all__ = [name for name in dir() if name.isupper() or name.startswith("MODEL_")
           or name.startswith("FEATURE_") or name.startswith("BOOTSTRAP_")
           or name.startswith("NON_") or name.startswith("PRIMARY_")
           or name in ("AIRFLOW_PREFIX", "AIRFLOW_STATISTICS",
                       "AIRFLOW_FEATURE_WHITELIST", "AUDIT_KEYS",
                       "FORBIDDEN_CATEGORY_TOKENS", "FORBIDDEN_COLUMNS",
                       "LABEL_COLUMN", "INHERITS_SPLIT", "REBUILDS_SPLIT",
                       "WINDOW_RANDOM_SPLIT", "APPROVAL_REQUIRED",
                       "AUDIO_BLOCK_REASON", "AUDIO_FEATURES_PRESENT",
                       "READ_WAV_FORBIDDEN", "TASK_NAME", "COHORT",
                       "MODELING_STAGE", "MODELING_VERSION",
                       "SPLIT_BALANCE_VERSION")]
