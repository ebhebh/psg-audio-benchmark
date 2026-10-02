"""Stage-9 robustness/sensitivity constants, analysis roles and column contracts.

Constants here are code-authoritative; the resolved-config loader cross-checks
the YAML against them and the code value always wins.
"""

from __future__ import annotations

# --- re-exported modeling constants (single source of truth) -----------------
from ...modeling import schema as MS  # noqa: F401  (stage-7 modeling schema)

MODEL_DUMMY_PRIOR = MS.MODEL_DUMMY_PRIOR
MODEL_LOGISTIC_REGRESSION = MS.MODEL_LOGISTIC_REGRESSION
MODEL_HIST_GRADIENT_BOOSTING = MS.MODEL_HIST_GRADIENT_BOOSTING
MODEL_FAMILIES = MS.MODEL_FAMILIES
PRIMARY_MODEL = MS.PRIMARY_MODEL
MODEL_ROLE = MS.MODEL_ROLE

# --- identity / framing ------------------------------------------------------
ROBUSTNESS_STAGE = "stage9"
MODELING_VERSION = "robustness_sensitivity_v1"
TASK_NAME = (
    "robustness and sensitivity analysis of the stage7/8 core/airflow CSV baseline"
)
COHORT = "core_hr_spo2_and_airflow_subcohort"
SPLIT_BALANCE_VERSION = "split_balance_v2"
APPROVAL_REQUIRED = MS.APPROVAL_REQUIRED  # "approved_for_modeling"

# --- audio block (mirrors modeling; audio is BLOCKED) ------------------------
AUDIO_COHORT_COUNT = MS.AUDIO_COHORT_COUNT            # 0
READ_WAV_FORBIDDEN = MS.READ_WAV_FORBIDDEN            # True
AUDIO_FEATURES_PRESENT = MS.AUDIO_FEATURES_PRESENT    # False
AUDIO_BLOCK_REASON = MS.AUDIO_BLOCK_REASON

NON_CLINICAL_NOTE = (
    "Internal patient-level cross-validation only (retrospective PSG-anchored CSV "
    "benchmark). Sensitivity outputs are NOT primary results and never replace "
    "them. Audio remains BLOCKED; the 34-patient airflow sub-cohort is selective "
    "and not extrapolated to the 50-patient main cohort. Not external validation, "
    "not clinical/diagnostic/deployment."
)

# --- nested CV (inherited from frozen runs) ----------------------------------
OUTER_N_FOLDS = MS.OUTER_N_FOLDS          # 5
INNER_N_FOLDS = MS.INNER_N_FOLDS          # 4
CV_UNIT = MS.CV_UNIT                      # "patient"

# --- analysis roles (every emitted artifact carries one) ---------------------
ROLE_PRIMARY_REPLICATION = "primary_replication"
ROLE_SENSITIVITY = "sensitivity"
ROLE_EXPLORATORY = "exploratory"
_ANALYSIS_ROLES = (ROLE_PRIMARY_REPLICATION, ROLE_SENSITIVITY, ROLE_EXPLORATORY)


def is_valid_role(role: str) -> bool:
    return role in _ANALYSIS_ROLES


# --- analysis keys (mirror config.analyses) ----------------------------------
A_PRIMARY_REPRODUCTION = "primary_reproduction"
A_PATIENT_WEIGHT = "patient_weight_uncertainty"
A_THRESHOLD = "threshold_sensitivity"
A_CALIBRATION = "calibration_sensitivity"
A_CLASS_WEIGHT = "class_weight_sensitivity"
A_AIRFLOW_PROFILE = "airflow_selection_profile"
A_ALTERNATIVE_LABEL = "alternative_label"
ALL_ANALYSES = (
    A_PRIMARY_REPRODUCTION, A_PATIENT_WEIGHT, A_THRESHOLD, A_CALIBRATION,
    A_CLASS_WEIGHT, A_AIRFLOW_PROFILE, A_ALTERNATIVE_LABEL,
)

# --- threshold rule labels ---------------------------------------------------
THRESHOLD_YOUDEN = MS.THRESHOLD_RULE_DATA_DRIVEN     # inner_oof_max_youden_j
THRESHOLD_FIXED_05 = "fixed_0_5"
THRESHOLD_BAL_ACC = "inner_oof_balanced_accuracy_max"
THRESHOLD_DUMMY = MS.THRESHOLD_RULE_DUMMY            # fixed_0.5 (dummy)
THRESHOLD_FIT_FROM = MS.THRESHOLD_FIT_FROM           # inner_oof_validation_predictions

# --- calibration method labels -----------------------------------------------
CAL_RAW = "raw_uncalibrated"
CAL_SIGMOID_PLATT = "sigmoid_platt"
CAL_ISOTONIC = "isotonic"

# --- class-weight variant labels ---------------------------------------------
CW_NONE = "none"
CW_BALANCED = "balanced"

# --- bootstrap (patient cluster; window-IID forbidden) -----------------------
BOOTSTRAP_UNIT_PATIENT = "patient"
BOOTSTRAP_DEFAULT_N = MS.BOOTSTRAP_DEFAULT_N         # 1000
BOOTSTRAP_DEFAULT_SEED = MS.BOOTSTRAP_DEFAULT_SEED   # 20250714
BOOTSTRAP_CI_LEVEL = MS.BOOTSTRAP_CI_LEVEL           # 0.95

# --- cohorts / feature sets --------------------------------------------------
COHORT_CORE = "core"                  # 50-patient main cohort (Stage 7)
COHORT_AIRFLOW = "airflow"            # 34-patient selective sub-cohort (Stage 8)
FEATURE_SET_CORE = "core_restricted"
FEATURE_SET_ENHANCED = "airflow_enhanced"

# --- alternative label -------------------------------------------------------
ALT_LABEL_DEFINITION = (
    "event_duration_interval_overlaps_window_by_at_least_min_event_overlap_seconds"
)
ALT_LABEL_NOT_INTERPRETABLE = "not_interpretable"

# --- frozen input defaults (verified at runtime, never trusted blindly) ------
DEFAULT_STAGE7_RUN_ID = "stage7-core-csv-baseline-20260808T025258Z"
DEFAULT_STAGE8_RUN_ID = "stage8-airflow-incremental-20260808T033421Z"
DEFAULT_CONSUMED_SPLIT_RUN_ID = "stage6b-split-balance-v2-20260808T020039Z"

# --- column contracts (pseudonymized tables; no feature values / raw paths) --
# reproduction / sensitivity long-form metric rows
METRIC_ROW_COLUMNS = (
    "run_id", "analysis", "analysis_role", "cohort", "feature_set", "model",
    "model_role", "metric_kind", "metric", "variant", "outer_fold",
    "value", "frozen_value", "abs_diff", "interpretable", "reason",
)
# OOF rows emitted by the re-run engine (for audit / bootstrap alignment)
OOF_ALLOWED_COLUMNS = (
    "run_id", "analysis_role", "cohort", "feature_set", "model", "model_role",
    "outer_fold", "window_id", "patient_id", "y_true", "y_prob",
    "threshold", "threshold_rule", "y_pred", "model_version",
)

__all__ = [name for name in globals() if name.isupper() or name.startswith(("is_",))]
