"""Constants, vocabularies and versions for Stage-7 core-CSV modeling.

This module is the single in-code authority for the things that must NEVER be
silently relaxed by editing YAML:

* the feature ALLOW-list (only these prefixed columns may enter an estimator);
* the forbidden feature category tokens (IDs / time / quality / availability /
  label / annotation / awake / sleep / airflow / audio / full-cohort-fit);
* the pre-specified model family names and their pre-registered roles;
* the inner-selection rule and the threshold rule;
* the audio block (audio cohort hard-coded 0; ``*.wav`` never read);
* the non-clinical task framing.

Everything here mirrors ``config/modeling_core_csv.yaml``; the loader
(:mod:`modeling.config`) cross-checks the two so a careless edit cannot relax a
rule.
"""

from __future__ import annotations

from typing import Tuple

# ---------------------------------------------------------------------------
# Versions / task framing / audio block
# ---------------------------------------------------------------------------

MODELING_STAGE: str = "stage7"
MODELING_VERSION: str = "core_csv_baseline_v1"
TASK_NAME: str = "window-level dataset-scored respiratory-event baseline"
COHORT: str = "core_hr_spo2"
SPLIT_BALANCE_VERSION: str = "split_balance_v2"

APPROVAL_REQUIRED: str = "approved_for_modeling"

AUDIO_BLOCK_REASON: str = "no_trustworthy_audio_time_anchor_unresolved"
AUDIO_COHORT_COUNT: int = 0
AUDIO_FEATURES_PRESENT: bool = False
READ_WAV_FORBIDDEN: bool = True

NON_CLINICAL_NOTE: str = (
    "Single public dataset, patient-level INTERNAL cross-validation only "
    "(retrospective PSG-anchored CSV benchmark; not external validation, not a "
    "clinical/diagnostic/deployment claim). HGB is exploratory only and is never "
    "declared the winner by outer-test score. Results must NOT be generalized to "
    "audio; audio remains BLOCKED."
)

# ---------------------------------------------------------------------------
# Nested CV structure (must match the inherited frozen v2 split)
# ---------------------------------------------------------------------------

OUTER_N_FOLDS: int = 5
INNER_N_FOLDS: int = 4
PATIENTS_PER_OUTER_FOLD: int = 10
CV_UNIT: str = "patient"

# ---------------------------------------------------------------------------
# Feature whitelist (ALLOW-list). Only these prefixed statistics may enter an
# estimator. The source columns live in Stage-5 hr/spo2 window-feature parquets.
# ---------------------------------------------------------------------------

FEATURE_MODALITIES: Tuple[str, ...] = ("heart_rate", "spo2")
FEATURE_PREFIX = {"heart_rate": "hr_", "spo2": "spo2_"}
FEATURE_STATISTICS: Tuple[str, ...] = (
    "mean", "median", "std_ddof1", "min", "max", "range", "iqr", "slope_per_second",
)

#: The complete, ordered list of allowed estimator input columns (8 stats x 2
#: modalities = 16). feature_assembly builds exactly this set, in this order.
FEATURE_WHITELIST: Tuple[str, ...] = tuple(
    f"{FEATURE_PREFIX[mod]}{stat}"
    for mod in FEATURE_MODALITIES
    for stat in FEATURE_STATISTICS
)

#: Stable join key between features and the cohort/label tables.
FEATURE_JOIN_KEY: str = "window_id"

# ---------------------------------------------------------------------------
# Forbidden feature category tokens. Any input column carrying one of these
#: substrings (or appearing verbatim in FORBIDDEN_COLUMNS) is rejected before
# it can reach an estimator. This is defence-in-depth on top of the allow-list.
# ---------------------------------------------------------------------------

FORBIDDEN_CATEGORY_TOKENS: Tuple[str, ...] = (
    "airflow", "audio", "awake", "sleep", "label", "annotation", "exclusion",
    "quality", "coverage", "missing", "value_unit", "duration", "relative_to_record",
    "absolute_cumulative", "window_index", "run_id", "modality",
)

#: Explicit forbidden column names (the IDs / time / availability / label
#: columns that share no substring with the category tokens).
FORBIDDEN_COLUMNS: Tuple[str, ...] = (
    FEATURE_JOIN_KEY, "patient_id", "window_index", "run_id", "modality",
    "modality_feature_version", "audio_features_present",
    "n_expected", "n_observed", "n_finite",
    "core_heart_rate_coverage", "core_spo2_coverage", "has_airflow_coverage",
    "core_coverage_complete",
)

#: Audit/identity keys retained on the model frame for traceability + patient
#: bootstrap ONLY; they are stripped before any estimator sees the matrix.
AUDIT_KEYS: Tuple[str, ...] = (
    "window_id", "patient_id", "outer_fold", "inner_validation_fold",
)
LABEL_COLUMN: str = "binary_event_label"

# ---------------------------------------------------------------------------
# Pre-specified model families and their pre-registered roles.
# ---------------------------------------------------------------------------

MODEL_DUMMY_PRIOR: str = "dummy_prior"
MODEL_LOGISTIC_REGRESSION: str = "logistic_regression"
MODEL_HIST_GRADIENT_BOOSTING: str = "hist_gradient_boosting"

MODEL_FAMILIES: Tuple[str, ...] = (
    MODEL_DUMMY_PRIOR, MODEL_LOGISTIC_REGRESSION, MODEL_HIST_GRADIENT_BOOSTING,
)
PRIMARY_MODEL: str = MODEL_LOGISTIC_REGRESSION
EXPLORATORY_MODELS: Tuple[str, ...] = (MODEL_HIST_GRADIENT_BOOSTING,)

#: A model may NOT be declared the winner by outer-test score. Roles are fixed.
MODEL_ROLE: dict = {
    MODEL_DUMMY_PRIOR: "lower_bound_baseline",
    MODEL_LOGISTIC_REGRESSION: "primary",
    MODEL_HIST_GRADIENT_BOOSTING: "exploratory",
}

# ---------------------------------------------------------------------------
# Selection + threshold rules.
# ---------------------------------------------------------------------------

SELECTION_PRIMARY_SCORE: str = "average_precision"
SELECTION_TIE_BREAK: Tuple[str, ...] = ("higher_auroc", "simpler_candidate")
THRESHOLD_RULE_DATA_DRIVEN: str = "inner_oof_max_youden_j"
THRESHOLD_RULE_DUMMY: str = "fixed_0.5"
THRESHOLD_DUMMY_VALUE: float = 0.5
THRESHOLD_FIT_FROM: str = "inner_oof_validation_predictions"

# ---------------------------------------------------------------------------
# Bootstrap contract.
# ---------------------------------------------------------------------------

BOOTSTRAP_UNIT_PATIENT: str = "patient"
BOOTSTRAP_DEFAULT_N: int = 1000
BOOTSTRAP_DEFAULT_SEED: int = 20250714
BOOTSTRAP_CI_LEVEL: float = 0.95

__all__ = [name for name in dir() if name.isupper() or name.startswith("MODEL_")]
