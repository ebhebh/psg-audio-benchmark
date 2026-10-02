"""Stage 6: patient-level experimental cohort + nested cross-validation split.

Builds the audited CSV-physiology research cohort and the patient-level nested
(outer x inner) cross-validation split, plus an input version signature and
audit reports. Consumed by every later modeling stage.

Hard rules (mirror the Stage 1-5 discipline and the prompt):

* Splitting is **patient-level**; window-random splitting is forbidden.
* Balance uses **only** pre-model patient-level summaries; never model output.
* ``audio`` is BLOCKED: ``audio_cohort_count`` is hard-coded 0; no ``*.wav`` is
  ever read.
* No model matrix / imputation / normalization / feature selection / resampling
  / threshold / metrics is computed here -- all are fitted fold-internally later.
* ``data/raw`` is read-only (proved by a before/after snapshot); Stage 1-5
  historical products are never rewritten.
"""

from __future__ import annotations

from .config import SplitsConfigError, load_splits_config, write_resolved_config_yaml
from .qc import (
    EvaluationContaminationError,
    assert_not_contaminating,
    assert_paths_clean,
    relativize,
    snapshot_raw,
)
from .runner import Stage6Options, Stage6Runner
from .schema import (
    AUDIO_BLOCK_REASON,
    AUDIO_COHORT_COUNT,
    PATIENT_SPLIT_VERSION,
    STAGE6_FORBIDDEN_COLUMNS,
    SplitSummary,
)
from .signature import SignatureMismatchError, build_input_signature, compare_signatures
from .split_balance_config import (
    SPLIT_BALANCE_VERSION,
    ResolvedSplitBalanceConfig,
    SplitBalanceConfigError,
    load_split_balance_config,
    write_resolved_split_balance_yaml,
)
from .balance_optimizer import (
    OptimizationResult,
    PatientBurden,
    optimize_patient_folds,
)
from .split_balance_runner import Stage6bOptions, Stage6bRunner
from .split_balance_report import SplitBalanceSummary
from .gate import APPROVED, NOT_APPROVED, GateResult

__all__ = [
    "Stage6Options",
    "Stage6Runner",
    "SplitSummary",
    "SplitsConfigError",
    "load_splits_config",
    "write_resolved_config_yaml",
    "EvaluationContaminationError",
    "assert_not_contaminating",
    "assert_paths_clean",
    "relativize",
    "snapshot_raw",
    "SignatureMismatchError",
    "build_input_signature",
    "compare_signatures",
    "PATIENT_SPLIT_VERSION",
    "AUDIO_COHORT_COUNT",
    "AUDIO_BLOCK_REASON",
    "STAGE6_FORBIDDEN_COLUMNS",
    # Stage 6B
    "SPLIT_BALANCE_VERSION",
    "ResolvedSplitBalanceConfig",
    "SplitBalanceConfigError",
    "load_split_balance_config",
    "write_resolved_split_balance_yaml",
    "OptimizationResult",
    "PatientBurden",
    "optimize_patient_folds",
    "Stage6bOptions",
    "Stage6bRunner",
    "SplitBalanceSummary",
    "GateResult",
    "APPROVED",
    "NOT_APPROVED",
]
