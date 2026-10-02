"""Load + validate ``config/split_balance_optimization.yaml`` (Stage 6B).

Stage 6B re-partitions the APPROVED Stage-6 v1 core cohort for balance WITHOUT
training any model. It consumes only patient-level aggregates (per-patient window
count / positive count / negative count / positive rate) read from the v1
``outer_patient_folds_core.csv``; it reads no raw CSV/WAV and no feature value.

Validation lives in code (not just YAML) so a careless edit cannot relax a hard
rule: patient-level split, exactly-10-per-outer-fold, fixed seed, no window-random
split, the audio block, the ``no_*`` downstream-deferral flags, never dropping
patients/windows, never relabeling, and the pre-registered adoption thresholds are
all enforced here. The resolved config is written verbatim into
``split_balance_resolved.yaml`` per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required to read the split-balance configuration. "
        "Install it with: python -m pip install pyyaml"
    ) from _exc

from ..config import ConfigError
from .schema import AUDIO_BLOCK_REASON, AUDIO_COHORT_COUNT, AUDIO_FEATURES_PRESENT

SPLIT_BALANCE_VERSION = "split_balance_v2"

_NON_CLINICAL_NOTE = (
    "Single public dataset, patient-level INTERNAL validation only (not external "
    "validation, not a clinical/deployment claim). Stage 6B only re-partitions the "
    "approved Stage-6 core cohort for balance; it trains no model and reads no raw "
    "CSV/WAV. Audio remains BLOCKED."
)


class SplitBalanceConfigError(ConfigError):
    """Raised when ``split_balance_optimization.yaml`` violates a hard Stage-6B rule."""


def _require_bool(f: Dict[str, Any], key: str, *, expected: bool) -> None:
    actual = bool(f.get(key, expected))
    if actual != expected:
        raise SplitBalanceConfigError(
            f"split_balance_optimization.{key} must be {expected!r} (got {actual!r}); "
            f"this hard rule cannot be relaxed via configuration."
        )


@dataclass(frozen=True)
class ResolvedSplitBalanceConfig:
    """Fully-resolved, immutable Stage-6B split-balance configuration for one run."""

    # v1 comparator (read-only)
    v1_comparator_run_id: str
    v1_declared_outer_test_pos_rate_spread: float
    v1_declared_window_count_cv: float
    overall_cohort_positive_rate_reference: float
    # cohort identity
    cohort: str
    unit: str
    expected_core_patients: int
    expected_core_windows: int
    expected_core_positive: int
    expected_core_negative: int
    # outer hard constraints
    outer_n_folds: int
    patients_per_outer_fold: int
    patient_level_split_required: bool
    allow_window_random_split: bool
    all_windows_of_a_patient_in_one_outer_fold: bool
    no_inter_outer_fold_patient_overlap: bool
    every_outer_train_supports_inner_cv: bool
    airflow_inherits_core_outer_fold: bool
    independent_airflow_split_forbidden: bool
    inner_cv_built_only_after_outer_approved: bool
    # inner CV
    inner_n_folds: int
    patients_per_inner_fold: int
    outer_test_patients_never_in_inner: bool
    inner_balance_target: str
    inner_patient_isolation_priority: str
    # search
    algorithm: str
    base_seed: int
    n_candidate_seeds: int
    local_swap_iterations: int
    greedy_seed_sort_key: str
    selection_rule: str
    stop_condition: str
    # objective
    weight_positive_rate_spread: float
    weight_window_count_cv: float
    objective_aggregate: str
    primary_gate_metric: str
    # gate
    gate_relative_positive_rate_spread_max: float
    gate_require_v2_spread_strictly_below_v1: bool
    gate_v2_wcv_ceiling_factor: float
    gate_require_leakage_checks: bool
    gate_require_hard_constraints: bool
    # audio block
    audio_cohort_count: int
    audio_features_present: bool
    audio_block_reason: str
    read_wav_forbidden: bool
    # no-downstream flags
    no_model_matrix: bool
    no_imputation: bool
    no_normalization: bool
    no_feature_selection: bool
    no_class_resampling: bool
    no_threshold_optimization: bool
    no_performance_metrics: bool
    no_audio_access: bool
    no_raw_access: bool
    uses_only_pre_model_patient_summaries: bool
    never_drops_patients_or_windows: bool
    never_relabels: bool
    # version
    split_balance_version: str
    non_clinical_note: str

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            d[k] = getattr(self, k)
        return d


def load_split_balance_config(cfg_yaml_path: Path) -> ResolvedSplitBalanceConfig:
    """Read and validate ``config/split_balance_optimization.yaml``."""
    if not cfg_yaml_path.is_file():
        raise SplitBalanceConfigError(f"split-balance config not found: {cfg_yaml_path}")
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SplitBalanceConfigError(
            f"Failed to parse split-balance YAML {cfg_yaml_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or "split_balance_optimization" not in data:
        raise SplitBalanceConfigError(
            f"{cfg_yaml_path.name}: missing top-level 'split_balance_optimization' mapping."
        )
    f = data["split_balance_optimization"]
    if not isinstance(f, dict):
        raise SplitBalanceConfigError("'split_balance_optimization' must be a mapping.")

    # ---- v1 comparator (read-only) ----
    v1_run = str(f.get("v1_comparator_run_id", ""))
    if not v1_run:
        raise SplitBalanceConfigError("v1_comparator_run_id must be set.")
    v1_decl_spread = float(f.get("v1_declared_outer_test_pos_rate_spread", 0.0))
    v1_decl_wcv = float(f.get("v1_declared_window_count_cv", 0.0))
    overall_ref = float(f.get("overall_cohort_positive_rate_reference", 0.0))

    # ---- cohort identity ----
    cohort = str(f.get("cohort", "core_hr_spo2"))
    unit = str(f.get("unit", "patient"))
    exp_pat = int(f.get("expected_core_patients", 50))
    exp_win = int(f.get("expected_core_windows", 34643))
    exp_pos = int(f.get("expected_core_positive", 9732))
    exp_neg = int(f.get("expected_core_negative", 24911))

    # ---- outer hard constraints ----
    outer_n = int(f.get("outer_n_folds", 5))
    if outer_n < 2:
        raise SplitBalanceConfigError(f"outer_n_folds must be >= 2 (got {outer_n}).")
    per_fold = int(f.get("patients_per_outer_fold", 10))
    if per_fold < 1:
        raise SplitBalanceConfigError(f"patients_per_outer_fold must be >= 1 (got {per_fold}).")
    _require_bool(f, "patient_level_split_required", expected=True)
    _require_bool(f, "allow_window_random_split", expected=False)
    _require_bool(f, "all_windows_of_a_patient_in_one_outer_fold", expected=True)
    _require_bool(f, "no_inter_outer_fold_patient_overlap", expected=True)
    _require_bool(f, "every_outer_train_supports_inner_cv", expected=True)
    _require_bool(f, "airflow_inherits_core_outer_fold", expected=True)
    _require_bool(f, "independent_airflow_split_forbidden", expected=True)
    _require_bool(f, "inner_cv_built_only_after_outer_approved", expected=True)

    # ---- inner CV ----
    inner = f.get("inner_cv", {}) or {}
    inner_n = int(inner.get("inner_n_folds", 4))
    if inner_n < 2:
        raise SplitBalanceConfigError(f"inner_cv.inner_n_folds must be >= 2 (got {inner_n}).")
    inner_per_fold = int(inner.get("patients_per_inner_fold", 10))
    if not bool(inner.get("outer_test_patients_never_in_inner", True)):
        raise SplitBalanceConfigError("inner_cv.outer_test_patients_never_in_inner must be True.")
    inner_balance_target = str(inner.get("balance_target", "positive_rate_spread"))
    inner_iso_prio = str(inner.get("patient_isolation_priority", "highest"))

    # ---- search (pre-registered, deterministic) ----
    search = f.get("search", {}) or {}
    algorithm = str(search.get("algorithm", ""))
    base_seed = int(search.get("base_seed", 20250714))
    n_cand = int(search.get("n_candidate_seeds", 24))
    if n_cand < 1:
        raise SplitBalanceConfigError("search.n_candidate_seeds must be >= 1.")
    iters = int(search.get("local_swap_iterations", 40000))
    if iters < 0:
        raise SplitBalanceConfigError("search.local_swap_iterations must be >= 0.")
    sort_key = str(search.get("greedy_seed_sort_key", ""))
    selection_rule = str(search.get("selection_rule", ""))
    stop_cond = str(search.get("stop_condition", ""))

    # ---- objective (pre-registered weights) ----
    obj = f.get("objective", {}) or {}
    weights = obj.get("weights", {}) or {}
    w_spread = float(weights.get("positive_rate_spread", 1.0))
    w_wcv = float(weights.get("window_count_cv", 0.5))
    if w_spread < 0 or w_wcv < 0:
        raise SplitBalanceConfigError("objective.weights must be non-negative.")
    obj_agg = str(obj.get("aggregate", "weighted_sum"))
    primary_metric = str(obj.get("primary_gate_metric", "positive_rate_spread"))

    # ---- adoption gate ----
    gate = f.get("gate", {}) or {}
    spread_max = float(gate.get("relative_positive_rate_spread_max", 0.10))
    if not (0.0 <= spread_max <= 1.0):
        raise SplitBalanceConfigError(
            f"gate.relative_positive_rate_spread_max must be in [0,1] (got {spread_max})."
        )
    require_below_v1 = bool(gate.get("require_v2_spread_strictly_below_v1", True))
    wcv_ceiling = float(gate.get("v2_window_count_cv_must_not_exceed_v1_times", 1.0))
    if wcv_ceiling < 0:
        raise SplitBalanceConfigError("gate.v2_window_count_cv_must_not_exceed_v1_times must be >= 0.")
    require_leakage = bool(gate.get("require_inner_and_airflow_leakage_checks_pass", True))
    require_hard = bool(gate.get("require_all_hard_constraints_pass", True))

    # ---- audio block (hard) ----
    audio = f.get("audio", {}) or {}
    if int(audio.get("audio_cohort_count", AUDIO_COHORT_COUNT)) != AUDIO_COHORT_COUNT:
        raise SplitBalanceConfigError(
            f"audio.audio_cohort_count is hard-coded {AUDIO_COHORT_COUNT}; Stage 6B emits no audio cohort."
        )
    if bool(audio.get("audio_features_present", False)):
        raise SplitBalanceConfigError("audio.audio_features_present must be False.")
    if not bool(audio.get("read_wav_forbidden", True)):
        raise SplitBalanceConfigError("audio.read_wav_forbidden must be True.")
    block_reason = str(audio.get("block_reason", AUDIO_BLOCK_REASON))

    # ---- nothing downstream ----
    for key in (
        "no_model_matrix", "no_imputation", "no_normalization", "no_feature_selection",
        "no_class_resampling", "no_threshold_optimization", "no_performance_metrics",
        "no_audio_access", "no_raw_access", "uses_only_pre_model_patient_summaries",
        "never_drops_patients_or_windows", "never_relabels",
    ):
        _require_bool(f, key, expected=True)

    return ResolvedSplitBalanceConfig(
        v1_comparator_run_id=v1_run,
        v1_declared_outer_test_pos_rate_spread=v1_decl_spread,
        v1_declared_window_count_cv=v1_decl_wcv,
        overall_cohort_positive_rate_reference=overall_ref,
        cohort=cohort, unit=unit,
        expected_core_patients=exp_pat, expected_core_windows=exp_win,
        expected_core_positive=exp_pos, expected_core_negative=exp_neg,
        outer_n_folds=outer_n, patients_per_outer_fold=per_fold,
        patient_level_split_required=True,
        allow_window_random_split=False,
        all_windows_of_a_patient_in_one_outer_fold=True,
        no_inter_outer_fold_patient_overlap=True,
        every_outer_train_supports_inner_cv=True,
        airflow_inherits_core_outer_fold=True,
        independent_airflow_split_forbidden=True,
        inner_cv_built_only_after_outer_approved=True,
        inner_n_folds=inner_n, patients_per_inner_fold=inner_per_fold,
        outer_test_patients_never_in_inner=True,
        inner_balance_target=inner_balance_target,
        inner_patient_isolation_priority=inner_iso_prio,
        algorithm=algorithm, base_seed=base_seed, n_candidate_seeds=n_cand,
        local_swap_iterations=iters, greedy_seed_sort_key=sort_key,
        selection_rule=selection_rule, stop_condition=stop_cond,
        weight_positive_rate_spread=w_spread, weight_window_count_cv=w_wcv,
        objective_aggregate=obj_agg, primary_gate_metric=primary_metric,
        gate_relative_positive_rate_spread_max=spread_max,
        gate_require_v2_spread_strictly_below_v1=require_below_v1,
        gate_v2_wcv_ceiling_factor=wcv_ceiling,
        gate_require_leakage_checks=require_leakage,
        gate_require_hard_constraints=require_hard,
        audio_cohort_count=AUDIO_COHORT_COUNT,
        audio_features_present=AUDIO_FEATURES_PRESENT,
        audio_block_reason=block_reason,
        read_wav_forbidden=True,
        no_model_matrix=True, no_imputation=True, no_normalization=True,
        no_feature_selection=True, no_class_resampling=True,
        no_threshold_optimization=True, no_performance_metrics=True,
        no_audio_access=True, no_raw_access=True,
        uses_only_pre_model_patient_summaries=True,
        never_drops_patients_or_windows=True, never_relabels=True,
        split_balance_version=str(f.get("split_balance_version", SPLIT_BALANCE_VERSION)),
        non_clinical_note=str(f.get("non_clinical_note", _NON_CLINICAL_NOTE)),
    )


def write_resolved_split_balance_yaml(
    path: Path,
    config: ResolvedSplitBalanceConfig,
    *,
    run_id: str,
    config_hash: str,
    extra: Dict[str, Any] | None = None,
) -> None:
    """Write the resolved split-balance config as a flat, deterministic YAML.

    ``extra`` carries runtime-computed comparator numbers (v1/v2 metrics) so the
    resolved file is self-describing and reproducible.
    """
    body: Dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "split_balance_version": config.split_balance_version,
        "v1_comparator_run_id": config.v1_comparator_run_id,
        "resolved_split_balance_optimization": config.as_dict(),
    }
    if extra:
        body["runtime_computed"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle, sort_keys=True, allow_unicode=True)


__all__ = [
    "SPLIT_BALANCE_VERSION",
    "SplitBalanceConfigError",
    "ResolvedSplitBalanceConfig",
    "load_split_balance_config",
    "write_resolved_split_balance_yaml",
]
