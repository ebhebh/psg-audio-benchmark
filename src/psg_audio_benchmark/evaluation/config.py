"""Load and validate ``config/splits.yaml`` into a resolved Stage-6 config.

Validation lives in code (not just YAML) so a careless edit cannot relax a hard
rule: patient-level split, no window-random split, the audio block, the fixed
seed, outer>=2 / inner>=2, the strict airflow membership rule, the airflow
inheritance, and the ``no_*`` downstream-deferral flags are all enforced here.
The resolved config is written verbatim into ``splits_resolved.yaml`` per run,
recording the seed, scikit-learn version, fold counts and the patient sort-order
convention so the exact fold assignment is reproducible from this file alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required to read the splits configuration. "
        "Install it with: python -m pip install pyyaml"
    ) from _exc

from ..config import ConfigError
from .schema import (
    AIRFLOW_MEMBERSHIP_RULE,
    AUDIO_BLOCK_REASON,
    AUDIO_COHORT_COUNT,
    AUDIO_FEATURES_PRESENT,
    COHORT_AIRFLOW,
    COHORT_CORE,
    PATIENT_SPLIT_VERSION,
    ResolvedSplitsConfig,
)

_DEFAULT_SEED = 20250714
_REQUIRED_STRATIFIER = "stratified_group_kfold_by_patient_label_burden"
_REQUIRED_GROUP = "patient_id"
_REQUIRED_LABEL = "binary_event_label"
_REQUIRED_SORT = "patient_id_ascending"
_REQUIRED_AIRFLOW_RULE = AIRFLOW_MEMBERSHIP_RULE  # core_hr_spo2_AND_airflow_available
_NON_CLINICAL_NOTE = (
    "Single public dataset, patient-level INTERNAL validation only (not external "
    "validation, not a clinical/deployment claim). CSV physiology features, "
    "PSG-anchored retrospective benchmark. Audio remains BLOCKED."
)


class SplitsConfigError(ConfigError):
    """Raised when ``splits.yaml`` violates a hard Stage-6 rule."""


def _get(d: Dict[str, Any], key: str, *, default=None):
    return d.get(key, default)


def _require_bool(f: Dict[str, Any], key: str, *, expected: bool) -> None:
    actual = bool(f.get(key, expected))
    if actual != expected:
        raise SplitsConfigError(
            f"splits.{key} must be {expected!r} (got {actual!r}); this hard rule "
            f"cannot be relaxed via configuration."
        )


def load_splits_config(cfg_yaml_path: Path) -> ResolvedSplitsConfig:
    """Read and validate ``config/splits.yaml``."""
    if not cfg_yaml_path.is_file():
        raise SplitsConfigError(f"splits config not found: {cfg_yaml_path}")
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SplitsConfigError(
            f"Failed to parse splits YAML {cfg_yaml_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or "splits" not in data:
        raise SplitsConfigError(
            f"{cfg_yaml_path.name}: missing top-level 'splits' mapping."
        )
    f = data["splits"]
    if not isinstance(f, dict):
        raise SplitsConfigError("'splits' must be a mapping.")

    # ---- hard leakage rules ----
    _require_bool(f, "patient_level_split_required", expected=True)
    _require_bool(f, "allow_window_random_split", expected=False)
    _require_bool(f, "split_uses_only_pre_model_patient_summaries", expected=True)
    _require_bool(f, "never_uses_model_output_or_predictions", expected=True)

    # ---- inputs (may be empty here; the runner pins/gates them) ----
    stage4 = str(f.get("stage4_input_run_id", ""))
    stage5 = str(f.get("stage5_input_run_id", ""))
    stage3 = str(f.get("stage3_input_run_id", ""))

    # ---- main analysis: 5-fold outer ----
    main = f.get("main_analysis", {}) or {}
    outer_n = int(main.get("outer_n_folds", 5))
    if outer_n < 2:
        raise SplitsConfigError(f"main_analysis.outer_n_folds must be >= 2 (got {outer_n}).")
    seed = int(main.get("outer_fixed_seed", _DEFAULT_SEED))
    stratifier = str(main.get("stratifier", _REQUIRED_STRATIFIER))
    if stratifier != _REQUIRED_STRATIFIER:
        raise SplitsConfigError(
            f"main_analysis.stratifier must be {_REQUIRED_STRATIFIER!r} (got {stratifier!r})."
        )
    group_col = str(main.get("group_column", _REQUIRED_GROUP))
    if group_col != _REQUIRED_GROUP:
        raise SplitsConfigError(f"main_analysis.group_column must be {_REQUIRED_GROUP!r}.")
    label_col = str(main.get("label_for_stratification", _REQUIRED_LABEL))
    if label_col != _REQUIRED_LABEL:
        raise SplitsConfigError(
            f"main_analysis.label_for_stratification must be {_REQUIRED_LABEL!r}."
        )
    sort_by = str(main.get("sort_patients_by", _REQUIRED_SORT))
    if sort_by != _REQUIRED_SORT:
        raise SplitsConfigError(
            f"main_analysis.sort_patients_by must be {_REQUIRED_SORT!r} for determinism."
        )

    # ---- secondary analysis: airflow inherits ----
    sec = f.get("secondary_analysis", {}) or {}
    airflow_rule = str(sec.get("airflow_membership_rule", _REQUIRED_AIRFLOW_RULE))
    if airflow_rule != _REQUIRED_AIRFLOW_RULE:
        raise SplitsConfigError(
            f"secondary_analysis.airflow_membership_rule must be "
            f"{_REQUIRED_AIRFLOW_RULE!r} (strict: HR+SpO2+airflow all available)."
        )
    inherits = str(sec.get("inherits_outer_fold_from", COHORT_CORE))
    if inherits != COHORT_CORE:
        raise SplitsConfigError(
            f"secondary_analysis.inherits_outer_fold_from must be {COHORT_CORE!r}."
        )
    airflow_indep_forbidden = bool(sec.get("independent_random_split_forbidden", True))
    if not airflow_indep_forbidden:
        raise SplitsConfigError(
            "secondary_analysis.independent_random_split_forbidden must be True."
        )

    # ---- nested inner CV: 4-fold ----
    inner = f.get("inner_cv", {}) or {}
    inner_n = int(inner.get("inner_n_folds", 4))
    if inner_n < 2:
        raise SplitsConfigError(f"inner_cv.inner_n_folds must be >= 2 (got {inner_n}).")
    inner_seed = int(inner.get("inner_fixed_seed", _DEFAULT_SEED))
    outer_test_never_in_inner = bool(inner.get("outer_test_patients_never_in_inner", True))
    if not outer_test_never_in_inner:
        raise SplitsConfigError("inner_cv.outer_test_patients_never_in_inner must be True.")

    # ---- balance ----
    bal = f.get("balance", {}) or {}
    spread_thr = float(bal.get("warn_when_test_fold_pos_rate_spread_exceeds", 0.10))
    if not (0.0 <= spread_thr <= 1.0):
        raise SplitsConfigError(
            f"balance.warn_when_test_fold_pos_rate_spread_exceeds must be in [0,1] (got {spread_thr})."
        )
    if not bool(bal.get("strict_patient_isolation_priority_highest", True)):
        raise SplitsConfigError("balance.strict_patient_isolation_priority_highest must be True.")
    if not bool(bal.get("never_silently_relabel_or_drop_minority", True)):
        raise SplitsConfigError("balance.never_silently_relabel_or_drop_minority must be True.")

    # ---- audio block (hard) ----
    audio = f.get("audio", {}) or {}
    if int(audio.get("audio_cohort_count", AUDIO_COHORT_COUNT)) != AUDIO_COHORT_COUNT:
        raise SplitsConfigError(
            f"audio.audio_cohort_count is hard-coded {AUDIO_COHORT_COUNT}; Stage 6 emits no audio cohort."
        )
    if bool(audio.get("audio_features_present", False)):
        raise SplitsConfigError("audio.audio_features_present must be False.")
    if not bool(audio.get("read_wav_forbidden", True)):
        raise SplitsConfigError("audio.read_wav_forbidden must be True.")

    # ---- nothing downstream ----
    for key in (
        "no_model_matrix", "no_imputation", "no_normalization", "no_feature_selection",
        "no_class_resampling", "no_threshold_optimization", "no_performance_metrics",
        "no_audio_access",
    ):
        _require_bool(f, key, expected=True)

    return ResolvedSplitsConfig(
        stage4_input_run_id=stage4,
        stage5_input_run_id=stage5,
        stage3_input_run_id=stage3,
        outer_n_folds=outer_n,
        inner_n_folds=inner_n,
        seed=seed,
        inner_seed=inner_seed,
        stratifier=stratifier,
        group_column=group_col,
        label_for_stratification=label_col,
        sort_patients_by=sort_by,
        airflow_membership_rule=airflow_rule,
        airflow_inherits_outer_fold_from=inherits,
        airflow_independent_split_forbidden=airflow_indep_forbidden,
        warn_pos_rate_spread_threshold=spread_thr,
        patient_level_split_required=True,
        allow_window_random_split=False,
        split_uses_only_pre_model_summaries=True,
        audio_cohort_count=AUDIO_COHORT_COUNT,
        audio_features_present=AUDIO_FEATURES_PRESENT,
        audio_block_reason=str(audio.get("block_reason", AUDIO_BLOCK_REASON)),
        read_wav_forbidden=True,
        no_model_matrix=True,
        no_imputation=True,
        no_normalization=True,
        no_feature_selection=True,
        no_class_resampling=True,
        no_threshold_optimization=True,
        no_performance_metrics=True,
        no_audio_access=True,
        patient_split_version=str(f.get("patient_split_version", PATIENT_SPLIT_VERSION)),
        non_clinical_note=str(f.get("non_clinical_note", _NON_CLINICAL_NOTE)),
    )


def write_resolved_config_yaml(
    path: Path,
    config: ResolvedSplitsConfig,
    *,
    run_id: str,
    config_hash: str,
) -> None:
    """Write the resolved split config as a flat, deterministic YAML."""
    body = {
        "run_id": run_id,
        "config_hash": config_hash,
        "patient_split_version": config.patient_split_version,
        "input_stage4_run_id": config.stage4_input_run_id,
        "input_stage5_run_id": config.stage5_input_run_id,
        "input_stage3_run_id": config.stage3_input_run_id,
        "sklearn_version": _sklearn_version(),
        "resolved_splits": config.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle, sort_keys=True, allow_unicode=True)


def _sklearn_version() -> str:
    try:
        import sklearn  # type: ignore
        return str(getattr(sklearn, "__version__", "unknown"))
    except Exception:  # pragma: no cover - dependency guard
        return "not_installed"


__all__ = [
    "SplitsConfigError",
    "load_splits_config",
    "write_resolved_config_yaml",
]
