"""Stage-7 core-CSV modeling configuration loader + resolved-config writer.

Mirrors the Stage-6B split-balance loader discipline:

* the YAML is declarative;
* the immutable scientific rules (patient-level inherited split, no window
  random split, audio BLOCKED, feature allow-list, selection/threshold rules)
  are cross-checked against :mod:`modeling.schema` so a careless YAML edit
  cannot relax them;
* a frozen ``ResolvedModelingConfig`` dataclass is produced for one run and
  written verbatim as ``modeling_resolved.yaml``.

Importing this module reads no data and writes no files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ..config import Config, ConfigError
from . import schema as S


class ModelingConfigError(ConfigError):
    """Raised when ``modeling_core_csv.yaml`` violates a hard Stage-7 rule."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_bool(f: Dict[str, Any], key: str, *, expected: bool, where: str) -> None:
    actual = bool(f.get(key, expected))
    if actual != expected:
        raise ModelingConfigError(
            f"{where}.{key} must be {expected!r} (got {actual!r}); this hard rule "
            f"cannot be relaxed via configuration."
        )


def _require_eq(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ModelingConfigError(
            f"{label}: YAML declares {value!r} but the immutable code constant is "
            f"{expected!r}. The code constant wins."
        )


# ---------------------------------------------------------------------------
# resolved config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedModelingConfig:
    """Fully-resolved, immutable Stage-7 core-CSV modeling configuration."""

    # identity / framing
    stage: str
    modeling_version: str
    task_name: str
    cohort: str
    split_balance_version: str
    non_clinical_note: str
    # gate
    require_license_passed: bool
    require_latest_run_v2: bool
    require_approval_status: str
    reject_v1: bool
    reject_unapproved: bool
    reject_signature_mismatch: bool
    verify_v2_file_sha256: bool
    verify_assignment_sha256: bool
    verify_patient_set_sha256: bool
    verify_lineage_run_ids: bool
    runtime_reverify_cohort: bool
    forbidden_hardcode_counts: bool
    # features
    feature_whitelist: Tuple[str, ...]
    feature_statistics: Tuple[str, ...]
    feature_modalities: Tuple[str, ...]
    feature_prefix: Dict[str, str]
    feature_join_key: str
    forbidden_columns: Tuple[str, ...]
    forbidden_category_tokens: Tuple[str, ...]
    # label
    label_column: str
    label_positive: int
    label_negative: int
    # nested CV (inherited)
    outer_n_folds: int
    inner_n_folds: int
    patients_per_outer_fold: int
    cv_unit: str
    inherits_split: bool
    rebuilds_split: bool
    window_random_split: bool
    outer_assignment_source: str
    inner_assignment_source: str
    # selection / threshold
    selection_primary_score: str
    selection_tie_break: Tuple[str, ...]
    threshold_rule_data_driven: str
    threshold_rule_dummy: str
    threshold_dummy_value: float
    threshold_fit_from: str
    # evaluation / bootstrap
    threshold_free_metrics: Tuple[str, ...]
    threshold_based_metrics: Tuple[str, ...]
    single_class_policy: str
    bootstrap_enabled: bool
    bootstrap_unit: str
    bootstrap_n_resamples: int
    bootstrap_seed: int
    bootstrap_ci_level: float
    bootstrap_apply_to: Tuple[str, ...]
    forbid_window_iid: bool
    figure_max_count: int
    figure_requested: Tuple[str, ...]
    # immutability / isolation
    read_wav_forbidden: bool
    audio_products_forbidden: bool
    airflow_in_core_model_forbidden: bool
    snapshot_raw_before_after: bool
    historical_products_unchanged: bool
    oof_pseudonymize: bool
    # raw model spec (verbatim, for the resolved YAML + audit)
    models_spec: Dict[str, Any] = field(default_factory=dict)
    selection_spec: Dict[str, Any] = field(default_factory=dict)
    threshold_spec: Dict[str, Any] = field(default_factory=dict)
    evaluation_spec: Dict[str, Any] = field(default_factory=dict)
    raw_yaml: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, tuple):
                v = list(v)
            d[k] = v
        return d


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def load_modeling_config(cfg_yaml_path: Path) -> ResolvedModelingConfig:
    """Load and validate ``modeling_core_csv.yaml`` against the code constants."""
    if not cfg_yaml_path.is_file():
        raise ModelingConfigError(f"Modeling config not found: {cfg_yaml_path}")
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ModelingConfigError(f"Failed to parse {cfg_yaml_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelingConfigError(f"{cfg_yaml_path} did not parse to a mapping.")

    meta = raw.get("meta", {}) or {}
    gate = raw.get("gate", {}) or {}
    feat = raw.get("features", {}) or {}
    lab = raw.get("label", {}) or {}
    ncv = raw.get("nested_cv", {}) or {}
    sel = raw.get("selection", {}) or {}
    thr = raw.get("threshold", {}) or {}
    ev = raw.get("evaluation", {}) or {}
    out = raw.get("output", {}) or {}

    # --- framing must match code constants ---
    _require_eq(meta.get("stage"), S.MODELING_STAGE, "meta.stage")
    _require_eq(meta.get("cohort"), S.COHORT, "meta.cohort")
    _require_eq(meta.get("split_balance_version"), S.SPLIT_BALANCE_VERSION,
                "meta.split_balance_version")
    _require_eq(meta.get("task_name"), S.TASK_NAME, "meta.task_name")

    # --- audio block (must stay blocked) ---
    _require_bool(out, "read_wav_forbidden", expected=True, where="output")
    _require_bool(out, "audio_products_forbidden", expected=True, where="output")
    _require_bool(out, "airflow_in_core_model_forbidden", expected=True, where="output")

    # --- nested CV must inherit the frozen split, never rebuild ---
    _require_bool(ncv, "inherits_split", expected=True, where="nested_cv")
    _require_bool(ncv, "rebuilds_split", expected=False, where="nested_cv")
    _require_bool(ncv, "window_random_split", expected=False, where="nested_cv")
    _require_eq(int(ncv.get("outer_n_folds")), S.OUTER_N_FOLDS, "nested_cv.outer_n_folds")
    _require_eq(int(ncv.get("inner_n_folds")), S.INNER_N_FOLDS, "nested_cv.inner_n_folds")
    _require_eq(int(ncv.get("patients_per_outer_fold")), S.PATIENTS_PER_OUTER_FOLD,
                "nested_cv.patients_per_outer_fold")
    _require_eq(ncv.get("unit"), S.CV_UNIT, "nested_cv.unit")

    # --- feature whitelist must exactly match the code allow-list ---
    yaml_stats = tuple(feat.get("statistics", []))
    _require_eq(yaml_stats, S.FEATURE_STATISTICS, "features.statistics")
    yaml_modalities = tuple(feat.get("modalities", []))
    _require_eq(yaml_modalities, S.FEATURE_MODALITIES, "features.modalities")
    yaml_prefix = dict(feat.get("prefix", {}))
    _require_eq(yaml_prefix, dict(S.FEATURE_PREFIX), "features.prefix")
    _require_eq(feat.get("join_key"), S.FEATURE_JOIN_KEY, "features.join_key")

    # --- selection / threshold rules must match code constants ---
    _require_eq(sel.get("primary_score"), S.SELECTION_PRIMARY_SCORE,
                "selection.primary_score")
    _require_eq(tuple(sel.get("tie_break_order", [])), S.SELECTION_TIE_BREAK,
                "selection.tie_break_order")
    _require_eq(thr.get("data_driven_rule"), S.THRESHOLD_RULE_DATA_DRIVEN,
                "threshold.data_driven_rule")
    _require_eq(thr.get("fit_from"), S.THRESHOLD_FIT_FROM, "threshold.fit_from")
    _require_eq(thr.get("dummy_fixed_value"), S.THRESHOLD_DUMMY_VALUE,
                "threshold.dummy_fixed_value")

    # --- bootstrap must be patient-cluster, never window IID ---
    bsp = ev.get("bootstrap", {}) or {}
    _require_bool(bsp, "enabled", expected=True, where="evaluation.bootstrap")
    _require_eq(bsp.get("unit"), S.BOOTSTRAP_UNIT_PATIENT, "evaluation.bootstrap.unit")
    _require_bool(bsp, "resample_bring_all_windows", expected=True,
                  where="evaluation.bootstrap")
    _require_bool(bsp, "forbid_window_iid", expected=True, where="evaluation.bootstrap")

    cfg = ResolvedModelingConfig(
        stage=S.MODELING_STAGE,
        modeling_version=S.MODELING_VERSION,
        task_name=S.TASK_NAME,
        cohort=S.COHORT,
        split_balance_version=S.SPLIT_BALANCE_VERSION,
        non_clinical_note=S.NON_CLINICAL_NOTE,
        require_license_passed=bool(gate.get("require_license_passed", True)),
        require_latest_run_v2=bool(gate.get("require_latest_run_v2", True)),
        require_approval_status=str(gate.get("require_approval_status",
                                             S.APPROVAL_REQUIRED)),
        reject_v1=bool(gate.get("reject_v1", True)),
        reject_unapproved=bool(gate.get("reject_unapproved", True)),
        reject_signature_mismatch=bool(gate.get("reject_signature_mismatch", True)),
        verify_v2_file_sha256=bool(gate.get("verify_v2_file_sha256", True)),
        verify_assignment_sha256=bool(gate.get("verify_assignment_sha256", True)),
        verify_patient_set_sha256=bool(gate.get("verify_patient_set_sha256", True)),
        verify_lineage_run_ids=bool(gate.get("verify_lineage_run_ids", True)),
        runtime_reverify_cohort=bool(gate.get("runtime_reverify_cohort", True)),
        forbidden_hardcode_counts=bool(gate.get("forbidden_hardcode_counts", True)),
        feature_whitelist=S.FEATURE_WHITELIST,
        feature_statistics=S.FEATURE_STATISTICS,
        feature_modalities=S.FEATURE_MODALITIES,
        feature_prefix=dict(S.FEATURE_PREFIX),
        feature_join_key=S.FEATURE_JOIN_KEY,
        forbidden_columns=S.FORBIDDEN_COLUMNS,
        forbidden_category_tokens=S.FORBIDDEN_CATEGORY_TOKENS,
        label_column=str(lab.get("column", S.LABEL_COLUMN)),
        label_negative=int((lab.get("values", {}) or {}).get("negative", 0)),
        label_positive=int((lab.get("values", {}) or {}).get("positive", 1)),
        outer_n_folds=S.OUTER_N_FOLDS,
        inner_n_folds=S.INNER_N_FOLDS,
        patients_per_outer_fold=S.PATIENTS_PER_OUTER_FOLD,
        cv_unit=S.CV_UNIT,
        inherits_split=True,
        rebuilds_split=False,
        window_random_split=False,
        outer_assignment_source=str(ncv.get("outer_assignment_source",
                                            "splits/outer_patient_folds_core_v2.csv")),
        inner_assignment_source=str(ncv.get("inner_assignment_source",
                                            "splits/inner_patient_folds_core_v2.parquet")),
        selection_primary_score=S.SELECTION_PRIMARY_SCORE,
        selection_tie_break=S.SELECTION_TIE_BREAK,
        threshold_rule_data_driven=S.THRESHOLD_RULE_DATA_DRIVEN,
        threshold_rule_dummy=S.THRESHOLD_RULE_DUMMY,
        threshold_dummy_value=S.THRESHOLD_DUMMY_VALUE,
        threshold_fit_from=S.THRESHOLD_FIT_FROM,
        threshold_free_metrics=tuple(ev.get("threshold_free_metrics", [])),
        threshold_based_metrics=tuple(ev.get("threshold_based_metrics", [])),
        single_class_policy=str(ev.get("single_class_policy", "na_with_reason")),
        bootstrap_enabled=True,
        bootstrap_unit=S.BOOTSTRAP_UNIT_PATIENT,
        bootstrap_n_resamples=int(bsp.get("n_resamples", S.BOOTSTRAP_DEFAULT_N)),
        bootstrap_seed=int(bsp.get("seed", S.BOOTSTRAP_DEFAULT_SEED)),
        bootstrap_ci_level=float(bsp.get("ci_level", S.BOOTSTRAP_CI_LEVEL)),
        bootstrap_apply_to=tuple(bsp.get("apply_to", [S.PRIMARY_MODEL])),
        forbid_window_iid=True,
        figure_max_count=int((ev.get("figures", {}) or {}).get("max_count", 5)),
        figure_requested=tuple((ev.get("figures", {}) or {}).get("requested", [])),
        read_wav_forbidden=True,
        audio_products_forbidden=True,
        airflow_in_core_model_forbidden=True,
        snapshot_raw_before_after=bool(out.get("snapshot_raw_before_after", True)),
        historical_products_unchanged=bool(out.get("historical_products_unchanged", True)),
        oof_pseudonymize=bool(out.get("oof_pseudonymize", True)),
        models_spec=dict(raw.get("models", {}) or {}),
        selection_spec=dict(sel),
        threshold_spec=dict(thr),
        evaluation_spec=dict(ev),
        raw_yaml=dict(raw),
    )
    return cfg


# ---------------------------------------------------------------------------
# resolved-config writer
# ---------------------------------------------------------------------------

def write_resolved_modeling_yaml(
    *, cfg: ResolvedModelingConfig, run_id: str, config_hash: str, path: Path,
    split_run_id: str, split_signature: Dict[str, Any],
    package_versions: Dict[str, str],
) -> None:
    """Write the resolved, reproducible config YAML into the run-dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sig_cohort = (split_signature or {}).get("cohort", {}) or {}
    sig_split = (split_signature or {}).get("split", {}) or {}
    doc: Dict[str, Any] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "modeling_version": cfg.modeling_version,
        "stage": cfg.stage,
        "task_name": cfg.task_name,
        "cohort": cfg.cohort,
        "non_clinical_note": cfg.non_clinical_note,
        "split_balance_version": cfg.split_balance_version,
        "consumed_split_run_id": split_run_id,
        "split_approval_status": (split_signature or {}).get("approval_status"),
        "split_assignment_sha256": sig_split.get("assignment_sha256"),
        "split_patient_set_sha256": sig_cohort.get("patient_set_sha256"),
        "frozen_cohort_fingerprint": {
            "n_patients": sig_cohort.get("n_patients"),
            "n_windows": sig_cohort.get("n_windows"),
            "n_positive": sig_cohort.get("n_positive"),
            "n_negative": sig_cohort.get("n_negative"),
            "overall_positive_rate": sig_cohort.get("overall_positive_rate"),
            "note": "read at runtime from the approved signature; never hard-coded",
        },
        "nested_cv": {
            "outer_n_folds": cfg.outer_n_folds,
            "inner_n_folds": cfg.inner_n_folds,
            "patients_per_outer_fold": cfg.patients_per_outer_fold,
            "unit": cfg.cv_unit,
            "inherits_split": cfg.inherits_split,
            "rebuilds_split": cfg.rebuilds_split,
            "window_random_split": cfg.window_random_split,
            "outer_assignment_source": cfg.outer_assignment_source,
            "inner_assignment_source": cfg.inner_assignment_source,
        },
        "features": {
            "whitelist": list(cfg.feature_whitelist),
            "modalities": list(cfg.feature_modalities),
            "prefix": cfg.feature_prefix,
            "statistics": list(cfg.feature_statistics),
            "join_key": cfg.feature_join_key,
            "forbidden_columns": list(cfg.forbidden_columns),
            "forbidden_category_tokens": list(cfg.forbidden_category_tokens),
        },
        "label": {
            "column": cfg.label_column,
            "negative": cfg.label_negative,
            "positive": cfg.label_positive,
        },
        "models": cfg.models_spec,
        "selection": {
            "primary_score": cfg.selection_primary_score,
            "tie_break_order": list(cfg.selection_tie_break),
            "record_all_candidates": True,
            "refit_on_full_outer_train": True,
            "predict_outer_test_once": True,
            "no_cherry_pick_by_outer_test": True,
        },
        "threshold": {
            "data_driven_rule": cfg.threshold_rule_data_driven,
            "dummy_rule": cfg.threshold_rule_dummy,
            "dummy_value": cfg.threshold_dummy_value,
            "fit_from": cfg.threshold_fit_from,
            "never_fit_from": "outer_test",
        },
        "evaluation": {
            "threshold_free_metrics": list(cfg.threshold_free_metrics),
            "threshold_based_metrics": list(cfg.threshold_based_metrics),
            "single_class_policy": cfg.single_class_policy,
            "pooled_oof": True,
            "bootstrap": {
                "enabled": cfg.bootstrap_enabled,
                "unit": cfg.bootstrap_unit,
                "n_resamples": cfg.bootstrap_n_resamples,
                "seed": cfg.bootstrap_seed,
                "ci_level": cfg.bootstrap_ci_level,
                "apply_to": list(cfg.bootstrap_apply_to),
                "resample_bring_all_windows": True,
                "forbid_window_iid": cfg.forbid_window_iid,
            },
            "figures": {"max_count": cfg.figure_max_count,
                        "requested": list(cfg.figure_requested)},
        },
        "output": {
            "run_id_prefix": "stage7-core-csv-baseline-",
            "run_dir_isolated": True,
            "snapshot_raw_before_after": cfg.snapshot_raw_before_after,
            "historical_products_unchanged": cfg.historical_products_unchanged,
            "read_wav_forbidden": cfg.read_wav_forbidden,
            "audio_products_forbidden": cfg.audio_products_forbidden,
            "airflow_in_core_model_forbidden": cfg.airflow_in_core_model_forbidden,
            "oof_pseudonymize": cfg.oof_pseudonymize,
        },
        "package_versions": package_versions,
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


__all__ = [
    "ModelingConfigError",
    "ResolvedModelingConfig",
    "load_modeling_config",
    "write_resolved_modeling_yaml",
]
