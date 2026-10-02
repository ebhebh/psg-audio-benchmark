"""Stage-8 airflow-incremental configuration loader + resolved-config writer.

Mirrors the Stage-7 loader discipline:

* the YAML is declarative;
* the immutable scientific rules (inherited v2 split, no window random split,
  audio BLOCKED, two feature sets on identical membership, fixed LR-primary /
  HGB-exploratory roles, no outer-test selection of model/feature-set/threshold,
  paired patient-cluster bootstrap) are cross-checked against
  :mod:`airflow_increment.schema` so a careless YAML edit cannot relax them;
* a frozen ``ResolvedAirflowConfig`` dataclass is produced for one run and
  written verbatim as ``airflow_incremental_resolved.yaml``.

Importing this module reads no data and writes no files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from ...config import ConfigError
from . import schema as S


class AirflowConfigError(ConfigError):
    """Raised when ``modeling_airflow_incremental.yaml`` violates a hard Stage-8 rule."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_bool(f: Dict[str, Any], key: str, *, expected: bool, where: str) -> None:
    actual = bool(f.get(key, expected))
    if actual != expected:
        raise AirflowConfigError(
            f"{where}.{key} must be {expected!r} (got {actual!r}); this hard rule "
            f"cannot be relaxed via configuration."
        )


def _require_eq(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise AirflowConfigError(
            f"{label}: YAML declares {value!r} but the immutable code constant is "
            f"{expected!r}. The code constant wins."
        )


# ---------------------------------------------------------------------------
# resolved config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedAirflowConfig:
    """Fully-resolved, immutable Stage-8 airflow-incremental configuration."""

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
    reject_v1: bool                 # read by the reused Stage-7 v2 gate
    reject_unapproved: bool
    reject_signature_mismatch: bool
    verify_v2_file_sha256: bool
    verify_assignment_sha256: bool
    verify_patient_set_sha256: bool
    verify_lineage_run_ids: bool
    runtime_reverify_cohort: bool
    runtime_reverify_airflow_cohort: bool
    require_airflow_feature_file: bool
    airflow_inherits_v2_folds: bool
    forbidden_hardcode_counts: bool
    # features
    feature_join_key: str
    feature_sets: Tuple[str, ...]
    core_feature_whitelist: Tuple[str, ...]
    enhanced_feature_whitelist: Tuple[str, ...]
    airflow_feature_whitelist: Tuple[str, ...]
    forbidden_columns: Tuple[str, ...]
    forbidden_category_tokens: Tuple[str, ...]
    # label
    label_column: str
    label_positive: int
    label_negative: int
    # nested CV (inherited)
    outer_n_folds: int
    inner_n_folds: int
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
    threshold_fit_from: str
    # evaluation / bootstrap
    threshold_free_metrics: Tuple[str, ...]
    threshold_based_metrics: Tuple[str, ...]
    single_class_policy: str
    bootstrap_enabled: bool
    bootstrap_kind: str
    bootstrap_unit: str
    bootstrap_n_resamples: int
    bootstrap_seed: int
    bootstrap_ci_level: float
    bootstrap_apply_to: Tuple[str, ...]
    bootstrap_forbid_window_iid: bool
    bootstrap_forbid_cross_cohort_pairing: bool
    bootstrap_threshold_free_delta_metrics: Tuple[str, ...]
    bootstrap_threshold_based_delta_metrics: Tuple[str, ...]
    main_comparison_model: str
    main_comparison_direction: str
    calibration_claim_calibrated: bool
    calibration_forbid_outer_test_post_calibration: bool
    figure_max_count: int
    figure_requested: Tuple[str, ...]
    # immutability / isolation
    read_wav_forbidden: bool
    audio_products_forbidden: bool
    airflow_only_in_enhanced: bool
    snapshot_raw_before_after: bool
    historical_products_unchanged: bool
    oof_pseudonymize: bool
    # raw model spec (verbatim, for the resolved YAML + audit)
    models_spec: Dict[str, Any] = field(default_factory=dict)
    selection_spec: Dict[str, Any] = field(default_factory=dict)
    threshold_spec: Dict[str, Any] = field(default_factory=dict)
    evaluation_spec: Dict[str, Any] = field(default_factory=dict)
    features_spec: Dict[str, Any] = field(default_factory=dict)
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

def load_airflow_config(cfg_yaml_path: Path) -> ResolvedAirflowConfig:
    """Load and validate ``modeling_airflow_incremental.yaml`` vs code constants."""
    if not cfg_yaml_path.is_file():
        raise AirflowConfigError(f"Airflow modeling config not found: {cfg_yaml_path}")
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise AirflowConfigError(f"Failed to parse {cfg_yaml_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise AirflowConfigError(f"{cfg_yaml_path} did not parse to a mapping.")

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
    _require_bool(out, "airflow_only_in_enhanced", expected=True, where="output")

    # --- nested CV must inherit the frozen split, never rebuild ---
    _require_bool(ncv, "inherits_split", expected=True, where="nested_cv")
    _require_bool(ncv, "rebuilds_split", expected=False, where="nested_cv")
    _require_bool(ncv, "window_random_split", expected=False, where="nested_cv")
    _require_eq(int(ncv.get("outer_n_folds")), S.OUTER_N_FOLDS, "nested_cv.outer_n_folds")
    _require_eq(int(ncv.get("inner_n_folds")), S.INNER_N_FOLDS, "nested_cv.inner_n_folds")
    _require_eq(ncv.get("unit"), S.CV_UNIT, "nested_cv.unit")

    # --- feature sets must match the code allow-lists ---
    _require_eq(feat.get("join_key"), S.FEATURE_JOIN_KEY, "features.join_key")
    sets = feat.get("sets", {}) or {}
    if set(sets.keys()) != set(S.FEATURE_SETS):
        raise AirflowConfigError(
            f"features.sets keys {sorted(sets.keys())} must be exactly "
            f"{list(S.FEATURE_SETS)}."
        )
    # core statistics (8 HR/SpO2 stats) must match Stage 7 exactly
    from ..schema import FEATURE_STATISTICS as _HR_SPO2_STATS
    core_yaml_stats = tuple(sets.get("core_restricted", {}).get("statistics", []))
    _require_eq(core_yaml_stats, _HR_SPO2_STATS, "features.sets.core_restricted.statistics")
    enh = sets.get("airflow_enhanced", {}) or {}
    air_yaml_stats = tuple(enh.get("airflow_statistics", []))
    _require_eq(air_yaml_stats, S.AIRFLOW_STATISTICS,
                "features.sets.airflow_enhanced.airflow_statistics")
    hr_yaml_stats = tuple(enh.get("heart_rate_statistics", []))
    _require_eq(hr_yaml_stats, _HR_SPO2_STATS,
                "features.sets.airflow_enhanced.heart_rate_statistics")
    spo2_yaml_stats = tuple(enh.get("spo2_statistics", []))
    _require_eq(spo2_yaml_stats, _HR_SPO2_STATS,
                "features.sets.airflow_enhanced.spo2_statistics")

    # --- selection / threshold rules must match code constants ---
    _require_eq(sel.get("primary_score"), S.SELECTION_PRIMARY_SCORE,
                "selection.primary_score")
    _require_eq(tuple(sel.get("tie_break_order", [])), S.SELECTION_TIE_BREAK,
                "selection.tie_break_order")
    _require_eq(thr.get("data_driven_rule"), S.THRESHOLD_RULE_DATA_DRIVEN,
                "threshold.data_driven_rule")
    _require_eq(thr.get("fit_from"), S.THRESHOLD_FIT_FROM, "threshold.fit_from")
    _require_bool(sel, "per_feature_set_independent", expected=True, where="selection")
    _require_bool(sel, "no_outer_test_selection", expected=True, where="selection")
    _require_bool(thr, "never_post_calibrate_on_outer_test", expected=True,
                  where="threshold")

    # --- bootstrap must be paired patient-cluster, never window IID / cross-cohort ---
    bsp = ev.get("bootstrap", {}) or {}
    _require_bool(bsp, "enabled", expected=True, where="evaluation.bootstrap")
    _require_eq(bsp.get("kind"), "paired_patient_cluster", "evaluation.bootstrap.kind")
    _require_eq(bsp.get("unit"), S.BOOTSTRAP_UNIT_PATIENT, "evaluation.bootstrap.unit")
    _require_bool(bsp, "resample_bring_all_windows", expected=True,
                  where="evaluation.bootstrap")
    _require_bool(bsp, "forbid_window_iid", expected=True, where="evaluation.bootstrap")
    _require_bool(bsp, "forbid_cross_cohort_pairing", expected=True,
                  where="evaluation.bootstrap")
    _require_eq(tuple(bsp.get("threshold_free_delta_metrics", [])),
                S.BOOTSTRAP_THRESHOLD_FREE_METRICS,
                "evaluation.bootstrap.threshold_free_delta_metrics")

    # --- calibration must not claim calibrated / must forbid outer-test post-cal ---
    cal = ev.get("calibration", {}) or {}
    _require_bool(cal, "claim_calibrated", expected=False, where="evaluation.calibration")
    _require_bool(cal, "forbid_outer_test_post_calibration", expected=True,
                  where="evaluation.calibration")

    main = ev.get("main_comparison", {}) or {}
    _require_eq(main.get("model"), S.PRIMARY_MODEL, "evaluation.main_comparison.model")

    cfg = ResolvedAirflowConfig(
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
        runtime_reverify_airflow_cohort=bool(gate.get("runtime_reverify_airflow_cohort", True)),
        require_airflow_feature_file=bool(gate.get("require_airflow_feature_file", True)),
        airflow_inherits_v2_folds=bool(gate.get("airflow_inherits_v2_folds", True)),
        forbidden_hardcode_counts=bool(gate.get("forbidden_hardcode_counts", True)),
        feature_join_key=S.FEATURE_JOIN_KEY,
        feature_sets=S.FEATURE_SETS,
        core_feature_whitelist=S.CORE_FEATURE_WHITELIST,
        enhanced_feature_whitelist=S.ENHANCED_FEATURE_WHITELIST,
        airflow_feature_whitelist=S.AIRFLOW_FEATURE_WHITELIST,
        forbidden_columns=S.FORBIDDEN_COLUMNS,
        forbidden_category_tokens=S.FORBIDDEN_CATEGORY_TOKENS,
        label_column=str(lab.get("column", S.LABEL_COLUMN)),
        label_negative=int((lab.get("values", {}) or {}).get("negative", 0)),
        label_positive=int((lab.get("values", {}) or {}).get("positive", 1)),
        outer_n_folds=S.OUTER_N_FOLDS,
        inner_n_folds=S.INNER_N_FOLDS,
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
        threshold_fit_from=S.THRESHOLD_FIT_FROM,
        threshold_free_metrics=tuple(ev.get("threshold_free_metrics", [])),
        threshold_based_metrics=tuple(ev.get("threshold_based_metrics", [])),
        single_class_policy=str(ev.get("single_class_policy", "na_with_reason")),
        bootstrap_enabled=True,
        bootstrap_kind="paired_patient_cluster",
        bootstrap_unit=S.BOOTSTRAP_UNIT_PATIENT,
        bootstrap_n_resamples=int(bsp.get("n_resamples", S.BOOTSTRAP_DEFAULT_N)),
        bootstrap_seed=int(bsp.get("seed", S.BOOTSTRAP_DEFAULT_SEED)),
        bootstrap_ci_level=float(bsp.get("ci_level", S.BOOTSTRAP_CI_LEVEL)),
        bootstrap_apply_to=tuple(bsp.get("apply_to", [S.PRIMARY_MODEL])),
        bootstrap_forbid_window_iid=True,
        bootstrap_forbid_cross_cohort_pairing=True,
        bootstrap_threshold_free_delta_metrics=S.BOOTSTRAP_THRESHOLD_FREE_METRICS,
        bootstrap_threshold_based_delta_metrics=S.BOOTSTRAP_THRESHOLD_BASED_METRICS,
        main_comparison_model=S.PRIMARY_MODEL,
        main_comparison_direction="enhanced_minus_core",
        calibration_claim_calibrated=False,
        calibration_forbid_outer_test_post_calibration=True,
        figure_max_count=int((ev.get("figures", {}) or {}).get("max_count", 5)),
        figure_requested=tuple((ev.get("figures", {}) or {}).get("requested", [])),
        read_wav_forbidden=True,
        audio_products_forbidden=True,
        airflow_only_in_enhanced=True,
        snapshot_raw_before_after=bool(out.get("snapshot_raw_before_after", True)),
        historical_products_unchanged=bool(out.get("historical_products_unchanged", True)),
        oof_pseudonymize=bool(out.get("oof_pseudonymize", True)),
        models_spec=dict(raw.get("models", {}) or {}),
        selection_spec=dict(sel),
        threshold_spec=dict(thr),
        evaluation_spec=dict(ev),
        features_spec=dict(feat),
        raw_yaml=dict(raw),
    )
    return cfg


# ---------------------------------------------------------------------------
# resolved-config writer
# ---------------------------------------------------------------------------

def write_resolved_airflow_yaml(
    *, cfg: ResolvedAirflowConfig, run_id: str, config_hash: str, path: Path,
    split_run_id: str, split_signature: Dict[str, Any],
    package_versions: Dict[str, str], airflow_cohort: Dict[str, Any],
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
        "frozen_core_cohort_fingerprint": {
            "n_patients": sig_cohort.get("n_patients"),
            "n_windows": sig_cohort.get("n_windows"),
            "n_positive": sig_cohort.get("n_positive"),
            "n_negative": sig_cohort.get("n_negative"),
            "overall_positive_rate": sig_cohort.get("overall_positive_rate"),
            "note": "the 50-patient core cohort from the approved v2 signature",
        },
        "runtime_airflow_cohort_fingerprint": {
            "n_patients": airflow_cohort.get("n_patients"),
            "n_windows": airflow_cohort.get("n_windows"),
            "n_positive": airflow_cohort.get("n_positive"),
            "n_negative": airflow_cohort.get("n_negative"),
            "overall_positive_rate": airflow_cohort.get("overall_positive_rate"),
            "note": ("re-derived at runtime: HR & SpO2 & airflow all available, "
                     "positive/negative windows; never hard-coded"),
        },
        "nested_cv": {
            "outer_n_folds": cfg.outer_n_folds,
            "inner_n_folds": cfg.inner_n_folds,
            "unit": cfg.cv_unit,
            "inherits_split": cfg.inherits_split,
            "rebuilds_split": cfg.rebuilds_split,
            "window_random_split": cfg.window_random_split,
            "outer_assignment_source": cfg.outer_assignment_source,
            "inner_assignment_source": cfg.inner_assignment_source,
            "airflow_inherits_v2_folds": True,
        },
        "features": {
            "sets": {
                "core_restricted": {
                    "role": S.FEATURE_SET_ROLE[S.FEATURE_SET_CORE],
                    "whitelist": list(S.CORE_FEATURE_WHITELIST),
                    "n_columns": len(S.CORE_FEATURE_WHITELIST),
                },
                "airflow_enhanced": {
                    "role": S.FEATURE_SET_ROLE[S.FEATURE_SET_ENHANCED],
                    "whitelist": list(S.ENHANCED_FEATURE_WHITELIST),
                    "n_columns": len(S.ENHANCED_FEATURE_WHITELIST),
                    "airflow_columns": list(S.AIRFLOW_FEATURE_WHITELIST),
                },
            },
            "join_key": cfg.feature_join_key,
            "membership_identical": True,
            "forbidden_columns": list(cfg.forbidden_columns),
            "forbidden_category_tokens": list(cfg.forbidden_category_tokens),
        },
        "label": {
            "column": cfg.label_column, "negative": cfg.label_negative,
            "positive": cfg.label_positive,
        },
        "models": cfg.models_spec,
        "selection": {
            "primary_score": cfg.selection_primary_score,
            "tie_break_order": list(cfg.selection_tie_break),
            "per_feature_set_independent": True,
            "no_outer_test_selection": True,
            "record_all_candidates": True,
            "refit_on_full_outer_train": True,
            "predict_outer_test_once": True,
        },
        "threshold": {
            "data_driven_rule": cfg.threshold_rule_data_driven,
            "fit_from": cfg.threshold_fit_from,
            "never_fit_from": "outer_test",
            "never_post_calibrate_on_outer_test": True,
        },
        "evaluation": {
            "threshold_free_metrics": list(cfg.threshold_free_metrics),
            "threshold_based_metrics": list(cfg.threshold_based_metrics),
            "single_class_policy": cfg.single_class_policy,
            "pooled_oof": True,
            "paired_same_window": True,
            "main_comparison": {
                "model": cfg.main_comparison_model,
                "delta": "airflow_enhanced_minus_core_restricted",
                "direction": cfg.main_comparison_direction,
            },
            "bootstrap": {
                "enabled": cfg.bootstrap_enabled,
                "kind": cfg.bootstrap_kind,
                "unit": cfg.bootstrap_unit,
                "n_resamples": cfg.bootstrap_n_resamples,
                "seed": cfg.bootstrap_seed,
                "ci_level": cfg.bootstrap_ci_level,
                "apply_to": list(cfg.bootstrap_apply_to),
                "resample_bring_all_windows": True,
                "forbid_window_iid": cfg.bootstrap_forbid_window_iid,
                "forbid_cross_cohort_pairing": cfg.bootstrap_forbid_cross_cohort_pairing,
                "threshold_free_delta_metrics":
                    list(cfg.bootstrap_threshold_free_delta_metrics),
                "threshold_based_delta_metrics":
                    list(cfg.bootstrap_threshold_based_delta_metrics),
            },
            "calibration": {
                "report_raw_brier_logloss": True,
                "claim_calibrated": cfg.calibration_claim_calibrated,
                "forbid_outer_test_post_calibration":
                    cfg.calibration_forbid_outer_test_post_calibration,
            },
            "figures": {"max_count": cfg.figure_max_count,
                        "requested": list(cfg.figure_requested)},
        },
        "output": {
            "run_id_prefix": "stage8-airflow-incremental-",
            "run_dir_isolated": True,
            "snapshot_raw_before_after": cfg.snapshot_raw_before_after,
            "historical_products_unchanged": cfg.historical_products_unchanged,
            "read_wav_forbidden": cfg.read_wav_forbidden,
            "audio_products_forbidden": cfg.audio_products_forbidden,
            "airflow_only_in_enhanced": cfg.airflow_only_in_enhanced,
            "oof_pseudonymize": cfg.oof_pseudonymize,
        },
        "package_versions": package_versions,
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


__all__ = [
    "AirflowConfigError",
    "ResolvedAirflowConfig",
    "load_airflow_config",
    "write_resolved_airflow_yaml",
]
