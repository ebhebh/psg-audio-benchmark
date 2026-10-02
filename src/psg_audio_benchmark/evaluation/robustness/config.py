"""Stage-9 resolved robustness/sensitivity config + loader + writer.

The YAML in ``config/robustness_sensitivity.yaml`` is parsed and cross-checked
against :mod:`robustness.schema`; the code constant always wins. The resolved
config is immutable for the run and is persisted to
``models/runs/<run-id>/sensitivity_resolved.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from ...config import ConfigError
from . import schema as S


class RobustnessConfigError(ConfigError):
    """Raised when the resolved config contradicts the code-authoritative rules."""


def _require_eq(section: str, key: str, got: Any, want: Any) -> None:
    if got != want:
        raise RobustnessConfigError(
            f"config {section}.{key} = {got!r} contradicts the required value {want!r}"
        )


def _require_bool(section: str, key: str, got: Any) -> None:
    if not isinstance(got, bool):
        raise RobustnessConfigError(
            f"config {section}.{key} must be a boolean, got {got!r}"
        )


@dataclass(frozen=True)
class ResolvedRobustnessConfig:
    # identity / framing
    stage: str
    modeling_version: str
    task_name: str
    cohort: str
    split_balance_version: str
    approval_required: str
    non_clinical_note: str
    # gate
    require_license_passed: bool
    require_latest_run_v2: bool
    require_approval_status: bool
    reject_v1: bool
    reject_unapproved: bool
    reject_signature_mismatch: bool
    verify_v2_file_sha256: bool
    verify_assignment_sha256: bool
    verify_patient_set_sha256: bool
    verify_lineage_run_ids: bool
    runtime_reverify_cohort: bool
    runtime_reverify_airflow_cohort: bool
    forbidden_hardcode_counts: bool
    # frozen inputs (verified, never trusted blindly)
    stage7_run_id: str
    stage8_run_id: str
    expected_consumed_split_run_id: str
    expected_assignment_sha256: str
    expected_patient_set_sha256: str
    # nested CV (inherited)
    outer_n_folds: int
    inner_n_folds: int
    cv_unit: str
    inherits_split: bool
    rebuilds_split: bool
    window_random_split: bool
    # analysis toggles + params
    primary_reproduction: Dict[str, Any]
    patient_weight_uncertainty: Dict[str, Any]
    threshold_sensitivity: Dict[str, Any]
    calibration_sensitivity: Dict[str, Any]
    class_weight_sensitivity: Dict[str, Any]
    airflow_selection_profile: Dict[str, Any]
    alternative_label: Dict[str, Any]
    # isolation
    read_wav_forbidden: bool
    audio_products_forbidden: bool
    airflow_in_core_model_forbidden: bool
    snapshot_raw_before_after: bool
    historical_products_unchanged: bool
    primary_results_read_only: bool
    output_run_dir_isolated: bool
    # output
    run_id_prefix: str
    figure_max_count: int
    oof_pseudonymize: bool
    # raw spec dump (for the resolved yaml)
    raw_yaml: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        import dataclasses

        out = dataclasses.asdict(self)
        return out


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    sec = raw.get(name, {})
    if not isinstance(sec, dict):
        raise RobustnessConfigError(f"config section {name!r} must be a mapping")
    return sec


def load_robustness_config(cfg_yaml_path: Path) -> ResolvedRobustnessConfig:
    raw = yaml.safe_load(Path(cfg_yaml_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RobustnessConfigError("robustness config root must be a mapping")

    meta = _section(raw, "meta")
    _require_eq("meta", "stage", meta.get("stage"), S.ROBUSTNESS_STAGE)
    _require_eq("meta", "modeling_version", meta.get("modeling_version"), S.MODELING_VERSION)
    _require_eq("meta", "split_balance_version", meta.get("split_balance_version"),
                S.SPLIT_BALANCE_VERSION)
    _require_eq("meta", "approval_required", meta.get("approval_required"), S.APPROVAL_REQUIRED)

    gate = _section(raw, "gate")
    for k in ("require_license_passed", "require_latest_run_v2", "require_approval_status",
              "reject_v1", "reject_unapproved", "reject_signature_mismatch",
              "verify_v2_file_sha256", "verify_assignment_sha256",
              "verify_patient_set_sha256", "verify_lineage_run_ids",
              "runtime_reverify_cohort", "runtime_reverify_airflow_cohort",
              "forbidden_hardcode_counts"):
        _require_bool("gate", k, gate.get(k))

    frozen = _section(raw, "frozen_inputs")
    for k in ("stage7_run_id", "stage8_run_id", "expected_consumed_split_run_id",
              "expected_assignment_sha256", "expected_patient_set_sha256"):
        if not isinstance(frozen.get(k), str) or not frozen.get(k):
            raise RobustnessConfigError(f"frozen_inputs.{k} must be a non-empty string")

    nested = _section(raw, "nested_cv")
    _require_eq("nested_cv", "unit", nested.get("unit"), S.CV_UNIT)
    _require_bool("nested_cv", "inherits_split", nested.get("inherits_split"))
    _require_bool("nested_cv", "rebuilds_split", nested.get("rebuilds_split"))
    _require_bool("nested_cv", "window_random_split", nested.get("window_random_split"))
    if nested.get("inherits_split") is not True:
        raise RobustnessConfigError("nested_cv.inherits_split must be true (frozen split)")
    if nested.get("rebuilds_split") is not False:
        raise RobustnessConfigError("nested_cv.rebuilds_split must be false")
    if nested.get("window_random_split") is not False:
        raise RobustnessConfigError("nested_cv.window_random_split must be false")
    _require_eq("nested_cv", "outer_n_folds", int(nested.get("outer_n_folds")), S.OUTER_N_FOLDS)
    _require_eq("nested_cv", "inner_n_folds", int(nested.get("inner_n_folds")), S.INNER_N_FOLDS)

    analyses = _section(raw, "analyses")
    for key in S.ALL_ANALYSES:
        if key not in analyses or not isinstance(analyses[key], dict):
            raise RobustnessConfigError(f"analyses.{key} must be present as a mapping")
        _require_bool(f"analyses.{key}", "enabled", analyses[key].get("enabled"))

    # patient-cluster bootstrap contract (window-IID forbidden)
    pw = analyses[S.A_PATIENT_WEIGHT]
    _require_eq("analyses.patient_weight_uncertainty", "bootstrap_unit",
                pw.get("bootstrap_unit"), S.BOOTSTRAP_UNIT_PATIENT)
    _require_bool("analyses.patient_weight_uncertainty", "forbid_window_iid",
                  pw.get("forbid_window_iid"))
    _require_bool("analyses.patient_weight_uncertainty", "resample_bring_all_windows",
                  pw.get("resample_bring_all_windows"))
    if pw.get("forbid_window_iid") is not True:
        raise RobustnessConfigError("window-IID bootstrap is forbidden")

    # threshold sensitivity: main rule unchanged; fit from inner OOF only
    ts = analyses[S.A_THRESHOLD]
    _require_eq("analyses.threshold_sensitivity", "primary_rule",
                ts.get("primary_rule"), S.THRESHOLD_YOUDEN)
    _require_eq("analyses.threshold_sensitivity", "fit_from",
                ts.get("fit_from"), S.THRESHOLD_FIT_FROM)
    _require_eq("analyses.threshold_sensitivity", "never_fit_from",
                ts.get("never_fit_from"), "outer_test")

    # calibration: fit from inner OOF only; never reselect model
    cal = analyses[S.A_CALIBRATION]
    _require_eq("analyses.calibration_sensitivity", "fit_from",
                cal.get("fit_from"), S.THRESHOLD_FIT_FROM)
    _require_bool("analyses.calibration_sensitivity", "do_not_reselect_model",
                  cal.get("do_not_reselect_model"))

    # class weight: LR only
    cw = analyses[S.A_CLASS_WEIGHT]
    _require_eq("analyses.class_weight_sensitivity", "model",
                cw.get("model"), S.MODEL_LOGISTIC_REGRESSION)

    # airflow: must not extrapolate to main cohort
    ap = analyses[S.A_AIRFLOW_PROFILE]
    _require_bool("analyses.airflow_selection_profile",
                  "do_not_extrapolate_to_main_cohort",
                  ap.get("do_not_extrapolate_to_main_cohort"))
    _require_eq("analyses.airflow_selection_profile", "subcohort_patients",
                int(ap.get("subcohort_patients")), 34)

    # alternative label: structured events only; do not change main label
    al = analyses[S.A_ALTERNATIVE_LABEL]
    _require_eq("analyses.alternative_label", "definition",
                al.get("definition"), S.ALT_LABEL_DEFINITION)
    _require_bool("analyses.alternative_label", "same_window_membership_as_primary",
                  al.get("same_window_membership_as_primary"))
    _require_bool("analyses.alternative_label", "same_frozen_split",
                  al.get("same_frozen_split"))
    _require_bool("analyses.alternative_label", "mirror_awake_setaside",
                  al.get("mirror_awake_setaside"))
    _require_bool("analyses.alternative_label",
                  "do_not_use_to_select_better_main_label",
                  al.get("do_not_use_to_select_better_main_label"))

    iso = _section(raw, "isolation")
    for k in ("read_wav_forbidden", "audio_products_forbidden",
              "airflow_in_core_model_forbidden", "snapshot_raw_before_after",
              "historical_products_unchanged", "primary_results_read_only",
              "output_run_dir_isolated"):
        _require_bool("isolation", k, iso.get(k))
    if iso.get("read_wav_forbidden") is not True:
        raise RobustnessConfigError("isolation.read_wav_forbidden must be true")

    out = _section(raw, "output")
    if not isinstance(out.get("run_id_prefix"), str) or not out.get("run_id_prefix"):
        raise RobustnessConfigError("output.run_id_prefix must be a non-empty string")
    fig_max = int(out.get("figure_max_count", 6))
    if fig_max < 1 or fig_max > 6:
        raise RobustnessConfigError("output.figure_max_count must be between 1 and 6")

    return ResolvedRobustnessConfig(
        stage=S.ROBUSTNESS_STAGE, modeling_version=S.MODELING_VERSION,
        task_name=S.TASK_NAME, cohort=S.COHORT,
        split_balance_version=S.SPLIT_BALANCE_VERSION,
        approval_required=S.APPROVAL_REQUIRED, non_clinical_note=S.NON_CLINICAL_NOTE,
        require_license_passed=gate["require_license_passed"],
        require_latest_run_v2=gate["require_latest_run_v2"],
        require_approval_status=gate["require_approval_status"],
        reject_v1=gate["reject_v1"], reject_unapproved=gate["reject_unapproved"],
        reject_signature_mismatch=gate["reject_signature_mismatch"],
        verify_v2_file_sha256=gate["verify_v2_file_sha256"],
        verify_assignment_sha256=gate["verify_assignment_sha256"],
        verify_patient_set_sha256=gate["verify_patient_set_sha256"],
        verify_lineage_run_ids=gate["verify_lineage_run_ids"],
        runtime_reverify_cohort=gate["runtime_reverify_cohort"],
        runtime_reverify_airflow_cohort=gate["runtime_reverify_airflow_cohort"],
        forbidden_hardcode_counts=gate["forbidden_hardcode_counts"],
        stage7_run_id=frozen["stage7_run_id"],
        stage8_run_id=frozen["stage8_run_id"],
        expected_consumed_split_run_id=frozen["expected_consumed_split_run_id"],
        expected_assignment_sha256=frozen["expected_assignment_sha256"],
        expected_patient_set_sha256=frozen["expected_patient_set_sha256"],
        outer_n_folds=S.OUTER_N_FOLDS, inner_n_folds=S.INNER_N_FOLDS,
        cv_unit=S.CV_UNIT, inherits_split=True, rebuilds_split=False,
        window_random_split=False,
        primary_reproduction=analyses[S.A_PRIMARY_REPRODUCTION],
        patient_weight_uncertainty=analyses[S.A_PATIENT_WEIGHT],
        threshold_sensitivity=analyses[S.A_THRESHOLD],
        calibration_sensitivity=analyses[S.A_CALIBRATION],
        class_weight_sensitivity=analyses[S.A_CLASS_WEIGHT],
        airflow_selection_profile=analyses[S.A_AIRFLOW_PROFILE],
        alternative_label=analyses[S.A_ALTERNATIVE_LABEL],
        read_wav_forbidden=iso["read_wav_forbidden"],
        audio_products_forbidden=iso["audio_products_forbidden"],
        airflow_in_core_model_forbidden=iso["airflow_in_core_model_forbidden"],
        snapshot_raw_before_after=iso["snapshot_raw_before_after"],
        historical_products_unchanged=iso["historical_products_unchanged"],
        primary_results_read_only=iso["primary_results_read_only"],
        output_run_dir_isolated=iso["output_run_dir_isolated"],
        run_id_prefix=out["run_id_prefix"], figure_max_count=fig_max,
        oof_pseudonymize=bool(out.get("oof_pseudonymize", True)),
        raw_yaml=raw,
    )


def write_sensitivity_resolved_yaml(
    *, cfg: ResolvedRobustnessConfig, run_id: str, config_hash: str, path: Path,
    split_run_id: str, split_signature: Dict[str, Any], package_versions: Dict[str, str],
    stage7_signature: Dict[str, Any], stage8_signature: Dict[str, Any],
    determinism: Dict[str, Any],
) -> None:
    """Persist the resolved config + frozen-input fingerprints to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "run_id": run_id,
        "config_hash": config_hash,
        "modeling_version": cfg.modeling_version,
        "stage": cfg.stage,
        "task_name": cfg.task_name,
        "cohort": cfg.cohort,
        "non_clinical_note": cfg.non_clinical_note,
        "split_balance_version": cfg.split_balance_version,
        "consumed_split_run_id": split_run_id,
        "approval_status": (split_signature or {}).get("approval_status"),
        "split_assignment_sha256": (split_signature or {}).get("split", {}).get("assignment_sha256"),
        "split_patient_set_sha256": (split_signature or {}).get("cohort", {}).get("patient_set_sha256"),
        "frozen_inputs": {
            "stage7_run_id": cfg.stage7_run_id,
            "stage8_run_id": cfg.stage8_run_id,
            "expected_consumed_split_run_id": cfg.expected_consumed_split_run_id,
        },
        "stage7_signature": stage7_signature,
        "stage8_signature": stage8_signature,
        "nested_cv": {
            "outer_n_folds": cfg.outer_n_folds, "inner_n_folds": cfg.inner_n_folds,
            "unit": cfg.cv_unit, "inherits_split": True, "rebuilds_split": False,
            "window_random_split": False,
        },
        "analyses": {
            "primary_reproduction": cfg.primary_reproduction,
            "patient_weight_uncertainty": cfg.patient_weight_uncertainty,
            "threshold_sensitivity": cfg.threshold_sensitivity,
            "calibration_sensitivity": cfg.calibration_sensitivity,
            "class_weight_sensitivity": cfg.class_weight_sensitivity,
            "airflow_selection_profile": cfg.airflow_selection_profile,
            "alternative_label": cfg.alternative_label,
        },
        "rerun_determinism": determinism,
        "audio_block": {
            "audio_cohort_count": S.AUDIO_COHORT_COUNT,
            "read_wav_forbidden": True,
            "audio_features_present": False,
        },
        "package_versions": package_versions,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


__all__ = [
    "RobustnessConfigError",
    "ResolvedRobustnessConfig",
    "load_robustness_config",
    "write_sensitivity_resolved_yaml",
]
