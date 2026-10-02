"""Load and validate ``config/physiology_features.yaml`` into a resolved config.

Validation lives in code (not just YAML) so a careless edit cannot relax a hard
rule: the audio block, the half-open boundary, the record-start coordinate, the
inherited clock, the coverage >= 0.80 gate, the no-imputation/no-split policy and
the forbidden label-inputs deny-list are all enforced here. The resolved config is
written verbatim into ``physiology_features_resolved.yaml`` per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required to read the physiology feature configuration. "
        "Install it with: python -m pip install pyyaml"
    ) from _exc

from ..config import ConfigError
from .schema import (
    AUDIO_BLOCK_REASON,
    AUDIO_FEATURES_PRESENT,
    CORE_MODALITIES,
    FORBIDDEN_FEATURE_INPUTS,
    MOD_AIRFLOW,
    MOD_HEART_RATE,
    MOD_SPO2,
    ResolvedPhysiologyFeatureConfig,
)

_REQUIRED_BOUNDARY = "half_open_start_inclusive_end_exclusive"
_REQUIRED_COORDINATE = "relative_to_record_start"
_REQUIRED_CLOCK = "inherits_stage3_verified_time_axis"


class PhysiologyFeatureConfigError(ConfigError):
    """Raised when ``physiology_features.yaml`` violates a hard Stage-5 rule."""


def _get(d: Dict[str, Any], key: str, *, default=None):
    return d.get(key, default)


def load_physiology_feature_config(
    cfg_yaml_path: Path,
) -> ResolvedPhysiologyFeatureConfig:
    """Read and validate ``config/physiology_features.yaml``."""
    if not cfg_yaml_path.is_file():
        raise PhysiologyFeatureConfigError(
            f"physiology feature config not found: {cfg_yaml_path}"
        )
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise PhysiologyFeatureConfigError(
            f"Failed to parse physiology feature YAML {cfg_yaml_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or "physiology_features" not in data:
        raise PhysiologyFeatureConfigError(
            f"{cfg_yaml_path.name}: missing top-level 'physiology_features' mapping."
        )
    f = data["physiology_features"]
    if not isinstance(f, dict):
        raise PhysiologyFeatureConfigError("'physiology_features' must be a mapping.")

    # ---- boundary / coordinate / clock (hard) ----
    boundary = str(f.get("boundary", _REQUIRED_BOUNDARY))
    if boundary != _REQUIRED_BOUNDARY:
        raise PhysiologyFeatureConfigError(
            f"physiology_features.boundary must be {_REQUIRED_BOUNDARY!r} (got {boundary!r})."
        )
    coordinate = str(f.get("coordinate", _REQUIRED_COORDINATE))
    if coordinate != _REQUIRED_COORDINATE:
        raise PhysiologyFeatureConfigError(
            f"physiology_features.coordinate must be {_REQUIRED_COORDINATE!r} (got {coordinate!r})."
        )
    clock = str(f.get("cross_midnight_clock", _REQUIRED_CLOCK))
    if clock != _REQUIRED_CLOCK:
        raise PhysiologyFeatureConfigError(
            f"physiology_features.cross_midnight_clock must be {_REQUIRED_CLOCK!r}."
        )
    time_mapping = str(f.get("time_mapping", "relative_position_plus_stage3_offset"))
    if "stage3" not in time_mapping and "offset" not in time_mapping:
        raise PhysiologyFeatureConfigError(
            "physiology_features.time_mapping must inherit the Stage-3 verified clock."
        )

    # ---- forbidden label inputs (the deny-list must include the hard core) ----
    forbidden = tuple(f.get("forbidden_feature_inputs", FORBIDDEN_FEATURE_INPUTS))
    for required in FORBIDDEN_FEATURE_INPUTS:
        if required not in forbidden:
            raise PhysiologyFeatureConfigError(
                f"forbidden_feature_inputs is missing required entry {required!r}."
            )

    # ---- coverage gate (>= 0.80 for every feature modality) ----
    min_cov_raw = f.get("min_coverage", {}) or {}
    min_cov: Dict[str, float] = {}
    for mod in (MOD_HEART_RATE, MOD_SPO2, MOD_AIRFLOW):
        v = float(min_cov_raw.get(mod, 0.80))
        if not (0.0 <= v <= 1.0):
            raise PhysiologyFeatureConfigError(
                f"min_coverage.{mod} must be in [0.0, 1.0] (got {v})."
            )
        if v < 0.80:
            raise PhysiologyFeatureConfigError(
                f"min_coverage.{mod} must be >= 0.80 (got {v}); lower thresholds are forbidden."
            )
        min_cov[mod] = v

    min_stats = int(f.get("min_samples_for_statistics", 2))
    min_slope = int(f.get("min_samples_for_slope", 2))
    min_zcr = int(f.get("min_samples_for_zcr", 2))
    for name, val in (("statistics", min_stats), ("slope", min_slope), ("zcr", min_zcr)):
        if val < 2:
            raise PhysiologyFeatureConfigError(
                f"min_samples_for_{name} must be >= 2 (got {val})."
            )

    # ---- statistics ----
    std_ddof = int(f.get("std_ddof", 1))
    if std_ddof != 1:
        raise PhysiologyFeatureConfigError(
            f"std_ddof must be 1 (sample std); got {std_ddof}."
        )
    slope = f.get("slope", {}) or {}
    airflow = f.get("airflow", {}) or {}
    rms_def = str((airflow.get("rms", {}) or {}).get("definition", ""))
    zcr = airflow.get("zero_crossing", {}) or {}
    zcr_def = str(zcr.get("definition", ""))
    zcr_unit = str(zcr.get("rate_unit", "crossings_per_second"))

    # ---- SpO2 optional descriptive desaturation ----
    desat = f.get("spo2_desaturation", {}) or {}
    desat_enabled = bool(desat.get("enabled", False))
    desat_thr = float(desat.get("threshold_pct", 90.0))
    if not (0.0 <= desat_thr <= 100.0):
        raise PhysiologyFeatureConfigError(
            f"spo2_desaturation.threshold_pct must be in [0,100] (got {desat_thr})."
        )
    desat_role = str(desat.get("role", "descriptive_count_not_clinical_diagnosis"))
    if "clinical" in desat_role.lower() and "not" not in desat_role.lower():
        raise PhysiologyFeatureConfigError(
            "spo2_desaturation.role must state it is NOT a clinical diagnosis."
        )

    # ---- physiological range warnings (warning only, never clip) ----
    prw = f.get("physiological_range_warnings", {}) or {}
    prw_enabled = bool(prw.get("enabled", True))
    ranges: Dict[str, Any] = {}
    for key in ("heart_rate_bpm", "spo2_pct"):
        rng = prw.get(key)
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = float(rng[0]), float(rng[1])
            if hi <= lo:
                raise PhysiologyFeatureConfigError(
                    f"physiological_range_warnings.{key} must be [lo, hi] with hi > lo."
                )
            ranges[key] = (lo, hi)
    # airflow is unbounded; never a hard range gate.

    # ---- audio block (hard) ----
    audio = f.get("audio", {}) or {}
    if bool(audio.get("audio_features_present", False)):
        raise PhysiologyFeatureConfigError(
            "audio.audio_features_present is hard-coded False; Stage 5 extracts NO "
            "audio features (all audio remains unresolved/excluded)."
        )
    if not bool(audio.get("read_wav_forbidden", True)):
        raise PhysiologyFeatureConfigError(
            "audio.read_wav_forbidden must be True: Stage 5 may never read *.wav."
        )

    # ---- no downstream steps ----
    for key in ("no_model_matrix", "no_imputation", "no_normalization", "no_split"):
        if not bool(f.get(key, True)):
            raise PhysiologyFeatureConfigError(
                f"physiology_features.{key} must be True; Stage 5 does not perform that step."
            )

    # ---- versions ----
    fs_ver = str(f.get("feature_set_version", "physiology_baseline_v1"))
    mod_vers_raw = f.get("modality_feature_version", {}) or {}
    _default_mod_vers = {MOD_HEART_RATE: "hr_v1", MOD_SPO2: "spo2_v1", MOD_AIRFLOW: "airflow_v1"}
    mod_vers = {str(k): str(v) for k, v in mod_vers_raw.items()}
    for mod in (MOD_HEART_RATE, MOD_SPO2, MOD_AIRFLOW):
        mod_vers.setdefault(mod, _default_mod_vers[mod])

    nominal = f.get("nominal_sampling_rate_hz", {}) or {}
    nominal_hz = {
        MOD_HEART_RATE: float(nominal.get(MOD_HEART_RATE, 1.0)),
        MOD_SPO2: float(nominal.get(MOD_SPO2, 1.0)),
        MOD_AIRFLOW: float(nominal.get(MOD_AIRFLOW, 2.0)),
    }

    feature_statuses = tuple(f.get("feature_only_for_label_statuses", ("positive", "negative")))

    return ResolvedPhysiologyFeatureConfig(
        stage4_input_run_id=str(f.get("stage4_input_run_id", "")),
        stage3_input_run_id=str(f.get("stage3_input_run_id", "")),
        feature_only_for_label_statuses=feature_statuses,
        boundary=boundary,
        coordinate=coordinate,
        cross_midnight_clock=clock,
        time_mapping=time_mapping,
        forbidden_feature_inputs=forbidden,
        min_coverage=min_cov,
        min_samples_for_statistics=min_stats,
        min_samples_for_slope=min_slope,
        min_samples_for_zcr=min_zcr,
        coverage_definition=str(f.get("coverage_definition", "n_finite_over_n_expected")),
        missing_definition=str(f.get("missing_definition", "one_minus_coverage_fraction")),
        expected_samples_from=str(f.get("expected_samples_from", "stage3_inferred_sampling_rate")),
        nominal_sampling_rate_hz=nominal_hz,
        std_ddof=std_ddof,
        slope_definition=str(slope.get("definition", "ordinary_least_squares_slope_of_value_vs_window_time")),
        slope_unit=str(slope.get("unit", "value_per_second")),
        airflow_rms_definition=rms_def or "root_mean_square_of_finite_samples",
        airflow_zcr_definition=zcr_def or "consecutive_finite_sample_sign_changes",
        airflow_zcr_rate_unit=zcr_unit,
        spo2_desaturation_enabled=desat_enabled,
        spo2_desaturation_threshold_pct=desat_thr,
        spo2_desaturation_role=desat_role,
        physiological_range_warnings_enabled=prw_enabled,
        physiological_ranges=ranges,  # type: ignore[arg-type]
        audio_features_present=AUDIO_FEATURES_PRESENT,
        audio_block_reason=str(audio.get("block_reason", AUDIO_BLOCK_REASON)),
        read_wav_forbidden=True,
        no_model_matrix=True,
        no_imputation=True,
        no_normalization=True,
        no_split=True,
        feature_set_version=fs_ver,
        modality_feature_version=mod_vers,
        non_clinical_note=str(
            f.get("non_clinical_note", "research benchmark input features only; not a diagnosis")
        ),
    )


def write_resolved_config_yaml(
    path: Path,
    config: ResolvedPhysiologyFeatureConfig,
    *,
    run_id: str,
    input_stage4_run_id: str,
    input_stage3_run_id: str,
    config_hash: str,
) -> None:
    """Write the resolved config as a flat, deterministic YAML for reproducibility."""
    body = {
        "run_id": run_id,
        "input_stage4_run_id": input_stage4_run_id,
        "input_stage3_run_id": input_stage3_run_id,
        "config_hash": config_hash,
        "resolved_physiology_features": config.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle, sort_keys=True, allow_unicode=True)


__all__ = [
    "PhysiologyFeatureConfigError",
    "load_physiology_feature_config",
    "write_resolved_config_yaml",
]
