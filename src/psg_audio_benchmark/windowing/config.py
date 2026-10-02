"""Load and validate ``config/windowing.yaml`` into a resolved config.

Validation lives in code (not just YAML) so a careless edit cannot relax a hard
rule: the audio block, the half-open boundary, the non-overlapping default and
the no-split policy are all enforced here. The resolved config is written
verbatim into ``windowing_config_resolved.yaml`` per run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # PyYAML
except ImportError as _exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "PyYAML is required to read windowing configuration. "
        "Install it with: python -m pip install pyyaml"
    ) from _exc

from ..config import ConfigError
from .schema import AUDIO_BLOCK_REASON, ResolvedWindowingConfig

_REQUIRED_BOUNDARY = "half_open_start_inclusive_end_exclusive"
_REQUIRED_COORDINATE = "relative_to_record_start"
_REQUIRED_DOMAIN = "core_signal_intersection"


class WindowingConfigError(ConfigError):
    """Raised when ``windowing.yaml`` violates a hard Stage-4 rule."""


def _get(d: Dict[str, Any], key: str, *, default=None):
    return d.get(key, default)


def load_windowing_config(cfg_yaml_path: Path) -> ResolvedWindowingConfig:
    """Read and validate ``config/windowing.yaml`` -> :class:`ResolvedWindowingConfig`."""
    if not cfg_yaml_path.is_file():
        raise WindowingConfigError(f"windowing config not found: {cfg_yaml_path}")
    try:
        with cfg_yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise WindowingConfigError(
            f"Failed to parse windowing YAML {cfg_yaml_path}: {exc}"
        ) from exc
    if not isinstance(data, dict) or "windowing" not in data:
        raise WindowingConfigError(
            f"{cfg_yaml_path.name}: missing top-level 'windowing' mapping."
        )
    w = data["windowing"]
    if not isinstance(w, dict):
        raise WindowingConfigError("'windowing' must be a mapping.")

    length = float(w.get("window_length_s", 30.0))
    step = float(w.get("hop_step_s", 30.0))
    if length <= 0:
        raise WindowingConfigError("windowing.window_length_s must be positive.")
    if step <= 0:
        raise WindowingConfigError("windowing.hop_step_s must be positive.")
    if step > length:
        raise WindowingConfigError(
            "windowing.hop_step_s must not exceed window_length_s "
            "(gaps between windows are not permitted)."
        )

    boundary = w.get("boundary", _REQUIRED_BOUNDARY)
    if boundary != _REQUIRED_BOUNDARY:
        raise WindowingConfigError(
            f"windowing.boundary must be {_REQUIRED_BOUNDARY!r} (got {boundary!r})."
        )
    coordinate = w.get("coordinate", _REQUIRED_COORDINATE)
    if coordinate != _REQUIRED_COORDINATE:
        raise WindowingConfigError(
            f"windowing.coordinate must be {_REQUIRED_COORDINATE!r} (got {coordinate!r})."
        )
    domain = w.get("candidate_time_domain", _REQUIRED_DOMAIN)
    if domain != _REQUIRED_DOMAIN:
        raise WindowingConfigError(
            f"windowing.candidate_time_domain must be {_REQUIRED_DOMAIN!r}."
        )

    core = tuple(w.get("core_signal_modalities", ["heart_rate", "spo2"]))
    if set(core) != {"heart_rate", "spo2"}:
        raise WindowingConfigError(
            "windowing.core_signal_modalities must be exactly "
            "['heart_rate', 'spo2']."
        )
    optional = tuple(w.get("optional_signal_modalities", ["airflow"]))

    audio = w.get("audio", {}) or {}
    audio_eligible = bool(audio.get("audio_window_eligible", False))
    if audio_eligible:
        raise WindowingConfigError(
            "windowing.audio.audio_window_eligible is hard-coded False; no audio "
            "file carries a trustworthy time anchor (all are unresolved/excluded "
            "in Stage 3). Enabling it is forbidden."
        )
    if bool(audio.get("allow_substitute_anchors", False)):
        raise WindowingConfigError(
            "windowing.audio.allow_substitute_anchors must be False: record_start, "
            "mtime, filename, order and duration may NOT substitute for an audio "
            "time anchor."
        )
    audio_block_reason = str(audio.get("block_reason", AUDIO_BLOCK_REASON))

    max_awake = float(w.get("max_awake_overlap_fraction", 0.0))
    if max_awake < 0.0 or max_awake > 1.0:
        raise WindowingConfigError(
            "windowing.max_awake_overlap_fraction must be in [0.0, 1.0]."
        )

    do_split = bool(w.get("perform_random_window_split", False))
    emit_split = bool(w.get("emit_train_val_test_split", False))
    if do_split or emit_split:
        raise WindowingConfigError(
            "Stage 4 performs NO random window split and emits NO "
            "train/val/test split (patient-level split only, later)."
        )

    return ResolvedWindowingConfig(
        window_length_s=length,
        hop_step_s=step,
        boundary=boundary,
        coordinate=coordinate,
        candidate_time_domain=domain,
        core_signal_modalities=core,
        optional_signal_modalities=optional,
        sleep_structure_role=str(
            w.get("sleep_structure_role", "observational_context_only_not_verified")
        ),
        require_complete_core_coverage=bool(w.get("require_complete_core_coverage", True)),
        max_awake_overlap_fraction=max_awake,
        event_link_rule=str(w.get("event_link_rule", "event_start_point_half_open")),
        require_event_not_awake_overlap=bool(w.get("require_event_not_awake_overlap", True)),
        binary_event_label_field=str(w.get("binary_event_label_field", "binary_event_label")),
        non_clinical_label_note=str(
            w.get("non_clinical_label_note", "research label only; not a clinical diagnosis")
        ),
        audio_window_eligible=False,
        audio_block_reason=audio_block_reason,
        allow_substitute_anchors=False,
        perform_random_window_split=False,
        emit_train_val_test_split=False,
    )


def write_resolved_config_yaml(
    path: Path,
    config: ResolvedWindowingConfig,
    *,
    run_id: str,
    input_run_id: str,
    input_run_config_hash: str,
    config_hash: str,
) -> None:
    """Write the resolved config as a flat, deterministic YAML for reproducibility."""
    body = {
        "run_id": run_id,
        "input_stage3_run_id": input_run_id,
        "input_stage3_config_hash": input_run_config_hash,
        "config_hash": config_hash,
        "resolved_windowing": config.as_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(body, handle, sort_keys=True, allow_unicode=True)


__all__ = [
    "WindowingConfigError",
    "load_windowing_config",
    "write_resolved_config_yaml",
]
