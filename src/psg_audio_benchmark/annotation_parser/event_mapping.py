"""Conservative, versioned event-type standardization (prompt 3.2).

Loads ``config/event_type_mapping.yaml`` and applies it to the raw event types
*actually observed* in the dataset. Design rules:

* Only ``hypo`` and ``osa`` were observed; they map to ``hypopnea`` and
  ``apnea_unspecified`` respectively.
* ``osa`` is deliberately mapped to ``apnea_unspecified`` (NOT
  ``obstructive_apnea``) — without the codebook or author confirmation it cannot
  distinguish obstructive / central / mixed.
* ``central`` / ``mixed`` / ``obstructive`` exist only as RESERVED placeholders
  in the config; this module never generates them from current data and rejects
  any raw type not in the observed set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from ..config import Config, ConfigError


@dataclass(frozen=True)
class EventTypeMapping:
    """Parsed event-type mapping."""

    mapping_version: str
    mapping_source: str
    observed_raw_types: Tuple[str, ...]
    #: raw -> standardized
    mapped: Dict[str, str]
    #: raw -> mapping_status
    status: Dict[str, str]
    reserved: Tuple[str, ...]
    binary_field: str
    binary_applies_to: Tuple[str, ...]

    def standardize(self, raw_type: str) -> Tuple[str, str, bool]:
        """Return ``(standardized_label, mapping_status, is_scored)``.

        Unknown raw types -> ``("unknown", "unknown_type", False)`` and are
        rejected by the parser; they never produce a subtype label.
        """
        key = (raw_type or "").strip()
        if key in self.mapped:
            return (
                self.mapped[key],
                self.status.get(key, "mapped"),
                key in self.binary_applies_to,
            )
        return "unknown", "unknown_type", False


def load_event_type_mapping(cfg: Config) -> EventTypeMapping:
    """Load and validate the versioned mapping file."""
    rel = "event_type_mapping.yaml"
    asy = cfg.config_data.get("annotation_sync", {}) or {}
    cfg_rel = asy.get("event_type_mapping_relpath")
    if cfg_rel:
        rel = str(cfg_rel)
    path: Path = cfg.config_dir / rel
    if not path.is_file():
        raise ConfigError(f"Event-type mapping file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - config guard
        raise ConfigError(f"Failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} did not parse to a mapping.")

    mapped: Dict[str, str] = {}
    status: Dict[str, str] = {}
    for raw, entry in (data.get("mapping") or {}).items():
        if not isinstance(entry, dict):
            continue
        mapped[raw] = str(entry.get("standardized", "unknown"))
        status[raw] = str(entry.get("mapping_status", "mapped"))
    observed = tuple(
        str(x) for x in (data.get("observed_raw_types") or [])
    )
    # The observed set must equal the mapped set for a self-consistent config.
    if set(observed) != set(mapped):
        raise ConfigError(
            f"{path}: observed_raw_types {sorted(observed)} != mapping keys "
            f"{sorted(mapped)}. Only observed types may be mapped."
        )
    reserved = tuple(
        str(x) for x in (data.get("reserved_for_future") or {})
    )
    binary = data.get("binary_candidate") or {}
    binary_field = str(binary.get("field", "is_any_scored_respiratory_event"))
    binary_applies = tuple(str(x) for x in (binary.get("applies_to_observed_types") or []))

    return EventTypeMapping(
        mapping_version=str(data.get("mapping_version", "unknown")),
        mapping_source=str(data.get("mapping_source", "")),
        observed_raw_types=observed,
        mapped=mapped,
        status=status,
        reserved=reserved,
        binary_field=binary_field,
        binary_applies_to=binary_applies,
    )


__all__ = ["EventTypeMapping", "load_event_type_mapping"]
