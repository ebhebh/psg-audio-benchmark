"""Stage-2 audit manifest schemas and atomic CSV writers (prompt section 5).

Six CSV products, each with a fixed schema. Every path field is stored relative
to the project root (production) or the fixture root (tests) -- never as an
absolute or pytest-temp path. Writes are atomic (temp file + replace), mirroring
``data_download.manifest``.

Products:

1. ``patient_manifest.csv``            -- per-patient modality inventory.
2. ``file_manifest_enriched.csv``      -- Stage-1 hash manifest + classification
   + audio/CSV/annotation structure summary.
3. ``signal_availability_matrix.csv``  -- per-patient modality availability.
4. ``annotation_structure_inventory.csv`` -- per annotation file structure.
5. ``event_distribution_preliminary.csv`` -- raw event-type counts (NOT final
   labels); a README sidecar reiterates this.
6. ``patient_id_resolution_issues.csv`` -- schema-only empty table when clean.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

#: Canonical schema for each Stage-2 CSV product.
SCHEMAS: Dict[str, Tuple[str, ...]] = {
    "patient_manifest": (
        "run_id",
        "patient_id_raw",
        "patient_id_canonical",
        "source_patient_dir_relative",
        "mapping_rule_version",
        "file_count",
        "n_smartphone_audio",
        "n_recorder_audio",
        "recorder_file_count",
        "n_spo2",
        "n_heart_rate",
        "n_airflow",
        "n_sleep_structure",
        "n_annotation_json",
        "n_annotation_txt",
        "n_unknown",
        "has_smartphone_audio",
        "has_recorder_audio",
        "has_spo2",
        "has_heart_rate",
        "has_airflow",
        "has_sleep_structure",
        "has_annotation_json",
        "has_annotation_txt",
        "annotation_format",
        "audit_status",
        "issue_flags",
    ),
    "file_manifest_enriched": (
        "run_id",
        "stage1_run_id",
        "stage1_sha256",
        "relative_path",
        "patient_id_canonical",
        "file_name",
        "extension",
        "size_bytes",
        "modality_candidate",
        "classification_evidence",
        "classification_confidence",
        "recorder_index",
        "audio_container",
        "audio_sample_rate_hz",
        "audio_channels",
        "audio_sample_width_bits",
        "audio_frame_count",
        "audio_header_duration_s",
        "audio_readability_status",
        "audio_error",
        "csv_inferred_role",
        "csv_row_count",
        "csv_sample_rate_estimate_hz",
        "csv_time_column_used",
        "csv_basic_cell_missingness_ratio",
        "annotation_format",
        "annotation_events_count",
        "annotation_event_field_set",
        "annotation_field_variants",
        "annotation_raw_event_type_freq",
        "parse_status",
        "anomalies",
    ),
    "signal_availability_matrix": (
        "run_id",
        "patient_id_canonical",
        "has_smartphone_audio",
        "recorder_file_count",
        "has_recorder_audio",
        "has_spo2",
        "has_heart_rate",
        "has_airflow",
        "has_sleep_structure",
        "has_annotation",
        "annotation_format",
    ),
    "annotation_structure_inventory": (
        "run_id",
        "relative_path",
        "patient_id_canonical",
        "annotation_format",
        "annotation_top_level_keys",
        "annotation_record_start_present",
        "annotation_awake_intervals_present",
        "annotation_awake_intervals_count",
        "annotation_events_count",
        "annotation_event_field_set",
        "annotation_field_variants",
        "annotation_raw_event_type_freq",
        "annotation_anomalies",
        "annotation_preliminary_note",
    ),
    "event_distribution_preliminary": (
        "run_id",
        "patient_id_canonical",
        "relative_path",
        "event_type_raw",
        "count",
        "preliminary_note",
    ),
    "patient_id_resolution_issues": (
        "run_id",
        "issue_type",
        "patient_id",
        "source_dir_relative",
        "description",
    ),
}

#: Header line stamped on the preliminary event-distribution README.
EVENT_DIST_PRELIMINARY_NOTE = (
    "PRELIMINARY STRUCTURE AUDIT ONLY — raw event-type counts from annotation "
    "files; NOT final standardized labels and NOT a label distribution. Final "
    "event parsing/standardization belongs to a later stage."
)


def write_csv(
    rows: Iterable[Dict[str, object]], out_path: Path, fieldnames: Sequence[str]
) -> Tuple[int, Path]:
    """Write ``rows`` to ``out_path`` with the given header, atomically.

    Each row is coerced to the schema: missing keys -> "", extra keys ignored.
    Returns ``(n_rows_written, out_path)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    rows_list = list(rows)
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows_list:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp.replace(out_path)
    return len(rows_list), out_path


def write_named(name: str, rows: Iterable[Dict[str, object]], out_dir: Path) -> Path:
    """Write a named product (``name`` in :data:`SCHEMAS`) into ``out_dir``."""
    if name not in SCHEMAS:
        raise KeyError(f"Unknown Stage-2 manifest product: {name!r}")
    out_path = Path(out_dir) / f"{name}.csv"
    write_csv(rows, out_path, SCHEMAS[name])
    return out_path


def write_event_distribution_readme(out_dir: Path) -> Path:
    """Write the README sidecar that stamps the preliminary warning."""
    out_path = Path(out_dir) / "event_distribution_preliminary.README.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        EVENT_DIST_PRELIMINARY_NOTE + "\n", encoding="utf-8"
    )
    return out_path


def schema(name: str) -> Tuple[str, ...]:
    """Return the canonical field tuple for a named product."""
    return SCHEMAS[name]


def all_product_names() -> List[str]:
    return list(SCHEMAS.keys())


__all__ = [
    "SCHEMAS",
    "EVENT_DIST_PRELIMINARY_NOTE",
    "write_csv",
    "write_named",
    "write_event_distribution_readme",
    "schema",
    "all_product_names",
]
