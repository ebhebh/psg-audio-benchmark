"""Patient-directory resolution and conflict detection (Stage 2, prompt 3.1).

A patient is identified from the *most stable original signal*: the directory
name itself (the de-identified numeric folder the dataset shipped with). Patient
IDs are NEVER inferred from the OS absolute path, modification time, or an
arbitrary file sort.

Rules encoded here (prompt 3.1):

* ``patient_id_raw`` = the original directory name verbatim.
* ``patient_id_canonical`` = the same name normalised (zero-padded numeric ids
  keep their width; the raw string is preserved unchanged).
* ``source_patient_dir_relative`` is relative to the relativization base
  (project root in production -> ``data/raw/V5/Data/01``; fixture root in tests).
* Any non-continuous numbering, duplicate ids, non-patient directories, a single
  patient spanning multiple folders, or id collisions are written to
  ``patient_id_resolution_issues.csv`` -- never silently merged.

The literature-reported ``data/raw/V5/Data/01..50`` layout is treated as a
hypothesis; this code discovers patient directories from the real tree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

#: Version of the patient-id mapping rule, recorded for traceability.
MAPPING_RULE_VERSION = "dir_name_v1"

#: A plausible de-identified patient id: all digits (optionally zero-padded).
_PATIENT_ID_RE = re.compile(r"^\d+$")

#: Extensions that count as patient "data files" inside a candidate directory.
_DATA_EXTS = {".csv", ".json", ".txt", ".wav", ".mp3", ".flac"}


@dataclass
class PatientRecord:
    """One resolved patient directory."""

    patient_id_raw: str
    patient_id_canonical: str
    source_patient_dir_relative: str
    mapping_rule_version: str = MAPPING_RULE_VERSION
    confidence: str = "high"  # high | medium | low

    def as_dict(self) -> dict:
        return {
            "patient_id_raw": self.patient_id_raw,
            "patient_id_canonical": self.patient_id_canonical,
            "source_patient_dir_relative": self.source_patient_dir_relative,
            "mapping_rule_version": self.mapping_rule_version,
            "confidence": self.confidence,
        }


@dataclass
class PatientIssue:
    """A patient-id resolution problem to record (never silently fixed)."""

    issue_type: str
    patient_id: str
    source_dir_relative: str
    description: str

    def as_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "patient_id": self.patient_id,
            "source_dir_relative": self.source_dir_relative,
            "description": self.description,
        }


def _relativize(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` POSIX-style; raise if impossible.

    Mirrors the Stage-1 rule: manifest paths are never absolute or pytest-temp.
    """
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Refusing to record a non-relativizable patient path {path!s} "
            f"(base={base})."
        ) from exc
    return str(rel).replace("\\", "/")


def _has_data_files(d: Path) -> bool:
    """True if directory ``d`` directly contains at least one data file."""
    try:
        for entry in d.iterdir():
            if entry.is_file() and entry.suffix.lower() in _DATA_EXTS:
                return True
    except OSError:
        return False
    return False


def _discover_patient_dirs(raw_root: Path) -> List[Path]:
    """Find every directory that directly holds patient data files.

    Structural-only directories (e.g. ``V5``, ``Data``) contain only subdirs
    and are intentionally NOT returned, so we do not hard-code ``V5/Data``.
    """
    found: List[Path] = []
    if not raw_root.exists():
        return found
    for dirpath, dirnames, filenames in __import__("os").walk(raw_root):
        # prune noise
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        d = Path(dirpath)
        data_here = any(
            Path(dirpath, fn).suffix.lower() in _DATA_EXTS for fn in filenames
        )
        if data_here:
            found.append(d)
    return sorted(found)


def resolve_patients(
    raw_root: Path, relpath_base: Path
) -> Tuple[List[PatientRecord], List[PatientIssue]]:
    """Resolve patient directories under ``raw_root``.

    Returns ``(patients, issues)``. ``issues`` may be empty but is always a
    real list so the caller can emit a schema-only ``patient_id_resolution_issues.csv``.
    """
    patients: List[PatientRecord] = []
    issues: List[PatientIssue] = []

    dirs = _discover_patient_dirs(raw_root)

    # Group candidate dirs by canonical id to detect duplicates / multi-folder.
    by_id: dict = {}
    for d in dirs:
        raw_id = d.name
        canonical = raw_id  # keep the original name; numeric width preserved
        rel = _relativize(d, relpath_base)
        confidence = "high" if _PATIENT_ID_RE.match(raw_id) else "low"
        if confidence == "low":
            issues.append(
                PatientIssue(
                    issue_type="non_standard_patient_id",
                    patient_id=raw_id,
                    source_dir_relative=rel,
                    description=(
                        f"Directory name {raw_id!r} is not a plain numeric id; "
                        "kept verbatim but flagged for human review."
                    ),
                )
            )
        rec = PatientRecord(
            patient_id_raw=raw_id,
            patient_id_canonical=canonical,
            source_patient_dir_relative=rel,
            confidence=confidence,
        )
        patients.append(rec)
        by_id.setdefault(canonical, []).append(rec)

    # Duplicate / multi-folder: same canonical id from >1 source dir.
    for canonical, group in by_id.items():
        if len(group) > 1:
            srcs = ", ".join(g.source_patient_dir_relative for g in group)
            for g in group:
                issues.append(
                    PatientIssue(
                        issue_type="patient_id_collision_multiple_dirs",
                        patient_id=canonical,
                        source_dir_relative=g.source_patient_dir_relative,
                        description=(
                            f"Canonical patient id {canonical!r} appears in "
                            f"{len(group)} directories: {srcs}. Not merged."
                        ),
                    )
                )

    # Non-continuous numeric ids (informational; reported, not blocking).
    numeric_ids = sorted(
        int(p.patient_id_canonical)
        for p in patients
        if _PATIENT_ID_RE.match(p.patient_id_canonical)
    )
    if numeric_ids:
        full_range = set(range(numeric_ids[0], numeric_ids[-1] + 1))
        missing_nums = sorted(full_range - set(numeric_ids))
        if missing_nums:
            issues.append(
                PatientIssue(
                    issue_type="non_continuous_patient_ids",
                    patient_id=",".join(str(n) for n in missing_nums[:20]),
                    source_dir_relative="(multiple)",
                    description=(
                        f"Numeric patient ids are not continuous; missing "
                        f"{len(missing_nums)} id(s) in range "
                        f"{numeric_ids[0]}..{numeric_ids[-1]}: "
                        f"{missing_nums[:20]}."
                    ),
                )
            )

    # Deterministic order by canonical id (numeric-aware where possible).
    def _sort_key(p: PatientRecord):
        m = _PATIENT_ID_RE.match(p.patient_id_canonical)
        return (0, int(p.patient_id_canonical)) if m else (1, p.patient_id_canonical)

    patients.sort(key=_sort_key)
    return patients, issues


__all__ = [
    "MAPPING_RULE_VERSION",
    "PatientRecord",
    "PatientIssue",
    "resolve_patients",
]
