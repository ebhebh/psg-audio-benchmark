#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safe Mode-B run-ID binding: per-stage entry (Stage-26 fix).

WHY THIS EXISTS
    The frozen ``config/*.yaml`` files and the ``--input-*-run-id`` script
    defaults record the AUTHORS' historical run ids as provenance pins. A
    reproducer's own Mode-B regeneration produces NEW run ids, and the chain
    must be advanced ONE stage at a time: after each upstream stage completes,
    bind YOUR validated run ids and get the exact next-stage command. The
    Stage-25 version of this tool validated the WHOLE chain (Stage 3/4/5/6b)
    before printing any command, which made the FIRST reproduction step
    (binding Stage 5 after only Stage 3/4 exist) impossible — a dependency
    cycle. This version takes ``--next-stage`` and requires ONLY the upstream
    products that stage genuinely needs:

        --next-stage 3   license gate only (chain start; Stage 3 has no
                         upstream run-id input)
        --next-stage 4   + YOUR Stage-3 run (pointer, run-dir, products)
        --next-stage 5   + YOUR Stage-4 run (and Stage-4's resolved config
                         must cite your Stage-3 run)
        --next-stage 6   + YOUR Stage-5 run; the Stage-5 resolved config MUST
                         cite your Stage-4 AND Stage-3 runs (non-empty;
                         empty lineage fields are refused, not skipped)
        --next-stage 6b  + YOUR Stage-6 v1 run (the comparator Stage 6b
                         re-partitions); its input_version_signature.json must
                         cite your Stage-4/5 runs. Prints 06b with
                         ``--v1-comparator-run-id <your stage-6 run>`` (the
                         frozen config pin is never edited)
        --next-stage 15  + YOUR APPROVED Stage-6b run: approval status, signed
                         v2 product hashes in BOTH the run dir and the fixed
                         mirror, run-dir/mirror signature identity, cohort
                         fingerprint, transitive 3/4/5 lineage, and the fixed
                         mirrors Stage 15b actually reads
        --next-stage all legacy whole-chain validation (3/4/5/6b + all gates)

    Stage 15 note (honest, verified): ``scripts/15_bspc_reanalysis.py`` reads
    the fixed mirrors (``splits/``, ``annotations/``,
    ``features/physiology/``) directly; the ``provenance_gate.split_run`` pin
    in ``config/bspc_revision_analysis.yaml`` is DECLARATIVE provenance, NOT a
    runtime gate in this release (pinned by
    ``tests/test_bind_run_ids_staged.py::test_stage15_provenance_gate_is_declarative``).
    The executable gate for Stage 15 is therefore the mirror/signature
    verification this tool performs for ``--next-stage 15``.

WHAT IT NEVER DOES
    * It never edits a frozen config, a signature file, or any approval record
      (the frozen provenance pins stay untouched as read-only provenance).
    * It never marks anything approved: a Stage-6b split that did not reach
      ``approved_for_modeling`` fails the ``--next-stage 15`` binding. That
      gate is not bypassable here — rerun Stage 6b or investigate; never
      hand-edit statuses.
    * It never writes outside ``reports/run_bindings/`` (git-ignored audit
      record; use --write-bindings to create it).
    * It never reads outside the project root: run ids must be a single safe
      path segment (letters/digits/``.``/``_``/``+``/``-`` only) and signature
      ``relpath`` values are validated the same way before any file is opened,
      so absolute paths, ``..`` traversal and shell metacharacters are refused
      before they can reach a path or a printed command.

FAIL-CLOSED REQUIRED FIELDS (missing/empty = refused, never silently skipped):
    license: the CURRENT evidence file must pass the real Stage-2 license gate
    (same code path as ``scripts/02_data_audit.py``) AND the
    ``license_gate_passed.md`` record must exist;
    stage5 lineage: ``input_stage3_run_id``/``input_stage4_run_id`` non-empty;
    6b signature: ``run_id``, ``approval_status``, non-empty ``v2_files``,
    complete ``cohort`` block, non-empty 3/4/5 ``lineage`` ids.

USAGE (from the repository root, after each upstream stage finishes)
    python tools/bind_run_ids.py --next-stage 5     # validate + print command
    python tools/bind_run_ids.py --next-stage 5 --write-bindings

EXIT CODES: 0 = gates pass and the command is printed; 2 = any gate failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark.data_audit.license_gate import (  # noqa: E402
    check_license_gate as _check_license_gate,
)

# Pointer/pattern/product map for the chain stages (all fixed mirror paths
# written by the production runners; see each script's docstring).
STAGES = {
    "stage3": {
        "pointer": "annotations/LATEST_RUN.txt",
        "prefix": "stage3-annotation-sync-",
        "run_root": "annotations/runs",
        # the exact artifact set the REAL Stage-4 CLI gate requires
        # (windowing.runner._REQUIRED_INPUT_ARTIFACTS)
        "products": ["parsed_events.parquet", "awake_intervals_canonical.parquet",
                     "signal_time_ranges.parquet", "record_time_anchors.parquet",
                     "audio_time_alignment_inventory.csv"],
    },
    "stage4": {
        "pointer": "reports/windowing/LATEST_RUN.txt",
        "prefix": "stage4-csv-window-index-",
        "run_root": "data/manifests/runs",
        "products": ["csv_window_index.parquet", "patient_window_summary.csv",
                     "windowing_config_resolved.yaml"],
    },
    "stage5": {
        "pointer": "reports/feature_extraction/LATEST_RUN.txt",
        "prefix": "stage5-physiology-features-",
        "run_root": "features/physiology/runs",
        "products": ["hr_window_features.parquet", "spo2_window_features.parquet",
                     "physiology_features_resolved.yaml"],
    },
    "stage6": {
        "pointer": "reports/evaluation/LATEST_RUN.txt",
        "prefix": "stage6-patient-splits-",
        "run_root": "splits/runs",
        "products": ["outer_patient_folds_core.csv",
                     "airflow_outer_fold_inheritance.csv",
                     "input_version_signature.json"],
    },
    "stage6b": {
        "pointer": "splits/LATEST_RUN.txt",
        "prefix": "stage6b-split-balance-v2-",
        "run_root": "splits/runs",
        "products": ["split_balance_version_signature.json"],
    },
}
APPROVED = "approved_for_modeling"

NEXT_STAGES = ("3", "4", "5", "6", "6b", "15", "all")

# Upstream pointer-stages each --next-stage entry genuinely requires.
REQUIREMENTS = {
    "3": (),
    "4": ("stage3",),
    "5": ("stage3", "stage4"),
    "6": ("stage3", "stage4", "stage5"),
    "6b": ("stage3", "stage4", "stage5", "stage6"),
    "15": ("stage3", "stage4", "stage5", "stage6b"),
    "all": ("stage3", "stage4", "stage5", "stage6b"),
}

# A run id / signature relpath segment must be a single safe path segment:
# alnum start, then alnum . _ + - only. This rejects absolute paths, drive
# letters, '/', '\', '..', spaces, quotes and every shell metacharacter BEFORE
# the value is used in a path or printed in a command.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def _is_safe_segment(value: str) -> bool:
    return bool(value) and ".." not in value and bool(_SAFE_SEGMENT.match(value))


def _safe_relpath(rel: str) -> bool:
    """A signature ``relpath``: 1-2 safe posix segments, relative, no escape."""
    if not isinstance(rel, str) or not rel:
        return False
    if "\\" in rel or ":" in rel:
        return False
    p = PurePosixPath(rel)
    if p.is_absolute():
        return False
    parts = p.parts
    if not 1 <= len(parts) <= 2:
        return False
    return all(_is_safe_segment(seg) for seg in parts)


def _q(value: str) -> str:
    """Quote a validated id for shell display (defensive; ids are pre-checked)."""
    return '"' + value + '"'


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _DocsOnlyCfg:
    """Minimal cfg shim so the REAL Stage-2 license gate runs against an
    arbitrary project root (it only needs ``path('docs')`` and project_root).
    Deliberately a plain class (not a dataclass) so the tool loads under any
    import mechanism (importlib spec loading without sys.modules registration)."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def path(self, key: str) -> Path:
        if key != "docs":
            raise KeyError(key)
        return self.project_root / "docs"


def _license_problems(root: Path, bindings: dict, problems: list) -> None:
    """License gate: the CURRENT evidence file must pass the REAL Stage-2 gate
    (complete non-placeholder fields, DOI match, affirmative basis) AND the
    passed-record mirror must exist. A stale passed record whose evidence was
    later removed/reverted is refused."""
    passed_record = root / "reports/data_audit/license_gate_passed.md"
    bindings["license_gate_passed_record"] = passed_record.is_file()
    if not passed_record.is_file():
        problems.append(
            "license gate: reports/data_audit/license_gate_passed.md absent — "
            "complete docs/license_evidence/ (template + README) and rerun "
            "scripts/02_data_audit.py until the gate PASSES")
    gate = _check_license_gate(_DocsOnlyCfg(project_root=root))
    bindings["license_gate_current_status"] = gate.status
    bindings["license_gate_evidence_relpath"] = gate.evidence_relpath
    if gate.status != "passed":
        problems.append(
            f"license gate: CURRENT evidence re-check {gate.status!r} "
            f"({', '.join(gate.missing_items) or 'not affirmative'}) — the "
            "passed record alone is not sufficient; fix "
            f"{gate.evidence_relpath} and rerun scripts/02_data_audit.py")


def _read_pointer(root: Path, rel: str, problems: list) -> str:
    p = root / rel
    if not p.is_file():
        problems.append(f"pointer missing: {rel} (run the corresponding stage first)")
        return ""
    val = p.read_text(encoding="utf-8").strip()
    if not val:
        problems.append(f"pointer empty: {rel}")
        return ""
    if "\n" in val or not _is_safe_segment(val):
        problems.append(
            f"unsafe run id in {rel}: {val!r} is not a single safe path "
            "segment (absolute paths, '..', separators, spaces, quotes and "
            "shell metacharacters are refused; nothing was read or printed)")
        return ""
    return val


def _validate_pointer_stage(root: Path, name: str, run_ids: dict,
                            problems: list) -> None:
    spec = STAGES[name]
    rid = _read_pointer(root, spec["pointer"], problems)
    run_ids[name] = rid
    if not rid:
        return
    if not rid.startswith(spec["prefix"]):
        if name == "stage6" and rid.startswith(STAGES["stage6b"]["prefix"]):
            problems.append(
                f"stage6: reports/evaluation/LATEST_RUN.txt tracks a Stage-6b "
                f"run ({rid!r}) — a 6b already published the mirrors, so "
                "there is no NEW Stage-6 v1 run to bind; rerun Stage 6 first "
                "if you intend a fresh v1 comparator")
        else:
            problems.append(
                f"{name}: run id {rid!r} does not start with the expected "
                f"prefix {spec['prefix']!r} — refusing to bind a wrong-stage run")
        return
    run_dir = root / spec["run_root"] / rid
    if not run_dir.is_dir():
        problems.append(f"{name}: run dir not found: {spec['run_root']}/{rid}")
        return
    for prod in spec["products"]:
        if not (run_dir / prod).is_file():
            problems.append(f"{name}: product missing: {run_dir / prod}")


def _require_yaml_lineage(root: Path, name: str, run_dir_rel: str, yaml_name: str,
                          fields: dict, run_ids: dict, problems: list) -> None:
    """Fail-closed lineage: each named resolved-config field must be present,
    non-empty, and equal the bound pointer id. An empty/missing field is a
    problem (never silently skipped)."""
    resolved = root / run_dir_rel / yaml_name
    if not resolved.is_file():
        problems.append(f"{name}: resolved config absent: {resolved}")
        return
    try:
        txt = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        problems.append(f"{name}: unreadable resolved config: {exc}")
        return
    for field, stage in fields.items():
        rec = str(txt.get(field, "") or "")
        bound = run_ids.get(stage, "")
        if not rec:
            problems.append(
                f"lineage: {name} resolved config field {field!r} is empty — "
                f"required lineage to {stage} is missing (fail closed)")
        elif bound and rec != bound:
            problems.append(
                f"lineage: {name} run cites {stage} {rec!r} but the "
                f"{stage} pointer is {bound!r}")


# ---------------------------------------------------------------------------
# Stage-6b / Stage-15 signature gates
# ---------------------------------------------------------------------------

def _signature_gate(root: Path, *, strict_stage15: bool, bindings: dict,
                    problems: list, run_ids: dict) -> None:
    """Validate the Stage-6b signature + products.

    Legacy mode (``strict_stage15=False``, used by --next-stage all): v2
    products are verified at the fixed mirror (historical behaviour, now with
    fail-closed required fields).

    Stage-15 mode (``strict_stage15=True``): additionally every v2 product
    must exist in the 6b run dir AND hash-match its fixed-mirror copy, and the
    fixed-mirror signature must be byte-identical (sha256) to the run-dir
    signature.
    """
    rid = run_ids.get("stage6b", "")
    sig_fixed = root / "splits/split_balance_version_signature.json"
    if not sig_fixed.is_file():
        problems.append("split signature missing: splits/"
                        "split_balance_version_signature.json (run Stage 6b)")
        return
    try:
        sig = json.loads(sig_fixed.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        problems.append(f"split signature unreadable: {exc}")
        return
    if not isinstance(sig, dict):
        problems.append("split signature is not a JSON object")
        return

    status = sig.get("approval_status")
    bindings["split_approval_status"] = status
    bindings["split_signature_run_id"] = sig.get("run_id", "")
    if not isinstance(status, str) or not status:
        problems.append("approval gate: signature approval_status is missing/"
                        "empty — required field (fail closed)")
    elif status != APPROVED:
        problems.append(
            f"approval gate: split signature status {status!r} != "
            f"{APPROVED!r} — the v2 split is NOT approved for modeling; "
            "binding refused (do not hand-edit the status)")
    sig_run_id = sig.get("run_id")
    if not isinstance(sig_run_id, str) or not sig_run_id or \
            not _is_safe_segment(sig_run_id):
        problems.append("split signature run_id missing/unsafe — required "
                        "field (fail closed)")
        sig_run_id = ""
    elif rid and sig_run_id != rid:
        problems.append(
            f"split signature run_id {sig_run_id!r} != splits/"
            f"LATEST_RUN.txt {rid!r}")

    # run-dir vs fixed-mirror signature identity (Stage-15 requirement).
    sig_run_path = root / STAGES["stage6b"]["run_root"] / rid / \
        "split_balance_version_signature.json"
    if rid and strict_stage15:
        if not sig_run_path.is_file():
            problems.append(
                f"mirror gate: 6b run-dir signature absent: {sig_run_path}")
        elif _sha256_file(sig_run_path) != _sha256_file(sig_fixed):
            problems.append(
                "mirror gate: fixed-mirror signature differs from the 6b "
                "run-dir signature — the published mirror must be the "
                "run-dir signature, byte for byte")

    # tamper rejection: every v2 product file must still hash to the sha256
    # recorded in the approved signature. v2_files must be non-empty and each
    # relpath must be a safe relative path under splits/ (no traversal).
    v2_files = sig.get("v2_files")
    if not isinstance(v2_files, dict) or not v2_files:
        problems.append("split signature v2_files missing/empty — required "
                        "field (fail closed; nothing was skipped)")
        v2_files = {}
    for key, rec in v2_files.items():
        if not isinstance(rec, dict):
            problems.append(f"tamper gate: v2_files[{key!r}] is not a record")
            continue
        rel = rec.get("relpath")
        if not isinstance(rel, str) or not _safe_relpath(rel):
            problems.append(
                f"tamper gate: v2_files[{key!r}] relpath {rel!r} is not a "
                "safe relative path under splits/ (absolute paths and '..' "
                "are refused; the file was not opened)")
            continue
        target = root / "splits" / rel
        mirror = target.is_file()
        if not mirror:
            problems.append(f"tamper gate: split product missing: {target}")
            continue
        if _sha256_file(target) != rec.get("sha256"):
            problems.append(
                f"tamper gate: SHA-256 mismatch for splits/{rel} (recorded "
                f"{str(rec.get('sha256'))[:12]}…, found "
                f"{_sha256_file(target)[:12]}…) — the split products changed "
                "after the signature was written; binding refused")
        if strict_stage15:
            run_copy = root / STAGES["stage6b"]["run_root"] / (rid or "") / rel
            if not run_copy.is_file():
                problems.append(
                    f"mirror gate: v2 product absent from the 6b run dir: "
                    f"{run_copy}")
            elif _sha256_file(run_copy) != rec.get("sha256"):
                problems.append(
                    f"mirror gate: 6b run-dir copy of {rel} does not match "
                    "the signed hash (mirror/run product mismatch)")

    # cohort block: required, complete, and equal to the live membership file
    sig_cohort = sig.get("cohort")
    cohort_fp = root / "splits/core_cohort_window_membership.parquet"
    if not isinstance(sig_cohort, dict) or not sig_cohort:
        problems.append("split signature cohort block missing/empty — required "
                        "field (fail closed)")
    else:
        for field in ("n_patients", "n_windows", "n_positive", "n_negative",
                      "patient_set_sha256"):
            if not sig_cohort.get(field):
                problems.append(
                    f"split signature cohort.{field} missing/empty — required "
                    "field (fail closed)")
    if not cohort_fp.is_file():
        problems.append(
            f"cohort mirror missing: {cohort_fp} (Stage 15 reads this mirror)")
    elif isinstance(sig_cohort, dict) and sig_cohort.get("patient_set_sha256"):
        df = pd.read_parquet(cohort_fp, columns=["patient_id"])
        pids = sorted({str(p) for p in df["patient_id"]})
        live_sha = hashlib.sha256(
            "\n".join(pids).encode("utf-8")).hexdigest()
        if live_sha != sig_cohort.get("patient_set_sha256"):
            problems.append(
                "patient-split gate: live cohort patient set does not "
                "match the approved signature fingerprint")
        bindings["cohort_n_patients"] = len(pids)

    # transitive lineage inside the signature: required non-empty + equal
    lin = sig.get("lineage")
    if not isinstance(lin, dict):
        lin = {}
    for key, stage in (("stage3_run_id", "stage3"), ("stage4_run_id", "stage4"),
                       ("stage5_run_id", "stage5")):
        rec = lin.get(key)
        if not isinstance(rec, str) or not rec:
            problems.append(
                f"lineage: signature {key} is empty — required field "
                "(fail closed)")
        elif run_ids.get(stage) and rec != run_ids[stage]:
            problems.append(
                f"lineage: signature {key}={rec!r} != pointer "
                f"{run_ids[stage]!r}")


def _stage15_mirrors(root: Path, problems: list) -> None:
    """The fixed mirrors Stage 15b actually reads (bspc_revision.runner.run)."""
    for rel in (
        "annotations/parsed_events.parquet",
        "data/manifests/csv_window_index.parquet",
        "features/physiology/hr_window_features.parquet",
        "features/physiology/spo2_window_features.parquet",
        "features/physiology/airflow_window_features.parquet",
        "splits/core_cohort_window_membership.parquet",
        "splits/outer_patient_folds_core_v2.csv",
        "splits/inner_patient_folds_core_v2.parquet",
    ):
        p = root / rel
        if not p.is_file() or p.stat().st_size == 0:
            problems.append(f"stage-15 mirror missing/empty: {rel}")


# ---------------------------------------------------------------------------
# Per-stage entry points
# ---------------------------------------------------------------------------

def _stage6b_lineage(root: Path, run_ids: dict, problems: list) -> None:
    """--next-stage 6b: the bound Stage-6 v1 run's input_version_signature.json
    must cite the bound Stage-4/5 (and Stage-3) runs, non-empty."""
    rid = run_ids.get("stage6", "")
    if not rid:
        return
    p = root / STAGES["stage6"]["run_root"] / rid / "input_version_signature.json"
    try:
        sig = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"stage6: unreadable input_version_signature.json: {exc}")
        return
    for key, stage in (("stage4_run_id", "stage4"), ("stage5_run_id", "stage5"),
                       ("stage3_run_id", "stage3")):
        rec = sig.get(key)
        if not isinstance(rec, str) or not rec:
            problems.append(
                f"lineage: stage6 v1 signature {key} is empty — required "
                "field (fail closed)")
        elif run_ids.get(stage) and rec != run_ids[stage]:
            problems.append(
                f"lineage: stage6 v1 signature {key}={rec!r} != pointer "
                f"{run_ids[stage]!r}")


def check_stage(root: Path, next_stage: str) -> tuple[dict, list]:
    """Validate exactly the upstream chain ``next_stage`` needs.

    Returns (bindings, problems); bindings['commands'] carries the validated
    next-stage command lines (only when problems == []).
    """
    if next_stage not in NEXT_STAGES:
        raise ValueError(f"unknown next_stage {next_stage!r}")
    problems: list[str] = []
    bindings: dict = {"project_root": str(root), "next_stage": next_stage,
                      "validated_utc": datetime.now(timezone.utc).strftime(
                          "%Y-%m-%dT%H:%M:%SZ")}

    _license_problems(root, bindings, problems)

    run_ids: dict[str, str] = {}
    for name in REQUIREMENTS[next_stage]:
        _validate_pointer_stage(root, name, run_ids, problems)
    bindings["run_ids"] = run_ids

    # per-stage extra gates
    if next_stage in ("5", "6", "all"):
        s4 = run_ids.get("stage4", "")
        if s4:
            _require_yaml_lineage(
                root, "stage4", f"data/manifests/runs/{s4}",
                "windowing_config_resolved.yaml",
                {"input_stage3_run_id": "stage3"}, run_ids, problems)
    if next_stage in ("6", "all"):
        s5 = run_ids.get("stage5", "")
        if s5:
            _require_yaml_lineage(
                root, "stage5", f"features/physiology/runs/{s5}",
                "physiology_features_resolved.yaml",
                {"input_stage3_run_id": "stage3",
                 "input_stage4_run_id": "stage4"}, run_ids, problems)
    if next_stage == "6b":
        _stage6b_lineage(root, run_ids, problems)
    if next_stage in ("15", "all"):
        _signature_gate(root, strict_stage15=(next_stage == "15"),
                        bindings=bindings, problems=problems, run_ids=run_ids)
    if next_stage == "15":
        _stage15_mirrors(root, problems)

    if not problems:
        bindings["commands"] = _stage_commands(next_stage, run_ids)
    return bindings, problems


def _stage_commands(next_stage: str, run_ids: dict) -> list[str]:
    s3 = run_ids.get("stage3", "")
    s4 = run_ids.get("stage4", "")
    s5 = run_ids.get("stage5", "")
    s6 = run_ids.get("stage6", "")
    if next_stage == "3":
        return ["python scripts/03_annotation_sync.py"]
    if next_stage == "4":
        return [f"python scripts/04_window_index.py "
                f"--input-run-id {_q(s3)}"]
    if next_stage == "5":
        return [f"python scripts/05_physiology_features.py "
                f"--input-stage4-run-id {_q(s4)} "
                f"--input-stage3-run-id {_q(s3)}"]
    if next_stage == "6":
        return [f"python scripts/06_patient_splits.py "
                f"--input-stage4-run-id {_q(s4)} "
                f"--input-stage5-run-id {_q(s5)}"]
    if next_stage == "6b":
        return [f"python scripts/06b_split_balance_optimize.py "
                f"--v1-comparator-run-id {_q(s6)}"]
    if next_stage == "15":
        return [
            "python scripts/15_bspc_reanalysis.py",
            "# NOTE: 15 reads the fixed mirrors verified above; the frozen",
            "# config's provenance_gate.split_run is declarative provenance,",
            "# not a runtime gate in this release (see tools/bind_run_ids.py",
            "# docstring and tests/test_bind_run_ids_staged.py).",
        ]
    return list(_commands(run_ids))


# ---------------------------------------------------------------------------
# Legacy whole-chain check (kept for --next-stage all; same semantics as the
# Stage-25 tool, now fail-closed on required fields)
# ---------------------------------------------------------------------------

def check(root: Path) -> tuple[dict, list]:
    """Validate the whole legacy chain (Stage 3/4/5/6b + gates)."""
    bindings, problems = check_stage(root, "all")
    bindings.pop("commands", None)
    bindings.pop("next_stage", None)
    return bindings, problems


def _commands(run_ids: dict) -> list[str]:
    s3, s4, s5 = (run_ids.get(k, "<run-id>") for k in ("stage3", "stage4", "stage5"))
    return [
        "# next-stage commands with YOUR validated run ids bound:",
        f"python scripts/05_physiology_features.py "
        f"--input-stage4-run-id {_q(s4)} --input-stage3-run-id {_q(s3)}",
        f"python scripts/06_patient_splits.py "
        f"--input-stage4-run-id {_q(s4)} --input-stage5-run-id {_q(s5)}",
        "python scripts/06b_split_balance_optimize.py   # re-freezes the v2 "
        "split + signature + LATEST pointers (bind YOUR stage-6 v1 run with "
        "--v1-comparator-run-id; per-stage entry: --next-stage 6b)",
        "python scripts/15_bspc_reanalysis.py           # reads the approved "
        "frozen mirror products (splits/, annotations/, features/)",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parents[1]
    ap.add_argument("--next-stage", required=True, choices=NEXT_STAGES,
                    help="The stage you want to run NEXT; only its genuinely "
                         "required upstream products are validated.")
    ap.add_argument("--project-root", default=str(here),
                    help="Repository root (default: this checkout).")
    ap.add_argument("--write-bindings", action="store_true",
                    help="Write the validated bindings audit record under "
                         "reports/run_bindings/ (git-ignored).")
    args = ap.parse_args(argv)
    root = Path(args.project_root).resolve()

    bindings, problems = check_stage(root, args.next_stage)
    print(f"[bind] project root : {root}")
    print(f"[bind] next stage   : {args.next_stage}")
    for k, v in bindings.get("run_ids", {}).items():
        print(f"[bind] {k:<8}: {v or '(missing)'}")
    print(f"[bind] license gate : record="
          f"{'present' if bindings.get('license_gate_passed_record') else 'MISSING'}"
          f", current-evidence={bindings.get('license_gate_current_status')}")
    if "split_approval_status" in bindings:
        print(f"[bind] approval     : {bindings.get('split_approval_status')}")
    if problems:
        print("\n[bind] GATES FAILED — binding refused:")
        for p in problems:
            print(f"  - {p}")
        return 2
    print("\n[bind] ALL GATES PASS — validated next-stage command:")
    for line in bindings.get("commands", []):
        print(f"  {line}")
    if args.write_bindings:
        out = root / "reports/run_bindings"
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"local_bindings_stage{args.next_stage}.json"
        f.write_text(json.dumps(bindings, indent=1), encoding="utf-8")
        print(f"\n[bind] audit record written: {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
