#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 2: real-data structure exploration and patient/modality audit.

This script is the Stage-2 orchestrator. It:

  * Enforces the license confirmation gate FIRST. If the human evidence at
    ``docs/license_evidence/primary_dataset_license_confirmation.md`` is
    missing/incomplete, it writes ``license_gate_blocked.md`` and stops --
    without scanning ``data/raw``.
  * If the gate passes: snapshots ``data/raw`` before; resolves patients;
    classifies every file; probes audio (header-only)/CSV/annotation metadata;
    aggregates; writes the six Stage-2 CSV products and reports; snapshots
    ``data/raw`` after and asserts immutability.

It NEVER modifies, moves, renames or deletes anything under ``data/raw``.

CLI:
    python scripts/02_data_audit.py [--config CONFIG] [--dry-run]
        [--run-id ID] [--output-root DIR] [--stage1-run-id ID]

``--output-root`` via the CLI must resolve INSIDE the project (production); an
external path is refused. Tests construct the runner directly with an isolated
``tmp`` output_root instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Make ``src`` importable when run directly (no install required).
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark import run_metadata  # noqa: E402
from psg_audio_benchmark.config import Config, ConfigError, default_project_root  # noqa: E402
from psg_audio_benchmark.data_audit.runner import (  # noqa: E402
    Stage2ContaminationError,
    Stage2Options,
    Stage2Runner,
)
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 data structure & modality audit.")
    p.add_argument("--config", default="config/config.yaml", help="Config YAML (relative to project root).")
    p.add_argument("--dry-run", action="store_true", help="License gate still runs; no artifacts written.")
    p.add_argument("--run-id", default="", help="Optional run id override.")
    p.add_argument(
        "--output-root",
        default="",
        help="Explicit output root. Via the CLI this must resolve INSIDE the "
        "project (production); an external path is refused. Tests build the "
        "runner directly with an isolated tmp output_root instead.",
    )
    p.add_argument(
        "--stage1-run-id",
        default="",
        help="Stage-1 production run id whose SHA-256 manifest is the content "
        "baseline. Defaults to data/manifests/LATEST_RUN.txt.",
    )
    return p


def _resolve_stage1_run_id(cfg: Config, explicit: str) -> str:
    if explicit:
        return explicit
    latest = cfg.path("data_manifests") / "LATEST_RUN.txt"
    if latest.is_file():
        try:
            return latest.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


def _build_metadata(project_root: Path, argv: Optional[list]) -> Dict[str, Any]:
    config_texts: Dict[str, str] = {}
    for name in ("paths.yaml", "config.yaml", "experiments.yaml"):
        fpath = project_root / "config" / name
        try:
            config_texts[name] = fpath.read_text(encoding="utf-8")
        except Exception:
            config_texts[name] = ""
    meta = run_metadata.build_run_metadata(
        project_root=project_root, config_texts=config_texts, argv=argv
    )
    run_metadata.assert_no_absolute_raw_path(meta)
    return meta


def _make_stage2_run_id(meta: Dict[str, Any], override: str) -> str:
    """Stage-2 run id: ``stage2-audit-<UTC timestamp>`` (prompt section 2.2)."""
    if override:
        return override
    ts = meta.get("utc_time", "").replace(":", "").replace("-", "") or "unknown"
    # utc_time is ISO 'YYYYmmddTHHMMSSZ' -> compact already; keep stamp prefix.
    return f"stage2-audit-{ts}"


def _mirror_to_fixed_paths(
    cfg: Config, summary, run_dir_paths: Dict[str, Path], run_id: str, gate_passed: bool
) -> Dict[str, str]:
    """Production only: mirror this run's artifacts to fixed paths + pointers.

    Gate-blocked runs only mirror the blocked/completion reports (no manifests).
    """
    project_root = cfg.project_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    written: Dict[str, str] = {}
    manifests_dir = cfg.path("data_manifests")
    reports_dir = cfg.path("reports_data_audit")

    # Reports always mirrored (blocked or passed).
    reports_src = run_dir_paths.get("reports_dir")
    if reports_src and reports_src.is_dir():
        reports_dir.mkdir(parents=True, exist_ok=True)
        if gate_passed:
            for name in (
                "license_gate_passed.md",
                "data_structure_exploration_report.md",
                "data_audit_report.md",
                "data_dictionary_observed.md",
                "phase_02_completion_report.md",
            ):
                src = reports_src / name
                if src.exists():
                    shutil.copyfile(src, reports_dir / name)
                    written[name] = _rel(reports_dir / name)
        else:
            for name in ("license_gate_blocked.md", "phase_02_completion_report.md"):
                src = reports_src / name
                if src.exists():
                    shutil.copyfile(src, reports_dir / name)
                    written[name] = _rel(reports_dir / name)

    # Manifests only on gate-passed production runs.
    if gate_passed:
        msrc = run_dir_paths.get("manifests_dir")
        if msrc and msrc.is_dir():
            manifests_dir.mkdir(parents=True, exist_ok=True)
            for name in (
                "patient_manifest.csv",
                "file_manifest_enriched.csv",
                "signal_availability_matrix.csv",
                "annotation_structure_inventory.csv",
                "event_distribution_preliminary.csv",
                "event_distribution_preliminary.README.txt",
                "patient_id_resolution_issues.csv",
            ):
                src = msrc / name
                if src.exists():
                    shutil.copyfile(src, manifests_dir / name)
                    written[name] = _rel(manifests_dir / name)

    (reports_dir / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    written["reports_latest"] = _rel(reports_dir / "LATEST_RUN.txt")
    return written


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = default_project_root()

    try:
        cfg = Config(project_root=project_root)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    meta = _build_metadata(project_root, argv)
    run_id = _make_stage2_run_id(meta, args.run_id)
    meta["run_id"] = run_id

    stage1_run_id = _resolve_stage1_run_id(cfg, args.stage1_run_id)
    logger = get_logger("stage2.data_audit", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 2 data audit starting (run_id=%s)", run_id)

    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the "
                f"project root {cfg.project_root}. Production output must stay "
                f"inside the project. Tests should construct Stage2Runner with "
                f"a tmp output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage2Options(
        dry_run=bool(args.dry_run),
        output_root=output_root,
        stage1_run_id=stage1_run_id,
    )

    runner = Stage2Runner(cfg=cfg, run_metadata=meta, options=options)
    try:
        summary = runner.run()
    except Stage2ContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    if not options.dry_run:
        gate_passed = summary.license_gate_passed
        run_dir_paths = runner.artifact_paths()
        if not options.isolated:
            written = _mirror_to_fixed_paths(
                cfg, summary, run_dir_paths, summary.run_id, gate_passed
            )
            for kind, p in written.items():
                logger.info("fixed-path %s -> %s", kind, p)

    logger.info(
        "overall_status=%s | license_gate=%s | raw_scanned=%s | raw_modified=%s",
        summary.overall_status,
        summary.license_gate.status if summary.license_gate else "n/a",
        summary.raw_scanned,
        summary.raw_modified,
    )

    return 1 if summary.overall_status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
