#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 6: patient-level experimental cohort + nested cross-validation split.

Orchestrates the Stage-6 runner. The license gate runs FIRST; if it blocks, the
runner writes a blocked completion report and stops. The input Stage-4 (window
index) and Stage-5 (availability) run gates run next. On a passing, non-dry
production run this CLI mirrors the run-dir split products to fixed paths under
``splits/`` and the reports to ``reports/evaluation/`` and writes
``reports/evaluation/LATEST_RUN.txt``.

It NEVER reads raw audio (``*.wav`` is never opened), NEVER reads ``data/raw``,
and NEVER modifies, moves, renames or deletes anything under ``data/raw``. It
does not train, impute, scale, select features, resample, tune thresholds or
evaluate. Every later transform is fitted fold-internally in the model stage.

CLI:
    python scripts/06_patient_splits.py [--config CONFIG] [--dry-run]
        [--run-id ID] [--output-root DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark import run_metadata  # noqa: E402
from psg_audio_benchmark.config import Config, ConfigError, default_project_root  # noqa: E402
from psg_audio_benchmark.evaluation import (  # noqa: E402
    EvaluationContaminationError,
    Stage6Options,
    Stage6Runner,
)
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 6 patient-level cohort + nested cross-validation split."
    )
    p.add_argument("--config", default="config/config.yaml",
                   help="Main config YAML (relative to project root).")
    p.add_argument("--dry-run", action="store_true",
                   help="License + input-run gates still run; no artifacts written.")
    p.add_argument("--run-id", default="", help="Optional run id override.")
    p.add_argument(
        "--output-root",
        default="",
        help="Explicit output root. Via the CLI this must resolve INSIDE the "
        "project (production); an external path is refused. Tests build the "
        "runner directly with an isolated tmp output_root instead.",
    )
    p.add_argument(
        "--input-stage4-run-id",
        default="stage4-csv-window-index-20260807T170636Z",
        help="Approved Stage-4 window-index run id (default: the pinned production run).",
    )
    p.add_argument(
        "--input-stage5-run-id",
        default="stage5-physiology-features-20260807T232931Z",
        help="Approved Stage-5 physiology-features run id (default: the pinned production run).",
    )
    return p


def _build_metadata(project_root: Path, argv: Optional[list]) -> Dict[str, Any]:
    config_texts: Dict[str, str] = {}
    for name in (
        "paths.yaml", "config.yaml", "experiments.yaml",
        "event_type_mapping.yaml", "windowing.yaml", "physiology_features.yaml",
        "splits.yaml",
    ):
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


def _make_stage6_run_id(meta: Dict[str, Any], override: str) -> str:
    if override:
        return override
    ts = meta.get("utc_time", "").replace(":", "").replace("-", "") or "unknown"
    return f"stage6-patient-splits-{ts}"


def _mirror_to_fixed_paths(cfg: Config, runner: Stage6Runner) -> Dict[str, str]:
    """Production only: mirror split products + reports + LATEST_RUN pointer."""
    project_root = cfg.project_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    written: Dict[str, str] = {}
    splits_fixed = cfg.path("splits")
    reports_fixed = cfg.path("reports_evaluation")
    run_splits = runner._splits_out_dir()
    run_rep = runner._reports_out_dir()

    splits_fixed.mkdir(parents=True, exist_ok=True)
    for name in (
        "core_cohort_window_membership.parquet",
        "airflow_cohort_window_membership.parquet",
        "outer_patient_folds_core.csv",
        "inner_patient_folds_core.parquet",
        "airflow_outer_fold_inheritance.csv",
        "patient_split_summary.csv",
        "cohort_exclusions.csv",
        "input_version_signature.json",
        "splits_resolved.yaml",
    ):
        src = run_splits / name
        if src.is_file():
            shutil.copyfile(src, splits_fixed / name)
            written[name] = _rel(splits_fixed / name)

    if run_rep.is_dir():
        reports_fixed.mkdir(parents=True, exist_ok=True)
        for name in (
            "cohort_definition_report.md",
            "split_balance_report.md",
            "phase_06_completion_report.md",
        ):
            rsrc = run_rep / name
            if rsrc.exists():
                shutil.copyfile(rsrc, reports_fixed / name)
                written[name] = _rel(reports_fixed / name)

    (reports_fixed / "LATEST_RUN.txt").write_text(runner.run_id + "\n", encoding="utf-8")
    written["evaluation_latest"] = _rel(reports_fixed / "LATEST_RUN.txt")
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
    run_id = _make_stage6_run_id(meta, args.run_id)
    meta["run_id"] = run_id

    logger = get_logger("stage6.patient_splits", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 6 patient splits starting (run_id=%s)", run_id)

    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the project "
                f"root {cfg.project_root}. Production output must stay inside the "
                f"project. Tests should construct Stage6Runner with a tmp "
                f"output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage6Options(
        dry_run=bool(args.dry_run),
        output_root=output_root,
        stage4_input_run_id=args.input_stage4_run_id,
        stage5_input_run_id=args.input_stage5_run_id,
    )
    runner = Stage6Runner(cfg=cfg, run_metadata=meta, options=options)
    try:
        summary = runner.run()
    except EvaluationContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    if (
        not options.dry_run
        and summary.license_gate_passed
        and summary.input_stage4_verified
        and summary.input_stage5_verified
        and not options.isolated
    ):
        written = _mirror_to_fixed_paths(cfg, runner)
        for kind, p in written.items():
            logger.info("fixed-path %s -> %s", kind, p)

    logger.info(
        "overall_status=%s | license_gate=%s | stage4_verified=%s | stage5_verified=%s | "
        "raw_modified=%s | core_patients=%d core_windows=%d (pos=%d neg=%d) | "
        "airflow_patients=%d airflow_windows=%d | audio_cohort=%d",
        summary.overall_status,
        "passed" if summary.license_gate_passed else "blocked",
        summary.input_stage4_verified,
        summary.input_stage5_verified,
        summary.raw_modified,
        summary.core_patients, summary.core_windows, summary.core_positive, summary.core_negative,
        summary.airflow_patients, summary.airflow_windows,
        summary.audio_cohort_count,
    )

    return 1 if summary.overall_status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
