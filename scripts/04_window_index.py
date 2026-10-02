#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 4: CSV/annotation-side window index + research label table.

Orchestrates the Stage-4 runner. The license gate runs FIRST; if it blocks, the
runner writes a blocked completion report and stops. The input Stage-3 run gate
(annotations/LATEST_RUN.txt + artifact integrity) runs next. On a passing,
non-dry production run this CLI mirrors the run-dir tables to fixed paths under
``data/manifests/`` and the reports to ``reports/windowing/`` and writes
``reports/windowing/LATEST_RUN.txt``.

It NEVER modifies, moves, renames or deletes anything under ``data/raw`` and
never reads raw audio or raw CSV content (it consumes only Stage-3 derived
parquet/csv tables).

CLI:
    python scripts/04_window_index.py [--config CONFIG] [--dry-run]
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
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402
from psg_audio_benchmark.windowing import (  # noqa: E402
    Stage4Options,
    Stage4Runner,
    WindowingContaminationError,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 4 CSV/annotation-side window index + research labels."
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
        "--input-run-id",
        default="stage3-annotation-sync-20260807T154558Z",
        help="Approved Stage-3 input run id (default: the pinned production run).",
    )
    return p


def _build_metadata(project_root: Path, argv: Optional[list]) -> Dict[str, Any]:
    config_texts: Dict[str, str] = {}
    for name in (
        "paths.yaml", "config.yaml", "experiments.yaml",
        "event_type_mapping.yaml", "windowing.yaml",
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


def _make_stage4_run_id(meta: Dict[str, Any], override: str) -> str:
    if override:
        return override
    ts = meta.get("utc_time", "").replace(":", "").replace("-", "") or "unknown"
    return f"stage4-csv-window-index-{ts}"


def _mirror_to_fixed_paths(cfg: Config, runner: Stage4Runner) -> Dict[str, str]:
    """Production only: mirror tables + reports + LATEST_RUN pointer."""
    project_root = cfg.project_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    written: Dict[str, str] = {}
    manifests_fixed = cfg.path("data_manifests")
    reports_fixed = cfg.path("reports_windowing")
    run_man = runner._manifests_out_dir()
    run_rep = runner._reports_out_dir()

    manifests_fixed.mkdir(parents=True, exist_ok=True)
    for name in (
        "csv_window_index.parquet",
        "window_event_links.parquet",
        "window_exclusions.csv",
        "patient_window_summary.csv",
        "windowing_config_resolved.yaml",
    ):
        src = run_man / name
        if src.is_file():
            shutil.copyfile(src, manifests_fixed / name)
            written[name] = _rel(manifests_fixed / name)

    if run_rep.is_dir():
        reports_fixed.mkdir(parents=True, exist_ok=True)
        for name in (
            "window_label_report.md",
            "windowing_qc_report.md",
            "phase_04_completion_report.md",
        ):
            rsrc = run_rep / name
            if rsrc.exists():
                shutil.copyfile(rsrc, reports_fixed / name)
                written[name] = _rel(reports_fixed / name)

    (reports_fixed / "LATEST_RUN.txt").write_text(runner.run_id + "\n", encoding="utf-8")
    written["windowing_latest"] = _rel(reports_fixed / "LATEST_RUN.txt")
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
    run_id = _make_stage4_run_id(meta, args.run_id)
    meta["run_id"] = run_id

    logger = get_logger("stage4.window_index", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 4 window index starting (run_id=%s)", run_id)

    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the project "
                f"root {cfg.project_root}. Production output must stay inside the "
                f"project. Tests should construct Stage4Runner with a tmp "
                f"output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage4Options(
        dry_run=bool(args.dry_run),
        output_root=output_root,
        input_run_id=args.input_run_id,
    )
    runner = Stage4Runner(cfg=cfg, run_metadata=meta, options=options)
    try:
        summary = runner.run()
    except WindowingContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    if not options.dry_run and summary.license_gate_passed and summary.input_run_verified and not options.isolated:
        written = _mirror_to_fixed_paths(cfg, runner)
        for kind, p in written.items():
            logger.info("fixed-path %s -> %s", kind, p)

    logger.info(
        "overall_status=%s | license_gate=%s | input_verified=%s | raw_modified=%s | "
        "cand=%d | usable=%d (pos=%d neg=%d) | excluded=%d | linked=%d/%d | audio_eligible=%d",
        summary.overall_status,
        "passed" if summary.license_gate_passed else "blocked",
        summary.input_run_verified,
        summary.raw_modified,
        summary.total_candidate_windows,
        summary.total_usable_windows,
        summary.total_positive_windows,
        summary.total_negative_windows,
        summary.total_excluded_windows,
        summary.total_linked_events,
        summary.total_events,
        summary.audio_window_eligible_count,
    )

    return 1 if summary.overall_status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
