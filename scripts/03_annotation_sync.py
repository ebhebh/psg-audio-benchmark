#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 3: controlled annotation parsing & time-axis synchronization.

Orchestrates the Stage-3 runner. The license gate runs FIRST; if it blocks, the
runner writes a blocked completion report and stops without scanning raw. On a
passing, non-dry production run this CLI mirrors ``parsed_events.parquet`` to the
fixed path ``annotations/parsed_events.parquet``, mirrors the reports to
``reports/annotations/`` and writes ``annotations/LATEST_RUN.txt``.

It NEVER modifies, moves, renames or deletes anything under ``data/raw``.

CLI:
    python scripts/03_annotation_sync.py [--config CONFIG] [--dry-run]
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
from psg_audio_benchmark.annotation_parser import (  # noqa: E402
    Stage3ContaminationError,
    Stage3Options,
    Stage3Runner,
)
from psg_audio_benchmark.config import Config, ConfigError, default_project_root  # noqa: E402
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 3 controlled annotation parsing & time-axis sync."
    )
    p.add_argument("--config", default="config/config.yaml",
                   help="Config YAML (relative to project root).")
    p.add_argument("--dry-run", action="store_true",
                   help="License gate still runs; no artifacts written.")
    p.add_argument("--run-id", default="", help="Optional run id override.")
    p.add_argument(
        "--output-root",
        default="",
        help="Explicit output root. Via the CLI this must resolve INSIDE the "
        "project (production); an external path is refused. Tests build the "
        "runner directly with an isolated tmp output_root instead.",
    )
    return p


def _build_metadata(project_root: Path, argv: Optional[list]) -> Dict[str, Any]:
    config_texts: Dict[str, str] = {}
    for name in ("paths.yaml", "config.yaml", "experiments.yaml", "event_type_mapping.yaml"):
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


def _make_stage3_run_id(meta: Dict[str, Any], override: str) -> str:
    if override:
        return override
    ts = meta.get("utc_time", "").replace(":", "").replace("-", "") or "unknown"
    return f"stage3-annotation-sync-{ts}"


def _mirror_to_fixed_paths(cfg: Config, runner: Stage3Runner) -> Dict[str, str]:
    """Production only: mirror parsed_events + reports + LATEST_RUN pointer."""
    project_root = cfg.project_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    written: Dict[str, str] = {}
    ann_fixed = cfg.path("annotations")
    reports_fixed = cfg.path("reports_annotations")
    run_ann = runner._annotations_out_dir()
    run_rep = runner._reports_out_dir()

    # parsed_events.parquet fixed mirror (prompt section 5).
    src = run_ann / "parsed_events.parquet"
    if src.is_file():
        ann_fixed.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, ann_fixed / "parsed_events.parquet")
        written["parsed_events_fixed"] = _rel(ann_fixed / "parsed_events.parquet")

    # reports mirror.
    if run_rep.is_dir():
        reports_fixed.mkdir(parents=True, exist_ok=True)
        for name in (
            "annotation_parse_report.md",
            "time_synchronization_report.md",
            "event_type_mapping_report.md",
            "phase_03_completion_report.md",
        ):
            rsrc = run_rep / name
            if rsrc.exists():
                shutil.copyfile(rsrc, reports_fixed / name)
                written[name] = _rel(reports_fixed / name)

    (ann_fixed / "LATEST_RUN.txt").write_text(runner.run_id + "\n", encoding="utf-8")
    written["annotations_latest"] = _rel(ann_fixed / "LATEST_RUN.txt")
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
    run_id = _make_stage3_run_id(meta, args.run_id)
    meta["run_id"] = run_id

    logger = get_logger("stage3.annotation_sync", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 3 annotation sync starting (run_id=%s)", run_id)

    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the project "
                f"root {cfg.project_root}. Production output must stay inside the "
                f"project. Tests should construct Stage3Runner with a tmp "
                f"output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage3Options(dry_run=bool(args.dry_run), output_root=output_root)
    runner = Stage3Runner(cfg=cfg, run_metadata=meta, options=options)
    try:
        summary = runner.run()
    except Stage3ContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    if not options.dry_run and summary.license_gate_passed and not options.isolated:
        written = _mirror_to_fixed_paths(cfg, runner)
        for kind, p in written.items():
            logger.info("fixed-path %s -> %s", kind, p)

    logger.info(
        "overall_status=%s | license_gate=%s | raw_scanned=%s | raw_modified=%s | "
        "events_std=%d | rejected=%d | audio_unresolved=%d | audio_excluded=%d",
        summary.overall_status,
        "passed" if summary.license_gate_passed else "blocked",
        summary.raw_scanned,
        summary.raw_modified,
        summary.n_events_standardized,
        summary.n_events_rejected,
        summary.audio_status_counts.get("unresolved_no_trustworthy_audio_time_anchor", 0),
        summary.n_audio_excluded_too_short,
    )

    return 1 if summary.overall_status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
