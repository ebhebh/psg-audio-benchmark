#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 6B: patient-level split balance optimization + freeze (NO model training).

Re-partitions the APPROVED Stage-6 v1 core cohort into a more-balanced v2
patient-level outer split (exactly 10 patients per outer fold), compares it to the
v1 comparator, runs the pre-registered adoption gate, and ONLY on
``approved_for_modeling`` builds the inner CV and mirrors v2 to the fixed paths +
``splits/LATEST_RUN.txt``. The v1 run-dir products and ``data/raw`` are never
modified.

It NEVER reads raw audio (``*.wav``), NEVER reads ``data/raw``, NEVER reads a
feature value, and NEVER trains / imputes / scales / selects features / resamples
/ tunes thresholds / evaluates. It operates ONLY on patient-level aggregates
(per-patient window count / positive count / negative count / positive rate).

CLI:
    python scripts/06b_split_balance_optimize.py [--config CONFIG] [--dry-run]
        [--run-id ID] [--output-root DIR] [--v1-comparator-run-id ID]

``--v1-comparator-run-id`` (Stage-26 fix) explicitly binds YOUR regenerated
Stage-6 v1 run as the immutable comparator. The frozen
``config/split_balance_optimization.yaml`` pin
(``v1_comparator_run_id: stage6-patient-splits-20260808T002136Z``) stays
untouched as the authors' provenance record; the override is recorded in the
run's resolved config and signature lineage. Get the bound id from
``tools/bind_run_ids.py --next-stage 6b`` (validates that the run exists,
its products are present, and its signature cites your Stage-4/5 runs).
"""

from __future__ import annotations

import argparse
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
    Stage6bOptions,
    Stage6bRunner,
)
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 6B patient-level split balance optimization + freeze (no model)."
    )
    p.add_argument(
        "--config", default="config/split_balance_optimization.yaml",
        help="Split-balance config YAML (relative to project root).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="v1-product + contamination gates only; no optimization, no products.")
    p.add_argument("--run-id", default="", help="Optional run id override.")
    p.add_argument(
        "--output-root", default="",
        help="Explicit output root. Via the CLI this must resolve INSIDE the "
        "project (production); an external path is refused. Tests build the "
        "runner directly with an isolated tmp output_root instead.",
    )
    p.add_argument(
        "--v1-comparator-run-id", default="",
        help="Explicit override of the frozen config pin: YOUR regenerated "
             "Stage-6 v1 run id (prefix 'stage6-patient-splits-'). Bind it "
             "via tools/bind_run_ids.py --next-stage 6b. Empty (default) = "
             "use the frozen config value.",
    )
    return p


def _build_metadata(project_root: Path, argv: Optional[list]) -> Dict[str, Any]:
    config_texts: Dict[str, str] = {}
    for name in (
        "paths.yaml", "config.yaml", "experiments.yaml",
        "event_type_mapping.yaml", "windowing.yaml", "physiology_features.yaml",
        "splits.yaml", "split_balance_optimization.yaml",
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


def _make_run_id(meta: Dict[str, Any], override: str) -> str:
    if override:
        return override
    ts = meta.get("utc_time", "").replace(":", "").replace("-", "")[:15] or "unknown"
    return f"stage6b-split-balance-v2-{ts}Z"


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = default_project_root()

    try:
        cfg = Config(project_root=project_root)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    meta = _build_metadata(project_root, argv)
    run_id = _make_run_id(meta, args.run_id)
    config_hash = str(meta.get("config_hash", ""))
    meta["run_id"] = run_id

    logger = get_logger("stage6b.split_balance", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 6B split-balance optimization starting (run_id=%s)", run_id)

    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the project "
                f"root {cfg.project_root}. Production output must stay inside the "
                f"project. Tests should construct Stage6bRunner with a tmp "
                f"output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage6bOptions(
        dry_run=bool(args.dry_run),
        output_root=output_root,
        config_path=cfg.project_root / args.config,
        run_id=run_id,
        config_hash=config_hash,
        v1_comparator_run_id=(args.v1_comparator_run_id or None),
    )
    runner = Stage6bRunner(cfg=cfg, options=options)
    if options.v1_comparator_run_id:
        logger.info(
            "v1 comparator EXPLICIT override: %s (frozen config pin kept "
            "untouched: %s)", options.v1_comparator_run_id,
            runner.frozen_v1_comparator_pin)
    try:
        summary = runner.run()
    except EvaluationContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    logger.info(
        "status=%s | approved=%s | v1_spread=%.6f v2_spread=%.6f | v1_wcv=%.6f v2_wcv=%.6f | "
        "latest_updated=%s | airflow_mismatch=%d | inner_anomalies=%d | "
        "raw_modified=%s | v1_unchanged=%s | audio_cohort=%d",
        summary.status, bool(summary.gate and summary.gate.approved),
        summary.v1.get("spread", 0.0), summary.v2.get("spread", 0.0),
        summary.v1.get("wcv", 0.0), summary.v2.get("wcv", 0.0),
        summary.latest_updated, summary.airflow_mismatch_count,
        len(summary.inner_anomalies), summary.raw_modified,
        summary.historical_products_unchanged, summary.audio_cohort_count,
    )

    # non-zero exit only on hard block (missing inputs / raw modified); an
    # unapproved balance result still exits 0 (it is an honest, valid outcome).
    if summary.status == "BLOCKED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
