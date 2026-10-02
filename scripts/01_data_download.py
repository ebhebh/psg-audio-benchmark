#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 1: data access, manual-download instructions and raw integrity audit.

This script is the Stage-1 orchestrator. It:

  * Verifies the primary dataset DOI identity against hard-coded constants
    (never from a filename).
  * Snapshots ``data/raw`` BEFORE and AFTER the run and asserts immutability.
  * Recursively scans ``data/raw`` (read-only) and ``data/external/incoming``
    (staging), building ``file_manifest_stage1.csv`` with mandatory SHA-256.
  * Runs shallow archive-safety + file-readability checks (no medical field is
    parsed).
  * Writes ``download_receipt.json`` and the Stage-1 reports.

It NEVER modifies, moves, renames or deletes anything under ``data/raw``.
Downloads default to ``manual_required``; opt-in auto-download is only attempted
for an explicit, verified no-auth URL with ``--allow-auto-download``.

CLI:
    python scripts/01_data_download.py [--config CONFIG] [--source-url URL]
        [--staging-dir DIR] [--manifest-out PATH] [--dry-run]
        [--verify-existing] [--allow-auto-download] [--run-id ID]
"""

from __future__ import annotations

import argparse
import json
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
from psg_audio_benchmark.data_download.identity import (  # noqa: E402
    PRIMARY_IDENTITY,
    verify_primary_identity,
)
from psg_audio_benchmark.data_download.runner import (  # noqa: E402
    Stage1ContaminationError,
    Stage1Options,
    Stage1Runner,
)
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 data download & integrity audit.")
    p.add_argument("--config", default="config/config.yaml", help="Config YAML (relative to project root).")
    p.add_argument("--source-url", default="", help="Optional verified public no-auth URL (cannot override DOI identity).")
    p.add_argument(
        "--staging-dir",
        default="",
        help="Optional staging dir; must resolve inside the project external area.",
    )
    p.add_argument("--manifest-out", default="", help="Optional manifest output path (relative to project root).")
    p.add_argument("--dry-run", action="store_true", help="Scan/read only; do not write manifest/receipt/reports.")
    p.add_argument("--verify-existing", action="store_true", help="Only audit existing files; do not download.")
    p.add_argument("--allow-auto-download", action="store_true", help="Permit opt-in auto-download from a verified no-auth URL.")
    p.add_argument("--run-id", default="", help="Optional run id override.")
    p.add_argument(
        "--output-root",
        default="",
        help="Explicit output root. Via the CLI this must resolve INSIDE the "
        "project (production); an external path is refused. Tests construct "
        "the runner directly with an isolated tmp output_root instead.",
    )
    return p


def _resolve_project_root() -> Path:
    return default_project_root()


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


def _render_main_report(summary, meta: Dict[str, Any]) -> str:
    v = summary.identity
    rows_meta = (
        f"- raw 文件数：{summary.n_files_raw}\n"
        f"- staging 文件数：{summary.n_files_staging}\n"
        f"- 总字节数：{summary.total_bytes}\n"
        f"- 压缩包数 / 损坏压缩包数：{summary.n_archives} / {summary.n_corrupt_archives}\n"
        f"- 重复内容记录数：{summary.n_duplicates}\n"
        f"- 冲突数：{summary.n_conflicts}\n"
        f"- 异常文件数：{summary.n_anomalies}\n"
        f"- 因依赖缺失未检查音频数：{summary.n_audio_not_checked}\n"
    )
    raw_eq = "未覆盖 / 未修改（前后轻量快照 size:mtime 完全一致）" if not summary.raw_modified else "检测到变更（BLOCKED）"
    return (
        "# 阶段 1 数据下载与原始数据完整性报告\n\n"
        f"- run_id：`{summary.run_id}`\n"
        f"- config_hash：`{meta.get('config_hash')}`\n"
        f"- 访问日期：{summary.access_date}\n"
        f"- 运行产物根：`{summary.out_base}`\n"
        f"- 总体状态：**{summary.overall_status}**\n\n"
        "## 1. 数据集身份与 DOI 核验\n"
        f"- 主数据集：{PRIMARY_IDENTITY.name}\n"
        f"- 论文 DOI：`{PRIMARY_IDENTITY.paper_doi}`；数据 DOI：`{PRIMARY_IDENTITY.data_doi}`\n"
        f"- 身份核验状态：`{v.status if v else 'not_run'}`\n"
        "- 背景 PSG-Audio 2021（`10.11922/sciencedb.00345`）标为 `background_reference_only`，未混入主清单。\n"
        "> 文献报告的患者数/采样率/airflow 子集等仅作 `literature_reported_hypothesis`，属阶段 2 审计范畴。\n\n"
        "## 2. 访问 / 下载方法与时间\n"
        f"- 下载状态：`{summary.download_status}`\n"
        "- 主数据集位于 Science Data Bank，需认证访问；默认产出人工下载说明，未假装公开无认证 URL。\n"
        "- 下载器仅在显式提供经核验的无认证公开 URL 且 `--allow-auto-download` 时才会发起网络请求。\n"
        "- `verify_existing` 表示本次为对已存在 raw 的只读复审，未下载、未移动、未修改任何 raw 文件。\n\n"
        "## 3. 许可状态及未决事项\n"
        "- 许可状态：`TBD_AFTER_MANUAL_LICENSE_REVIEW`（详见 license_review_report.md）。\n\n"
        "## 4. 暂存区 / 已有 raw 文件概况\n"
        f"{rows_meta}\n"
        f"- manifest：`{summary.manifest_path or '(dry-run 未写入)'}`\n"
        f"- receipt：`{summary.receipt_path or '(dry-run 未写入)'}`\n\n"
        "## 5. 文件 / 压缩包 / 音频完整性\n"
        f"- 详见 `reports/data_download/file_integrity_report.md`。\n"
        "- 音频仅做容器级/魔数检查；ffmpeg/soundfile 缺失时标注限制，不宣称音频内容完整。\n\n"
        "## 6. 冲突、损坏、缺失与人工操作项\n"
        f"- 损坏压缩包：{summary.n_corrupt_archives}；冲突：{summary.n_conflicts}；"
        f"重复内容：{summary.n_duplicates}\n"
        f"- 人工下载步骤详见 `docs/manual_download_instructions.md`。\n\n"
        "## 7. raw 目录是否被修改\n"
        f"- 结论：**{raw_eq}**\n"
        f"- raw 前/后文件数：{len(summary.raw_before)} / {len(summary.raw_after)}\n"
        "- 注：前/后为轻量快照（相对路径 + size + mtime_ns）；每个文件的完整 SHA-256 在 manifest 中生成一次。\n\n"
        "## 8. 进入阶段 2 的准入建议\n"
        "- 在人工下载并核验数据、许可状态确认前，数据集科学审计（阶段 2）不应开始。\n"
        "- 工程准入：身份核验通过、raw 未被修改、manifest 与 receipt 已生成。\n"
    )


def _mirror_run_to_fixed_paths(
    cfg: Config, summary, run_dir_files: Dict[str, Path], run_id: str
) -> Dict[str, str]:
    """Production only: copy this run's artifacts to the fixed legacy paths so
    humans can find them, and write LATEST_RUN pointers. Content always comes
    from the just-completed real production run (never from a test).

    Returns a mapping of legacy-path-kind -> project-relative path written.
    """
    project_root = cfg.project_root.resolve()

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(project_root)).replace("\\", "/")
        except ValueError:
            return str(p).replace("\\", "/")

    manifests_dir = cfg.path("data_manifests")
    reports_dir = cfg.path("reports_data_download")
    written: Dict[str, str] = {}

    # Manifests
    if run_dir_files.get("manifest"):
        dst = manifests_dir / "file_manifest_stage1.csv"
        shutil.copyfile(run_dir_files["manifest"], dst)
        written["manifest"] = _rel(dst)
    if run_dir_files.get("receipt"):
        dst = manifests_dir / "download_receipt.json"
        shutil.copyfile(run_dir_files["receipt"], dst)
        written["receipt"] = _rel(dst)
    (manifests_dir / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    written["manifests_latest"] = _rel(manifests_dir / "LATEST_RUN.txt")

    # Reports (the four sub-reports + the main report)
    reports_src = run_dir_files.get("reports_dir")
    main_src = run_dir_files.get("main_report")
    if reports_src and reports_src.is_dir():
        reports_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "dataset_identity_report.md",
            "license_review_report.md",
            "file_integrity_report.md",
            "known_issues.md",
        ):
            src = reports_src / name
            if src.exists():
                shutil.copyfile(src, reports_dir / name)
                written[name] = _rel(reports_dir / name)
    if main_src and main_src.exists():
        dst = cfg.path("reports") / "data_download_report.md"
        shutil.copyfile(main_src, dst)
        written["main_report"] = _rel(dst)
    (reports_dir / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    written["reports_latest"] = _rel(reports_dir / "LATEST_RUN.txt")
    return written


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = _resolve_project_root()

    try:
        cfg = Config(project_root=project_root)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    meta = _build_metadata(project_root, argv)
    run_id = args.run_id or meta.get("run_id", "stage1")
    # Propagate the (possibly overridden) run id into the metadata so the runner,
    # the run-dir layout and the manifest's run_id column all agree on it.
    meta["run_id"] = run_id
    logger = get_logger("stage1.data_download", cfg.path("logs"), run_id=run_id)
    logger.info("Stage 1 data download & integrity audit starting (run_id=%s)", run_id)

    # --output-root via the CLI must resolve INSIDE the project (production).
    # An external path is refused so a stray CLI flag can never scatter outputs
    # outside the repo. Tests do not use this path; they build the runner
    # directly with an isolated tmp output_root.
    output_root: Optional[Path] = None
    if args.output_root:
        candidate = Path(args.output_root).resolve()
        try:
            candidate.relative_to(cfg.project_root.resolve())
        except ValueError:
            print(
                f"[CONFIG ERROR] --output-root {candidate} is outside the "
                f"project root {cfg.project_root}. Production output must stay "
                f"inside the project. Tests should construct Stage1Runner with "
                f"a tmp output_root instead.",
                file=sys.stderr,
            )
            return 2
        output_root = candidate

    options = Stage1Options(
        dry_run=bool(args.dry_run),
        verify_existing=bool(args.verify_existing),
        source_url=args.source_url,
        allow_auto_download=bool(args.allow_auto_download),
        access_date=meta.get("utc_time", "")[:10] or "",
        output_root=output_root,
    )

    runner = Stage1Runner(cfg=cfg, run_metadata=meta, options=options)
    try:
        summary = runner.run()
    except Stage1ContaminationError as exc:
        print(f"[CONTAMINATION GUARD] {exc}", file=sys.stderr)
        logger.error("contamination guard refused run: %s", exc)
        return 2

    artifacts = runner.artifact_paths()
    main_report_path = artifacts["main_report"]

    if not options.dry_run:
        main_report_path.parent.mkdir(parents=True, exist_ok=True)
        main_report_path.write_text(_render_main_report(summary, meta), encoding="utf-8")
        logger.info("main report -> %s", main_report_path)
        logger.info("manifest -> %s", summary.manifest_path)
        logger.info("receipt  -> %s", summary.receipt_path)
        logger.info("out_base  -> %s", summary.out_base)

        # Production only: mirror this run to the fixed legacy paths + pointers.
        # Isolated/test runs (output_root set) never touch fixed paths.
        if not options.isolated:
            run_dir_files = {
                "manifest": artifacts["manifest"],
                "receipt": artifacts["receipt"],
                "reports_dir": artifacts["reports_dir"],
                "main_report": artifacts["main_report"],
            }
            written = _mirror_run_to_fixed_paths(cfg, summary, run_dir_files, summary.run_id)
            for kind, p in written.items():
                logger.info("fixed-path %s -> %s", kind, p)
    else:
        logger.info("dry-run: no artifacts written")

    logger.info("overall_status=%s | raw_modified=%s | download_status=%s",
                summary.overall_status, summary.raw_modified, summary.download_status)

    # Exit non-zero only on hard BLOCKED (raw modified or identity failure).
    return 1 if summary.overall_status == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
