#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 1 helper: batch-download files from a scidb.cn URL list into staging.

This drives the safe, non-destructive downloader over a user-provided list of
*verified, no-auth* direct download URLs (the Stage-1 default was
``manual_required``; the user has now supplied working public links, which was
probed to return HTTP 200 without authentication).

Contract:
  * Downloads ONLY into ``data/external/incoming/`` (staging), preserving the
    source tree ``V5/Data/<patient>/<file>``. NEVER writes to ``data/raw``.
  * Per file: temporary ``.part`` + atomic rename + SHA-256 + no-overwrite
    (already_present / conflict) + resume (when server supports ranges) +
    bounded retry. No credentials, no cookies, no access-control bypass.
  * After downloading, re-runs the Stage-1 audit so the manifest / receipt /
  reports reflect the staged files.

CLI:
    python scripts/02_download_from_urllist.py --url-list LINKS.txt \
        --patients 01-05 --workers 4 --allow-auto-download
    python scripts/02_download_from_urllist.py --url-list LINKS.txt --list-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark import run_metadata  # noqa: E402
from psg_audio_benchmark.config import Config, ConfigError, default_project_root  # noqa: E402
from psg_audio_benchmark.data_download.downloader import (  # noqa: E402
    download_url_list,
    filter_entries,
    parse_url_list,
)
from psg_audio_benchmark.data_download.runner import (  # noqa: E402
    Stage1Options,
    Stage1Runner,
)
from psg_audio_benchmark.logging_utils import get_logger  # noqa: E402


def _parse_patients(spec: str) -> Optional[List[str]]:
    if not spec:
        return None
    out: List[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            for n in range(int(a), int(b) + 1):
                out.append(str(n).zfill(2))
        else:
            out.append(part.zfill(2))
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch-download scidb.cn URL list into staging.")
    p.add_argument("--url-list", required=True, help="Path to the .txt file of download URLs.")
    p.add_argument("--patients", default="", help="Patient filter, e.g. '01-05' or '01,02,03'.")
    p.add_argument("--include-script", dest="include_script", action="store_true", default=True)
    p.add_argument("--no-include-script", dest="include_script", action="store_false")
    p.add_argument("--audio-only", dest="audio_only", action="store_true", default=None)
    p.add_argument("--non-audio-only", dest="audio_only", action="store_false")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--backoff", type=float, default=5.0,
                   help="Base seconds for exponential backoff between retries (0 disables).")
    p.add_argument("--staging-dir", default="", help="Override staging dir (must be inside project).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-only", action="store_true", help="Print the filtered plan and exit.")
    p.add_argument("--allow-auto-download", action="store_true",
                   help="REQUIRED to actually fetch. Without it, only --list-only / dry-run run.")
    p.add_argument("--no-audit", action="store_true", help="Skip re-running the Stage-1 audit afterwards.")
    p.add_argument("--run-id", default="")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = default_project_root()
    try:
        cfg = Config(project_root=project_root)
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return 2

    url_list_path = Path(args.url_list)
    if not url_list_path.is_absolute():
        url_list_path = (project_root / url_list_path).resolve() if (project_root / args.url_list).exists() else url_list_path.resolve()
    if not url_list_path.is_file():
        print(f"[ERROR] url-list not found: {url_list_path}", file=sys.stderr)
        return 2
    text = url_list_path.read_text(encoding="utf-8")
    entries = parse_url_list(text)
    patients = _parse_patients(args.patients)
    selected = filter_entries(
        entries,
        patients=patients,
        include_script=args.include_script,
        audio_only=args.audio_only,
    )

    staging_dir = cfg.path("data_external_incoming")
    meta = run_metadata.build_run_metadata(
        project_root=project_root,
        config_texts={
            "paths.yaml": (project_root / "config" / "paths.yaml").read_text(encoding="utf-8"),
            "config.yaml": (project_root / "config" / "config.yaml").read_text(encoding="utf-8"),
        },
        argv=argv,
    )
    run_id = args.run_id or meta.get("run_id", "dl")
    logger = get_logger("stage1.url_download", cfg.path("logs"), run_id=run_id)
    logger.info("parsed %d URLs; selected %d (patients=%s, audio_only=%s, include_script=%s)",
                len(entries), len(selected), patients, args.audio_only, args.include_script)

    n_audio = sum(1 for e in selected if e.is_audio)
    n_other = len(selected) - n_audio
    print(f"selected: {len(selected)} files ({n_audio} audio, {n_other} other) -> staging: {cfg.relative_path('data_external_incoming')}")
    if args.list_only:
        for e in selected:
            kind = "AUDIO" if e.is_audio else "data "
            print(f"  [{kind}] {e.relpath}")
        return 0

    if not args.allow_auto_download:
        print("\n[SAFETY GATE] Pass --allow-auto-download to actually fetch these URLs.\n"
              "Without it this run is a no-op (use --list-only to inspect the plan).",
              file=sys.stderr)
        return 0

    print(f"downloading {len(selected)} files with {args.workers} workers, retry={args.retry}, resume={args.resume} ...")

    done = {"count": 0}

    def _progress(entry, res, attempt):
        done["count"] += 1
        logger.info("[%d/%d] %s -> %s (bytes=%s attempt=%s)",
                    done["count"], len(selected), entry.relpath, res.status, res.size_bytes, attempt)

    results, tally = download_url_list(
        selected,
        staging_dir,
        allow_auto_download=True,
        dry_run=args.dry_run,
        workers=args.workers,
        retry=args.retry,
        resume=args.resume,
        timeout=args.timeout,
        backoff_base=args.backoff,
        progress=_progress,
    )
    print("\ndownload tally:")
    for status, count in sorted(tally.items()):
        print(f"  {status}: {count}")
    logger.info("download tally: %s", tally)

    # Re-run Stage-1 audit so manifest/receipt/reports include staged files.
    if not args.no_audit and not args.dry_run:
        logger.info("re-running Stage-1 audit over staging + raw ...")
        runner = Stage1Runner(cfg=cfg, run_metadata=meta,
                              options=Stage1Options(verify_existing=True, allow_auto_download=False))
        summary = runner.run()
        print(f"\naudit: overall_status={summary.overall_status} "
              f"staging_files={summary.n_files_staging} raw_files={summary.n_files_raw} "
              f"raw_modified={summary.raw_modified}")
        print(f"manifest: {summary.manifest_path}")
        print(f"receipt : {summary.receipt_path}")

    # Non-zero only if every file failed.
    return 0 if tally.get("downloaded", 0) or tally.get("already_present", 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
