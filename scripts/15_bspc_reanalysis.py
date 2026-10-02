#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 15b: BSPC three-round revision -- STRICT RE-ANALYSIS (Round 2).

Re-derives window labels from the duration-overlap definition on the INHERITED
frozen v2 patient split, runs the pre-registered model/feature/lag battery
through patient-level 5x4 nested cross-validation with strict fold-internal
leakage control, and reports window / patient-macro / event-level / calibration
metrics with patient-cluster and paired patient-cluster bootstrap uncertainty,
plus the airflow shared-source secondary analysis.

Hard constraints (enforced, never bypassed):

* INHERITED frozen v2 split -- no re-splitting, no relabelling of Stage 2-14;
* audio is BLOCKED (count 0); no audio features are read;
* all outputs are ISOLATED under the Stage-15b run dir; Stage 2-14 products are
  never modified;
* no external download, no submission / upload;
* the protocol/config are the frozen analysis registration (Round 1).

CLI:
    python scripts/15_bspc_reanalysis.py [--config CONFIG] [--run-id ID]
        [--output-root DIR]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psg_audio_benchmark.config import default_project_root  # noqa: E402
from psg_audio_benchmark.bspc_revision.runner import run  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 15b BSPC strict re-analysis (read-only; isolated outputs).")
    p.add_argument("--config", default="config/bspc_revision_analysis.yaml",
                   help="Frozen BSPC revision analysis registration (relative to project root).")
    p.add_argument("--run-id", default="",
                   help="Optional run id override (else stage15b-bspc-reanalysis-<UTC>).")
    p.add_argument("--output-root", default="",
                   help="Reserved for tests (explicit isolated output root INSIDE the project).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    project_root = default_project_root()

    cfg_path = project_root / args.config
    if not cfg_path.exists():
        print(f"[CONFIG ERROR] frozen registration not found: {cfg_path}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"stage15b-bspc-reanalysis-{_utc_stamp()}"
    print(f"[stage15b] run_id={run_id}")
    print(f"[stage15b] frozen registration={cfg_path}")
    print("[stage15b] constraints: inherited v2 split | audio BLOCKED | outputs isolated | "
          "no download | no submit")

    result = run(project_root=project_root, run_id=run_id)
    print(f"[stage15b] onset_matches_stage4={result['onset_matches_stage4']}")
    print(f"[stage15b] results_dir={result['res_dir']}")
    print("[stage15b] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
