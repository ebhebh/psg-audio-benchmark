"""Shared synthetic fixtures for Stage-6 (patient-level split) tests.

Stage 6 consumes ONLY the Stage-4 window index + the Stage-5 availability parquet
(+ each run's resolved-config yaml, for the lineage cross-check). It reads no raw
CSV content and no audio. These fixtures therefore build a fully-determined
Stage-4 + Stage-5 run pair under the isolated Config -- with **>=10 patients of
varied positive rate (including one all-negative patient, replicating real
patient 24) and an airflow subcohort** -- so ``StratifiedGroupKFold`` has groups
to stratify and every leakage invariant is assertable. No real patient data is
used (all content is fabricated).

Imports ``_stage5_fixtures`` (which chains ``_stage2_fixtures`` for the license
evidence) so the license gate passes; only ``splits`` + ``reports_evaluation``
are added to the isolated Config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from _stage5_fixtures import make_isolated_cfg as _make_iso_cfg  # chains license evidence

from psg_audio_benchmark.config import Config

# Synthetic run ids + a config-hash that is NOT a test token ("abcd..." passes the
# contamination guard's production-mode check, though we always run isolated here).
STAGE3_ID = "stage3-synth-stage6-20260808T000000Z"
STAGE4_ID = "stage4-synth-stage6-20260808T000000Z"
STAGE5_ID = "stage5-synth-stage6-20260808T000000Z"
S6_RUN_ID = "stage6-synth-20260808T000000Z-abcd1234"
S6_CONFIG_HASH = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
S4_CONFIG_HASH = "1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff"
S5_CONFIG_HASH = "ffff0000eeee1111dddd2222cccc3333bbbb4444aaaa55559999888877776666"

# Stage-4 window-index column set (mirrors the real Stage-4 schema).
_WIN_COLUMNS = [
    "run_id", "patient_id", "window_index",
    "start_absolute_cumulative_seconds", "end_absolute_cumulative_seconds",
    "start_relative_to_record_start", "end_relative_to_record_start",
    "duration_seconds", "core_heart_rate_coverage", "core_spo2_coverage",
    "has_airflow_coverage", "core_coverage_complete", "awake_overlap_fraction",
    "audio_window_eligible", "audio_block_reason", "label_status",
    "binary_event_label", "exclusion_reason", "n_linked_retained_events",
    "sleep_structure_observational_status",
]

# Stage-5 availability column set (mirrors the real Stage-5 schema).
_AVAIL_COLUMNS = [
    "run_id", "window_id", "patient_id", "window_index",
    "window_start_relative_to_record_start", "window_end_relative_to_record_start",
    "duration_seconds", "hr_available", "hr_coverage_fraction", "hr_quality_status",
    "spo2_available", "spo2_coverage_fraction", "spo2_quality_status",
    "airflow_modality_present_for_patient", "airflow_available",
    "airflow_coverage_fraction", "airflow_quality_status",
    "core_hr_spo2_available", "audio_features_present",
]


def make_isolated_cfg(tmp_path: Path) -> Config:
    """Isolated Config with license evidence + every Stage-6 read/write root
    redirected under ``tmp_path`` (adds ``splits`` + ``reports_evaluation``)."""
    cfg = _make_iso_cfg(tmp_path)
    for key, parts in (("splits", ("splits",)),
                       ("reports_evaluation", ("reports", "evaluation"))):
        p = tmp_path.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        cfg._resolved_paths[key] = p
    return cfg


# ---------------------------------------------------------------------------
# Window-spec helpers
# ---------------------------------------------------------------------------

def _window(pid: str, idx: int, status: str, airflow: bool) -> dict:
    """One fabricated window spec. Candidates (positive/negative) are
    HR+SpO2-core-available; ``airflow`` toggles airflow availability. Excluded
    windows keep availability rows for realism but never enter the cohort."""
    candidate = status in ("positive", "negative")
    hr = bool(candidate)
    sp = bool(candidate)
    return {
        "patient_id": pid,
        "window_index": int(idx),
        "label_status": status,
        "hr_available": hr,
        "spo2_available": sp,
        "airflow_available": bool(airflow),
    }


def make_windows(plan: List[tuple]) -> List[dict]:
    """Expand a ``[(patient_id, n_pos, n_neg, n_excluded, airflow)]`` plan into
    a flat window-spec list with sequential per-patient ``window_index``."""
    windows: List[dict] = []
    for pid, pos, neg, exc, af in plan:
        idx = 0
        for _ in range(pos):
            windows.append(_window(pid, idx, "positive", af)); idx += 1
        for _ in range(neg):
            windows.append(_window(pid, idx, "negative", af)); idx += 1
        for _ in range(exc):
            windows.append(_window(pid, idx, "excluded", af)); idx += 1
    return windows


def default_windows() -> List[dict]:
    """10-patient default world: varied pos-rate (incl. one all-negative
    patient ``09``), an excluded window on ``03``, airflow subcohort on
    ``01/02/05/08/10``. Yields 2 patients per 5-fold outer + 8 outer-train per
    fold (enough for 4-fold inner) under the pinned seed."""
    return make_windows([
        ("01", 4, 1, 0, True),
        ("02", 3, 2, 0, True),
        ("03", 3, 3, 1, False),
        ("04", 2, 3, 0, False),
        ("05", 2, 4, 0, True),
        ("06", 1, 4, 0, False),
        ("07", 1, 5, 0, False),
        ("08", 3, 3, 0, True),
        ("09", 0, 5, 0, False),   # all-negative patient (replicates real patient 24)
        ("10", 4, 2, 0, True),
    ])


# ---------------------------------------------------------------------------
# Synthetic Stage-4 + Stage-5 run writers
# ---------------------------------------------------------------------------

def _window_id(pid: str, idx: int) -> str:
    return f"{pid}-{int(idx):05d}"


def _write_stage4_run(cfg: Config, run_id: str, stage3_run_id: str, windows: List[dict]) -> Path:
    run_dir = cfg.path("data_manifests") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in windows:
        pid, idx, status = w["patient_id"], w["window_index"], w["label_status"]
        start = float(idx) * 30.0
        end = start + 30.0
        rows.append({
            "run_id": run_id, "patient_id": pid, "window_index": int(idx),
            "start_absolute_cumulative_seconds": 0.0, "end_absolute_cumulative_seconds": 0.0,
            "start_relative_to_record_start": start, "end_relative_to_record_start": end,
            "duration_seconds": 30.0,
            "core_heart_rate_coverage": True, "core_spo2_coverage": True,
            "has_airflow_coverage": bool(w["airflow_available"]),
            "core_coverage_complete": True, "awake_overlap_fraction": 0.0,
            "audio_window_eligible": False,
            "audio_block_reason": "no_trustworthy_audio_time_anchor_unresolved",
            "label_status": status,
            "binary_event_label": 1.0 if status == "positive" else 0.0,
            "exclusion_reason": "stage4_excluded" if status == "excluded" else "",
            "n_linked_retained_events": 0, "sleep_structure_observational_status": "not_verified",
        })
    pd.DataFrame(rows, columns=_WIN_COLUMNS).to_parquet(
        run_dir / "csv_window_index.parquet", index=False
    )
    (run_dir / "windowing_config_resolved.yaml").write_text(
        yaml.safe_dump({
            "run_id": run_id,
            "config_hash": S4_CONFIG_HASH,
            "input_stage3_run_id": stage3_run_id,
            "input_stage3_config_hash": "stage3hash" * 6 + "ab",
            "resolved_windowing": {"synthetic": True},
        }, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return run_dir


def _write_stage5_run(
    cfg: Config, run_id: str, stage4_run_id: str, stage3_run_id: str, windows: List[dict]
) -> Path:
    run_dir = cfg.path("features_physiology") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in windows:
        pid, idx = w["patient_id"], w["window_index"]
        start = float(idx) * 30.0
        end = start + 30.0
        hr, sp, af = w["hr_available"], w["spo2_available"], w["airflow_available"]
        core = bool(hr and sp)
        rows.append({
            "run_id": run_id, "window_id": _window_id(pid, idx), "patient_id": pid,
            "window_index": int(idx),
            "window_start_relative_to_record_start": start,
            "window_end_relative_to_record_start": end, "duration_seconds": 30.0,
            "hr_available": hr, "hr_coverage_fraction": 1.0 if hr else 0.0,
            "hr_quality_status": "available" if hr else "all_missing",
            "spo2_available": sp, "spo2_coverage_fraction": 1.0 if sp else 0.0,
            "spo2_quality_status": "available" if sp else "all_missing",
            "airflow_modality_present_for_patient": bool(af),
            "airflow_available": af,
            "airflow_coverage_fraction": 1.0 if af else 0.0,
            "airflow_quality_status": "available" if af else "modality_not_available_for_patient",
            "core_hr_spo2_available": core,
            "audio_features_present": False,
        })
    pd.DataFrame(rows, columns=_AVAIL_COLUMNS).to_parquet(
        run_dir / "physiology_feature_availability.parquet", index=False
    )
    (run_dir / "physiology_features_resolved.yaml").write_text(
        yaml.safe_dump({
            "run_id": run_id,
            "config_hash": S5_CONFIG_HASH,
            "input_stage4_run_id": stage4_run_id,
            "input_stage3_run_id": stage3_run_id,
            "resolved_physiology_features": {"synthetic": True},
        }, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return run_dir


def build_stage6_world(
    cfg: Config,
    *,
    windows: Optional[List[dict]] = None,
    stage4_run_id: str = STAGE4_ID,
    stage5_run_id: str = STAGE5_ID,
    stage3_run_id: str = STAGE3_ID,
) -> Dict[str, str]:
    """Write a fully-determined synthetic Stage-4 + Stage-5 run pair.

    Returns ``{"stage3_run_id", "stage4_run_id", "stage5_run_id"}``. Stage 6
    reads only these two run dirs (window index + availability + resolved yamls);
    no raw CSV/audio is written or read.
    """
    if windows is None:
        windows = default_windows()
    _write_stage4_run(cfg, stage4_run_id, stage3_run_id, windows)
    _write_stage5_run(cfg, stage5_run_id, stage4_run_id, stage3_run_id, windows)
    return {
        "stage3_run_id": stage3_run_id,
        "stage4_run_id": stage4_run_id,
        "stage5_run_id": stage5_run_id,
    }


# ---------------------------------------------------------------------------
# Runner harness + output-path helpers
# ---------------------------------------------------------------------------

def run_stage6(
    cfg: Config,
    tmp_path: Path,
    *,
    ids: Dict[str, str],
    dry_run: bool = False,
    run_id: str = S6_RUN_ID,
    output_root: Optional[Path] = None,
    config_path: Optional[Path] = None,
):
    """Construct + run Stage6Runner isolated under ``tmp_path/out``; return
    ``(summary, runner, options)``."""
    from psg_audio_benchmark.evaluation import Stage6Options, Stage6Runner

    options = Stage6Options(
        dry_run=dry_run,
        output_root=output_root if output_root is not None else tmp_path / "out",
        relpath_base=tmp_path,
        stage4_input_run_id=ids["stage4_run_id"],
        stage5_input_run_id=ids["stage5_run_id"],
        stage3_input_run_id=ids["stage3_run_id"],
        config_path=config_path,
    )
    meta = {"run_id": run_id, "config_hash": S6_CONFIG_HASH}
    runner = Stage6Runner(cfg=cfg, run_metadata=meta, options=options)
    summary = runner.run()
    return summary, runner, options


def splits_out(tmp_path: Path, run_id: str = S6_RUN_ID) -> Path:
    """Isolated split-product dir for the synthetic run id."""
    return tmp_path / "out" / "splits" / "runs" / run_id


def reports_out(tmp_path: Path, run_id: str = S6_RUN_ID) -> Path:
    """Isolated evaluation-report dir for the synthetic run id."""
    return tmp_path / "out" / "reports" / "evaluation" / "runs" / run_id


__all__ = [
    "STAGE3_ID", "STAGE4_ID", "STAGE5_ID", "S6_RUN_ID",
    "S6_CONFIG_HASH", "S4_CONFIG_HASH", "S5_CONFIG_HASH",
    "make_isolated_cfg", "make_windows", "default_windows", "_window",
    "build_stage6_world", "run_stage6", "splits_out", "reports_out",
]
