"""Shared synthetic fixtures for Stage-5 tests.

Stage 5 is the FIRST stage that reads raw physiological CSV *content* (HR/SpO2/
airflow) inside Stage-4 windows, so these fixtures build three things together,
fully determined so tests can assert EXACT feature values:

* raw modality CSVs with controlled sample times and values (1 Hz HR/SpO2,
  2 Hz airflow, with optional missing/"-" cells);
* a synthetic Stage-3 run dir whose ``signal_time_ranges.parquet`` carries the
  FULL column set the Stage-5 runner reads (incl. ``first_relative_seconds``,
  ``sample_count``, ``relative_duration_seconds``) plus ``record_time_anchors``;
* a synthetic Stage-4 run dir whose ``csv_window_index.parquet`` carries the
  windows (positive/negative/excluded) the Stage-5 runner consumes.

An isolated Config redirects every read/write root under ``tmp_path``; no real
patient data is used (all content is fabricated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from _stage2_fixtures import write_complete_license_evidence

from psg_audio_benchmark.config import Config, default_project_root

STAGE3_ID = "stage3-synth-input-20260808T000000Z"
STAGE4_ID = "stage4-synth-input-20260808T000000Z"

# modality -> (file suffix, value-column header)
_MOD_FILE = {
    "heart_rate": ("HR", 'Heart Rate ("bpm")'),
    "spo2": ("SpO2", 'OSat ("%")'),
    "airflow": ("Flow_DR", "Flow_DR"),
}


def make_isolated_cfg(tmp_path: Path) -> Config:
    """Real Config whose every read/write root points under tmp_path."""
    cfg = Config(project_root=default_project_root())

    def _ensure(rel_key: str, *parts: str) -> Path:
        p = tmp_path.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        cfg._resolved_paths[rel_key] = p
        return p

    _ensure("data_raw", "raw")
    _ensure("docs", "docs")
    write_complete_license_evidence(cfg.path("docs"))  # license gate passes
    _ensure("annotations", "annotations")
    _ensure("data_manifests", "data", "manifests")
    _ensure("reports_windowing", "reports", "windowing")
    _ensure("reports_annotations", "reports", "annotations")
    _ensure("reports_feature_extraction", "reports", "feature_extraction")
    _ensure("features_physiology", "features", "physiology")
    return cfg


def _hhmmss(seconds: float) -> str:
    s = float(seconds)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def write_modality_csv(
    path: Path, *, modality: str, rel_seconds: List[float], values: List[Optional[float]]
) -> Path:
    """Write a controlled modality CSV (relative col + absolute col + value col).

    ``values`` entries that are ``None`` become the dataset's ``"-"`` missing
    marker. The reader only consumes the relative column + the value column.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _suffix, value_header = _MOD_FILE[modality]
    rel_header = "relative position (hh:mm:ss.ms)"
    abs_header = "absolute position (hh:mm:ss.ms)"
    lines = [f"{rel_header},{abs_header},{value_header}"]
    # absolute column starts at an arbitrary late-evening base (unused by reader)
    abs_base = 80000.0
    for r, v in zip(rel_seconds, values):
        cell = "-" if v is None else repr(float(v))
        lines.append(f"{_hhmmss(r)},{_hhmmss(abs_base + r)},{cell}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _patient_spec(
    *,
    patient_id: str = "01",
    offset: float = 0.0,
    hr: Optional[List[Optional[float]]] = None,
    spo2: Optional[List[Optional[float]]] = None,
    airflow: Optional[List[Optional[float]]] = None,
) -> dict:
    if hr is None:
        hr = [60.0] * 10
    if spo2 is None:
        spo2 = [97.0] * 10
    return dict(patient_id=patient_id, offset=offset, hr=hr, spo2=spo2, airflow=airflow)


def _write_stage3_run(
    cfg: Config, run_id: str, patients: List[dict], record_start_cum: float = 80000.0
) -> Path:
    run_dir = cfg.path("annotations") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_root = cfg.path("data_raw")

    anchor_rows, signal_rows = [], []
    for spec in patients:
        pid = spec["patient_id"]
        offset = float(spec.get("offset", 0.0))
        anchor_rows.append({
            "run_id": run_id, "patient_id": pid,
            "record_start_cumulative_seconds": record_start_cum + offset,
            "record_start_present": True,
        })
        for key, mod in (("hr", "heart_rate"), ("spo2", "spo2")):
            vals = spec.get(key)
            if vals is None:
                continue
            n = len(vals)
            last_rel = float(n - 1)            # 1 Hz: samples at 0..n-1
            suffix = _MOD_FILE[mod][0]
            signal_rows.append(_signal_row(
                run_id, pid, mod,
                source_relpath=f"V5/Data/{pid}/{pid}_{suffix}.csv",
                first_abs_rel=offset, last_abs_rel=offset + last_rel,
                first_rel=0.0, last_rel=last_rel,
                sample_count=n, rel_duration=float(n),  # fs = n/n = 1.0
            ))
            write_modality_csv(
                raw_root / "V5" / "Data" / pid / f"{pid}_{suffix}.csv",
                modality=mod, rel_seconds=[float(i) for i in range(n)], values=vals,
            )
        af = spec.get("airflow")
        if af is not None:
            n = len(af)
            last_rel = (n - 1) * 0.5           # 2 Hz
            suffix = _MOD_FILE["airflow"][0]
            signal_rows.append(_signal_row(
                run_id, pid, "airflow",
                source_relpath=f"V5/Data/{pid}/{pid}_{suffix}.csv",
                first_abs_rel=offset, last_abs_rel=offset + last_rel,
                first_rel=0.0, last_rel=last_rel,
                sample_count=n, rel_duration=float(n) / 2.0,  # fs = n/(n/2) = 2.0
            ))
            write_modality_csv(
                raw_root / "V5" / "Data" / pid / f"{pid}_{suffix}.csv",
                modality="airflow",
                rel_seconds=[i * 0.5 for i in range(n)], values=af,
            )

    sig_cols = [
        "run_id", "patient_id", "modality", "source_relpath",
        "first_absolute_relative_to_record_start", "last_absolute_relative_to_record_start",
        "first_relative_seconds", "last_relative_seconds", "sample_count",
        "relative_duration_seconds", "verification_status", "data_quality_status",
    ]
    pd.DataFrame(signal_rows, columns=sig_cols).to_parquet(
        run_dir / "signal_time_ranges.parquet", index=False
    )
    pd.DataFrame(anchor_rows, columns=[
        "run_id", "patient_id", "record_start_cumulative_seconds", "record_start_present",
    ]).to_parquet(run_dir / "record_time_anchors.parquet", index=False)
    return run_dir


def _signal_row(run_id, pid, mod, *, source_relpath, first_abs_rel, last_abs_rel,
                first_rel, last_rel, sample_count, rel_duration) -> dict:
    return {
        "run_id": run_id, "patient_id": pid, "modality": mod,
        "source_relpath": source_relpath,
        "first_absolute_relative_to_record_start": first_abs_rel,
        "last_absolute_relative_to_record_start": last_abs_rel,
        "first_relative_seconds": first_rel, "last_relative_seconds": last_rel,
        "sample_count": sample_count, "relative_duration_seconds": rel_duration,
        "verification_status": "verified", "data_quality_status": "ok",
    }


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


def _write_stage4_run(cfg: Config, run_id: str, windows: List[dict], patients: List[dict]) -> Path:
    run_dir = cfg.path("data_manifests") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    has_airflow = {p["patient_id"]: (p.get("airflow") is not None) for p in patients}

    rows = []
    for w in windows:
        pid = w["patient_id"]
        s = float(w["start_rel"]); e = float(w["end_rel"])
        status = w["label_status"]
        rows.append({
            "run_id": run_id, "patient_id": pid, "window_index": int(w["window_index"]),
            "start_absolute_cumulative_seconds": 0.0, "end_absolute_cumulative_seconds": 0.0,
            "start_relative_to_record_start": s, "end_relative_to_record_start": e,
            "duration_seconds": e - s,
            "core_heart_rate_coverage": True, "core_spo2_coverage": True,
            "has_airflow_coverage": bool(has_airflow.get(pid, False)),
            "core_coverage_complete": True, "awake_overlap_fraction": 0.0,
            "audio_window_eligible": False,
            "audio_block_reason": "no_trustworthy_audio_time_anchor_unresolved",
            "label_status": status,
            "binary_event_label": 1.0 if status == "positive" else 0.0,
            "exclusion_reason": str(w.get("exclusion_reason", "")),
            "n_linked_retained_events": 0, "sleep_structure_observational_status": "not_verified",
        })
    pd.DataFrame(rows, columns=_WIN_COLUMNS).to_parquet(
        run_dir / "csv_window_index.parquet", index=False
    )

    # patient_window_summary.csv (existence gate; minimal well-formed content)
    pws_cols = [
        "run_id", "patient_id", "n_candidate_windows", "n_core_complete_windows",
        "n_awake_excluded_windows", "n_coverage_excluded_windows", "n_excluded_windows",
        "n_positive_windows", "n_negative_windows", "n_usable_windows", "n_linked_events",
        "has_airflow_coverage", "sleep_structure_observational_status",
        "zero_usable_windows", "zero_reason",
    ]
    pws_rows = [{
        "run_id": run_id, "patient_id": p["patient_id"], "n_candidate_windows": 0,
        "n_core_complete_windows": 0, "n_awake_excluded_windows": 0,
        "n_coverage_excluded_windows": 0, "n_excluded_windows": 0,
        "n_positive_windows": 0, "n_negative_windows": 0, "n_usable_windows": 0,
        "n_linked_events": 0, "has_airflow_coverage": bool(p.get("airflow") is not None),
        "sleep_structure_observational_status": "not_verified",
        "zero_usable_windows": True, "zero_reason": "synthetic",
    } for p in patients]
    pd.DataFrame(pws_rows, columns=pws_cols).to_csv(
        run_dir / "patient_window_summary.csv", index=False
    )

    (run_dir / "windowing_config_resolved.yaml").write_text(
        "run_id: %s\nconfig_hash: deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
        % run_id,
        encoding="utf-8",
    )
    return run_dir


def build_stage5_world(
    cfg: Config,
    *,
    patients: List[dict],
    windows: List[dict],
    stage3_run_id: str = STAGE3_ID,
    stage4_run_id: str = STAGE4_ID,
) -> Dict[str, object]:
    """Write a fully-determined synthetic Stage-3 + Stage-4 + raw world.

    Returns ``{"stage3_run_id", "stage4_run_id"}``. Raw CSVs are written under
    ``cfg.path("data_raw")`` with ``source_relpath`` like ``V5/Data/<pid>/...``;
    tests pass ``options.raw_base = cfg.path("data_raw")`` so the runner reads
    them and the raw snapshot covers the same tree.
    """
    _write_stage3_run(cfg, stage3_run_id, patients)
    _write_stage4_run(cfg, stage4_run_id, windows, patients)
    return {"stage3_run_id": stage3_run_id, "stage4_run_id": stage4_run_id}


def run_stage5(cfg: Config, tmp_path: Path, *, stage3_run_id: str, stage4_run_id: str,
               dry_run: bool = False):
    """Construct + run Stage5Runner isolated under tmp_path/out; return summary."""
    from psg_audio_benchmark.feature_extraction import Stage5Options, Stage5Runner

    options = Stage5Options(
        dry_run=dry_run,
        output_root=tmp_path / "out",
        relpath_base=tmp_path,
        raw_base=cfg.path("data_raw"),
        stage4_input_run_id=stage4_run_id,
        stage3_input_run_id=stage3_run_id,
    )
    meta = {
        "run_id": "stage5-synth-20260808T000000Z-abcd1234",
        "config_hash": "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234",
    }
    runner = Stage5Runner(cfg=cfg, run_metadata=meta, options=options)
    summary = runner.run()
    return summary, runner, options


def features_out(tmp_path: Path) -> Path:
    """The isolated feature-product dir for the synthetic run id."""
    return tmp_path / "out" / "features" / "physiology" / "runs" / "stage5-synth-20260808T000000Z-abcd1234"


def reports_out(tmp_path: Path) -> Path:
    return tmp_path / "out" / "reports" / "feature_extraction" / "runs" / "stage5-synth-20260808T000000Z-abcd1234"


__all__ = [
    "STAGE3_ID", "STAGE4_ID",
    "make_isolated_cfg", "write_modality_csv", "_hhmmss", "_patient_spec",
    "build_stage5_world", "run_stage5", "features_out", "reports_out",
]
