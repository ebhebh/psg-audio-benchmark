"""Shared synthetic fixtures for Stage-4 tests.

Stage 4 consumes only Stage-3 DERIVED artifacts (parquet/csv), never raw audio
or raw CSV content. These fixtures therefore build a small synthetic Stage-3 run
dir under ``tmp_path`` (with controlled signal extents, awake intervals and
events) plus an isolated Config whose ``data_raw`` / ``docs`` / ``annotations``
paths point under ``tmp_path`` so production dirs are never touched.

No real patient data is used; all content is fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from _stage2_fixtures import write_complete_license_evidence

from psg_audio_benchmark.config import Config, default_project_root

DEFAULT_INPUT_RUN_ID = "stage3-synth-input-20260808T000000Z"


def make_isolated_cfg(tmp_path: Path) -> Config:
    """Real Config whose raw/docs/annotations/reports_annotations point under tmp."""
    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    write_complete_license_evidence(docs)  # license gate passes
    ann = tmp_path / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    rep_ann = tmp_path / "reports" / "annotations"
    rep_ann.mkdir(parents=True, exist_ok=True)
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["docs"] = docs
    cfg._resolved_paths["annotations"] = ann
    cfg._resolved_paths["reports_annotations"] = rep_ann
    return cfg


def _patient_spec(
    *,
    patient_id: str = "01",
    record_start_cumulative: float = 80000.0,
    hr: Tuple[float, float] = (100.0, 400.0),
    spo2: Tuple[float, float] = (100.0, 400.0),
    airflow: Optional[Tuple[float, float]] = None,
    sleep_structure_status: str = "not_verified",
    awake: Optional[List[Tuple[float, float]]] = None,
    events: Optional[List[dict]] = None,
    audio_count: int = 3,
) -> dict:
    if awake is None:
        awake = []
    if events is None:
        events = []
    return dict(
        patient_id=patient_id,
        record_start_cumulative=record_start_cumulative,
        hr=hr,
        spo2=spo2,
        airflow=airflow,
        sleep_structure_status=sleep_structure_status,
        awake=awake,
        events=events,
        audio_count=audio_count,
    )


def write_stage3_run(ann_root: Path, run_id: str, patients: List[dict]) -> Path:
    """Write a synthetic Stage-3 run dir with all artifacts the Stage-4 runner reads."""
    run_dir = ann_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    anchor_rows, signal_rows, awake_rows, event_rows, audio_rows = [], [], [], [], []
    for spec in patients:
        pid = spec["patient_id"]
        rs = float(spec["record_start_cumulative"])
        anchor_rows.append({
            "run_id": run_id, "patient_id": pid,
            "record_start_cumulative_seconds": rs,
            "record_start_present": True,
        })
        for mod, ext, verified in (
            ("heart_rate", spec["hr"], True),
            ("spo2", spec["spo2"], True),
            ("sleep_structure", spec.get("sleep_structure_ext", (None, None)),
             spec["sleep_structure_status"] == "verified"),
        ):
            f, l = ext
            signal_rows.append({
                "run_id": run_id, "patient_id": pid, "modality": mod,
                "source_relpath": f"data/raw/V5/Data/{pid}/{pid}_{mod}.csv",
                "verification_status": "verified" if verified else spec.get("sleep_structure_status", "not_verified"),
                "first_absolute_relative_to_record_start": f,
                "last_absolute_relative_to_record_start": l,
            })
        af = spec.get("airflow")
        if af is not None:
            signal_rows.append({
                "run_id": run_id, "patient_id": pid, "modality": "airflow",
                "source_relpath": f"data/raw/V5/Data/{pid}/{pid}_airflow.csv",
                "verification_status": "verified",
                "first_absolute_relative_to_record_start": af[0],
                "last_absolute_relative_to_record_start": af[1],
            })
        for (s, e) in spec["awake"]:
            awake_rows.append({
                "run_id": run_id, "patient_id": pid,
                "start_relative_to_record_start": float(s),
                "end_relative_to_record_start": float(e),
                "cross_midnight": False,
            })
        for ev in spec["events"]:
            event_rows.append({
                "run_id": run_id, "patient_id": pid,
                "source_event_index": ev["source_event_index"],
                "event_type_standardized": ev.get("type", "hypopnea"),
                "event_start_relative_to_record_start": float(ev["start"]),
                "event_duration_seconds": float(ev.get("dur", 10.0)),
                "event_end_relative_to_record_start": float(ev.get("end", ev["start"] + 10.0)),
                "overlaps_awake_interval": bool(ev.get("overlaps_awake", False)),
            })
        for i in range(spec.get("audio_count", 0)):
            audio_rows.append({
                "run_id": run_id, "patient_id": pid,
                "modality": "smartphone_audio" if i == 0 else "recorder_audio",
                "source_relpath": f"data/raw/V5/Data/{pid}/{pid}_audio_{i}.wav",
                "alignment_status": "unresolved_no_trustworthy_audio_time_anchor",
            })

    # Explicit column schemas so an EMPTY table (e.g. a patient with no awake
    # intervals / no events) still serialises as a well-formed, column-complete
    # parquet rather than a columnless one that would break downstream groupbys.
    _columns = {
        "record_time_anchors.parquet": [
            "run_id", "patient_id", "record_start_cumulative_seconds",
            "record_start_present",
        ],
        "signal_time_ranges.parquet": [
            "run_id", "patient_id", "modality", "source_relpath",
            "verification_status", "first_absolute_relative_to_record_start",
            "last_absolute_relative_to_record_start",
        ],
        "awake_intervals_canonical.parquet": [
            "run_id", "patient_id", "start_relative_to_record_start",
            "end_relative_to_record_start", "cross_midnight",
        ],
        "parsed_events.parquet": [
            "run_id", "patient_id", "source_event_index",
            "event_type_standardized", "event_start_relative_to_record_start",
            "event_duration_seconds", "event_end_relative_to_record_start",
            "overlaps_awake_interval",
        ],
    }

    def _to_parquet(rows, name):
        pd.DataFrame(rows, columns=_columns[name]).to_parquet(
            run_dir / name, index=False
        )

    _to_parquet(anchor_rows, "record_time_anchors.parquet")
    _to_parquet(signal_rows, "signal_time_ranges.parquet")
    _to_parquet(awake_rows, "awake_intervals_canonical.parquet")
    _to_parquet(event_rows, "parsed_events.parquet")
    pd.DataFrame(audio_rows).to_csv(run_dir / "audio_time_alignment_inventory.csv", index=False)

    # LATEST_RUN pointer + a completion report carrying a config hash.
    (ann_root / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    rep = ann_root.parent / "reports" / "annotations" / "runs" / run_id
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "phase_03_completion_report.md").write_text(
        f"# x\n- config_hash：`deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef`\n- run_id：`{run_id}`\n",
        encoding="utf-8",
    )
    return run_dir


def build_input_run(cfg: Config, run_id: str, patients: List[dict]) -> Path:
    """Write a synthetic Stage-3 run into the isolated Config's annotations root."""
    return write_stage3_run(cfg.path("annotations"), run_id, patients)


__all__ = [
    "DEFAULT_INPUT_RUN_ID",
    "make_isolated_cfg",
    "_patient_spec",
    "write_stage3_run",
    "build_input_run",
]
