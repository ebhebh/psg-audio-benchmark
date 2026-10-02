"""Stage-5 orchestration: CSV physiology feature extraction + quality audit.

Read-only over the approved Stage-4 window index and Stage-3 signal extents;
reads raw physiological CSVs (HR/SpO2/airflow) inside each Stage-4
positive/negative window and emits small, interpretable, input-side feature
tables plus a per-window quality/availability audit.

Pipeline (mirrors the Stage 1-4 read-only, run-dir-isolated discipline):

1. **License gate** runs FIRST. If it fails, write a blocked report and STOP.
2. **Input-run gate**: verify the approved Stage-4 run (window index) and the
   Stage-3 run (signal extents / anchors) exist and bind to the right run ids.
3. Snapshot ``data/raw`` before; read Stage-4 windows + Stage-3 extents + raw
   physiological CSVs; compute per-window per-modality features; snapshot after
   and assert immutability.
4. Write the feature/availability/exclusion tables, the resolved config, the
   reports and the (<=5) anonymous QC figures.

No audio is read (``*.wav`` is refused by the reader); no labels/annotations are
used as features; no modelling / split / tuning / evaluation; no imputation or
normalization. ``audio_features_present`` is hard-coded False.
"""

from __future__ import annotations

import csv as csvlib
import dataclasses
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..data_audit import license_gate as gate_mod
from ..data_download.identity import PRIMARY_IDENTITY
from . import config as cfg_mod
from . import figures as figures_mod
from . import qc as qc_mod
from . import report as report_mod
from .config import load_physiology_feature_config, write_resolved_config_yaml
from .csv_reader import CSVReadError, ModalitySignal, read_modality_csv
from .quality import build_feature_row, is_available
from .schema import (
    AUDIO_BLOCK_REASON,
    AUDIO_FEATURES_PRESENT,
    FEATURE_CANDIDATE_STATUSES,
    FEATURE_SET_VERSION,
    MOD_AIRFLOW,
    MOD_HEART_RATE,
    MOD_SPO2,
    AvailabilityRow,
    FeatureExclusion,
    FeatureRow,
    FeatureSummary,
    ModalityExtent,
    SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE,
    SCOPE_CORE_HR_SPO2_FEATURE_QUALITY,
    SCOPE_STAGE4_WINDOW_NOT_CANDIDATE,
)

_DEFAULT_STAGE4_RUN_ID = "stage4-csv-window-index-20260807T170636Z"
_DEFAULT_STAGE3_RUN_ID = "stage3-annotation-sync-20260807T154558Z"

_REQUIRED_STAGE4_ARTIFACTS = (
    "csv_window_index.parquet",
    "patient_window_summary.csv",
    "windowing_config_resolved.yaml",
)
_REQUIRED_STAGE3_ARTIFACTS = (
    "signal_time_ranges.parquet",
    "record_time_anchors.parquet",
)

# Stable column order for each feature table.
_ID_COLS = [
    "run_id", "window_id", "patient_id", "window_index", "modality",
    "window_start_relative_to_record_start", "window_end_relative_to_record_start",
    "duration_seconds", "modality_feature_version", "audio_features_present",
]
_Q_COLS = [
    "n_expected", "n_observed", "n_finite", "coverage_fraction",
    "missing_fraction", "quality_status", "quality_warnings",
]
_STAT_COLS = [
    "mean", "median", "std_ddof1", "min", "max", "range", "iqr",
    "slope_per_second", "value_unit",
]
_AIRFLOW_EXTRA = ["rms", "zero_crossing_count", "zero_crossing_rate"]
_DESAT_COLS = ["n_below_desaturation_threshold", "desaturation_threshold_pct"]


@dataclass
class Stage5Options:
    dry_run: bool = False
    output_root: Optional[Path] = None
    relpath_base: Optional[Path] = None
    raw_base: Optional[Path] = None
    stage4_input_run_id: str = ""
    stage3_input_run_id: str = ""
    config_path: Optional[Path] = None

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


class Stage5Runner:
    """Runs the Stage-5 CSV physiology feature extraction. Construct with a Config."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage5Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage5Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.summary = FeatureSummary(run_id=self.run_id, access_date=self.access_date)
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )
        fpath = self.options.config_path or (self.cfg.path("config") / "physiology_features.yaml")
        self.feature_config = load_physiology_feature_config(Path(fpath))
        self.summary.config_hash = str(run_metadata.get("config_hash", ""))

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _raw_base(self) -> Path:
        if self.options.raw_base is not None:
            return Path(self.options.raw_base).resolve()
        return self.cfg.project_root.resolve()

    def _stage4_run_id(self) -> str:
        return self.options.stage4_input_run_id or self.feature_config.stage4_input_run_id or _DEFAULT_STAGE4_RUN_ID

    def _stage3_run_id(self) -> str:
        return self.options.stage3_input_run_id or self.feature_config.stage3_input_run_id or _DEFAULT_STAGE3_RUN_ID

    def _stage4_run_dir(self) -> Path:
        return self.cfg.path("data_manifests") / "runs" / self._stage4_run_id()

    def _stage3_run_dir(self) -> Path:
        return self.cfg.path("annotations") / "runs" / self._stage3_run_id()

    def _features_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "features" / "physiology" / "runs" / self.run_id
        return self.cfg.path("features_physiology") / "runs" / self.run_id

    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "feature_extraction" / "runs" / self.run_id
        return self.cfg.path("reports_feature_extraction") / "runs" / self.run_id

    def _rel(self, path: Path) -> str:
        try:
            return qc_mod.relativize(path, self._relpath_base())
        except qc_mod.FeatureContaminationError:
            return path.name

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> FeatureSummary:
        qc_mod.assert_not_contaminating(
            cfg=self.cfg,
            run_id=self.run_id,
            config_hash=str(self.run_metadata.get("config_hash", "")),
            output_root=self._output_root,
        )

        # 1. LICENSE GATE -- first and decisive.
        gate = gate_mod.check_license_gate(self.cfg, PRIMARY_IDENTITY)
        self.summary.license_gate_passed = gate.passed
        self.summary.license_status = gate.license_status
        if not gate.passed:
            self.summary.overall_status = "BLOCKED"
            if not self.options.dry_run:
                self._write_blocked_report(gate)
            return self.summary

        # 2. INPUT-RUN GATE (Stage-4 window index + Stage-3 extents).
        self.summary.input_stage4_run_id = self._stage4_run_id()
        self.summary.input_stage3_run_id = self._stage3_run_id()
        if not self._verify_input_runs():
            self.summary.overall_status = "BLOCKED"
            if not self.options.dry_run:
                self._write_blocked_report(gate)
            return self.summary

        self.summary.input_stage4_config_hash = self._read_stage4_config_hash()

        if self.options.dry_run:
            # Dry-run does not read raw CSV content; stop before computing features.
            self.summary.overall_status = "PASS WITH WARNINGS"
            return self.summary

        # 3. full input integrity (run_id binding)
        if not self._verify_input_integrity():
            self.summary.overall_status = "BLOCKED"
            self._write_blocked_report(gate)
            return self.summary
        self.summary.input_stage4_verified = True

        # 4. raw snapshot BEFORE
        raw_before = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())

        # 5. build features
        (
            hr_rows, spo2_rows, af_rows,
            avail_rows, excl_rows,
            coverage_by_mod, per_patient_core, per_patient_airflow,
        ) = self._build_all()

        # 6. raw snapshot AFTER + immutability
        raw_after = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        self.summary.raw_modified = raw_before != raw_after
        self.summary.raw_scanned = True

        # 7. aggregate
        self._aggregate(
            hr_rows, spo2_rows, af_rows, avail_rows, excl_rows,
            coverage_by_mod, per_patient_core, per_patient_airflow,
        )

        # 8. status
        self._derive_status()

        # 9. write products
        self._write_products(
            hr_rows, spo2_rows, af_rows, avail_rows, excl_rows,
            coverage_by_mod, per_patient_core, per_patient_airflow,
        )
        return self.summary

    # ------------------------------------------------------------------
    # Input-run gate
    # ------------------------------------------------------------------
    def _verify_input_runs(self) -> bool:
        ok = True
        s4 = self._stage4_run_dir()
        if not s4.is_dir():
            self.summary.anomalies.append(f"stage4_run_dir_absent:{self._stage4_run_id()}")
            return False
        for name in _REQUIRED_STAGE4_ARTIFACTS:
            p = s4 / name
            if not p.is_file() or p.stat().st_size == 0:
                self.summary.anomalies.append(f"stage4_artifact_missing_or_empty:{name}")
                ok = False
        s3 = self._stage3_run_dir()
        if not s3.is_dir():
            self.summary.anomalies.append(f"stage3_run_dir_absent:{self._stage3_run_id()}")
            return False
        for name in _REQUIRED_STAGE3_ARTIFACTS:
            p = s3 / name
            if not p.is_file() or p.stat().st_size == 0:
                self.summary.anomalies.append(f"stage3_artifact_missing_or_empty:{name}")
                ok = False
        # informational LATEST_RUN pointer for Stage 4 (reports/windowing).
        latest = self.cfg.path("reports_windowing") / "LATEST_RUN.txt"
        if latest.is_file():
            try:
                txt = latest.read_text(encoding="utf-8").strip()
            except OSError:
                txt = ""
            if txt and txt != self._stage4_run_id():
                self.summary.anomalies.append(
                    f"stage4_latest_pointer_mismatch:{txt}!={self._stage4_run_id()}"
                )
        return ok

    def _verify_input_integrity(self) -> bool:
        """Strong input integrity via CONSISTENCY, not a magic patient count.

        The Stage-4 window patient set must equal the Stage-3 anchored patient
        set (Stage-4 windows are built FROM those anchors); every label_status
        must be a known value; and the Stage-4/Stage-3 run_id columns must bind
        to the pinned run ids. This catches a corrupted/wrong Stage-4 input
        without hard-coding the dataset's patient count (50 in production),
        which is recorded in the summary/report from the real data instead.
        """
        ok = True
        rid4 = self._stage4_run_id()
        try:
            win = pd.read_parquet(self._stage4_run_dir() / "csv_window_index.parquet")
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"stage4_window_index_unreadable:{type(exc).__name__}")
            return False
        if "run_id" in win.columns and win.shape[0] > 0:
            bad = int((win["run_id"] != rid4).sum())
            if bad:
                self.summary.anomalies.append(f"stage4_run_id_mismatch:{bad}rows")
                ok = False
        # label statuses must all be known values (no unknown/blank leakage).
        if "label_status" in win.columns:
            statuses = set(win["label_status"].dropna().astype(str).unique())
            allowed = {"positive", "negative", "excluded"}
            unknown = statuses - allowed
            if unknown:
                self.summary.anomalies.append(
                    f"stage4_unknown_label_statuses:{sorted(unknown)}"
                )
                ok = False
        # patient-set consistency: Stage-4 windows must cover exactly the
        # Stage-3 anchored patients.
        win_patients = (
            {str(p) for p in win["patient_id"].dropna().astype(str).unique()}
            if "patient_id" in win.columns else set()
        )
        if not win_patients:
            self.summary.anomalies.append("stage4_window_index_has_no_patients")
            ok = False
        try:
            anchors = pd.read_parquet(self._stage3_run_dir() / "record_time_anchors.parquet")
            anchor_patients = (
                {str(p) for p in anchors["patient_id"].dropna().astype(str).unique()}
                if "patient_id" in anchors.columns and anchors.shape[0] else set()
            )
        except Exception:  # pragma: no cover - defensive
            anchor_patients = set()
        if win_patients and anchor_patients and win_patients != anchor_patients:
            self.summary.anomalies.append(
                "stage4_stage3_patient_set_mismatch:"
                f"only_stage4={sorted(win_patients - anchor_patients)}"
                f":only_stage3={sorted(anchor_patients - win_patients)}"
            )
            ok = False
        # Stage-3 signal-time-ranges run_id binding.
        rid3 = self._stage3_run_id()
        try:
            sig = pd.read_parquet(self._stage3_run_dir() / "signal_time_ranges.parquet")
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"stage3_signal_ranges_unreadable:{type(exc).__name__}")
            return False
        if "run_id" in sig.columns and sig.shape[0] > 0:
            bad = int((sig["run_id"] != rid3).sum())
            if bad:
                self.summary.anomalies.append(f"stage3_run_id_mismatch:{bad}rows")
                ok = False
        return ok

    def _read_stage4_config_hash(self) -> str:
        p = self._stage4_run_dir() / "windowing_config_resolved.yaml"
        if not p.is_file():
            return ""
        try:
            import yaml as _yaml
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return ""
        return str(data.get("config_hash", ""))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build_all(
        self,
    ) -> Tuple[
        List[FeatureRow], List[FeatureRow], List[FeatureRow],
        List[AvailabilityRow], List[FeatureExclusion],
        Dict[str, List[float]], List[int], List[int],
    ]:
        win = pd.read_parquet(self._stage4_run_dir() / "csv_window_index.parquet")
        sig = pd.read_parquet(self._stage3_run_dir() / "signal_time_ranges.parquet")
        raw_base = self._raw_base()

        # per (patient, modality) extent
        extents: Dict[Tuple[str, str], ModalityExtent] = {}
        for _, r in sig.iterrows():
            mod = str(r.get("modality", ""))
            if mod not in (MOD_HEART_RATE, MOD_SPO2, MOD_AIRFLOW):
                continue
            pid = str(r.get("patient_id"))

            def _opt(col):
                v = r.get(col)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return None
                return float(v)

            sc = r.get("sample_count")
            extents[(pid, mod)] = ModalityExtent(
                patient_id=pid, modality=mod,
                source_relpath=str(r.get("source_relpath", "")),
                first_absolute_relative_to_record_start=_opt("first_absolute_relative_to_record_start"),
                last_absolute_relative_to_record_start=_opt("last_absolute_relative_to_record_start"),
                first_relative_seconds=_opt("first_relative_seconds"),
                last_relative_seconds=_opt("last_relative_seconds"),
                sample_count=(None if (sc is None or (isinstance(sc, float) and np.isnan(sc))) else int(sc)),
                relative_duration_seconds=_opt("relative_duration_seconds"),
                verification_status=str(r.get("verification_status", "")),
                data_quality_status=str(r.get("data_quality_status", "")),
            )

        hr_rows: List[FeatureRow] = []
        spo2_rows: List[FeatureRow] = []
        af_rows: List[FeatureRow] = []
        avail_rows: List[AvailabilityRow] = []
        excl_rows: List[FeatureExclusion] = []

        # Stage-4 excluded windows are NOT feature candidates (never faked as negative).
        excluded = win[win["label_status"] == "excluded"]
        for _, w in excluded.iterrows():
            pid = str(w["patient_id"])
            widx = int(w["window_index"])
            excl_rows.append(FeatureExclusion(
                run_id=self.run_id, window_id=self._window_id(pid, widx),
                patient_id=pid, window_index=widx,
                scope=SCOPE_STAGE4_WINDOW_NOT_CANDIDATE, modality="",
                reason=str(w.get("exclusion_reason", "stage4_excluded")),
                detail="Stage-4 excluded window is not a feature candidate; not faked as negative.",
            ))

        candidates = win[win["label_status"].isin(list(FEATURE_CANDIDATE_STATUSES))]
        self.summary.positive_windows = int((win["label_status"] == "positive").sum())
        self.summary.negative_windows = int((win["label_status"] == "negative").sum())
        self.summary.total_candidate_windows = int(len(candidates))
        self.summary.total_excluded_stage4_windows = int(len(excluded))
        self.summary.total_stage4_windows = int(len(win))

        coverage_by_mod: Dict[str, List[float]] = {MOD_HEART_RATE: [], MOD_SPO2: [], MOD_AIRFLOW: []}
        per_patient_core: Dict[str, int] = {}
        per_patient_airflow: Dict[str, int] = {}

        cfg = self.feature_config
        patients = sorted(str(p) for p in candidates["patient_id"].unique())
        self.summary.n_patients = len(patients)
        airflow_patient_set = {
            pid for (pid, mod) in extents
            if mod == MOD_AIRFLOW and extents[(pid, mod)].verification_status == "verified"
        }
        self.summary.airflow_patients_with_modality = len(airflow_patient_set)

        # cache loaded CSVs per (patient, modality) so each file is read once
        signal_cache: Dict[Tuple[str, str], Optional[ModalitySignal]] = {}

        def _load(pid: str, mod: str) -> Optional[ModalitySignal]:
            key = (pid, mod)
            if key in signal_cache:
                return signal_cache[key]
            ext = extents.get(key)
            sig_obj: Optional[ModalitySignal] = None
            if (
                ext is not None
                and ext.verification_status == "verified"
                and ext.offset is not None
                and ext.source_relpath
            ):
                path = raw_base / ext.source_relpath
                try:
                    sig_obj = read_modality_csv(
                        path, patient_id=pid, modality=mod,
                        source_relpath=ext.source_relpath, offset=ext.offset,
                    )
                except (CSVReadError, OSError) as exc:
                    self.summary.anomalies.append(
                        f"modality_csv_unreadable:{pid}:{mod}:{type(exc).__name__}"
                    )
                    sig_obj = None
            signal_cache[key] = sig_obj
            return sig_obj

        for pid in patients:
            pdf = candidates[candidates["patient_id"] == pid]
            hr_sig = _load(pid, MOD_HEART_RATE)
            sp_sig = _load(pid, MOD_SPO2)
            af_sig = _load(pid, MOD_AIRFLOW) if pid in airflow_patient_set else None

            for _, w in pdf.iterrows():
                widx = int(w["window_index"])
                wid = self._window_id(pid, widx)
                start = float(w["start_relative_to_record_start"])
                end = float(w["end_relative_to_record_start"])
                dur = float(w["duration_seconds"])

                hr_row = self._row_for(
                    MOD_HEART_RATE, hr_sig, extents.get((pid, MOD_HEART_RATE)),
                    wid, pid, widx, start, end, dur,
                )
                sp_row = self._row_for(
                    MOD_SPO2, sp_sig, extents.get((pid, MOD_SPO2)),
                    wid, pid, widx, start, end, dur,
                )
                af_row = self._row_for(
                    MOD_AIRFLOW, af_sig, extents.get((pid, MOD_AIRFLOW)),
                    wid, pid, widx, start, end, dur,
                )
                hr_rows.append(hr_row)
                spo2_rows.append(sp_row)
                af_rows.append(af_row)

                # availability + exclusions
                hr_avail = is_available(hr_row.quality_status)
                sp_avail = is_available(sp_row.quality_status)
                af_present = pid in airflow_patient_set
                af_avail = af_present and is_available(af_row.quality_status)
                core = hr_avail and sp_avail

                if hr_row.coverage_fraction is not None:
                    coverage_by_mod[MOD_HEART_RATE].append(hr_row.coverage_fraction)
                if sp_row.coverage_fraction is not None:
                    coverage_by_mod[MOD_SPO2].append(sp_row.coverage_fraction)
                if af_row.coverage_fraction is not None:
                    coverage_by_mod[MOD_AIRFLOW].append(af_row.coverage_fraction)

                if not core:
                    why = []
                    if not hr_avail:
                        why.append(f"heart_rate:{hr_row.quality_status}")
                    if not sp_avail:
                        why.append(f"spo2:{sp_row.quality_status}")
                    excl_rows.append(FeatureExclusion(
                        run_id=self.run_id, window_id=wid, patient_id=pid,
                        window_index=widx, scope=SCOPE_CORE_HR_SPO2_FEATURE_QUALITY,
                        modality=",".join(m.split(":")[0] for m in why),
                        reason="core_hr_spo2_feature_quality_insufficient",
                        detail=";".join(why) or "core_unavailable",
                    ))
                if af_present and not af_avail:
                    excl_rows.append(FeatureExclusion(
                        run_id=self.run_id, window_id=wid, patient_id=pid,
                        window_index=widx, scope=SCOPE_AIRFLOW_OPTIONAL_NOT_AVAILABLE,
                        modality=MOD_AIRFLOW, reason=f"airflow_{af_row.quality_status}",
                        detail="airflow optional modality unavailable for this window",
                    ))

                avail_rows.append(AvailabilityRow(
                    run_id=self.run_id, window_id=wid, patient_id=pid, window_index=widx,
                    window_start_relative_to_record_start=start,
                    window_end_relative_to_record_start=end, duration_seconds=dur,
                    hr_available=hr_avail, hr_coverage_fraction=hr_row.coverage_fraction,
                    hr_quality_status=hr_row.quality_status,
                    spo2_available=sp_avail, spo2_coverage_fraction=sp_row.coverage_fraction,
                    spo2_quality_status=sp_row.quality_status,
                    airflow_modality_present_for_patient=af_present,
                    airflow_available=af_avail, airflow_coverage_fraction=af_row.coverage_fraction,
                    airflow_quality_status=af_row.quality_status,
                    core_hr_spo2_available=core, audio_features_present=AUDIO_FEATURES_PRESENT,
                ))

                per_patient_core[pid] = per_patient_core.get(pid, 0) + (1 if core else 0)
                if af_present:
                    per_patient_airflow[pid] = per_patient_airflow.get(pid, 0) + (1 if af_avail else 0)

        # ordered per-patient lists (anonymous rank by patient id)
        core_list = [per_patient_core[p] for p in patients]
        airflow_list = [per_patient_airflow[p] for p in patients if p in airflow_patient_set]
        return (
            hr_rows, spo2_rows, af_rows, avail_rows, excl_rows,
            coverage_by_mod, core_list, airflow_list,
        )

    def _window_id(self, patient_id: str, window_index: int) -> str:
        return f"{patient_id}-{int(window_index):05d}"

    def _row_for(
        self, modality: str, signal: Optional[ModalitySignal], extent: Optional[ModalityExtent],
        window_id: str, patient_id: str, window_index: int,
        start: float, end: float, dur: float,
    ) -> FeatureRow:
        modality_present = (
            extent is not None
            and extent.verification_status == "verified"
            and extent.offset is not None
        )
        # n_expected from the per-patient inferred sampling rate
        fs = extent.sampling_rate_hz if extent is not None else None
        n_expected = int(round(dur * fs)) if (fs and fs > 0) else 0
        time_axis_ok = modality_present and (fs is not None and fs > 0)

        if signal is not None and modality_present:
            t_win, v_win = signal.select(start, end)
        else:
            t_win = np.asarray([], dtype=float)
            v_win = np.asarray([], dtype=float)

        return build_feature_row(
            run_id=self.run_id, window_id=window_id, patient_id=patient_id,
            window_index=window_index, modality=modality,
            window_start_rel=start, window_end_rel=end, duration_seconds=dur,
            t_win=t_win, v_win=v_win, n_expected=n_expected,
            modality_present_for_patient=modality_present,
            time_axis_ok=bool(time_axis_ok), config=self.feature_config,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(
        self, hr_rows, spo2_rows, af_rows, avail_rows, excl_rows,
        coverage_by_mod, per_patient_core, per_patient_airflow,
    ) -> None:
        s = self.summary
        s.hr_rows_extracted = len(hr_rows)
        s.spo2_rows_extracted = len(spo2_rows)
        s.airflow_rows_extracted = len(af_rows)

        def _count(rows, status):
            return sum(1 for r in rows if r.quality_status == status)

        s.hr_low_coverage = _count(hr_rows, "low_coverage")
        s.spo2_low_coverage = _count(spo2_rows, "low_coverage")
        s.airflow_low_coverage = _count(af_rows, "low_coverage")

        qcounts: Dict[str, Dict[str, int]] = {MOD_HEART_RATE: {}, MOD_SPO2: {}, MOD_AIRFLOW: {}}
        for mod, rows in ((MOD_HEART_RATE, hr_rows), (MOD_SPO2, spo2_rows), (MOD_AIRFLOW, af_rows)):
            qcounts[mod] = dict(Counter(r.quality_status for r in rows))
        s.quality_status_counts = qcounts

        s.core_hr_spo2_available_windows = sum(1 for a in avail_rows if a.core_hr_spo2_available)
        s.core_hr_spo2_unavailable_windows = len(avail_rows) - s.core_hr_spo2_available_windows

        n_airflow_patients = len(per_patient_airflow)
        s.airflow_patients_with_modality = n_airflow_patients
        s.airflow_patients_without_modality = max(0, s.n_patients - n_airflow_patients)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _derive_status(self) -> None:
        s = self.summary
        s.audio_features_present = AUDIO_FEATURES_PRESENT
        if s.raw_modified or not s.license_gate_passed or not s.input_stage4_verified:
            s.overall_status = "BLOCKED"
            return
        if s.anomalies:
            s.overall_status = "PASS WITH WARNINGS"
            return
        s.overall_status = "PASS"

    # ------------------------------------------------------------------
    # Product writers
    # ------------------------------------------------------------------
    def _write_products(
        self, hr_rows, spo2_rows, af_rows, avail_rows, excl_rows,
        coverage_by_mod, per_patient_core, per_patient_airflow,
    ) -> None:
        fdir = self._features_out_dir()
        rdir = self._reports_out_dir()
        fdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        desat_on = self.feature_config.spo2_desaturation_enabled
        self._write_feature_parquet(hr_rows, fdir / "hr_window_features.parquet", MOD_HEART_RATE, desat_on)
        self._write_feature_parquet(spo2_rows, fdir / "spo2_window_features.parquet", MOD_SPO2, desat_on)
        self._write_feature_parquet(af_rows, fdir / "airflow_window_features.parquet", MOD_AIRFLOW, desat_on)
        self._write_parquet_rows(avail_rows, fdir / "physiology_feature_availability.parquet")
        self._write_csv_rows(excl_rows, fdir / "feature_exclusions.csv")

        write_resolved_config_yaml(
            fdir / "physiology_features_resolved.yaml",
            self.feature_config,
            run_id=self.run_id,
            input_stage4_run_id=self.summary.input_stage4_run_id,
            input_stage3_run_id=self.summary.input_stage3_run_id,
            config_hash=self.summary.config_hash,
        )

        # path-purity guard over every machine-readable product
        self._assert_products_clean(fdir)

        # figures (best-effort; must not crash the run) -- written first so their
        # paths can be recorded before the markdown reports reference product_paths.
        try:
            low_or_unav = {
                MOD_HEART_RATE: self.summary.hr_low_coverage,
                MOD_SPO2: self.summary.spo2_low_coverage,
                MOD_AIRFLOW: self.summary.airflow_low_coverage,
            }
            written = figures_mod.render_physiology_qc_figures(
                rdir, run_id=self.run_id,
                coverage_by_modality=coverage_by_mod,
                quality_counts_by_modality=self.summary.quality_status_counts,
                per_patient_core_available=per_patient_core,
                per_patient_airflow_available=per_patient_airflow,
                low_or_unavailable_by_modality=low_or_unav,
            )
            for name in written:
                self.summary.product_paths[name] = self._rel(rdir / name)
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"figures_failed:{type(exc).__name__}")

        # record product paths (features + reports) BEFORE rendering reports, so
        # the completion report can list every product path.
        for name in (
            "hr_window_features.parquet", "spo2_window_features.parquet",
            "airflow_window_features.parquet", "physiology_feature_availability.parquet",
            "feature_exclusions.csv", "physiology_features_resolved.yaml",
        ):
            self.summary.product_paths[name] = self._rel(fdir / name)
        for name in (
            "physiology_feature_report.md", "physiology_feature_qc_report.md",
            "phase_05_completion_report.md",
        ):
            self.summary.product_paths[name] = self._rel(rdir / name)

        # reports (rendered last; reference the now-populated product_paths)
        self._write_report(rdir, "physiology_feature_report.md",
                           report_mod.render_physiology_feature_report(self.summary))
        self._write_report(rdir, "physiology_feature_qc_report.md",
                           report_mod.render_physiology_qc_report(self.summary))
        self._write_report(rdir, "phase_05_completion_report.md", self._render_completion())

    def _write_feature_parquet(
        self, rows: List[FeatureRow], path: Path, modality: str, desat_on: bool
    ) -> None:
        cols = list(_ID_COLS) + list(_Q_COLS) + list(_STAT_COLS)
        if modality == MOD_AIRFLOW:
            cols = list(_ID_COLS) + list(_Q_COLS) + list(_STAT_COLS) + list(_AIRFLOW_EXTRA)
        elif modality == MOD_SPO2 and desat_on:
            cols = cols + list(_DESAT_COLS)
        if not rows:
            pd.DataFrame(columns=cols).to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        # keep only declared columns, in order
        for c in cols:
            if c not in df.columns:
                df[c] = None
        df = df[cols]
        df.to_parquet(path, index=False)

    def _write_parquet_rows(self, rows: List[Any], path: Path) -> None:
        # Always used for AvailabilityRow; keep a well-formed header on empty input.
        names = [f.name for f in dataclasses.fields(AvailabilityRow)]
        if not rows:
            pd.DataFrame(columns=names).to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df.to_parquet(path, index=False)

    def _write_csv_rows(self, rows: List[Any], path: Path) -> None:
        names = [f.name for f in dataclasses.fields(FeatureExclusion)]
        with open(path, "w", encoding="utf-8", newline="") as h:
            writer = csvlib.DictWriter(h, fieldnames=names)
            writer.writeheader()
            for r in rows:
                writer.writerow(dataclasses.asdict(r))

    def _write_report(self, rdir: Path, name: str, content: str) -> None:
        (rdir / name).write_text(content, encoding="utf-8")

    def _write_blocked_report(self, gate) -> None:
        rdir = self._reports_out_dir()
        rdir.mkdir(parents=True, exist_ok=True)
        s = self.summary
        missing = ", ".join(gate.missing_items) if gate.missing_items else (
            "; ".join(s.anomalies) if s.anomalies else "(未给出具体缺项)"
        )
        content = (
            "# 阶段 5 完成报告（phase_05_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- 阶段 5 状态：**BLOCKED**\n"
            f"- 许可门状态：{'BLOCKED' if not s.license_gate_passed else 'PASSED'}"
            f"（license_status=`{s.license_status}`）\n"
            f"- 阻断/缺项：{missing}\n\n"
            "## 准入下一阶段\n\n"
            "- [待决] 当前 **未满足**：许可门未通过或输入 Stage-4/Stage-3 run 完整性校验失败。\n"
            "- 本次不自行进入下一阶段。\n"
        )
        (rdir / "phase_05_completion_report.md").write_text(content, encoding="utf-8")

    def _render_completion(self) -> str:
        s = self.summary
        prods = "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.product_paths.items())) or "- （无）"
        return (
            "# 阶段 5 完成报告（phase_05_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- config_hash：`{s.config_hash or '(n/a)'}`\n"
            f"- feature_set_version：`{FEATURE_SET_VERSION}`\n"
            f"- 访问日期：{s.access_date}\n"
            f"- 阶段 5 状态：**{s.overall_status}**\n"
            f"- 许可门：{'PASSED' if s.license_gate_passed else 'BLOCKED'}（license_status=`{s.license_status}`）\n"
            f"- 输入 Stage-4 run_id：`{s.input_stage4_run_id}`（verified={s.input_stage4_verified}）\n"
            f"- 输入 Stage-3 run_id：`{s.input_stage3_run_id}`\n\n"
            "## 关键计数\n\n"
            f"- Stage-4 窗口：{s.total_stage4_windows}（正 {s.positive_windows} / 负 {s.negative_windows} / excluded {s.total_excluded_stage4_windows}）；候选 {s.total_candidate_windows}。\n"
            f"- HR/SpO2/airflow 特征行：{s.hr_rows_extracted}/{s.spo2_rows_extracted}/{s.airflow_rows_extracted}。\n"
            f"- HR+SpO2 核心共同可用窗口：{s.core_hr_spo2_available_windows}；不可用：{s.core_hr_spo2_unavailable_windows}。\n"
            f"- airflow 可用患者：{s.airflow_patients_with_modality}；无 airflow 患者：{s.airflow_patients_without_modality}。\n"
            f"- `audio_features_present=true` 的产物：0（恒为 false）。\n\n"
            "## 产物路径\n\n"
            f"{prods}\n\n"
            "## 准入下一阶段\n\n"
            "- [实测] CSV 生理特征（HR/SpO2/airflow）与质量审计已完成；无音频特征、无标签作为特征、未建模/划分/调参/评价。\n"
            "- [待决] **下一阶段（受限 CSV 路线）**：可讨论患者级数据划分与传统模型基线；仍须患者级划分、不得引入音频特征。\n"
            "- [待决] **音频路线：BLOCKED** —— 所有音频仍为 unresolved/excluded，须单独单列。\n"
            "- 本次不自行进入下一阶段。\n"
        )

    def _assert_products_clean(self, fdir: Path) -> None:
        values: List[str] = []
        for p in fdir.iterdir():
            if p.suffix == ".csv":
                with open(p, "r", encoding="utf-8", newline="") as h:
                    for row in csvlib.DictReader(h):
                        for v in row.values():
                            if isinstance(v, str):
                                values.append(v)
            elif p.suffix == ".parquet":
                try:
                    df = pd.read_parquet(p)
                except Exception:
                    continue
                for col in df.columns:
                    if df[col].dtype == object:
                        values.extend(str(v) for v in df[col].dropna().tolist())
            elif p.suffix in (".yaml", ".yml"):
                values.extend(p.read_text(encoding="utf-8").splitlines())
        qc_mod.assert_paths_clean(values)


__all__ = ["Stage5Options", "Stage5Runner"]
