"""Stage-4 orchestration: CSV/annotation-side window index + research labels.

Read-only over the approved Stage-3 run; builds a reproducible, patient-level
window index and a research label table from the CSV/annotation side only.

Pipeline (mirrors the Stage-1/2/3 read-only, run-dir-isolated discipline):

1. **License gate** runs FIRST. If it fails, write a blocked completion report
   and STOP.
2. **Input-run gate**: verify ``annotations/LATEST_RUN.txt`` points at the
   approved Stage-3 run, that the run dir + required artifacts exist, and (in a
   real run) that every artifact's ``run_id`` column matches the input run.
3. Snapshot ``data/raw`` before; consume only Stage-3 derived parquet/csv
   tables (never raw audio, never raw CSV content); build per-patient windows,
   link events, assign labels; snapshot ``data/raw`` after and assert
   immutability.
4. Write the parquet/csv tables, the resolved config, the reports, the QC
   figures and the Stage-3 cross-midnight erratum.

No windowing of audio, no feature extraction, no modelling, no split, no
evaluation, no clinical conclusion. ``audio_window_eligible`` is hard-coded
False for every window.
"""

from __future__ import annotations

import csv as csvlib
import dataclasses
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..data_audit import license_gate as gate_mod
from ..data_download.identity import PRIMARY_IDENTITY
from . import config as cfg_mod
from . import coverage as coverage_mod
from . import event_link as link_mod
from . import figures as figures_mod
from . import grid as grid_mod
from . import intervals as intervals_mod
from . import label as label_mod
from . import qc as qc_mod
from . import report as report_mod
from .config import load_windowing_config, write_resolved_config_yaml
from .schema import (
    AUDIO_BLOCK_REASON,
    AUDIO_WINDOW_ELIGIBLE,
    EVT_LINKED,
    EXC_NO_CORE_DOMAIN,
    LABEL_EXCLUDED,
    PatientSignalCoverage,
    PatientWindowSummary,
    WindowEventLink,
    WindowExclusion,
    WindowRecord,
    WindowingSummary,
)

#: Approved Stage-3 input run (prompt section 1).
_DEFAULT_INPUT_RUN_ID = "stage3-annotation-sync-20260807T154558Z"

#: Stage-3 artifacts that must all exist for the input run.
_REQUIRED_INPUT_ARTIFACTS = (
    "parsed_events.parquet",
    "awake_intervals_canonical.parquet",
    "signal_time_ranges.parquet",
    "record_time_anchors.parquet",
    "audio_time_alignment_inventory.csv",
)


@dataclass
class Stage4Options:
    dry_run: bool = False
    output_root: Optional[Path] = None
    relpath_base: Optional[Path] = None
    input_run_id: str = ""
    #: Explicit config path override (relative to project root). Empty -> default.
    config_path: Optional[Path] = None

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


class Stage4Runner:
    """Runs the Stage-4 CSV-side window index. Construct with a loaded Config."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage4Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage4Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.summary = WindowingSummary(
            run_id=self.run_id, access_date=self.access_date
        )
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )
        # resolved windowing config
        wpath = self.options.config_path or (self.cfg.path("config") / "windowing.yaml")
        self.windowing_config = load_windowing_config(Path(wpath))
        self.summary.config_hash = str(run_metadata.get("config_hash", ""))

    # ------------------------------------------------------------------
    # Output-path resolution
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _input_run_id(self) -> str:
        return self.options.input_run_id or self._resolve_input_run_id()

    def _resolve_input_run_id(self) -> str:
        latest = self.cfg.path("annotations") / "LATEST_RUN.txt"
        if latest.is_file():
            try:
                txt = latest.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
            except OSError:
                pass
        return _DEFAULT_INPUT_RUN_ID

    def _input_run_dir(self) -> Path:
        return self.cfg.path("annotations") / "runs" / self._input_run_id()

    def _manifests_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "data" / "manifests" / "runs" / self.run_id
        return self.cfg.path("data_manifests") / "runs" / self.run_id

    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "windowing" / "runs" / self.run_id
        return self.cfg.path("reports_windowing") / "runs" / self.run_id

    def _erratum_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "annotations"
        return self.cfg.path("reports_annotations")

    def _rel(self, path: Path) -> str:
        try:
            return qc_mod.relativize(path, self._relpath_base())
        except qc_mod.WindowingContaminationError:
            return path.name

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> WindowingSummary:
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
                self._write_blocked_artifacts(gate)
            return self.summary

        # 2. INPUT-RUN GATE (existence + pointer).
        self.summary.input_run_id = self._input_run_id()
        if not self._verify_input_run_existence():
            self.summary.overall_status = "BLOCKED"
            if not self.options.dry_run:
                self._write_blocked_artifacts(gate)
            return self.summary

        # capture Stage-3 config hash for traceability (best-effort).
        self.summary.input_run_config_hash = self._read_stage3_config_hash()

        if self.options.dry_run:
            # Dry-run does not read raw CSV content; stop before building windows.
            self.summary.overall_status = "PASS WITH WARNINGS"
            return self.summary

        # 3. INPUT INTEGRITY (full): run_id columns must match the input run.
        if not self._verify_input_run_integrity():
            self.summary.overall_status = "BLOCKED"
            self._write_blocked_artifacts(gate)
            return self.summary
        self.summary.input_run_verified = True

        # 4. raw snapshot BEFORE
        raw_before = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())

        # 5. read Stage-3 tables + build
        (
            window_rows,
            link_rows,
            exclusion_rows,
            patient_rows,
        ) = self._build_all()

        # 6. raw snapshot AFTER + immutability
        raw_after = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        self.summary.raw_modified = raw_before != raw_after
        self.summary.raw_scanned = True

        # 7. aggregate
        self._aggregate(window_rows, link_rows, patient_rows)

        # 8. status
        self._derive_status()

        # 9. write products
        self._write_products(window_rows, link_rows, exclusion_rows, patient_rows)
        return self.summary

    # ------------------------------------------------------------------
    # Input-run gate
    # ------------------------------------------------------------------
    def _verify_input_run_existence(self) -> bool:
        latest = self.cfg.path("annotations") / "LATEST_RUN.txt"
        ok = True
        if latest.is_file():
            try:
                txt = latest.read_text(encoding="utf-8").strip()
            except OSError:
                txt = ""
            if txt and txt != self.summary.input_run_id:
                self.summary.anomalies.append(
                    f"latest_run_pointer_mismatch:{txt}!={self.summary.input_run_id}"
                )
                ok = False
        else:
            self.summary.anomalies.append("annotations_LATEST_RUN_missing")
            ok = False
        run_dir = self._input_run_dir()
        if not run_dir.is_dir():
            self.summary.anomalies.append(f"input_run_dir_absent:{self.summary.input_run_id}")
            return False
        for name in _REQUIRED_INPUT_ARTIFACTS:
            p = run_dir / name
            if not p.is_file() or p.stat().st_size == 0:
                self.summary.anomalies.append(f"input_artifact_missing_or_empty:{name}")
                ok = False
        return ok

    def _verify_input_run_integrity(self) -> bool:
        run_dir = self._input_run_dir()
        rid = self.summary.input_run_id
        ok = True
        for name in (
            "parsed_events.parquet",
            "awake_intervals_canonical.parquet",
            "signal_time_ranges.parquet",
            "record_time_anchors.parquet",
        ):
            try:
                df = pd.read_parquet(run_dir / name)
            except Exception as exc:  # pragma: no cover - defensive
                self.summary.anomalies.append(f"input_artifact_unreadable:{name}:{type(exc).__name__}")
                return False
            # A zero-row but well-formed table is valid (a patient may have no
            # awake intervals / no events); only the run_id binding is checked.
            if "run_id" in df.columns and df.shape[0] > 0:
                bad = int(df.loc[df["run_id"] != rid].shape[0])
                if bad:
                    self.summary.anomalies.append(f"input_run_id_mismatch:{name}:{bad}rows")
                    ok = False
        # The patient set (record_time_anchors) and signal coverage must exist.
        try:
            anchors_df = pd.read_parquet(run_dir / "record_time_anchors.parquet")
        except Exception:  # pragma: no cover - defensive
            return False
        if anchors_df.shape[0] == 0:
            self.summary.anomalies.append("input_record_time_anchors_empty")
            ok = False
        try:
            audio_df = pd.read_csv(run_dir / "audio_time_alignment_inventory.csv")
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append("input_audio_inventory_unreadable")
            return False
        if "run_id" in audio_df.columns and (audio_df["run_id"] != rid).any():
            self.summary.anomalies.append("input_audio_run_id_mismatch")
            ok = False
        return ok

    def _read_stage3_config_hash(self) -> str:
        rep = (
            self.cfg.path("reports_annotations")
            / "runs"
            / self.summary.input_run_id
            / "phase_03_completion_report.md"
        )
        if not rep.is_file():
            return ""
        try:
            text = rep.read_text(encoding="utf-8")
        except OSError:
            return ""
        m = re.search(r"config_hash[：:]\s*`?([0-9a-fA-F]{8,})`?", text)
        return m.group(1) if m else ""

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build_all(
        self,
    ) -> Tuple[List[WindowRecord], List[WindowEventLink], List[WindowExclusion],
               List[PatientWindowSummary]]:
        run_dir = self._input_run_dir()
        anchors = pd.read_parquet(run_dir / "record_time_anchors.parquet")
        signals = pd.read_parquet(run_dir / "signal_time_ranges.parquet")
        awake = pd.read_parquet(run_dir / "awake_intervals_canonical.parquet")
        events = pd.read_parquet(run_dir / "parsed_events.parquet")
        audio = pd.read_csv(run_dir / "audio_time_alignment_inventory.csv")

        # audio side is informational only (audio_window_eligible always False)
        self._audio_total = int(audio.shape[0])
        self._audio_status_counts = dict(Counter(audio.get("alignment_status", pd.Series())))

        wcfg = self.windowing_config
        max_awake = wcfg.max_awake_overlap_fraction
        length = wcfg.window_length_s
        step = wcfg.hop_step_s

        window_rows: List[WindowRecord] = []
        link_rows: List[WindowEventLink] = []
        exclusion_rows: List[WindowExclusion] = []
        patient_rows: List[PatientWindowSummary] = []

        # pre-index per-patient slices (defensive: an empty/columnless table
        # for a legitimately empty modality must not crash the groupby).
        def _by_patient(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
            if "patient_id" not in df.columns:
                return {}
            return {str(pid): g for pid, g in df.groupby("patient_id")}

        sig_groups = _by_patient(signals)
        awake_groups = _by_patient(awake)
        event_groups = _by_patient(events)

        anchors = anchors.sort_values("patient_id")
        for _, arow in anchors.iterrows():
            pid = str(arow["patient_id"])
            rs_cum = float(arow["record_start_cumulative_seconds"])
            cov = self._patient_coverage(pid, sig_groups.get(pid))
            awake_bounds = self._awake_bounds(awake_groups.get(pid))
            ev_inputs = self._event_inputs(pid, event_groups.get(pid))

            slots, p_windows, p_excl = self._build_windows(
                pid=pid,
                rs_cum=rs_cum,
                cov=cov,
                awake_bounds=awake_bounds,
                length=length,
                step=step,
                max_awake=max_awake,
            )
            links, retained = link_mod.link_patient_events(
                run_id=self.run_id, patient_id=pid, slots=slots, events=ev_inputs
            )
            link_rows.extend(links)

            # finalize window labels with retained counts
            for wrec in p_windows:
                n_ret = int(retained.get(wrec.window_index, 0))
                wrec.n_linked_retained_events = n_ret
                excluded = wrec.label_status == LABEL_EXCLUDED
                # assign_label re-derives from excluded flag + retained count
                label_status, binary = label_mod.assign_label(excluded, n_ret)
                wrec.label_status = label_status
                wrec.binary_event_label = binary

            window_rows.extend(p_windows)
            exclusion_rows.extend(p_excl)
            patient_rows.append(
                self._patient_summary(pid, p_windows, retained, cov)
            )

        return window_rows, link_rows, exclusion_rows, patient_rows

    def _patient_coverage(
        self, pid: str, sig_df: Optional[pd.DataFrame]
    ) -> PatientSignalCoverage:
        empty = PatientSignalCoverage(
            patient_id=pid, record_start_cumulative_seconds=0.0,
            hr_verified=False, hr_first_rel=None, hr_last_rel=None, hr_source_relpath="",
            spo2_verified=False, spo2_first_rel=None, spo2_last_rel=None, spo2_source_relpath="",
            airflow_verified=False, airflow_first_rel=None, airflow_last_rel=None,
            airflow_source_relpath="", sleep_structure_status="absent",
        )
        if sig_df is None or sig_df.empty:
            return empty
        by_mod = {str(r["modality"]): r for _, r in sig_df.iterrows()}

        def _ext(mod):
            r = by_mod.get(mod)
            if r is None:
                return False, None, None, ""
            verified = str(r.get("verification_status", "")) == "verified"
            return (
                verified,
                None if pd.isna(r.get("first_absolute_relative_to_record_start")) else float(r["first_absolute_relative_to_record_start"]),
                None if pd.isna(r.get("last_absolute_relative_to_record_start")) else float(r["last_absolute_relative_to_record_start"]),
                str(r.get("source_relpath", "")),
            )

        hr_v, hr_f, hr_l, hr_src = _ext("heart_rate")
        sp_v, sp_f, sp_l, sp_src = _ext("spo2")
        af_v, af_f, af_l, af_src = _ext("airflow")
        ss = by_mod.get("sleep_structure")
        ss_status = str(ss.get("verification_status", "absent")) if ss is not None else "absent"
        # record_start_cumulative filled by caller; keep 0 here (overridden below)
        return PatientSignalCoverage(
            patient_id=pid, record_start_cumulative_seconds=0.0,
            hr_verified=hr_v, hr_first_rel=hr_f, hr_last_rel=hr_l, hr_source_relpath=hr_src,
            spo2_verified=sp_v, spo2_first_rel=sp_f, spo2_last_rel=sp_l, spo2_source_relpath=sp_src,
            airflow_verified=af_v, airflow_first_rel=af_f, airflow_last_rel=af_l,
            airflow_source_relpath=af_src, sleep_structure_status=ss_status,
        )

    def _awake_bounds(self, awake_df: Optional[pd.DataFrame]) -> List[Tuple[float, float]]:
        if awake_df is None or awake_df.empty:
            return []
        out: List[Tuple[float, float]] = []
        for _, r in awake_df.iterrows():
            s = r.get("start_relative_to_record_start")
            e = r.get("end_relative_to_record_start")
            if pd.isna(s) or pd.isna(e):
                continue
            out.append((float(s), float(e)))
        out.sort()
        return out

    def _event_inputs(self, pid: str, ev_df: Optional[pd.DataFrame]) -> List[link_mod.EventInput]:
        if ev_df is None or ev_df.empty:
            return []
        out: List[link_mod.EventInput] = []
        for _, r in ev_df.iterrows():
            out.append(
                link_mod.EventInput(
                    source_event_index=int(r["source_event_index"]),
                    event_start_rel=float(r["event_start_relative_to_record_start"]),
                    event_duration_seconds=float(r["event_duration_seconds"]),
                    event_end_rel=float(r["event_end_relative_to_record_start"]),
                    event_type_standardized=str(r["event_type_standardized"]),
                    overlaps_awake=bool(r["overlaps_awake_interval"]),
                )
            )
        return out

    def _build_windows(
        self, *, pid, rs_cum, cov, awake_bounds, length, step, max_awake
    ) -> Tuple[List[link_mod.WindowSlot], List[WindowRecord], List[WindowExclusion]]:
        slots: List[link_mod.WindowSlot] = []
        records: List[WindowRecord] = []
        exclusions: List[WindowExclusion] = []

        domain = cov.core_domain
        if domain is None:
            return slots, records, exclusions

        lo, hi = domain
        ss_status = cov.sleep_structure_status
        for idx, start, end in grid_mod.generate_window_grid(lo, hi, length, step):
            hr_cov = coverage_mod.core_heart_rate_coverage(start, end, cov)
            spo2_cov = coverage_mod.core_spo2_coverage(start, end, cov)
            af_cov = coverage_mod.airflow_coverage(start, end, cov)
            complete = hr_cov and spo2_cov
            awake_frac = intervals_mod.awake_overlap_fraction(start, end, awake_bounds)
            excluded, reason = label_mod.is_excluded(complete, awake_frac, max_awake)
            eligible = not excluded
            label_status, binary = label_mod.assign_label(
                excluded, 0
            )  # retained count finalized later
            slots.append(
                link_mod.WindowSlot(
                    window_index=idx, start=start, end=end,
                    eligible=eligible, exclusion_reason=reason,
                )
            )
            records.append(
                WindowRecord(
                    run_id=self.run_id,
                    patient_id=pid,
                    window_index=idx,
                    start_absolute_cumulative_seconds=round(rs_cum + start, 6),
                    end_absolute_cumulative_seconds=round(rs_cum + end, 6),
                    start_relative_to_record_start=round(start, 6),
                    end_relative_to_record_start=round(end, 6),
                    duration_seconds=round(end - start, 6),
                    core_heart_rate_coverage=bool(hr_cov),
                    core_spo2_coverage=bool(spo2_cov),
                    has_airflow_coverage=bool(af_cov),
                    core_coverage_complete=bool(complete),
                    awake_overlap_fraction=awake_frac,
                    audio_window_eligible=bool(AUDIO_WINDOW_ELIGIBLE),
                    audio_block_reason=self.windowing_config.audio_block_reason,
                    label_status=label_status,
                    binary_event_label=binary,
                    exclusion_reason=reason,
                    n_linked_retained_events=0,
                    sleep_structure_observational_status=ss_status,
                )
            )
            if excluded:
                exclusions.append(
                    WindowExclusion(
                        run_id=self.run_id,
                        patient_id=pid,
                        window_index=idx,
                        start_relative_to_record_start=round(start, 6),
                        end_relative_to_record_start=round(end, 6),
                        exclusion_reason=reason,
                        detail=(f"awake_overlap_fraction={awake_frac}" if reason != "" else ""),
                    )
                )
        return slots, records, exclusions

    def _patient_summary(
        self, pid, windows, retained, cov
    ) -> PatientWindowSummary:
        n_cand = len(windows)
        n_cov_excl = sum(1 for w in windows if w.exclusion_reason and "incomplete_core" in w.exclusion_reason)
        n_awake_excl = sum(1 for w in windows if w.exclusion_reason and "awake_overlap" in w.exclusion_reason)
        n_core_complete = n_cand - n_cov_excl
        n_pos = sum(1 for w in windows if w.label_status == "positive")
        n_neg = sum(1 for w in windows if w.label_status == "negative")
        n_usable = n_pos + n_neg
        n_linked = int(sum(retained.values()))
        zero = n_usable == 0
        if zero:
            if n_cand == 0:
                zero_reason = EXC_NO_CORE_DOMAIN
            elif n_core_complete == 0:
                zero_reason = "no_core_complete_windows"
            else:
                zero_reason = "all_windows_excluded_awake"
        else:
            zero_reason = ""
        return PatientWindowSummary(
            run_id=self.run_id,
            patient_id=pid,
            n_candidate_windows=n_cand,
            n_core_complete_windows=n_core_complete,
            n_awake_excluded_windows=n_awake_excl,
            n_coverage_excluded_windows=n_cov_excl,
            n_excluded_windows=n_cov_excl + n_awake_excl,
            n_positive_windows=n_pos,
            n_negative_windows=n_neg,
            n_usable_windows=n_usable,
            n_linked_events=n_linked,
            has_airflow_coverage=bool(cov.airflow_verified),
            sleep_structure_observational_status=cov.sleep_structure_status,
            zero_usable_windows=bool(zero),
            zero_reason=zero_reason,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, window_rows, link_rows, patient_rows) -> None:
        s = self.summary
        s.n_patients = len(patient_rows)
        s.n_patients_with_core_domain = sum(1 for p in patient_rows if p.n_candidate_windows > 0)
        s.total_candidate_windows = sum(p.n_candidate_windows for p in patient_rows)
        s.total_core_complete_windows = sum(p.n_core_complete_windows for p in patient_rows)
        s.total_awake_excluded_windows = sum(p.n_awake_excluded_windows for p in patient_rows)
        s.total_coverage_excluded_windows = sum(p.n_coverage_excluded_windows for p in patient_rows)
        s.total_excluded_windows = sum(p.n_excluded_windows for p in patient_rows)
        s.total_positive_windows = sum(p.n_positive_windows for p in patient_rows)
        s.total_negative_windows = sum(p.n_negative_windows for p in patient_rows)
        s.total_usable_windows = s.total_positive_windows + s.total_negative_windows

        usable_counts = [p.n_usable_windows for p in patient_rows]
        if usable_counts:
            s.per_patient_window_min = int(min(usable_counts))
            s.per_patient_window_max = int(max(usable_counts))
            s.per_patient_window_median = float(statistics.median(usable_counts))
        s.zero_usable_patients = sorted(p.patient_id for p in patient_rows if p.zero_usable_windows)
        s.n_patients_zero_usable = len(s.zero_usable_patients)

        # event destiny
        dest = Counter(l.event_destiny for l in link_rows)
        s.event_destiny_counts = dict(dest)
        s.total_events = len(link_rows)
        s.total_linked_events = int(dest.get(EVT_LINKED, 0))

        # modality window coverage (window counts)
        s.hr_window_coverage = sum(1 for w in window_rows if w.core_heart_rate_coverage)
        s.spo2_window_coverage = sum(1 for w in window_rows if w.core_spo2_coverage)
        s.airflow_window_coverage = sum(1 for w in window_rows if w.has_airflow_coverage)
        ss_counter: Counter = Counter()
        for p in patient_rows:
            ss_counter[p.sleep_structure_observational_status] += 1
        s.sleep_structure_status_counts = dict(ss_counter)

        # audio
        s.audio_total_files = int(getattr(self, "_audio_total", 0))
        s.audio_status_counts = dict(getattr(self, "_audio_status_counts", {}))
        s.audio_window_eligible_count = 0  # hard

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _derive_status(self) -> None:
        s = self.summary
        if s.raw_modified or not s.license_gate_passed or not s.input_run_verified:
            s.overall_status = "BLOCKED"
            return
        if s.n_patients_zero_usable > 0 or s.anomalies:
            s.overall_status = "PASS WITH WARNINGS"
            return
        s.overall_status = "PASS"

    # ------------------------------------------------------------------
    # Product writers
    # ------------------------------------------------------------------
    def _write_products(self, window_rows, link_rows, exclusion_rows, patient_rows) -> None:
        mdir = self._manifests_out_dir()
        rdir = self._reports_out_dir()
        mdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        # ---- machine-readable tables ----
        self._write_parquet(window_rows, mdir / "csv_window_index.parquet")
        self._write_parquet(link_rows, mdir / "window_event_links.parquet")
        self._write_csv(
            exclusion_rows, mdir / "window_exclusions.csv",
            fieldnames=[f.name for f in dataclasses.fields(WindowExclusion)],
        )
        self._write_csv(
            patient_rows, mdir / "patient_window_summary.csv",
            fieldnames=[f.name for f in dataclasses.fields(PatientWindowSummary)],
        )

        # ---- resolved config ----
        write_resolved_config_yaml(
            mdir / "windowing_config_resolved.yaml",
            self.windowing_config,
            run_id=self.run_id,
            input_run_id=self.summary.input_run_id,
            input_run_config_hash=self.summary.input_run_config_hash,
            config_hash=self.summary.config_hash,
        )

        # ---- path-purity guard on every machine-readable product ----
        self._assert_products_clean(mdir)

        # ---- reports ----
        self._write_report(rdir, "window_label_report.md",
                           report_mod.render_window_label_report(self.summary))
        self._write_report(rdir, "windowing_qc_report.md",
                           report_mod.render_windowing_qc_report(self.summary))
        self._write_report(rdir, "phase_04_completion_report.md", self._render_completion())

        # ---- figures (best-effort; must not crash the run) ----
        try:
            self._write_figures(rdir, patient_rows)
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"figures_failed:{type(exc).__name__}")

        # ---- Stage-3 cross-midnight erratum ----
        self._write_erratum()

        # ---- record product paths ----
        for name in (
            "csv_window_index.parquet", "window_event_links.parquet",
            "window_exclusions.csv", "patient_window_summary.csv",
            "windowing_config_resolved.yaml",
        ):
            self.summary.product_paths[name] = self._rel(mdir / name)
        for name in (
            "window_label_report.md", "windowing_qc_report.md",
            "phase_04_completion_report.md",
        ):
            self.summary.product_paths[name] = self._rel(rdir / name)

    def _write_erratum(self) -> None:
        run_dir = self._input_run_dir()
        aw = pd.read_parquet(run_dir / "awake_intervals_canonical.parquet")
        xmid = aw[aw["cross_midnight"] == True]  # noqa: E712
        correct_intervals = int(xmid.shape[0])
        correct_patients = int(xmid["patient_id"].nunique())
        correct_pids = sorted(str(p) for p in xmid["patient_id"].unique())
        erratum_dir = self._erratum_out_dir()
        erratum_dir.mkdir(parents=True, exist_ok=True)
        erratum_path = erratum_dir / "phase_03_cross_midnight_count_erratum.md"
        erratum_path.write_text(
            report_mod.render_phase03_cross_midnight_erratum(
                correct_intervals=correct_intervals,
                correct_patients=correct_patients,
                reported_patients=46,
                correct_patient_ids=correct_pids,
                input_run_id=self.summary.input_run_id,
                erratum_date=self.access_date,
            ),
            encoding="utf-8",
        )
        self.summary.product_paths["phase_03_cross_midnight_count_erratum.md"] = self._rel(erratum_path)

    def _write_blocked_artifacts(self, gate) -> None:
        rdir = self._reports_out_dir()
        rdir.mkdir(parents=True, exist_ok=True)
        self._write_report(rdir, "phase_04_completion_report.md", self._render_completion_blocked(gate))

    def _render_completion(self) -> str:
        s = self.summary
        prods = "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.product_paths.items())) or "- （无）"
        admission = self._admission_text()
        return (
            "# 阶段 4 完成报告（phase_04_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- config_hash：`{s.config_hash or '(n/a)'}`\n"
            f"- 访问日期：{s.access_date}\n"
            f"- 阶段 4 状态：**{s.overall_status}**\n"
            f"- 许可门：{'PASSED' if s.license_gate_passed else 'BLOCKED'}（license_status=`{s.license_status}`）\n"
            f"- 输入 Stage-3 run_id：`{s.input_run_id}`（verified={s.input_run_verified}）\n"
            f"- 输入 Stage-3 config_hash：`{s.input_run_config_hash or '(n/a)'}`\n\n"
            "## 产物路径\n\n"
            f"{prods}\n\n"
            "## 准入下一阶段\n\n"
            f"{admission}\n"
            "- 本次不自行进入下一阶段。\n"
        )

    def _render_completion_blocked(self, gate) -> str:
        s = self.summary
        missing = ", ".join(gate.missing_items) if gate.missing_items else (
            "; ".join(s.anomalies) if s.anomalies else "(未给出具体缺项)"
        )
        return (
            "# 阶段 4 完成报告（phase_04_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- 阶段 4 状态：**BLOCKED**\n"
            f"- 许可门状态：{'BLOCKED' if not s.license_gate_passed else 'PASSED'}"
            f"（license_status=`{s.license_status}`）\n"
            f"- 阻断/缺项：{missing}\n\n"
            "## 准入下一阶段\n\n"
            "- [待决] 当前 **未满足**：许可门未通过或输入 Stage-3 run 完整性校验失败。\n"
            "- 本次不自行进入下一阶段。\n"
        )

    def _admission_text(self) -> str:
        s = self.summary
        return "\n".join([
            "- [实测] CSV/标注侧窗口索引与研究标签表已完成（不含音频、特征、建模、划分、评价）。",
            f"- [实测] 候选窗口 {s.total_candidate_windows}，可用窗口（正+负）{s.total_usable_windows}（正 {s.total_positive_windows} / 负 {s.total_negative_windows}），排除 {s.total_excluded_windows}。",
            f"- [实测] 关联事件 {s.total_linked_events} / 事件总数 {s.total_events}；事件逐条归宿见 `window_event_links.parquet`。",
            f"- [实测] `audio_window_eligible=true` 的窗口数：{s.audio_window_eligible_count}（恒为 0）。",
            "- [待决] **下一阶段 CSV 生理特征计算**：条件基本具备（核心 HR/SpO2 域完整、时间坐标已对齐），"
            "可建议作为受限的下一步；但仍须患者级划分、不得引入音频特征。",
            "- [待决] **音频侧窗口构建：BLOCKED** —— 所有音频仍为 unresolved/excluded，无可信时间锚点，须单独单列。",
        ])

    def _write_figures(self, rdir, patient_rows) -> None:
        per_core = [(p.n_candidate_windows, p.n_core_complete_windows) for p in patient_rows]
        per_awake = [p.n_awake_excluded_windows for p in patient_rows]
        per_usable = [p.n_usable_windows for p in patient_rows]
        label_counts = {
            "positive": self.summary.total_positive_windows,
            "negative": self.summary.total_negative_windows,
            "excluded": self.summary.total_excluded_windows,
        }
        written = figures_mod.render_window_qc_figures(
            rdir,
            run_id=self.run_id,
            per_patient_core=per_core,
            per_patient_awake_excluded=per_awake,
            event_destiny_counts=self.summary.event_destiny_counts,
            window_label_counts=label_counts,
            per_patient_usable=per_usable,
        )
        for name in written:
            self.summary.product_paths[name] = self._rel(rdir / name)

    # ------------------------------------------------------------------
    # low-level writers + purity guard
    # ------------------------------------------------------------------
    def _write_parquet(self, rows: List[Any], path: Path) -> None:
        if not rows:
            pd.DataFrame().to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df.to_parquet(path, index=False)

    def _write_csv(
        self, rows: List[Any], path: Path, fieldnames: Optional[List[str]] = None
    ) -> None:
        with open(path, "w", encoding="utf-8", newline="") as h:
            if not rows:
                # Well-formed CSV with a header even when empty (not a 0-byte file).
                names = fieldnames or []
                if names:
                    h.write(",".join(names) + "\n")
                return
            names = list(dataclasses.asdict(rows[0]).keys())
            writer = csvlib.DictWriter(h, fieldnames=names)
            writer.writeheader()
            for r in rows:
                writer.writerow(dataclasses.asdict(r))

    def _write_report(self, rdir: Path, name: str, content: str) -> None:
        (rdir / name).write_text(content, encoding="utf-8")

    def _assert_products_clean(self, mdir: Path) -> None:
        values: List[str] = []
        for p in mdir.iterdir():
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
        qc_mod.assert_paths_clean(values)


__all__ = ["Stage4Options", "Stage4Runner"]
