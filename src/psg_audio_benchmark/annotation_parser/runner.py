"""Stage-3 orchestration: license gate, controlled annotation parsing,
awake normalization, CSV signal ranges and strict audio-anchor judgment.

Mirrors the Stage-1/Stage-2 read-only, run-dir-isolated discipline:

1. **License gate** runs FIRST. If it fails, write a blocked report + completion
   report and STOP without scanning ``data/raw``.
2. If the gate passes: snapshot ``data/raw`` before; resolve patients; parse each
   annotation JSON (fidelity + rejections); canonicalize awake intervals; compute
   CSV signal time ranges; judge every audio file's time anchor; finalize event
   overlap with awake + signals; snapshot ``data/raw`` after and assert
   immutability; write the parquet/CSV products, reports and QC figures.

Nothing under ``data/raw`` is ever written, moved, renamed or deleted. No
windowing, no feature extraction, no modelling. Audio is never assumed to share
the PSG clock.
"""

from __future__ import annotations

import csv as csvlib
import dataclasses
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import Config
from ..data_audit import license_gate as gate_mod
from ..data_audit import patient_resolution as patient_mod
from ..data_audit.classifier import classify_file
from ..data_download.identity import PRIMARY_IDENTITY
from . import audio_alignment as audio_mod
from . import awake as awake_mod
from . import event_mapping as mapping_mod
from . import parse as parse_mod
from . import qc as qc_mod
from . import report as report_mod
from . import signals as signals_mod
from . import figures as figures_mod
from .schema import (
    ALIGN_EXCLUDED_TOO_SHORT,
    ALIGN_UNRESOLVED,
    ParsedEvent,
    RecordTimeAnchor,
    RejectionRecord,
    SEV_ERROR,
    Stage3Summary,
)
from .time_axis import segment_overlaps_awake

_IGNORED_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}

#: Signal modalities handled as CSV time ranges.
_SIGNAL_MODALITIES = ("spo2", "heart_rate", "airflow", "sleep_structure")
#: Audio modalities handled by the anchor judge.
_AUDIO_MODALITIES = ("smartphone_audio", "recorder_audio")

#: Default Stage-2 production run id (used only if the LATEST_RUN pointer is absent).
_DEFAULT_STAGE2_RUN_ID = "stage2-audit-licensed-20260807T145303Z"


@dataclass
class Stage3Options:
    dry_run: bool = False
    output_root: Optional[Path] = None
    relpath_base: Optional[Path] = None
    #: Explicit Stage-2 production run id for the erratum/traceability.
    stage2_run_id: str = ""

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


class Stage3Runner:
    """Runs the Stage-3 annotation sync. Construct with a loaded :class:`Config`."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage3Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage3Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.summary = Stage3Summary(
            run_id=self.run_id, access_date=self.access_date
        )
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )
        self._mapping = mapping_mod.load_event_type_mapping(cfg)
        self.summary.mapping_version = self._mapping.mapping_version
        self.summary.config_hash = str(run_metadata.get("config_hash", ""))
        asy = self.cfg.config_data.get("annotation_sync", {}) or {}
        self._min_duration = float(asy.get("minimum_source_audio_duration_seconds", 30.0))
        self._csv_tol = float(asy.get("csv_abs_rel_duration_tolerance_s", 1.0))

    # ------------------------------------------------------------------
    # Output-path resolution
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _annotations_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "annotations" / "runs" / self.run_id
        return self.cfg.path("annotations") / "runs" / self.run_id

    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "annotations" / "runs" / self.run_id
        return self.cfg.path("reports_annotations") / "runs" / self.run_id

    def _erratum_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "data_audit"
        return self.cfg.path("reports_data_audit")

    def _rel(self, path: Path) -> str:
        try:
            return qc_mod.relativize(path, self._relpath_base())
        except qc_mod.Stage3ContaminationError:
            return path.name

    # ------------------------------------------------------------------
    # Stage-2 run id resolution
    # ------------------------------------------------------------------
    def _resolve_stage2_run_id(self) -> str:
        if self.options.stage2_run_id:
            return self.options.stage2_run_id
        latest = self.cfg.path("reports_data_audit") / "LATEST_RUN.txt"
        if latest.is_file():
            try:
                txt = latest.read_text(encoding="utf-8").strip()
                if txt:
                    return txt
            except OSError:
                pass
        return _DEFAULT_STAGE2_RUN_ID

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Stage3Summary:
        qc_mod.assert_not_contaminating(
            cfg=self.cfg,
            run_id=self.run_id,
            config_hash=str(self.run_metadata.get("config_hash", "")),
            output_root=self._output_root,
        )
        self.summary.stage2_production_run_id = self._resolve_stage2_run_id()

        # 1. LICENSE GATE -- first and decisive.
        gate = gate_mod.check_license_gate(self.cfg, PRIMARY_IDENTITY)
        self.summary.license_gate_passed = gate.passed
        self.summary.license_status = gate.license_status
        if not gate.passed:
            self.summary.overall_status = "BLOCKED"
            if not self.options.dry_run:
                self._write_blocked_artifacts(gate)
            return self.summary

        if self.options.dry_run:
            self.summary.overall_status = "PASS WITH WARNINGS"
            return self.summary

        # 2. raw snapshot BEFORE
        raw_before = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())

        # 3. resolve patients
        patients, _issues = patient_mod.resolve_patients(
            self.cfg.path("data_raw"), self._relpath_base()
        )
        self.summary.n_patients = len(patients)

        # 4. build per-patient file inventory
        per_patient_files = self._inventory_files(patients)

        # 5. process each patient
        (
            parsed_events,
            rejections,
            awake_raw_rows,
            awake_canonical_rows,
            record_anchors,
            signal_ranges,
            audio_records,
            warnings,
        ) = self._process_all(patients, per_patient_files)

        # 6. raw snapshot AFTER + immutability
        raw_after = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        self.summary.raw_modified = raw_before != raw_after
        self.summary.raw_scanned = True

        # 7. aggregate
        self._aggregate(
            parsed_events=parsed_events,
            rejections=rejections,
            awake_raw_rows=awake_raw_rows,
            awake_canonical_rows=awake_canonical_rows,
            signal_ranges=signal_ranges,
            audio_records=audio_records,
            warnings=warnings,
            patients=patients,
        )

        # 8. derive status
        self._derive_status()

        # 9. write products
        self._write_products(
            parsed_events=parsed_events,
            rejections=rejections,
            awake_raw_rows=awake_raw_rows,
            awake_canonical_rows=awake_canonical_rows,
            record_anchors=record_anchors,
            signal_ranges=signal_ranges,
            audio_records=audio_records,
            warnings=warnings,
        )
        return self.summary

    # ------------------------------------------------------------------
    # File inventory
    # ------------------------------------------------------------------
    def _inventory_files(
        self, patients: List[patient_mod.PatientRecord]
    ) -> Dict[str, List[Tuple[Path, str, str]]]:
        """Map patient_id -> list of (path, relpath, modality)."""
        base = self._relpath_base()
        by_pid: Dict[str, List[Tuple[Path, str, str]]] = {
            p.patient_id_canonical: [] for p in patients
        }
        dir_rel_by_pid = {p.patient_id_canonical: p.source_patient_dir_relative for p in patients}
        raw = self.cfg.path("data_raw")
        if not raw.exists():
            return by_pid
        for dirpath, dirnames, filenames in os.walk(raw):
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
            d = Path(dirpath)
            for fname in filenames:
                if fname in _IGNORED_NAMES:
                    continue
                path = d / fname
                try:
                    rel = qc_mod.relativize(path, base)
                except qc_mod.Stage3ContaminationError:
                    continue
                parent_rel = rel.rsplit("/", 1)[0] if "/" in rel else ""
                pid = None
                for cand_pid, dir_rel in dir_rel_by_pid.items():
                    if dir_rel == parent_rel:
                        pid = cand_pid
                        break
                if pid is None:
                    continue
                ext = path.suffix.lower().lstrip(".")
                cls = classify_file(rel, fname, ext)
                by_pid[pid].append((path, rel, cls.modality_candidate))
        return by_pid

    # ------------------------------------------------------------------
    # Per-patient processing
    # ------------------------------------------------------------------
    def _process_all(
        self,
        patients: List[patient_mod.PatientRecord],
        per_patient_files: Dict[str, List[Tuple[Path, str, str]]],
    ) -> Tuple[list, ...]:
        parsed_events: List[ParsedEvent] = []
        rejections: List = []
        awake_raw_rows: List = []
        awake_canonical_rows: List = []
        record_anchors: List[RecordTimeAnchor] = []
        signal_ranges: List = []
        audio_records: List = []
        warnings: List = []

        min_dur = self._min_duration
        tol = self._csv_tol

        for prec in patients:
            pid = prec.patient_id_canonical
            files = per_patient_files.get(pid, [])
            ann_files = [(p, r) for (p, r, m) in files if m == "annotation_json"]
            csv_files = [(p, r, m) for (p, r, m) in files if m in _SIGNAL_MODALITIES]
            wav_files = [(p, r, m) for (p, r, m) in files if m in _AUDIO_MODALITIES]

            if not ann_files:
                self.summary.anomalies.append(f"patient_{pid}_no_annotation_json")
                self.summary.n_patients_with_parse_failures += 1
                # still process signals/audio for this patient if present
            else:
                ann_path, ann_rel = ann_files[0]
                result = parse_mod.parse_annotation_file(
                    ann_path,
                    run_id=self.run_id,
                    patient_id=pid,
                    source_annotation_relpath=ann_rel,
                    mapping=self._mapping,
                )
                rejections.extend(result.rejections)
                self.summary.n_events_raw += result.n_events_in_file
                if result.file_anomalies:
                    self.summary.anomalies.append(
                        f"patient_{pid}_annotation:" + ";".join(result.file_anomalies[:3])
                    )
                    self.summary.n_patients_with_parse_failures += 1
                else:
                    self.summary.n_patients_parsed_ok += 1

                rs_point = result.record_start_point
                record_anchors.append(
                    RecordTimeAnchor(
                        run_id=self.run_id,
                        patient_id=pid,
                        source_annotation_relpath=ann_rel,
                        record_start_present=result.record_start_present,
                        record_start_raw_seconds=rs_point.raw_value if rs_point else None,
                        record_start_seconds_of_day=rs_point.seconds_of_day if rs_point else None,
                        record_start_day_offset=rs_point.day_offset if rs_point else None,
                        record_start_cumulative_seconds=rs_point.cumulative_seconds if rs_point else None,
                        time_basis="annotation_record_start_clock",
                    )
                )

                # awake canonicalization
                if rs_point is not None:
                    aw = awake_mod.canonicalize_awake(
                        run_id=self.run_id,
                        patient_id=pid,
                        source_annotation_relpath=ann_rel,
                        awake_raw=result.awake_raw,
                        rs_point=rs_point,
                    )
                    awake_raw_rows.extend(aw.raw_rows)
                    awake_canonical_rows.extend(aw.canonical_rows)
                    warnings.extend(aw.warnings)
                    # accumulate provenance-backed per-patient dedup/merge counts
                    self.summary.awake_exact_duplicates_removed += int(
                        aw.stats.get("exact_duplicates_removed", 0)
                    )
                    self.summary.awake_overlaps_or_adjacent_merged += int(
                        aw.stats.get("overlaps_or_adjacent_merged", 0)
                    )
                    canonical_bounds_rel = [
                        (c.start_relative_to_record_start, c.end_relative_to_record_start)
                        for c in aw.canonical_rows
                    ]
                else:
                    canonical_bounds_rel = []

                # signals for this patient
                ranges_by_modality: Dict[str, Any] = {}
                for cpath, crel, cmod in csv_files:
                    rng, swarns = signals_mod.compute_signal_time_range(
                        cpath,
                        run_id=self.run_id,
                        patient_id=pid,
                        modality=cmod,
                        source_relpath=crel,
                        rs_point=rs_point,
                        duration_tolerance_s=tol,
                    )
                    warnings.extend(swarns)
                    if rng is not None:
                        signal_ranges.append(rng)
                        ranges_by_modality[cmod] = rng

                # finalize events: awake overlap + signal coverage
                for ev in result.parsed_events:
                    if canonical_bounds_rel:
                        overlaps, cov, frac = segment_overlaps_awake(
                            ev.event_start_relative_to_record_start,
                            ev.event_end_relative_to_record_start,
                            canonical_bounds_rel,
                        )
                        ev.overlaps_awake_interval = bool(overlaps)
                        ev.awake_overlap_seconds = float(cov)
                        ev.awake_overlap_fraction = float(frac)
                    covering, n_cov = signals_mod.event_signal_overlap(
                        ev.event_start_relative_to_record_start,
                        ev.event_end_relative_to_record_start,
                        ranges_by_modality,
                    )
                    ev.overlaps_signals = covering
                    ev.n_signals_covering = int(n_cov)
                    parsed_events.append(ev)

            # audio alignment (independent of annotation availability)
            for wpath, wrel, wmod in wav_files:
                rec, warn = audio_mod.judge_audio_alignment(
                    wpath,
                    run_id=self.run_id,
                    patient_id=pid,
                    modality=wmod,
                    source_relpath=wrel,
                    minimum_source_audio_duration_seconds=min_dur,
                )
                audio_records.append(rec)
                if warn is not None:
                    warnings.append(warn)

        return (
            parsed_events,
            rejections,
            awake_raw_rows,
            awake_canonical_rows,
            record_anchors,
            signal_ranges,
            audio_records,
            warnings,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(
        self,
        *,
        parsed_events: List[ParsedEvent],
        rejections: List,
        awake_raw_rows: List,
        awake_canonical_rows: List,
        signal_ranges: List,
        audio_records: List,
        warnings: List,
        patients: List[patient_mod.PatientRecord],
    ) -> None:
        s = self.summary
        s.n_events_standardized = len(parsed_events)
        s.n_events_rejected = len(rejections)
        s.raw_event_type_counts = dict(Counter(e.event_type_raw for e in parsed_events))
        s.standardized_type_counts = dict(
            Counter(e.event_type_standardized for e in parsed_events)
        )
        s.rejection_reason_counts = dict(Counter(r.reason for r in rejections))
        s.awake_raw_total = len(awake_raw_rows)
        s.awake_canonical_total = len(awake_canonical_rows)
        # exact_duplicates_removed / overlaps_or_adjacent_merged were accumulated
        # per-patient from canonicalization provenance in _process_all.
        # cross-midnight roll-up from canonical rows:
        pids_xmid: set = set()
        xmid_int = 0
        for c in awake_canonical_rows:
            if c.cross_midnight:
                xmid_int += 1
                pids_xmid.add(c.patient_id)
        s.awake_cross_midnight_intervals = xmid_int
        s.events_overlapping_awake = sum(
            1 for e in parsed_events if e.overlaps_awake_interval
        )

        # signals
        sig_mod: Counter = Counter()
        sig_ver: Counter = Counter()
        for r in signal_ranges:
            sig_mod[r.modality] += 1
            if r.verification_status == "verified":
                sig_ver[r.modality] += 1
            if r.cross_midnight:
                pids_xmid.add(r.patient_id)
        s.signal_modality_counts = dict(sig_mod)
        s.signal_verified_counts = dict(sig_ver)
        s.patients_cross_midnight = len(pids_xmid)

        # audio
        s.audio_total = len(audio_records)
        s.audio_status_counts = dict(Counter(r.alignment_status for r in audio_records))
        s.n_audio_excluded_too_short = sum(
            1 for r in audio_records if r.alignment_status == ALIGN_EXCLUDED_TOO_SHORT
        )
        s.short_audio_files = sorted(
            r.source_relpath for r in audio_records if r.alignment_status == ALIGN_EXCLUDED_TOO_SHORT
        )

        s.warnings_total = len(warnings)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _derive_status(self) -> None:
        s = self.summary
        if s.raw_modified:
            s.overall_status = "BLOCKED"
            return
        if (
            s.warnings_total > 0
            or s.n_events_rejected > 0
            or s.n_patients_with_parse_failures > 0
        ):
            s.overall_status = "PASS WITH WARNINGS"
            return
        s.overall_status = "PASS"

    # ------------------------------------------------------------------
    # Product writers
    # ------------------------------------------------------------------
    def _write_products(
        self,
        *,
        parsed_events: List[ParsedEvent],
        rejections: List,
        awake_raw_rows: List,
        awake_canonical_rows: List,
        record_anchors: List[RecordTimeAnchor],
        signal_ranges: List,
        audio_records: List,
        warnings: List,
    ) -> None:
        adir = self._annotations_out_dir()
        rdir = self._reports_out_dir()
        adir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        # ---- machine-readable tables ----
        self._write_parquet(parsed_events, adir / "parsed_events.parquet")
        self._write_csv(
            rejections,
            adir / "event_parse_rejections.csv",
            fieldnames=[f.name for f in dataclasses.fields(RejectionRecord)],
        )
        self._write_parquet(awake_raw_rows, adir / "awake_intervals_raw.parquet")
        self._write_parquet(awake_canonical_rows, adir / "awake_intervals_canonical.parquet")
        self._write_parquet(record_anchors, adir / "record_time_anchors.parquet")
        self._write_parquet(signal_ranges, adir / "signal_time_ranges.parquet")
        self._write_csv(audio_records, adir / "audio_time_alignment_inventory.csv")
        self._write_csv(warnings, adir / "synchronization_warnings.csv")

        # ---- path purity guard on every machine-readable product ----
        self._assert_products_clean(adir)

        # ---- reports ----
        self._write_report(rdir, "annotation_parse_report.md",
                           report_mod.render_annotation_parse_report(self.summary))
        self._write_report(rdir, "time_synchronization_report.md",
                           report_mod.render_time_synchronization_report(self.summary))
        self._write_report(rdir, "event_type_mapping_report.md",
                           report_mod.render_event_type_mapping_report(self.summary, self._mapping))
        self._write_report(rdir, "phase_03_completion_report.md",
                           self._render_completion())

        # ---- figures (best-effort; must not crash the run) ----
        try:
            self._write_figures(rdir, parsed_events, awake_raw_rows, awake_canonical_rows, signal_ranges)
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"figures_failed:{type(exc).__name__}")

        # ---- phase-02 input-QC erratum (written by Stage 3; fixed path) ----
        erratum_dir = self._erratum_out_dir()
        erratum_dir.mkdir(parents=True, exist_ok=True)
        erratum_path = erratum_dir / "phase_02_input_qc_erratum.md"
        erratum_path.write_text(
            report_mod.render_phase02_input_qc_erratum(
                threshold_seconds=self._min_duration,
                affected_files_relpaths=self.summary.short_audio_files,
                stage2_production_run_id=self.summary.stage2_production_run_id,
                access_date=self.access_date,
            ),
            encoding="utf-8",
        )
        self.summary.product_paths["phase_02_input_qc_erratum.md"] = self._rel(erratum_path)

        # ---- record product paths ----
        for name in (
            "parsed_events.parquet", "event_parse_rejections.csv",
            "awake_intervals_raw.parquet", "awake_intervals_canonical.parquet",
            "record_time_anchors.parquet", "signal_time_ranges.parquet",
            "audio_time_alignment_inventory.csv", "synchronization_warnings.csv",
        ):
            self.summary.product_paths[name] = self._rel(adir / name)
        for name in (
            "annotation_parse_report.md", "time_synchronization_report.md",
            "event_type_mapping_report.md", "phase_03_completion_report.md",
        ):
            self.summary.product_paths[name] = self._rel(rdir / name)

    def _write_blocked_artifacts(self, gate) -> None:
        rdir = self._reports_out_dir()
        rdir.mkdir(parents=True, exist_ok=True)
        self._write_report(rdir, "phase_03_completion_report.md", self._render_completion_blocked(gate))

    def _render_completion(self) -> str:
        s = self.summary
        prods = "\n".join(f"- `{k}`：{v}" for k, v in sorted(s.product_paths.items())) or "- （无）"
        admission = self._admission_text()
        return (
            "# 阶段 3 完成报告（phase_03_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- config_hash：`{s.config_hash or '(n/a)'}`\n"
            f"- 访问日期：{s.access_date}\n"
            f"- 阶段 3 状态：**{s.overall_status}**\n"
            f"- 许可门：{'PASSED' if s.license_gate_passed else 'BLOCKED'}（license_status=`{s.license_status}`）\n"
            f"- 阶段 2 输入 run_id：`{s.stage2_production_run_id}`\n\n"
            "## 产物路径\n\n"
            f"{prods}\n\n"
            "## 准入下一阶段（窗口标签构建）\n\n"
            f"{admission}\n"
            "- 本次不自行进入阶段 4。\n"
        )

    def _render_completion_blocked(self, gate) -> str:
        s = self.summary
        missing = ", ".join(gate.missing_items) if gate.missing_items else "(未给出具体缺项)"
        return (
            "# 阶段 3 完成报告（phase_03_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- 阶段 3 状态：**BLOCKED**\n"
            f"- 许可门状态：BLOCKED（license_status=`{s.license_status}`）\n"
            f"- 缺项：{missing}\n\n"
            "## 准入下一阶段\n\n"
            f"- [待决] 当前 **未满足**：许可门未通过。需先人工依据 Science Data Bank 正式条款补齐"
            "许可证据并重跑阶段 2/3，方可评估进入阶段 4。\n"
            "- 本次不自行进入阶段 4。\n"
        )

    def _admission_text(self) -> str:
        s = self.summary
        n_audio = s.audio_status_counts.get(ALIGN_UNRESOLVED, 0)
        lines = [
            "- [实测] CSV/标注侧：时间轴已对齐到 record_start 时钟（CSV 各模态可验证数见 time_synchronization_report.md）。",
            f"- [实测] 音频侧：{n_audio} 个音频文件为 `unresolved_no_trustworthy_audio_time_anchor`，"
            f"{s.n_audio_excluded_too_short} 个为 `excluded_audio_too_short_for_analysis`。",
            "- [待决] **阶段 4 音频窗口构建：BLOCKED** —— 没有任何音频文件具有可证明的时间锚点。",
            "  因此不得把 PSG 事件直接映射到音频窗口，也不得静默以 record_start 代替音频采集起点。",
            "- [待决] 可受限推进的仅 CSV/标注侧工作（如基于 CSV 信号的时间索引），须以受限范围单列，"
            "不可混称为完整多模态准入。",
        ]
        return "\n".join(lines)

    def _write_figures(self, rdir, parsed_events, awake_raw_rows, awake_canonical_rows, signal_ranges) -> None:
        event_rels = [e.event_start_relative_to_record_start for e in parsed_events]
        # per-patient raw vs canonical counts
        raw_by_pid: Counter = Counter(c.patient_id for c in awake_raw_rows)
        can_by_pid: Counter = Counter(c.patient_id for c in awake_canonical_rows)
        pairs = sorted(
            (raw_by_pid[pid], can_by_pid[pid]) for pid in raw_by_pid
        )
        sig_dur: Dict[str, List[float]] = {}
        for r in signal_ranges:
            if r.relative_duration_seconds is not None:
                sig_dur.setdefault(r.modality, []).append(r.relative_duration_seconds)
        fracs = [e.awake_overlap_fraction for e in parsed_events if e.awake_overlap_fraction > 0]
        written = figures_mod.render_timeline_qc_figures(
            rdir,
            run_id=self.run_id,
            event_start_rels=event_rels,
            awake_raw_canonical_pairs=pairs,
            signal_durations_by_modality=sig_dur,
            n_cross_midnight_patients=self.summary.patients_cross_midnight,
            n_total_patients=self.summary.n_patients,
            awake_overlap_fractions=fracs,
        )
        for name in written:
            self.summary.product_paths[name] = self._rel(rdir / name)

    # ------------------------------------------------------------------
    # low-level writers + purity guard
    # ------------------------------------------------------------------
    def _write_parquet(self, rows: List[Any], path: Path) -> None:
        if not rows:
            # empty table with explicit columns is still useful; write empty frame
            pd.DataFrame().to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df.to_parquet(path, index=False)

    def _write_csv(self, rows: List[Any], path: Path, fieldnames: Optional[List[str]] = None) -> None:
        with open(path, "w", encoding="utf-8", newline="") as h:
            if not rows:
                # Still emit a header so an empty table (e.g. zero rejections) is
                # a well-formed CSV with a schema, not a 0-byte mystery file.
                names = fieldnames
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

    def _assert_products_clean(self, adir: Path) -> None:
        """Scan every CSV/parquet-relative string for absolute / test-path tokens."""
        values: List[str] = []
        for p in adir.iterdir():
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


__all__ = ["Stage3Options", "Stage3Runner"]
