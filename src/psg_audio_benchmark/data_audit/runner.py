"""Stage-2 orchestration: license gate, structure audit, manifests and reports.

The runner ties the data_audit modules into a single read-only pass:

1. **License gate** (prompt section 1) — the FIRST thing checked. If the human
   evidence is missing/incomplete, write ``license_gate_blocked.md`` and STOP
   without scanning ``data/raw``.
2. If the gate passes: snapshot ``data/raw`` before; resolve patients; classify
   every file; probe audio/CSV/annotation metadata; aggregate; snapshot after
   and assert immutability; write the six CSV products + reports.

Nothing under ``data/raw`` is ever written, moved, renamed or deleted. Audio is
probed header-only (stdlib ``wave``); CSV is stream-probed (bounded sample); JSON
is parsed structure-only with desensitised examples. No final event table, no
window labels, no synchronisation, no features, no models (prompt "禁止" list).

Output isolation mirrors Stage 1 (``data_download.runner``):

* **Production** (``output_root`` unset): artifacts under per-run dirs
  ``data/manifests/runs/<run_id>/`` and ``reports/data_audit/runs/<run_id>/``.
  Fixed legacy paths are regenerated only by the CLI ``main()`` after a
  successful production run.
* **Isolated / test** (``output_root`` set): every artifact strictly under
  ``<output_root>``; production dirs never touched. The contamination guard
  refuses an ``output_root`` that points at (or into) a production dir.
* A production run that *looks* like a test fixture (test run_id / config_hash /
  tmp raw) is refused, so a test can never masquerade as production.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..data_download.identity import PRIMARY_IDENTITY
from . import annotation_probe as annotation_mod
from . import audio_metadata as audio_mod
from . import classifier as classifier_mod
from . import csv_probe as csv_mod
from . import license_gate as gate_mod
from . import manifests as manifests_mod
from . import patient_resolution as patient_mod
from . import reports as reports_mod

_IGNORED_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache"}

_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "tmp", "temp")

# Modality -> short label used in availability/patient manifests.
_MOD_LABELS = {
    classifier_mod.MODALITY_SMARTPHONE_AUDIO: "smartphone_audio",
    classifier_mod.MODALITY_RECORDER_AUDIO: "recorder_audio",
    classifier_mod.MODALITY_SPO2: "spo2",
    classifier_mod.MODALITY_HEART_RATE: "heart_rate",
    classifier_mod.MODALITY_AIRFLOW: "airflow",
    classifier_mod.MODALITY_SLEEP_STRUCTURE: "sleep_structure",
    classifier_mod.MODALITY_ANNOTATION_JSON: "annotation_json",
    classifier_mod.MODALITY_ANNOTATION_TXT: "annotation_txt",
    classifier_mod.MODALITY_UNKNOWN: "unknown",
}


class Stage2ContaminationError(RuntimeError):
    """Raised when a run would write test output into production paths, or
    when a production run looks like a test fixture."""


@dataclass
class Stage2Options:
    dry_run: bool = False
    #: Isolated output root (tests). When set, ALL artifacts live under it.
    output_root: Optional[Path] = None
    #: Base for relativizing data paths (project root in production).
    relpath_base: Optional[Path] = None
    #: Stage-1 production run id whose SHA-256 manifest is the content baseline.
    stage1_run_id: str = ""
    #: Optional explicit Stage-1 manifest path (tests point this at a fixture).
    stage1_manifest_path: Optional[Path] = None

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


@dataclass
class Stage2Summary:
    run_id: str
    access_date: str = ""
    overall_status: str = "BLOCKED"
    license_gate: Optional[gate_mod.LicenseGateResult] = None
    license_status: str = "TBD_AFTER_MANUAL_LICENSE_REVIEW"
    stage1_run_id: str = ""
    stage1_manifest_relpath: str = ""
    n_patients: int = 0
    patients: List[patient_mod.PatientRecord] = field(default_factory=list)
    patient_issues: List[patient_mod.PatientIssue] = field(default_factory=list)
    n_files: int = 0
    file_rows: List[Dict[str, Any]] = field(default_factory=list)
    audio_summary: Dict[str, Any] = field(default_factory=dict)
    csv_summary: Dict[str, Any] = field(default_factory=dict)
    annotation_summary: Dict[str, Any] = field(default_factory=dict)
    airflow_patients: int = 0
    anomalies: List[str] = field(default_factory=list)
    raw_before: Dict[str, str] = field(default_factory=dict)
    raw_after: Dict[str, str] = field(default_factory=dict)
    raw_modified: bool = False
    raw_scanned: bool = False  # did this run actually read raw content?
    product_paths: Dict[str, str] = field(default_factory=dict)
    out_base: str = ""

    @property
    def license_gate_passed(self) -> bool:
        return bool(self.license_gate and self.license_gate.passed)


class Stage2Runner:
    """Runs the Stage-2 audit. Construct with a loaded :class:`Config`."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage2Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage2Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date = (
            datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        self.summary = Stage2Summary(run_id=self.run_id, access_date=self.access_date)
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )
        self._stage1_hashes: Dict[str, str] = {}
        self._stage1_sizes: Dict[str, int] = {}
        # Per-relative-path annotation probes, retained so the annotation
        # structure inventory and the aggregate presence flags (record_start /
        # awake_intervals / events) report the REAL top-level structure rather
        # than a hardcoded placeholder (prompt 4.3, 5.4).
        self._ann_probes: Dict[str, annotation_mod.AnnotationProbe] = {}

    # ------------------------------------------------------------------
    # Output-path resolution (isolation vs production run-dir)
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _manifests_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "manifests"
        return self.cfg.path("data_manifests") / "runs" / self.run_id

    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "data_audit"
        return self.cfg.path("reports_data_audit") / "runs" / self.run_id

    def _out_base_summary(self) -> str:
        base = self._manifests_out_dir()
        try:
            return str(base.resolve().relative_to(
                self.cfg.project_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(base).replace("\\", "/")

    def artifact_paths(self) -> Dict[str, Path]:
        return {
            "manifests_dir": self._manifests_out_dir(),
            "reports_dir": self._reports_out_dir(),
        }

    # ------------------------------------------------------------------
    # Contamination guard (mirrors Stage 1)
    # ------------------------------------------------------------------
    def _assert_not_contaminating(self) -> None:
        run_id_low = (self.run_id or "").lower()
        config_hash = str(self.run_metadata.get("config_hash", "") or "")

        if self._output_root is None:
            hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
            if hit:
                raise Stage2ContaminationError(
                    f"Production-mode run refused: run_id {self.run_id!r} "
                    f"contains test marker(s) {hit}."
                )
            if config_hash.lower() in ("test", "fixture", "pytest"):
                raise Stage2ContaminationError(
                    f"Production-mode run refused: config_hash {config_hash!r} "
                    f"looks like a test value."
                )
            raw = self.cfg.path("data_raw").resolve()
            try:
                raw.relative_to(self.cfg.project_root.resolve())
            except ValueError:
                raise Stage2ContaminationError(
                    f"Production-mode run refused: data_raw {raw} is outside "
                    f"the project root."
                ) from None
        else:
            root = self._output_root
            forbidden = [
                self.cfg.project_root.resolve(),
                self.cfg.path("data_manifests").resolve(),
                self.cfg.path("reports_data_audit").resolve(),
                self.cfg.path("reports_data_download").resolve(),
            ]
            for f in forbidden:
                try:
                    root.relative_to(f)
                    inside = True
                except ValueError:
                    inside = (root == f)
                if inside:
                    raise Stage2ContaminationError(
                        f"Isolated output_root {root} must not be (or be inside) "
                        f"the production path {f}."
                    )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _rel_to_base(self, path: Path) -> str:
        base = self._relpath_base()
        try:
            rel = path.resolve().relative_to(base)
        except ValueError as exc:
            raise Stage2ContaminationError(
                f"Refusing to record a non-relativizable path {path!s} "
                f"(base={base})."
            ) from exc
        return str(rel).replace("\\", "/")

    def _scan_dir(self, root: Path) -> List[Path]:
        out: List[Path] = []
        if not root.exists():
            return out
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
            for fname in filenames:
                if fname in _IGNORED_NAMES:
                    continue
                out.append(Path(dirpath) / fname)
        return out

    def _snapshot_raw(self) -> Dict[str, str]:
        snap: Dict[str, str] = {}
        for path in self._scan_dir(self.cfg.path("data_raw")):
            rel = self._rel_to_base(path)
            try:
                st = path.stat()
                snap[rel] = f"{st.st_size}:{st.st_mtime_ns}"
            except OSError as exc:
                snap[rel] = f"-1:stat_failed:{type(exc).__name__}"
        return snap

    # ------------------------------------------------------------------
    # Stage-1 baseline loading
    # ------------------------------------------------------------------
    def _resolve_stage1_manifest(self) -> Optional[Path]:
        if self.options.stage1_manifest_path is not None:
            return Path(self.options.stage1_manifest_path)
        rid = self.options.stage1_run_id
        if rid:
            cand = self.cfg.path("data_manifests") / "runs" / rid / "file_manifest_stage1.csv"
            if cand.is_file():
                return cand
        # fixed path
        fixed = self.cfg.path("data_manifests") / "file_manifest_stage1.csv"
        if fixed.is_file():
            return fixed
        return None

    def _load_stage1_baseline(self) -> None:
        path = self._resolve_stage1_manifest()
        if path is None:
            self.summary.anomalies.append(
                "stage1_manifest_not_found; stage1_sha256 left empty"
            )
            self.summary.stage1_manifest_relpath = ""
            return
        try:
            self.summary.stage1_manifest_relpath = self._rel_to_base(path)
        except Stage2ContaminationError:
            # Fixture-provided manifest outside the base: record name only.
            self.summary.stage1_manifest_relpath = path.name
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    rel = (row.get("relative_path") or "").replace("\\", "/")
                    sha = (row.get("sha256") or "").strip()
                    size = row.get("size_bytes", "")
                    if rel and sha:
                        self._stage1_hashes[rel] = sha
                        try:
                            self._stage1_sizes[rel] = int(size)
                        except (TypeError, ValueError):
                            self._stage1_sizes[rel] = -1
        except OSError as exc:
            self.summary.anomalies.append(
                f"stage1_manifest_read_failed:{type(exc).__name__}:{exc}"
            )
        if self.options.stage1_run_id:
            self.summary.stage1_run_id = self.options.stage1_run_id

    # ------------------------------------------------------------------
    # Per-file audit
    # ------------------------------------------------------------------
    def _audit_file(
        self, path: Path, patient_id: str, rel: str
    ) -> tuple:
        """Probe one file and return (enriched_row, cls, am, csvp, annp)."""
        ext = path.suffix.lower().lstrip(".")
        name = path.name
        try:
            size = path.stat().st_size
        except OSError:
            size = -1

        am: Optional[audio_mod.AudioMetadata] = None
        csvp: Optional[csv_mod.CsvProbe] = None
        annp: Optional[annotation_mod.AnnotationProbe] = None
        anomalies: List[str] = []
        parse_status = "ok"

        if ext in ("wav", "mp3", "flac", "ogg", "m4a"):
            am = audio_mod.probe_wav(path) if ext == "wav" else audio_mod.probe_wav(path)
            cls = classifier_mod.classify_file(
                rel, name, ext, audio_kind=(am.container if am else ext)
            )
            if am and am.readability_status not in (
                audio_mod.AUDIO_READABLE,
            ):
                anomalies.append(f"audio:{am.readability_status}:{am.error}")
                parse_status = "metadata_unavailable" if am.readability_status == audio_mod.AUDIO_METADATA_UNAVAILABLE else "anomaly"
        elif ext in ("csv", "tsv"):
            csvp = csv_mod.probe_csv(path)
            cls = classifier_mod.classify_file(
                rel, name, ext, csv_header=csvp.columns
            )
            if csvp.parse_anomalies:
                anomalies.append("csv:" + ";".join(csvp.parse_anomalies[:4]))
                parse_status = "anomaly"
        elif ext == "json":
            annp = annotation_mod.probe_annotation(path)
            cls = classifier_mod.classify_file(
                rel, name, ext, json_keys=annp.top_level_keys
            )
            if annp.anomalies:
                anomalies.append("annotation:" + ";".join(annp.anomalies[:4]))
                parse_status = "anomaly"
        elif ext in ("txt", "text"):
            annp = annotation_mod.probe_annotation(path)
            cls = classifier_mod.classify_file(rel, name, ext)
            if annp.anomalies:
                anomalies.append("annotation:" + ";".join(annp.anomalies[:4]))
        else:
            cls = classifier_mod.classify_file(rel, name, ext)
            anomalies.append(f"unhandled_extension:{ext}")
            parse_status = "anomaly"

        row: Dict[str, Any] = {
            "run_id": self.run_id,
            "stage1_run_id": self.summary.stage1_run_id,
            "stage1_sha256": self._stage1_hashes.get(rel, ""),
            "relative_path": rel,
            "patient_id_canonical": patient_id,
            "file_name": name,
            "extension": ext,
            "size_bytes": size,
            "modality_candidate": cls.modality_candidate,
            "classification_evidence": "; ".join(cls.classification_evidence),
            "classification_confidence": cls.classification_confidence,
            "recorder_index": "" if cls.recorder_index is None else str(cls.recorder_index),
            "audio_container": am.container if am else "",
            "audio_sample_rate_hz": _opt(am.sample_rate_hz) if am else "",
            "audio_channels": _opt(am.channels) if am else "",
            "audio_sample_width_bits": _opt(am.sample_width_bits) if am else "",
            "audio_frame_count": _opt(am.frame_count) if am else "",
            "audio_header_duration_s": _opt(am.header_duration_s) if am else "",
            "audio_readability_status": am.readability_status if am else "",
            "audio_error": (am.error if am else ""),
            "csv_inferred_role": csvp.inferred_role if csvp else "",
            "csv_row_count": csvp.row_count if csvp else "",
            "csv_sample_rate_estimate_hz": _opt(csvp.sample_rate_estimate_hz) if csvp else "",
            "csv_time_column_used": csvp.time_column_used if csvp else "",
            "csv_basic_cell_missingness_ratio": _opt(csvp.basic_cell_missingness_ratio) if csvp else "",
            "annotation_format": annp.format if annp else "",
            "annotation_events_count": annp.events_count if annp else "",
            "annotation_event_field_set": "; ".join(annp.event_field_set) if annp else "",
            "annotation_field_variants": "; ".join(annp.field_variants) if annp else "",
            "annotation_raw_event_type_freq": "; ".join(
                f"{k}={v}" for k, v in sorted((annp.raw_event_type_freq or {}).items())
            ) if annp else "",
            "parse_status": parse_status,
            "anomalies": "; ".join(anomalies),
        }
        if annp and annp.format:
            self._ann_probes[rel] = annp
        return row, cls, am, csvp, annp

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(
        self,
        per_patient: Dict[str, Dict[str, Any]],
    ) -> None:
        """Fill audio/csv/annotation summaries + airflow patient count."""
        audio_containers: Counter = Counter()
        audio_rates: Counter = Counter()
        audio_chans: Counter = Counter()
        audio_widths: Counter = Counter()
        durs: List[float] = []
        n_unreadable = 0
        csv_roles: Counter = Counter()
        csv_rates: Counter = Counter()
        missing_vals: List[float] = []
        n_no_time = 0
        ann_formats: Counter = Counter()
        event_fields: set = set()
        field_variants: set = set()
        raw_event_types: set = set()
        n_annotations = 0
        airflow_patients = 0

        for pid, info in per_patient.items():
            if info["availability"].get("airflow"):
                airflow_patients += 1

        for row in self.summary.file_rows:
            mod = row["modality_candidate"]
            if row.get("audio_container"):
                audio_containers[row["audio_container"]] += 1
            if row.get("audio_sample_rate_hz"):
                audio_rates[row["audio_sample_rate_hz"]] += 1
            if row.get("audio_channels"):
                audio_chans[row["audio_channels"]] += 1
            if row.get("audio_sample_width_bits"):
                audio_widths[row["audio_sample_width_bits"]] += 1
            if row.get("audio_header_duration_s"):
                try:
                    durs.append(float(row["audio_header_duration_s"]))
                except ValueError:
                    pass
            if row.get("audio_readability_status") and row["audio_readability_status"] != audio_mod.AUDIO_READABLE:
                n_unreadable += 1
            if row.get("csv_inferred_role"):
                csv_roles[row["csv_inferred_role"]] += 1
            if row.get("csv_sample_rate_estimate_hz"):
                csv_rates[row["csv_sample_rate_estimate_hz"]] += 1
            if row.get("csv_basic_cell_missingness_ratio") not in ("", None):
                try:
                    missing_vals.append(float(row["csv_basic_cell_missingness_ratio"]))
                except ValueError:
                    pass
            if row.get("csv_time_column_used") == "" and row.get("csv_inferred_role"):
                # CSV with a role but no usable time column.
                pass
            if row.get("annotation_format"):
                ann_formats[row["annotation_format"]] += 1
                n_annotations += 1

        # annotation booleans / sets come from the annotation rows directly
        ann_rows = [r for r in self.summary.file_rows if r.get("annotation_format")]
        for r in ann_rows:
            fs = [s.strip() for s in (r.get("annotation_event_field_set") or "").split(";") if s.strip()]
            event_fields.update(fs)
            fv = [s.strip() for s in (r.get("annotation_field_variants") or "").split(";") if s.strip()]
            field_variants.update(fv)
            for tok in (r.get("annotation_raw_event_type_freq") or "").split(";"):
                tok = tok.strip()
                if "=" in tok:
                    raw_event_types.add(tok.split("=", 1)[0].strip())
        # presence flags: measured directly from the annotation probes (prompt
        # 4.3). record_start / awake_intervals / events are top-level JSON keys
        # whose real presence is reported from the files, never assumed from the
        # literature and never left as a hardcoded placeholder.
        probes = self._ann_probes.values()
        has_record_start = any(p.record_start_present for p in probes)
        has_awake = any(p.awake_intervals_present for p in probes)
        has_events_any = any(p.events_present for p in probes)

        n_csv_no_time = sum(
            1
            for r in self.summary.file_rows
            if r.get("csv_inferred_role") and not r.get("csv_time_column_used")
        )

        self.summary.audio_summary = {
            "containers": dict(audio_containers),
            "sample_rates": dict(audio_rates),
            "channels": dict(audio_chans),
            "sample_widths": dict(audio_widths),
            "dur_min": round(min(durs), 3) if durs else None,
            "dur_max": round(max(durs), 3) if durs else None,
            "n_unreadable": n_unreadable,
        }
        self.summary.csv_summary = {
            "n_csv": sum(csv_roles.values()),
            "roles": dict(csv_roles),
            "sample_rates": dict(csv_rates),
            "missing_min": round(min(missing_vals), 6) if missing_vals else None,
            "missing_max": round(max(missing_vals), 6) if missing_vals else None,
            "n_no_time": n_csv_no_time,
        }
        self.summary.annotation_summary = {
            "n_annotations": n_annotations,
            "formats": dict(ann_formats),
            "has_record_start": has_record_start,
            "has_awake": has_awake,
            "has_events": has_events_any,
            "event_field_set": sorted(event_fields),
            "field_variants": sorted(field_variants),
            "raw_event_types": sorted(raw_event_types),
            "time_surface_comparable": "未判定（需同步阶段核验）",
        }
        self.summary.airflow_patients = airflow_patients

    # ------------------------------------------------------------------
    # Manifest + report writers
    # ------------------------------------------------------------------
    def _write_products(self, per_patient: Dict[str, Dict[str, Any]]) -> None:
        mdir = self._manifests_out_dir()
        rdir = self._reports_out_dir()
        mdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        # 1. patient_manifest.csv
        pm_rows: List[Dict[str, Any]] = []
        for pid, info in per_patient.items():
            counts = info["counts"]
            avail = info["availability"]
            ann_fmts = sorted(info["annotation_formats"])
            pm_rows.append({
                "run_id": self.run_id,
                "patient_id_raw": info["raw"],
                "patient_id_canonical": pid,
                "source_patient_dir_relative": info["dir_rel"],
                "mapping_rule_version": patient_mod.MAPPING_RULE_VERSION,
                "file_count": info["n_files"],
                "n_smartphone_audio": counts.get(_MOD_LABELS[classifier_mod.MODALITY_SMARTPHONE_AUDIO], 0),
                "n_recorder_audio": counts.get(_MOD_LABELS[classifier_mod.MODALITY_RECORDER_AUDIO], 0),
                "recorder_file_count": info["recorder_count"],
                "n_spo2": counts.get(_MOD_LABELS[classifier_mod.MODALITY_SPO2], 0),
                "n_heart_rate": counts.get(_MOD_LABELS[classifier_mod.MODALITY_HEART_RATE], 0),
                "n_airflow": counts.get(_MOD_LABELS[classifier_mod.MODALITY_AIRFLOW], 0),
                "n_sleep_structure": counts.get(_MOD_LABELS[classifier_mod.MODALITY_SLEEP_STRUCTURE], 0),
                "n_annotation_json": counts.get(_MOD_LABELS[classifier_mod.MODALITY_ANNOTATION_JSON], 0),
                "n_annotation_txt": counts.get(_MOD_LABELS[classifier_mod.MODALITY_ANNOTATION_TXT], 0),
                "n_unknown": counts.get(_MOD_LABELS[classifier_mod.MODALITY_UNKNOWN], 0),
                "has_smartphone_audio": bool(avail["smartphone_audio"]),
                "has_recorder_audio": bool(avail["recorder_audio"]),
                "has_spo2": bool(avail["spo2"]),
                "has_heart_rate": bool(avail["heart_rate"]),
                "has_airflow": bool(avail["airflow"]),
                "has_sleep_structure": bool(avail["sleep_structure"]),
                "has_annotation_json": bool(avail["annotation_json"]),
                "has_annotation_txt": bool(avail["annotation_txt"]),
                "annotation_format": "+".join(ann_fmts) if ann_fmts else "",
                "audit_status": "audited",
                "issue_flags": ";".join(info["issue_flags"]),
            })
        manifests_mod.write_named("patient_manifest", pm_rows, mdir)

        # 2. file_manifest_enriched.csv
        manifests_mod.write_named("file_manifest_enriched", self.summary.file_rows, mdir)

        # 3. signal_availability_matrix.csv
        sam_rows = []
        for pid, info in per_patient.items():
            avail = info["availability"]
            ann_fmts = sorted(info["annotation_formats"])
            sam_rows.append({
                "run_id": self.run_id,
                "patient_id_canonical": pid,
                "has_smartphone_audio": bool(avail["smartphone_audio"]),
                "recorder_file_count": info["recorder_count"],
                "has_recorder_audio": bool(avail["recorder_audio"]),
                "has_spo2": bool(avail["spo2"]),
                "has_heart_rate": bool(avail["heart_rate"]),
                "has_airflow": bool(avail["airflow"]),
                "has_sleep_structure": bool(avail["sleep_structure"]),
                "has_annotation": bool(avail["annotation_json"] or avail["annotation_txt"]),
                "annotation_format": "+".join(ann_fmts) if ann_fmts else "",
            })
        manifests_mod.write_named("signal_availability_matrix", sam_rows, mdir)

        # 4. annotation_structure_inventory.csv
        inv_rows = []
        for r in self.summary.file_rows:
            if not r.get("annotation_format"):
                continue
            ap = self._ann_probes.get(r["relative_path"])
            inv_rows.append({
                "run_id": self.run_id,
                "relative_path": r["relative_path"],
                "patient_id_canonical": r["patient_id_canonical"],
                "annotation_format": r["annotation_format"],
                "annotation_top_level_keys": "; ".join(ap.top_level_keys) if ap else "",
                "annotation_record_start_present": bool(ap.record_start_present) if ap else "",
                "annotation_awake_intervals_present": bool(ap.awake_intervals_present) if ap else "",
                "annotation_awake_intervals_count": ap.awake_intervals_count if ap else "",
                "annotation_events_count": r.get("annotation_events_count", ""),
                "annotation_event_field_set": r.get("annotation_event_field_set", ""),
                "annotation_field_variants": r.get("annotation_field_variants", ""),
                "annotation_raw_event_type_freq": r.get("annotation_raw_event_type_freq", ""),
                "annotation_anomalies": "; ".join(ap.anomalies) if ap else "",
                "annotation_preliminary_note": manifests_mod.EVENT_DIST_PRELIMINARY_NOTE,
            })
        manifests_mod.write_named("annotation_structure_inventory", inv_rows, mdir)

        # 5. event_distribution_preliminary.csv (+ README)
        ev_rows = []
        for r in self.summary.file_rows:
            if not r.get("annotation_format"):
                continue
            freq = r.get("annotation_raw_event_type_freq") or ""
            for tok in freq.split(";"):
                tok = tok.strip()
                if "=" not in tok:
                    continue
                et, cnt = tok.split("=", 1)
                try:
                    count = int(cnt)
                except ValueError:
                    continue
                ev_rows.append({
                    "run_id": self.run_id,
                    "patient_id_canonical": r["patient_id_canonical"],
                    "relative_path": r["relative_path"],
                    "event_type_raw": et.strip(),
                    "count": count,
                    "preliminary_note": manifests_mod.EVENT_DIST_PRELIMINARY_NOTE,
                })
        manifests_mod.write_named("event_distribution_preliminary", ev_rows, mdir)
        manifests_mod.write_event_distribution_readme(mdir)

        # 6. patient_id_resolution_issues.csv
        issue_rows = [
            {
                "run_id": self.run_id,
                "issue_type": it.issue_type,
                "patient_id": it.patient_id,
                "source_dir_relative": it.source_dir_relative,
                "description": it.description,
            }
            for it in self.summary.patient_issues
        ]
        manifests_mod.write_named("patient_id_resolution_issues", issue_rows, mdir)

        # Record product paths (project-relative where possible).
        for name in manifests_mod.all_product_names():
            p = mdir / f"{name}.csv"
            self.summary.product_paths[name] = self._rel_or_name(p)

        # Reports.
        self._write_report(rdir, "license_gate_passed.md",
                           reports_mod.render_license_gate_passed(self.summary))
        self._write_report(rdir, "data_structure_exploration_report.md",
                           reports_mod.render_data_structure_exploration(self.summary))
        self._write_report(rdir, "data_audit_report.md",
                           reports_mod.render_data_audit_report(self.summary))
        self._write_report(rdir, "data_dictionary_observed.md",
                           reports_mod.render_data_dictionary(self.summary))
        self._write_report(rdir, "phase_02_completion_report.md",
                           reports_mod.render_completion_report(
                               self.summary,
                               config_hash=str(self.run_metadata.get("config_hash", "")),
                           ))

    def _write_report(self, rdir: Path, name: str, content: str) -> None:
        p = rdir / name
        p.write_text(content, encoding="utf-8")
        self.summary.product_paths[name] = self._rel_or_name(p)

    def _rel_or_name(self, p: Path) -> str:
        try:
            return self._rel_to_base(p)
        except Stage2ContaminationError:
            return p.name

    def _write_blocked_artifacts(self) -> None:
        """License-gate-blocked path: only the blocked + completion reports."""
        rdir = self._reports_out_dir()
        rdir.mkdir(parents=True, exist_ok=True)
        self._write_report(rdir, "license_gate_blocked.md",
                           reports_mod.render_license_gate_blocked(self.summary))
        self._write_report(rdir, "phase_02_completion_report.md",
                           reports_mod.render_completion_report(
                               self.summary,
                               config_hash=str(self.run_metadata.get("config_hash", "")),
                           ))

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Stage2Summary:
        self._assert_not_contaminating()
        self.summary.out_base = self._out_base_summary()

        # 1. LICENSE GATE -- the first and decisive step.
        gate = gate_mod.check_license_gate(self.cfg, PRIMARY_IDENTITY)
        self.summary.license_gate = gate
        self.summary.license_status = gate.license_status
        if not gate.passed:
            self.summary.overall_status = "BLOCKED"
            self.summary.raw_scanned = False
            if not self.options.dry_run:
                self._write_blocked_artifacts()
            return self.summary

        # Gate passed -> may read raw (read-only).
        if self.options.dry_run:
            # Dry-run still respects the gate but writes nothing.
            self.summary.overall_status = "PASS WITH WARNINGS"
            return self.summary

        # 2. raw snapshot BEFORE
        self.summary.raw_before = self._snapshot_raw()

        # 3. Stage-1 content baseline
        self._load_stage1_baseline()

        # 4. patient resolution
        patients, issues = patient_mod.resolve_patients(
            self.cfg.path("data_raw"), self._relpath_base()
        )
        self.summary.patients = patients
        self.summary.patient_issues = issues
        self.summary.n_patients = len(patients)

        # 5. per-file audit
        per_patient: Dict[str, Dict[str, Any]] = {}
        issue_by_pid = {}
        for it in issues:
            issue_by_pid.setdefault(it.patient_id, []).append(it.issue_type)

        for prec in patients:
            pdir = self._relpath_base() / prec.source_patient_dir_relative
            per_patient[prec.patient_id_canonical] = {
                "raw": prec.patient_id_raw,
                "dir_rel": prec.source_patient_dir_relative,
                "n_files": 0,
                "counts": Counter(),
                "availability": {lbl: False for lbl in _MOD_LABELS.values()},
                "recorder_count": 0,
                "annotation_formats": set(),
                "issue_flags": issue_by_pid.get(prec.patient_id_raw, []),
            }

        for path in self._scan_dir(self.cfg.path("data_raw")):
            rel = self._rel_to_base(path)
            # which patient? find the patient whose dir_rel is a prefix of rel.
            pid = self._patient_for_path(rel, per_patient)
            if pid is None:
                self.summary.anomalies.append(f"file_not_under_any_patient_dir:{rel}")
                continue
            row, cls, am, csvp, annp = self._audit_file(path, pid, rel)
            self.summary.file_rows.append(row)
            self.summary.n_files += 1
            info = per_patient[pid]
            info["n_files"] += 1
            info["counts"][_MOD_LABELS.get(cls.modality_candidate, "unknown")] += 1
            info["availability"][_MOD_LABELS.get(cls.modality_candidate, "unknown")] = True
            if cls.modality_candidate == classifier_mod.MODALITY_RECORDER_AUDIO:
                info["recorder_count"] += 1
            if annp and annp.format:
                info["annotation_formats"].add(annp.format)

        self.summary.raw_scanned = True

        # 6. aggregate
        self._aggregate(per_patient)

        # 7. raw snapshot AFTER + immutability
        self.summary.raw_after = self._snapshot_raw()
        self.summary.raw_modified = self.summary.raw_before != self.summary.raw_after

        # 8. status -- derived BEFORE writing so the completion report carries
        #    the true overall_status (PASS / PASS WITH WARNINGS / BLOCKED) rather
        #    than the dataclass default 'BLOCKED'.
        self._derive_status()

        # 9. write products
        self._write_products(per_patient)
        return self.summary

    def _patient_for_path(
        self, rel: str, per_patient: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Return the patient whose source dir is the parent of ``rel``."""
        # rel looks like '<base>/data/raw/V5/Data/01/01_HR.csv' (prod) or
        # 'raw/01/01_HR.csv' (fixture). The patient dir is the immediate parent.
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        for pid, info in per_patient.items():
            if info["dir_rel"] == parent:
                return pid
        return None

    def _derive_status(self) -> None:
        if self.summary.raw_modified:
            self.summary.overall_status = "BLOCKED"
            return
        if any(
            "audio:truncated" in a or "audio:not_wav" in a or "audio:empty" in a
            for a in self.summary.anomalies
        ) or any(
            "truncated" in str(r.get("audio_readability_status", ""))
            for r in self.summary.file_rows
        ):
            self.summary.overall_status = "PASS WITH WARNINGS"
            return
        self.summary.overall_status = "PASS"


def _opt(v) -> str:
    return "" if v is None else str(v)


__all__ = [
    "Stage2Options",
    "Stage2Summary",
    "Stage2Runner",
    "Stage2ContaminationError",
]
