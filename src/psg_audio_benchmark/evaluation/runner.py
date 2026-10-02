"""Stage-6 orchestration: patient-level cohort + nested cross-validation split.

Read-only over the approved Stage-4 (window index) and Stage-5 (availability)
outputs; builds the audited CSV-physiology research cohorts, the patient-level
nested (outer x inner) cross-validation split, the input version signature and
the audit reports.

Pipeline (mirrors the Stage 1-5 read-only, run-dir-isolated discipline):

1. **License gate** runs FIRST. If it fails, write a blocked report and STOP.
2. **Input-run gate**: verify the approved Stage-4 and Stage-5 runs exist, bind
   to the right run ids, carry known label statuses, and that Stage 5 was built
   from this Stage 4 (lineage).
3. **Dry-run** stops here (gates only, no products, no raw snapshot).
4. Snapshot ``data/raw`` before; assemble cohorts; assign splits; build
   signature; snapshot after and assert immutability.
5. Write the membership / fold / inheritance / summary / exclusion / signature /
   resolved-config products and the three reports.

No audio is read (``*.wav`` is never opened); no model matrix / imputation /
normalization / feature selection / resampling / threshold / metrics. ``audio``
cohort count is hard-coded 0.
"""

from __future__ import annotations

import csv as csvlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Config
from ..data_audit import license_gate as gate_mod
from ..data_download.identity import PRIMARY_IDENTITY
from . import config as cfg_mod
from . import cohort as cohort_mod
from . import qc as qc_mod
from . import report as report_mod
from . import signature as sig_mod
from . import split as split_mod
from .config import load_splits_config, write_resolved_config_yaml
from .schema import (
    ALLOWED_LABEL_COLUMN,
    AUDIO_BLOCK_REASON,
    AUDIO_COHORT_COUNT,
    CohortExclusion,
    AirflowInheritanceRow,
    OuterPatientFold,
    PATIENT_SPLIT_VERSION,
    STAGE6_FORBIDDEN_COLUMNS,
    SplitSummary,
)

_DEFAULT_STAGE4_RUN_ID = "stage4-csv-window-index-20260807T170636Z"
_DEFAULT_STAGE5_RUN_ID = "stage5-physiology-features-20260807T232931Z"
_DEFAULT_STAGE3_RUN_ID = "stage3-annotation-sync-20260807T154558Z"

_REQUIRED_STAGE4_ARTIFACTS = ("csv_window_index.parquet", "windowing_config_resolved.yaml")
_REQUIRED_STAGE5_ARTIFACTS = ("physiology_feature_availability.parquet", "physiology_features_resolved.yaml")


@dataclass
class Stage6Options:
    dry_run: bool = False
    output_root: Optional[Path] = None
    relpath_base: Optional[Path] = None
    stage4_input_run_id: str = ""
    stage5_input_run_id: str = ""
    stage3_input_run_id: str = ""
    config_path: Optional[Path] = None

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


class Stage6Runner:
    """Runs the Stage-6 patient-level split. Construct with a Config."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage6Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage6Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.summary = SplitSummary(run_id=self.run_id, access_date=self.access_date)
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )
        fpath = self.options.config_path or (self.cfg.path("config") / "splits.yaml")
        self.splits_config = load_splits_config(Path(fpath))
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

    def _stage4_run_id(self) -> str:
        return (self.options.stage4_input_run_id or self.splits_config.stage4_input_run_id
                or _DEFAULT_STAGE4_RUN_ID)
    def _stage5_run_id(self) -> str:
        return (self.options.stage5_input_run_id or self.splits_config.stage5_input_run_id
                or _DEFAULT_STAGE5_RUN_ID)
    def _stage3_run_id(self) -> str:
        return (self.options.stage3_input_run_id or self.splits_config.stage3_input_run_id
                or _DEFAULT_STAGE3_RUN_ID)

    def _stage4_run_dir(self) -> Path:
        return self.cfg.path("data_manifests") / "runs" / self._stage4_run_id()
    def _stage5_run_dir(self) -> Path:
        return self.cfg.path("features_physiology") / "runs" / self._stage5_run_id()

    def _splits_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "splits" / "runs" / self.run_id
        return self.cfg.path("splits") / "runs" / self.run_id
    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "evaluation" / "runs" / self.run_id
        return self.cfg.path("reports_evaluation") / "runs" / self.run_id

    def _rel(self, path: Path) -> str:
        try:
            return qc_mod.relativize(path, self._relpath_base())
        except qc_mod.EvaluationContaminationError:
            return path.name

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> SplitSummary:
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

        # 2. INPUT-RUN GATE (Stage-4 window index + Stage-5 availability).
        self.summary.input_stage4_run_id = self._stage4_run_id()
        self.summary.input_stage5_run_id = self._stage5_run_id()
        self.summary.input_stage3_run_id = self._stage3_run_id()
        if not self._verify_input_runs():
            self.summary.overall_status = "BLOCKED"
            if not self.options.dry_run:
                self._write_blocked_report(gate)
            return self.summary

        self.summary.input_stage4_config_hash = self._read_stage4_config_hash()
        self.summary.input_stage5_config_hash = self._read_stage5_config_hash()

        if self.options.dry_run:
            # Dry-run does not read input content; stop before cohort/splits.
            self.summary.overall_status = "PASS WITH WARNINGS"
            return self.summary

        # 3. full input integrity (run_id binding + lineage)
        if not self._verify_input_integrity():
            self.summary.overall_status = "BLOCKED"
            self._write_blocked_report(gate)
            return self.summary
        self.summary.input_stage4_verified = True
        self.summary.input_stage5_verified = True

        # 4. raw snapshot BEFORE (Stage 6 reads no raw; proof of immutability)
        raw_before = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())

        # 5. cohort assembly + splits + signature
        cohort_res = cohort_mod.build_cohorts(
            stage4_window_index_path=self._stage4_run_dir() / "csv_window_index.parquet",
            stage5_availability_path=self._stage5_run_dir() / "physiology_feature_availability.parquet",
            run_id=self.run_id,
        )
        self.summary.anomalies.extend(cohort_res.anomalies)

        split_res = split_mod.assign_patient_splits(
            core_df=cohort_res.core_df,
            airflow_df=cohort_res.airflow_df,
            patient_burden=cohort_res.patient_burden,
            config=self.splits_config,
            run_id=self.run_id,
        )
        self.summary.anomalies.extend(split_res.anomalies)
        self.summary.balance_warnings.extend(split_res.balance_warnings)

        signature, sig_anomalies = sig_mod.build_input_signature(
            cfg=self.cfg,
            stage4_run_id=self._stage4_run_id(),
            stage5_run_id=self._stage5_run_id(),
            stage3_run_id=self._stage3_run_id(),
            run_id=self.run_id,
            project_root=self.cfg.project_root,
        )
        self.summary.anomalies.extend(sig_anomalies)

        # 6. raw snapshot AFTER + immutability
        raw_after = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        self.summary.raw_modified = raw_before != raw_after
        self.summary.raw_scanned = True

        # 7. aggregate
        self._aggregate(cohort_res, split_res, signature)

        # 8. status
        self._derive_status()

        # 9. write products
        self._write_products(cohort_res, split_res, signature)
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
        s5 = self._stage5_run_dir()
        if not s5.is_dir():
            self.summary.anomalies.append(f"stage5_run_dir_absent:{self._stage5_run_id()}")
            return False
        for name in _REQUIRED_STAGE5_ARTIFACTS:
            p = s5 / name
            if not p.is_file() or p.stat().st_size == 0:
                self.summary.anomalies.append(f"stage5_artifact_missing_or_empty:{name}")
                ok = False
        return ok

    def _verify_input_integrity(self) -> bool:
        ok = True
        rid4 = self._stage4_run_id()
        try:
            win = pd.read_parquet(self._stage4_run_dir() / "csv_window_index.parquet")
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"stage4_window_index_unreadable:{type(exc).__name__}")
            return False
        if "run_id" in win.columns and win.shape[0] > 0:
            bad = int((win["run_id"].astype(str) != rid4).sum())
            if bad:
                self.summary.anomalies.append(f"stage4_run_id_mismatch:{bad}rows")
                ok = False
        if "label_status" in win.columns:
            statuses = set(win["label_status"].dropna().astype(str).unique())
            allowed = {"positive", "negative", "excluded"}
            unknown = statuses - allowed
            if unknown:
                self.summary.anomalies.append(f"stage4_unknown_label_statuses:{sorted(unknown)}")
                ok = False
        rid5 = self._stage5_run_id()
        try:
            avail = pd.read_parquet(self._stage5_run_dir() / "physiology_feature_availability.parquet")
        except Exception as exc:  # pragma: no cover - defensive
            self.summary.anomalies.append(f"stage5_availability_unreadable:{type(exc).__name__}")
            return False
        if "run_id" in avail.columns and avail.shape[0] > 0:
            bad = int((avail["run_id"].astype(str) != rid5).sum())
            if bad:
                self.summary.anomalies.append(f"stage5_run_id_mismatch:{bad}rows")
                ok = False
        # lineage: Stage 5 must have been built from THIS Stage 4
        s5_input_stage4 = self._read_stage5_input_stage4()
        if s5_input_stage4 and s5_input_stage4 != rid4:
            self.summary.anomalies.append(
                f"lineage_stage5_input_stage4_mismatch:{s5_input_stage4}!={rid4}"
            )
            ok = False
        return ok

    def _read_yaml_field(self, run_dir: Path, yaml_name: str, field: str) -> str:
        p = run_dir / yaml_name
        if not p.is_file():
            return ""
        try:
            import yaml
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return ""
        return str(data.get(field, ""))

    def _read_stage4_config_hash(self) -> str:
        return self._read_yaml_field(self._stage4_run_dir(), "windowing_config_resolved.yaml", "config_hash")

    def _read_stage5_config_hash(self) -> str:
        return self._read_yaml_field(self._stage5_run_dir(), "physiology_features_resolved.yaml", "config_hash")

    def _read_stage5_input_stage4(self) -> str:
        return self._read_yaml_field(self._stage5_run_dir(), "physiology_features_resolved.yaml", "input_stage4_run_id")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, cohort_res, split_res, signature) -> None:
        try:
            s4 = pd.read_parquet(self._stage4_run_dir() / "csv_window_index.parquet")
            s5 = pd.read_parquet(self._stage5_run_dir() / "physiology_feature_availability.parquet")
        except Exception:  # pragma: no cover - defensive
            s4 = pd.DataFrame()
            s5 = pd.DataFrame()
        fills = cohort_mod.summary_fillers(cohort_res, s4=s4, s5=s5)
        for k, v in fills.items():
            setattr(self.summary, k, v)
        self.summary.outer_fold_stats = split_res.outer_fold_stats
        self.summary.inner_fold_stats = split_res.inner_fold_stats
        self.summary.airflow_fold_stats = split_res.airflow_fold_stats
        self.summary.audio_cohort_count = AUDIO_COHORT_COUNT
        self.summary.audio_block_reason = AUDIO_BLOCK_REASON

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _derive_status(self) -> None:
        s = self.summary
        if s.raw_modified or not s.license_gate_passed or not s.input_stage4_verified:
            s.overall_status = "BLOCKED"
            return
        # cohort must be non-empty and splits must have assigned folds
        if s.core_patients == 0 or not s.outer_fold_stats:
            s.overall_status = "BLOCKED"
            return
        if s.anomalies:
            s.overall_status = "PASS WITH WARNINGS"
            return
        s.overall_status = "PASS"

    # ------------------------------------------------------------------
    # Product writers
    # ------------------------------------------------------------------
    def _write_products(self, cohort_res, split_res, signature) -> None:
        sdir = self._splits_out_dir()
        rdir = self._reports_out_dir()
        sdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        self._write_membership_parquet(cohort_res.core_rows, sdir / "core_cohort_window_membership.parquet")
        self._write_membership_parquet(cohort_res.airflow_rows, sdir / "airflow_cohort_window_membership.parquet")
        self._write_csv_rows(split_res.outer_rows, OuterPatientFold, sdir / "outer_patient_folds_core.csv")
        self._write_parquet_rows(split_res.inner_rows, sdir / "inner_patient_folds_core.parquet")
        self._write_csv_rows(split_res.airflow_rows, AirflowInheritanceRow, sdir / "airflow_outer_fold_inheritance.csv")
        self._write_summary_csv(sdir / "patient_split_summary.csv")
        self._write_csv_rows(cohort_res.exclusions, CohortExclusion, sdir / "cohort_exclusions.csv")

        (sdir / "input_version_signature.json").write_text(
            json.dumps(signature, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_resolved_config_yaml(
            sdir / "splits_resolved.yaml", self.splits_config,
            run_id=self.run_id, config_hash=self.summary.config_hash,
        )

        # no-model-matrix + label-placement guard over every output
        self._assert_outputs_have_no_model_matrix(sdir)

        # path-purity guard over machine-readable products
        self._assert_products_clean(sdir)

        # record product paths (splits + reports)
        for name in (
            "core_cohort_window_membership.parquet", "airflow_cohort_window_membership.parquet",
            "outer_patient_folds_core.csv", "inner_patient_folds_core.parquet",
            "airflow_outer_fold_inheritance.csv", "patient_split_summary.csv",
            "cohort_exclusions.csv", "input_version_signature.json", "splits_resolved.yaml",
        ):
            self.summary.product_paths[name] = self._rel(sdir / name)
        for name in (
            "cohort_definition_report.md", "split_balance_report.md",
            "phase_06_completion_report.md",
        ):
            self.summary.product_paths[name] = self._rel(rdir / name)

        # reports (rendered last; reference product_paths)
        self._write_report(rdir, "cohort_definition_report.md",
                           report_mod.render_cohort_definition_report(self.summary))
        self._write_report(rdir, "split_balance_report.md",
                           report_mod.render_split_balance_report(self.summary))
        self._write_report(rdir, "phase_06_completion_report.md",
                           report_mod.render_phase_06_completion_report(self.summary))

    def _write_membership_parquet(self, rows, path: Path) -> None:
        import dataclasses
        from .schema import CohortMembershipRow
        names = [f.name for f in dataclasses.fields(CohortMembershipRow)]
        if not rows:
            pd.DataFrame(columns=names).to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df = df[names]
        df.to_parquet(path, index=False)

    def _write_csv_rows(self, rows, row_cls, path: Path) -> None:
        import dataclasses
        names = [f.name for f in dataclasses.fields(row_cls)]
        with open(path, "w", encoding="utf-8", newline="") as h:
            writer = csvlib.DictWriter(h, fieldnames=names)
            writer.writeheader()
            for r in rows:
                writer.writerow(dataclasses.asdict(r))

    def _write_parquet_rows(self, rows, path: Path) -> None:
        import dataclasses
        if not rows:
            pd.DataFrame().to_parquet(path, index=False)
            return
        df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
        df.to_parquet(path, index=False)

    def _write_summary_csv(self, path: Path) -> None:
        df = pd.DataFrame(self.summary.outer_fold_stats)
        df.to_csv(path, index=False, encoding="utf-8")

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
            "# 阶段 6 完成报告（phase_06_completion_report）\n\n"
            f"- run_id：`{s.run_id}`\n"
            f"- 阶段 6 状态：**BLOCKED**\n"
            f"- 许可门状态：{'BLOCKED' if not s.license_gate_passed else 'PASSED'}"
            f"（license_status=`{s.license_status}`）\n"
            f"- 阻断/缺项：{missing}\n\n"
            "## 准入下一阶段\n\n"
            "- [待决] 当前 **未满足**：许可门未通过或输入 Stage-4/Stage-5 run 完整性/谱系校验失败。\n"
            "- 本次不自行进入下一阶段。\n"
        )
        (rdir / "phase_06_completion_report.md").write_text(content, encoding="utf-8")

    def _assert_outputs_have_no_model_matrix(self, sdir: Path) -> None:
        """Every Stage-6 output is a membership/assignment/audit table: no feature
        values, and the stratification label only in the membership tables."""
        membership_only_label = {"core_cohort_window_membership.parquet",
                                 "airflow_cohort_window_membership.parquet"}
        for p in sdir.iterdir():
            if p.suffix == ".parquet":
                cols = set(pd.read_parquet(p).columns)
            elif p.suffix == ".csv":
                cols = set(pd.read_csv(p, nrows=0).columns)
            else:
                continue
            bad = cols & set(STAGE6_FORBIDDEN_COLUMNS)
            if bad:
                raise AssertionError(
                    f"{p.name}: forbidden feature-value column(s) {sorted(bad)}; "
                    f"Stage-6 outputs must not form a model matrix."
                )
            if p.name not in membership_only_label:
                if ALLOWED_LABEL_COLUMN in cols:
                    raise AssertionError(
                        f"{p.name}: {ALLOWED_LABEL_COLUMN} may appear only in the "
                        f"cohort membership tables, not in fold/exclusion/summary outputs."
                    )

    def _assert_products_clean(self, sdir: Path) -> None:
        values: List[str] = []
        for p in sdir.iterdir():
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
            elif p.suffix in (".yaml", ".yml", ".json"):
                values.extend(p.read_text(encoding="utf-8").splitlines())
        qc_mod.assert_paths_clean(values)


__all__ = ["Stage6Options", "Stage6Runner"]
