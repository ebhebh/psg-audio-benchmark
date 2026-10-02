"""Stage-6B orchestration: optimize + compare + gate + freeze a v2 patient split.

Read-only over the APPROVED Stage-6 v1 run products (the immutable comparator);
builds a more-balanced v2 patient-level outer split, compares it to v1, runs the
pre-registered adoption gate, and ONLY on approval builds the inner CV + mirrors
v2 to the fixed paths + updates ``splits/LATEST_RUN.txt``. v1 run-dir products and
raw are never modified.

Hard rules (mirror Stage 6 + the Stage-6B contract):

* patient-level only; exactly ``patients_per_outer_fold`` per outer fold; no
  window-random split; all windows of a patient in one fold;
* balance uses ONLY pre-model patient-level aggregates; no feature value, no model
  output; never drops a patient/window; never relabels;
* ``audio`` is BLOCKED (cohort count hard-coded 0); no ``*.wav`` / raw is read;
* no model matrix / imputation / normalization / feature selection / resampling /
  threshold / metrics.
"""

from __future__ import annotations

import csv as csvlib
import json
from dataclasses import dataclass, replace as dataclasses_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Config
from . import qc as qc_mod
from .balance_optimizer import (
    PatientBurden,
    burden_from_outer_folds_df,
    optimize_patient_folds,
)
from .candidate_comparison import (
    compute_assignment_metrics,
    compute_v1_metrics_from_outer_csv,
    build_comparison_rows,
    build_comparison_summary,
)
from .gate import Check, evaluate_gate, verify_hard_constraints
from .inner_split import build_inner_folds
from .split_balance_config import (
    ResolvedSplitBalanceConfig,
    SplitBalanceConfigError,
    load_split_balance_config,
    write_resolved_split_balance_yaml,
)
from .split_balance_report import (
    SplitBalanceSummary,
    render_phase_06b_completion_report,
    render_split_balance_optimization_report,
)
from .split_balance_signature import build_split_balance_signature, write_signature
from .schema import STAGE6_FORBIDDEN_COLUMNS


@dataclass
class Stage6bOptions:
    dry_run: bool = False
    output_root: Optional[Path] = None
    relpath_base: Optional[Path] = None
    config_path: Optional[Path] = None
    run_id: str = ""
    config_hash: str = ""
    # optional budget overrides (tests / fast inner already handled internally)
    n_candidate_seeds: Optional[int] = None
    local_swap_iterations: Optional[int] = None
    # Stage-26 fix: explicit override of the frozen config pin
    # ``v1_comparator_run_id`` (a reproducer's OWN Stage-6 v1 run). The frozen
    # YAML pin is never edited; when set, the override must be a v1 run id
    # (prefix 'stage6-patient-splits-') and is recorded in the run's resolved
    # config and signature lineage.
    v1_comparator_run_id: Optional[str] = None

    @property
    def isolated(self) -> bool:
        return self.output_root is not None


class Stage6bRunner:
    """Runs the Stage-6B split-balance optimization. Construct with a Config."""

    def __init__(self, cfg: Config, options: Optional[Stage6bOptions] = None) -> None:
        self.cfg = cfg
        self.options = options or Stage6bOptions()
        self.run_id: str = self.options.run_id or "stage6b-split-balance-v2-unknown"
        self.access_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fpath = self.options.config_path or (self.cfg.path("config") / "split_balance_optimization.yaml")
        loaded: ResolvedSplitBalanceConfig = load_split_balance_config(Path(fpath))
        # provenance: the untouched frozen-config pin (read-only record)
        self.frozen_v1_comparator_pin: str = loaded.v1_comparator_run_id
        self.v1_comparator_source = "frozen_config"
        if self.options.v1_comparator_run_id:
            override = str(self.options.v1_comparator_run_id)
            if not override.startswith("stage6-patient-splits-"):
                raise SplitBalanceConfigError(
                    f"--v1-comparator-run-id {override!r} is not a Stage-6 v1 "
                    "run id (expected prefix 'stage6-patient-splits-'); "
                    "refusing to bind a wrong-stage comparator")
            loaded = dataclasses_replace(
                loaded, v1_comparator_run_id=override)
            self.v1_comparator_source = "cli_override"
        self.config: ResolvedSplitBalanceConfig = loaded
        self.summary = SplitBalanceSummary(
            run_id=self.run_id, access_date=self.access_date,
            config_hash=self.options.config_hash,
            v1_comparator_run_id=self.config.v1_comparator_run_id,
            audio_cohort_count=self.config.audio_cohort_count,
            audio_block_reason=self.config.audio_block_reason,
            algorithm=self.config.algorithm, base_seed=self.config.base_seed,
            n_candidate_seeds=self.config.n_candidate_seeds,
            local_swap_iterations=self.config.local_swap_iterations,
            selection_rule=self.config.selection_rule,
        )
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve() if self.options.output_root is not None else None
        )

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _v1_run_dir(self) -> Path:
        return self.cfg.path("splits") / "runs" / self.config.v1_comparator_run_id

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
    def run(self) -> SplitBalanceSummary:
        qc_mod.assert_not_contaminating(
            cfg=self.cfg, run_id=self.run_id,
            config_hash=self.options.config_hash, output_root=self._output_root,
        )

        s = self.summary
        v1_dir = self._v1_run_dir()
        v1_outer_csv = v1_dir / "outer_patient_folds_core.csv"
        v1_airflow_csv = v1_dir / "airflow_outer_fold_inheritance.csv"
        v1_sig_json = v1_dir / "input_version_signature.json"

        # ---- gates: v1 products present ----
        for p in (v1_outer_csv, v1_airflow_csv, v1_sig_json):
            if not p.is_file():
                s.anomalies.append(f"v1_product_missing:{p.name}")
                s.status = "BLOCKED"
                return s

        if self.options.dry_run:
            s.status = "PASS WITH WARNINGS"
            s.anomalies.append("dry_run: gates only; no optimization, no products.")
            return s

        # ---- immutability: raw + v1 run-dir snapshot BEFORE ----
        raw_before = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        v1_before = self._hash_dir(v1_dir)

        # ---- read v1 patient burden (patient-level aggregates ONLY) ----
        burden = burden_from_outer_folds_df(v1_outer_csv)
        burden_by_pid = {b.patient_id: b for b in burden}
        s.cohort = {
            "n_patients": len(burden),
            "n_windows": sum(b.n_windows for b in burden),
            "n_positive": sum(b.n_positive for b in burden),
            "n_negative": sum(b.n_negative for b in burden),
        }

        # ---- optimize v2 outer split ----
        opt = optimize_patient_folds(
            burden, config=self.config,
            n_candidate_seeds=self.options.n_candidate_seeds,
            local_swap_iterations=self.options.local_swap_iterations,
        )
        s.anomalies.extend(opt.anomalies)
        s.n_candidates_evaluated = len(opt.all_candidates)
        s.candidate_spreads = [c.positive_rate_spread for c in opt.all_candidates]
        s.candidate_wcvs = [c.window_count_cv for c in opt.all_candidates]
        patient_to_fold = opt.patient_to_fold
        s.v2_per_fold = list(opt.best.per_fold)

        # ---- v1 vs v2 metrics + comparison ----
        v1_metrics = compute_v1_metrics_from_outer_csv(v1_outer_csv, n_folds=self.config.outer_n_folds)
        v2_metrics = compute_assignment_metrics(
            patient_to_fold=patient_to_fold, burden_by_pid=burden_by_pid,
            n_folds=self.config.outer_n_folds,
        )
        s.v1_per_fold = list(v1_metrics.per_fold)
        s.v1 = {"spread": v1_metrics.positive_rate_spread, "wcv": v1_metrics.window_count_cv,
                "max_abs_deviation": v1_metrics.max_abs_deviation,
                "overall_rate": v1_metrics.overall_positive_rate}
        s.v2 = {"spread": v2_metrics.positive_rate_spread, "wcv": v2_metrics.window_count_cv,
                "max_abs_deviation": v2_metrics.max_abs_deviation,
                "overall_rate": v2_metrics.overall_positive_rate}
        s.comparison_rows = build_comparison_rows(v1_metrics, v2_metrics)
        s.comparison_summary = build_comparison_summary(v1_metrics, v2_metrics, self.config)

        # ---- hard constraints ----
        hard_checks = verify_hard_constraints(
            patient_to_fold=patient_to_fold, burden_by_pid=burden_by_pid, config=self.config,
        )
        s.hard_checks = hard_checks

        # ---- airflow inheritance (v2 core fold) + leakage ----
        airflow_rows, airflow_stats, af_mismatch, af_anomalies = self._build_airflow(
            v1_airflow_csv, patient_to_fold,
        )
        s.airflow_per_fold = airflow_stats
        s.airflow_mismatch_count = af_mismatch
        s.airflow_patients = len(airflow_rows)
        s.anomalies.extend(af_anomalies)
        airflow_check = Check("airflow_inherits_v2_outer_fold_zero_mismatch",
                              af_mismatch == 0 and not af_anomalies,
                              f"mismatch={af_mismatch}, anomalies={af_anomalies}")

        # ---- inner leakage placeholder (built only on approval) ----
        inner_leakage_check = Check("inner_leakage_outer_test_never_in_inner", True,
                                    "built only after outer approval")

        # ---- gate (pre-registered; decides approval) ----
        leakage_checks = [airflow_check, inner_leakage_check]
        gate = evaluate_gate(
            v2=v2_metrics, v1=v1_metrics, hard_checks=hard_checks,
            leakage_checks=leakage_checks, config=self.config,
        )
        s.gate = gate
        s.status = gate.status

        # ---- inner CV only on approval ----
        if gate.approved:
            inner = build_inner_folds(
                burden=burden, outer_patient_to_fold=patient_to_fold,
                config=self.config, run_id=self.run_id,
            )
            s.inner_fold_stats = inner.fold_stats
            s.inner_warnings = inner.warnings
            s.inner_anomalies = inner.anomalies
            s.anomalies.extend(inner.anomalies)
            inner_leakage_check = Check(
                "inner_leakage_outer_test_never_in_inner",
                len(inner.anomalies) == 0,
                f"inner anomalies={inner.anomalies} or 0",
            )
            # rebuild gate so the leakage check reflects the actual inner result
            leakage_checks = [airflow_check, inner_leakage_check]
            gate = evaluate_gate(
                v2=v2_metrics, v1=v1_metrics, hard_checks=hard_checks,
                leakage_checks=leakage_checks, config=self.config,
            )
            s.gate = gate
            s.status = gate.status
            inner_rows = inner.rows
        else:
            inner_rows = []

        # ---- immutability: raw + v1 run-dir snapshot AFTER ----
        raw_after = qc_mod.snapshot_raw(self.cfg.path("data_raw"), self._relpath_base())
        s.raw_scanned = True
        s.raw_modified = raw_before != raw_after
        v1_after = self._hash_dir(v1_dir)
        s.historical_products_unchanged = (v1_before == v1_after) and not s.raw_modified
        if s.raw_modified:
            s.anomalies.append("raw_modified_during_run")
        if v1_before != v1_after:
            s.anomalies.append("v1_run_dir_modified_during_run")

        # ---- write products (isolated run-dir) ----
        self._write_products(
            burden=burden, patient_to_fold=patient_to_fold,
            airflow_rows=airflow_rows, inner_rows=inner_rows,
            v1_metrics=v1_metrics, v2_metrics=v2_metrics,
        )

        # ---- reports (run-dir) ----
        # written BEFORE the mirror so the fixed-path mirror can copy them too.
        self._write_reports()

        # ---- mirror + LATEST only on approval & production ----
        if gate.approved and not self.options.isolated:
            s.latest_updated, s.latest_path = self._mirror_to_fixed_paths()
        else:
            s.latest_updated = False

        return s

    # ------------------------------------------------------------------
    # Airflow inheritance
    # ------------------------------------------------------------------
    def _build_airflow(
        self, v1_airflow_csv: Path, patient_to_fold: Dict[str, int],
    ) -> tuple:
        import pandas as pd
        df = pd.read_csv(v1_airflow_csv, dtype={"patient_id": str})
        rows: List[Dict[str, Any]] = []
        stats: Dict[int, Dict[str, Any]] = {f: {
            "outer_fold": f, "n_airflow_patients": 0, "n_airflow_windows": 0,
            "airflow_positive": 0, "airflow_negative": 0, "airflow_positive_rate": 0.0,
        } for f in range(self.config.outer_n_folds)}
        mismatch = 0
        anomalies: List[str] = []
        for _, r in df.iterrows():
            pid = str(r["patient_id"])
            if pid not in patient_to_fold:
                mismatch += 1
                anomalies.append(f"airflow_patient_not_in_core_v2:{pid}")
                continue
            fold = int(patient_to_fold[pid])
            nw = int(r["n_airflow_windows"]); npos = int(r["n_airflow_positive"])
            rows.append({
                "run_id": self.run_id, "patient_id": pid, "outer_fold": fold,
                "n_airflow_windows": nw, "n_airflow_positive": npos,
                "n_airflow_negative": int(r["n_airflow_negative"]),
                "airflow_positive_rate": float(npos / nw) if nw else 0.0,
            })
            st = stats[fold]
            st["n_airflow_patients"] += 1
            st["n_airflow_windows"] += nw
            st["airflow_positive"] += npos
            st["airflow_negative"] += int(r["n_airflow_negative"])
        for st in stats.values():
            st["airflow_positive_rate"] = (
                float(st["airflow_positive"] / st["n_airflow_windows"])
                if st["n_airflow_windows"] else 0.0
            )
        return rows, list(stats.values()), mismatch, anomalies

    # ------------------------------------------------------------------
    # Product writers
    # ------------------------------------------------------------------
    def _write_products(
        self, *, burden: List[PatientBurden], patient_to_fold: Dict[str, int],
        airflow_rows: List[Dict[str, Any]], inner_rows: List[Dict[str, Any]],
        v1_metrics, v2_metrics,
    ) -> None:
        s = self.summary
        sdir = self._splits_out_dir()
        sdir.mkdir(parents=True, exist_ok=True)

        # outer_patient_folds_core_v2.csv (same schema as v1)
        outer_cols = ["run_id", "patient_id", "outer_fold", "n_windows",
                      "n_positive", "n_negative", "positive_rate"]
        outer_rows = []
        for b in burden:
            f = int(patient_to_fold[b.patient_id])
            outer_rows.append({
                "run_id": self.run_id, "patient_id": b.patient_id, "outer_fold": f,
                "n_windows": b.n_windows, "n_positive": b.n_positive,
                "n_negative": b.n_negative, "positive_rate": b.positive_rate,
            })
        outer_rows.sort(key=lambda r: str(r["patient_id"]))
        self._write_csv(sdir / "outer_patient_folds_core_v2.csv", outer_cols, outer_rows)

        # patient_level_balance_inputs.csv (inputs + assigned v2 fold)
        inp_cols = ["run_id", "patient_id", "n_windows", "n_positive", "n_negative",
                    "positive_rate", "assigned_outer_fold_v2"]
        inp_rows = [{
            "run_id": self.run_id, "patient_id": b.patient_id, "n_windows": b.n_windows,
            "n_positive": b.n_positive, "n_negative": b.n_negative,
            "positive_rate": b.positive_rate,
            "assigned_outer_fold_v2": int(patient_to_fold[b.patient_id]),
        } for b in burden]
        inp_rows.sort(key=lambda r: str(r["patient_id"]))
        self._write_csv(sdir / "patient_level_balance_inputs.csv", inp_cols, inp_rows)

        # split_candidate_comparison.csv
        comp_cols = ["scope", "fold", "metric", "v1", "v2", "delta"]
        self._write_csv(sdir / "split_candidate_comparison.csv", comp_cols, s.comparison_rows)

        # airflow_outer_fold_inheritance_v2.csv
        af_cols = ["run_id", "patient_id", "outer_fold", "n_airflow_windows",
                   "n_airflow_positive", "n_airflow_negative", "airflow_positive_rate"]
        airflow_rows = sorted(airflow_rows, key=lambda r: str(r["patient_id"]))
        self._write_csv(sdir / "airflow_outer_fold_inheritance_v2.csv", af_cols, airflow_rows)

        # inner_patient_folds_core_v2.parquet (approved only)
        inner_cols = ["run_id", "outer_fold", "patient_id", "inner_validation_fold",
                      "n_windows", "n_positive", "n_negative", "positive_rate"]
        if inner_rows:
            pd.DataFrame(inner_rows, columns=inner_cols).to_parquet(
                sdir / "inner_patient_folds_core_v2.parquet", index=False
            )

        # version signature
        v2_filenames = [
            "outer_patient_folds_core_v2.csv", "patient_level_balance_inputs.csv",
            "split_candidate_comparison.csv", "airflow_outer_fold_inheritance_v2.csv",
        ]
        if inner_rows:
            v2_filenames.append("inner_patient_folds_core_v2.parquet")
        sig = build_split_balance_signature(
            v1_run_dir=self._v1_run_dir(), v2_product_dir=sdir, v2_filenames=v2_filenames,
            config=self.config, run_id=self.run_id, config_hash=s.config_hash,
            patient_ids=[b.patient_id for b in burden], patient_to_fold=patient_to_fold,
            n_windows=s.cohort["n_windows"], n_positive=s.cohort["n_positive"],
            n_negative=s.cohort["n_negative"],
            positive_rate_spread=v2_metrics.positive_rate_spread,
            window_count_cv=v2_metrics.window_count_cv,
            approval_status=s.gate.status if s.gate else "not_approved_balance_target_not_met",
        )
        write_signature(sdir / "split_balance_version_signature.json", sig)

        # resolved config (with runtime-computed comparators)
        extra = {
            "v1_comparator_run_id_source": self.v1_comparator_source,
            "frozen_config_v1_comparator_pin": self.frozen_v1_comparator_pin,
            "v1_computed_positive_rate_spread": v1_metrics.positive_rate_spread,
            "v1_computed_window_count_cv": v1_metrics.window_count_cv,
            "v1_computed_overall_positive_rate": v1_metrics.overall_positive_rate,
            "v2_positive_rate_spread": v2_metrics.positive_rate_spread,
            "v2_window_count_cv": v2_metrics.window_count_cv,
            "approval_status": s.gate.status if s.gate else "not_approved_balance_target_not_met",
            "gate_checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                            for c in (s.gate.checks if s.gate else [])],
        }
        write_resolved_split_balance_yaml(
            sdir / "split_balance_resolved.yaml", self.config,
            run_id=self.run_id, config_hash=s.config_hash, extra=extra,
        )

        # ---- guards: no model matrix; path purity ----
        self._assert_no_model_matrix(sdir)
        self._assert_products_clean(sdir)

        # record product paths
        for name in v2_filenames + [
            "split_balance_version_signature.json", "split_balance_resolved.yaml",
        ]:
            s.product_paths[name] = self._rel(sdir / name)

    def _write_csv(self, path: Path, cols: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as h:
            w = csvlib.DictWriter(h, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})

    def _assert_no_model_matrix(self, sdir: Path) -> None:
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
                    f"Stage-6B outputs must not form a model matrix."
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

    # ------------------------------------------------------------------
    # Mirror + LATEST (production, approved only)
    # ------------------------------------------------------------------
    def _mirror_to_fixed_paths(self) -> tuple:
        import shutil
        project_root = self.cfg.project_root.resolve()
        splits_fixed = self.cfg.path("splits")
        reports_fixed = self.cfg.path("reports_evaluation")
        run_splits = self._splits_out_dir()
        run_rep = self._reports_out_dir()

        splits_fixed.mkdir(parents=True, exist_ok=True)
        mirrored: List[str] = []
        for name in (
            "outer_patient_folds_core_v2.csv", "inner_patient_folds_core_v2.parquet",
            "airflow_outer_fold_inheritance_v2.csv", "patient_level_balance_inputs.csv",
            "split_candidate_comparison.csv", "split_balance_version_signature.json",
            "split_balance_resolved.yaml",
        ):
            src = run_splits / name
            if src.is_file():
                shutil.copyfile(src, splits_fixed / name)
                mirrored.append(str((splits_fixed / name).resolve().relative_to(project_root)).replace("\\", "/"))

        # LATEST pointers (the "current blessed version")
        latest_splits = splits_fixed / "LATEST_RUN.txt"
        latest_splits.write_text(self.run_id + "\n", encoding="utf-8")
        (reports_fixed / "LATEST_RUN.txt").write_text(self.run_id + "\n", encoding="utf-8")

        # mirror reports to fixed paths
        reports_fixed.mkdir(parents=True, exist_ok=True)
        for name in ("split_balance_optimization_report.md", "phase_06b_completion_report.md"):
            rsrc = run_rep / name
            if rsrc.exists():
                shutil.copyfile(rsrc, reports_fixed / name)

        self.summary.mirrored_fixed_paths = mirrored
        return True, str(latest_splits.resolve().relative_to(project_root)).replace("\\", "/")

    def _write_reports(self) -> None:
        rdir = self._reports_out_dir()
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "split_balance_optimization_report.md").write_text(
            render_split_balance_optimization_report(self.summary), encoding="utf-8")
        (rdir / "phase_06b_completion_report.md").write_text(
            render_phase_06b_completion_report(self.summary), encoding="utf-8")
        s = self.summary
        for name in ("split_balance_optimization_report.md", "phase_06b_completion_report.md"):
            s.product_paths[name] = self._rel(rdir / name)

    # ------------------------------------------------------------------
    # Hashing helpers (immutability proof)
    # ------------------------------------------------------------------
    def _hash_dir(self, d: Path) -> Dict[str, str]:
        """Size+mtime fingerprint of every file under ``d`` (immutability proof).

        Keys are absolute resolved paths (unique); values carry a best-effort
        relative label plus size+mtime. The label is robust to paths that live
        outside ``relpath_base`` (e.g. the real v1 run-dir / raw when an isolated
        run sets relpath_base to a temp dir) — it falls back to project-relative,
        then to the bare file name. The before/after equality check is unaffected
        because the keys are stable absolute paths.
        """
        out: Dict[str, str] = {}
        if not d.is_dir():
            return out
        base = self._relpath_base()
        proj = self.cfg.project_root.resolve()
        for p in sorted(d.rglob("*")):
            if p.is_file():
                try:
                    label = qc_mod.relativize(p, base)
                except Exception:
                    try:
                        label = str(p.resolve().relative_to(proj)).replace("\\", "/")
                    except ValueError:
                        label = p.name
                out[str(p.resolve())] = f"{label}|{p.stat().st_size}|{int(p.stat().st_mtime_ns)}"
        return out


__all__ = ["Stage6bOptions", "Stage6bRunner"]
