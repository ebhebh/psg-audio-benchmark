"""Stage-9 robustness/sensitivity orchestrator.

Drives the end-to-end robustness pass on the FROZEN Stage 7/8 results:

1. contamination guard + resolved-config load;
2. snapshot ``data/raw`` + protected prior-stage products BEFORE;
3. frozen-input gate (:mod:`robustness.signature`) -- re-derive both cohorts,
   re-check split hashes + Stage 7/8 signatures (audio=0 enforced);
4. **2.1** primary reproduction from the frozen OOF (pure, authoritative) +
   re-run determinism check;
5. shared frozen nested-CV re-run -> the bundle feeding every sensitivity;
6. **2.3/2.4/2.5/2.2/2.6/3.0** sensitivities (each pre-registered, never
   re-selecting the primary model/threshold, never fitting on outer-test);
7. write all run-dir artefacts (pseudonymized, path-pure, isolated);
8. figures (<=6) + markdown reports;
9. path purity; snapshot AFTER -> prove raw + prior-stage products unchanged.

Audio is BLOCKED throughout. The Stage 7/8 primary results are read-only. A dry
run does the gate + frozen-OOF reproduction only (no re-training, no writes).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...config import Config
from ...run_metadata import package_versions
from ...windowing.qc import assert_paths_clean, snapshot_raw
from . import schema as S
from .airflow_profile import airflow_profile
from .alternative_label import alternative_label
from .bootstrap_sensitivity import bootstrap_sensitivity
from .calibration import calibration_sensitivity
from .class_weight import class_weight_sensitivity
from .config import (ResolvedRobustnessConfig, load_robustness_config,
                     write_sensitivity_resolved_yaml)
from .reproduction import compare_rerun_to_frozen, reproduce_from_frozen
from .rerun import run_frozen_nested_cv
from .signature import (RobustnessInputError, build_robustness_input_signature,
                        verify_robustness_inputs)
from .threshold_sensitivity import threshold_sensitivity


class RobustnessContaminationError(RuntimeError):
    """Raised when a run would write into production paths or a production run
    looks like a test fixture."""


_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "\\tmp\\", "/tmp/")
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

_PROTECTED_PATH_KEYS = (
    "data_raw", "splits", "features_physiology", "features_audio",
    "data_manifests", "annotations", "reports_data_audit", "reports_annotations",
    "reports_data_download", "reports_windowing", "reports_feature_extraction",
    "reports_evaluation", "reports_leakage_audit", "reports_model_audit",
    "reports_manuscript_outputs",
)
_STAGE9_OUTPUT_KEYS = ("models", "results", "reports_modeling")


@dataclass
class Stage9Options:
    dry_run: bool
    output_root: Optional[Path]
    config_path: Path
    run_id: str
    config_hash: str
    split_run_id: Optional[str] = None


@dataclass
class Stage9Summary:
    status: str  # OK | DRY_RUN | BLOCKED
    run_id: str
    split_run_id: str
    approval_status: str
    gate_checks: List[dict] = field(default_factory=list)
    core_cohort: Dict[str, Any] = field(default_factory=dict)
    airflow_cohort: Dict[str, Any] = field(default_factory=dict)
    reproduction: Dict[str, Any] = field(default_factory=dict)
    rerun_determinism: Dict[str, Any] = field(default_factory=dict)
    analyses_executed: Dict[str, str] = field(default_factory=dict)
    audio_cohort_count: int = S.AUDIO_COHORT_COUNT
    read_wav: bool = False
    airflow_in_core_model: bool = False
    audio_features_in_matrix: bool = False
    raw_modified: bool = False
    historical_products_unchanged: bool = True
    production_paths_clean: bool = True
    n_figures: int = 0
    artifacts: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# contamination guard
# ---------------------------------------------------------------------------

def assert_not_contaminating_robustness(
    *, cfg: Config, run_id: str, config_hash: str, output_root: Optional[Path],
) -> None:
    run_id_low = (run_id or "").lower()
    if output_root is None:
        hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
        if hit:
            raise RobustnessContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}.")
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise RobustnessContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value.")
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise RobustnessContaminationError(
                f"Production-mode run refused: data_raw {raw} is outside the "
                f"project root.") from exc
    else:
        root = Path(output_root).resolve()
        for key in _PROTECTED_PATH_KEYS + _STAGE9_OUTPUT_KEYS:
            forbidden = cfg.path(key).resolve()
            try:
                root.relative_to(forbidden)
                inside = True
            except ValueError:
                inside = root == forbidden
            if inside:
                raise RobustnessContaminationError(
                    f"Isolated output_root {root} must not be (or be inside) the "
                    f"production path {forbidden} ({key}).")


def _run_dirs(*, cfg: Config, run_id: str, output_root: Optional[Path]) -> Dict[str, Path]:
    base = {k: (Path(output_root) if output_root else cfg.project_root)
            for k in ("models", "results", "reports")}
    rel = {
        "models": Path("models") / "runs" / run_id,
        "results": Path("results") / "runs" / run_id,
        "reports": Path("reports") / "modeling" / "runs" / run_id,
    }
    return {k: base[k] / rel[k] for k in base}


def _snapshot_protected(cfg: Config, *, skip_run_id: str) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for key in _PROTECTED_PATH_KEYS:
        root = cfg.path(key)
        if root.exists():
            for rel, val in snapshot_raw(root, cfg.project_root).items():
                snap[f"{key}/{rel}"] = val
    for key in ("models", "results"):
        runs = cfg.path(key) / "runs"
        if runs.exists():
            for rel, val in snapshot_raw(runs, runs).items():
                top = rel.split("/", 1)[0]
                if top == skip_run_id:
                    continue
                snap[f"{key}/runs/{rel}"] = val
    return snap


def _audit_audio_airflow(bundle) -> Dict[str, bool]:
    """Runtime audit: no audio columns anywhere; no airflow columns in the CORE frame."""
    def _has(cols, token):
        return any(token in str(c).lower() for c in cols)
    audio_any = False
    airflow_in_core = False
    for label, frame in (("core", bundle.core_frame),
                         ("air_core", bundle.airflow_core_frame),
                         ("air_enh", bundle.airflow_enh_frame)):
        if frame is None:
            continue
        cols = list(getattr(frame, "feature_names", []))
        if _has(cols, "audio"):
            audio_any = True
        if label == "core" and _has(cols, "airflow"):
            airflow_in_core = True
    return {"audio_features_in_matrix": audio_any,
            "airflow_in_core_model": airflow_in_core}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class Stage9Runner:
    def __init__(self, cfg: Config, options: Stage9Options) -> None:
        self.cfg = cfg
        self.options = options
        self.summary = Stage9Summary(
            status="BLOCKED", run_id=options.run_id, split_run_id="",
            approval_status="",
        )
        self._mc: Optional[ResolvedRobustnessConfig] = None
        self._rin = None
        self._bundle = None

    def _load_mc(self) -> ResolvedRobustnessConfig:
        return load_robustness_config(self.options.config_path)

    def _log(self, msg: str) -> None:
        print(msg)

    # -- main --------------------------------------------------------------
    def run(self) -> Stage9Summary:
        t0 = time.perf_counter()
        cfg = self.cfg
        opts = self.options
        assert_not_contaminating_robustness(
            cfg=cfg, run_id=opts.run_id, config_hash=opts.config_hash,
            output_root=opts.output_root,
        )
        mc = self._load_mc()
        self._mc = mc

        raw_before: Dict[str, str] = {}
        prot_before: Dict[str, str] = {}
        if opts.output_root is None:
            raw_before = snapshot_raw(cfg.path("data_raw"), cfg.project_root)
            prot_before = _snapshot_protected(cfg, skip_run_id=opts.run_id)

        # 1. frozen-input gate
        rin = verify_robustness_inputs(
            cfg=cfg, mc=mc, split_run_id_override=opts.split_run_id,
        )
        self._rin = rin
        self.summary.split_run_id = rin.split_run_id
        self.summary.approval_status = rin.split_signature.get("approval_status", "")
        self.summary.gate_checks = [
            {"name": c["name"], "passed": bool(c["passed"]), "detail": c["detail"]}
            for c in rin.checks
        ]
        cg, ag = rin.core_gate, rin.airflow_gate
        self.summary.core_cohort = {
            "n_windows": int(len(cg.cohort_windows)),
            "n_patients": int(cg.cohort_windows["patient_id"].nunique()),
            "n_positive": int((cg.cohort_windows["binary_event_label"] == 1).sum()),
            "n_negative": int((cg.cohort_windows["binary_event_label"] == 0).sum()),
        }
        self.summary.airflow_cohort = {
            "n_windows": int(ag.airflow_cohort["n_windows"]),
            "n_patients": int(ag.airflow_cohort["n_patients"]),
            "n_positive": int(ag.airflow_cohort["n_positive"]),
            "n_negative": int(ag.airflow_cohort["n_negative"]),
        }

        # 2. primary reproduction from the FROZEN OOF (pure, cheap)
        repro = reproduce_from_frozen(run_id=opts.run_id, rin=rin, mc=mc)
        self.summary.reproduction = self._repro_summary(repro)

        # dry run: gate + frozen-OOF reproduction only; no training, no writes
        if opts.dry_run:
            self.summary.status = "DRY_RUN"
            self.summary.analyses_executed = {
                S.A_PRIMARY_REPRODUCTION: "executed",
                **{a: "skipped_dry_run" for a in S.ALL_ANALYSES
                   if a != S.A_PRIMARY_REPRODUCTION},
            }
            self.summary.raw_modified = False
            self.summary.historical_products_unchanged = True
            self.summary.elapsed_seconds = float(time.perf_counter() - t0)
            return self.summary

        # 3. shared frozen nested-CV re-run
        bundle = run_frozen_nested_cv(cfg=cfg, mc=mc, rin=rin, log=self._log)
        self._bundle = bundle
        audit = _audit_audio_airflow(bundle)
        self.summary.audio_features_in_matrix = audit["audio_features_in_matrix"]
        self.summary.airflow_in_core_model = audit["airflow_in_core_model"]
        if audit["audio_features_in_matrix"]:
            self.summary.reasons.append("audio features detected in a feature matrix")
        if audit["airflow_in_core_model"]:
            self.summary.reasons.append("airflow features leaked into the core model matrix")

        # re-run determinism vs frozen
        determ = compare_rerun_to_frozen(bundle=bundle, rin=rin, mc=mc)
        self.summary.rerun_determinism = determ.get("summary", {})

        analyses_executed: Dict[str, str] = {S.A_PRIMARY_REPRODUCTION: "executed"}

        # 4. threshold sensitivity (2.3)
        thr_df = threshold_sensitivity(bundle=bundle, run_id=opts.run_id, mc=mc)
        analyses_executed[S.A_THRESHOLD] = "executed"

        # 5. calibration sensitivity (2.4)
        cal_df = calibration_sensitivity(bundle=bundle, run_id=opts.run_id, mc=mc)
        analyses_executed[S.A_CALIBRATION] = "executed"

        # 6. class-weight sensitivity (2.5)
        cw_df = class_weight_sensitivity(bundle=bundle, run_id=opts.run_id, mc=mc)
        analyses_executed[S.A_CLASS_WEIGHT] = "executed"

        # 7. patient-cluster bootstrap + macro (2.2)
        boot = bootstrap_sensitivity(bundle=bundle, rin=rin, run_id=opts.run_id, mc=mc)
        analyses_executed[S.A_PATIENT_WEIGHT] = "executed"

        # 8. airflow selection profile (2.6)
        air = airflow_profile(cfg=cfg, rin=rin, run_id=opts.run_id, mc=mc)
        analyses_executed[S.A_AIRFLOW_PROFILE] = "executed"

        # 9. alternative label (3.0)
        alt = alternative_label(cfg=cfg, bundle=bundle, rin=rin, run_id=opts.run_id, mc=mc)
        alt_interp = bool(alt["alt_label_prevalence"]["interpretable"].all()) \
            if len(alt["alt_label_prevalence"]) else False
        analyses_executed[S.A_ALTERNATIVE_LABEL] = (
            "executed" if alt_interp else "executed_partial_na")
        self.summary.analyses_executed = analyses_executed

        # 10. write artefacts (CSVs / JSON / resolved YAML) -- populates the
        # logical artifact-path list consumed by the path-purity check.
        run_dirs = _run_dirs(cfg=cfg, run_id=opts.run_id, output_root=opts.output_root)
        artifacts = self._write_artifacts(
            run_dirs=run_dirs, mc=mc, rin=rin, determ=determ, repro=repro,
            thr_df=thr_df, cal_df=cal_df, cw_df=cw_df, boot=boot, air=air, alt=alt,
        )
        self.summary.artifacts = artifacts

        # 11. path purity (validate the emitted artifact strings + ids).
        try:
            assert_paths_clean(self._all_output_strings())
            self.summary.production_paths_clean = True
        except AssertionError as exc:
            self.summary.production_paths_clean = False
            self.summary.reasons.append(f"path_purity: {exc}")

        # 12. immutability AFTER (production only): raw + prior-stage products
        # byte-identical. Runs before report writing so the completion report
        # records the TRUE post-run immutability / path-purity state (not the
        # field defaults).
        if opts.output_root is None:
            raw_after = snapshot_raw(cfg.path("data_raw"), cfg.project_root)
            self.summary.raw_modified = (raw_before != raw_after)
            prot_after = _snapshot_protected(cfg, skip_run_id=opts.run_id)
            self.summary.historical_products_unchanged = (prot_before == prot_after)
            if self.summary.raw_modified:
                self.summary.reasons.append("data/raw was modified during the run")
            if not self.summary.historical_products_unchanged:
                self.summary.reasons.append("prior-stage products were modified")

        # 13. finalize the outcome BEFORE reports so every generated artefact
        # (signature JSON is already written; the markdown reports next) reflects
        # the true status / elapsed / immutability / path-purity state.
        self.summary.status = "OK"
        self.summary.elapsed_seconds = float(time.perf_counter() - t0)
        if (opts.output_root is None
                and not self.summary.raw_modified
                and self.summary.historical_products_unchanged
                and self.summary.production_paths_clean
                and not self.summary.audio_features_in_matrix
                and not self.summary.airflow_in_core_model):
            self._update_latest()

        # 14. figures + markdown reports (consume the finalized summary).
        from .report import write_reports
        n_fig = write_reports(
            run_dirs=run_dirs, mc=mc, rin=rin, summary=self.summary, repro=repro,
            determ=determ, thr_df=thr_df, cal_df=cal_df, cw_df=cw_df, boot=boot,
            air=air, alt=alt, run_id=opts.run_id, figure_max=mc.figure_max_count,
        )
        self.summary.n_figures = int(n_fig)
        return self.summary

    # -- summaries ---------------------------------------------------------
    def _repro_summary(self, repro: pd.DataFrame) -> Dict[str, Any]:
        if repro.empty:
            return {}
        diffs = repro["abs_diff"].dropna()
        mism = int((repro["reason"] == "mismatch").sum())
        return {
            "n_rows": int(len(repro)),
            "n_mismatches": mism,
            "max_abs_diff": (float(diffs.max()) if len(diffs) else 0.0),
            "mean_abs_diff": (float(diffs.mean()) if len(diffs) else 0.0),
            "all_match": bool(mism == 0),
        }

    # -- artefact writing --------------------------------------------------
    def _write_artifacts(self, *, run_dirs, mc, rin, determ, repro, thr_df,
                         cal_df, cw_df, boot, air, alt) -> List[str]:
        opts = self.options
        mdir = run_dirs["models"]
        rdir = run_dirs["results"]
        mdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)
        out: List[str] = []

        # signature (results/) + resolved config (models/)
        sig = build_robustness_input_signature(
            run_id=opts.run_id, mc=mc, rin=rin, determinism=determ.get("summary", {}),
            analyses_executed=self.summary.analyses_executed,
        )
        (rdir / "robustness_input_signature.json").write_text(
            json.dumps(sig, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8")
        out.append("results/runs/<run-id>/robustness_input_signature.json")

        write_sensitivity_resolved_yaml(
            cfg=mc, run_id=opts.run_id, config_hash=opts.config_hash,
            path=mdir / "sensitivity_resolved.yaml",
            split_run_id=rin.split_run_id, split_signature=rin.split_signature,
            package_versions=package_versions(),
            stage7_signature=rin.frozen.stage7_signature,
            stage8_signature=rin.frozen.stage8_signature,
            determinism=determ.get("summary", {}),
        )
        out.append("models/runs/<run-id>/sensitivity_resolved.yaml")

        # results/ (names fixed by the prompt Section 4 contract)
        def _csv(df, name):
            df.to_csv(rdir / name, index=False)
            out.append(f"results/runs/<run-id>/{name}")

        repro.to_csv(rdir / "primary_reproduction_metrics.csv", index=False)
        out.append("results/runs/<run-id>/primary_reproduction_metrics.csv")
        (rdir / "rerun_determinism.json").write_text(
            json.dumps(determ, indent=2, default=_json_default), encoding="utf-8")
        out.append("results/runs/<run-id>/rerun_determinism.json")
        _csv(thr_df, "threshold_sensitivity_metrics.csv")
        _csv(cal_df, "calibration_sensitivity_metrics.csv")
        _csv(cw_df, "class_weight_sensitivity_metrics.csv")
        _csv(boot["bootstrap_ci"], "patient_cluster_bootstrap_sensitivity.csv")
        _csv(boot["patient_macro"], "patient_macro_descriptives.csv")
        _csv(boot["paired_fold_delta"], "paired_fold_delta.csv")
        _csv(air["per_patient"], "airflow_selection_profile.csv")
        _csv(air["cohort_summary"], "airflow_cohort_summary.csv")
        _csv(air["feature_quality"], "airflow_feature_quality.csv")
        _csv(alt["alt_label_metrics"], "alternative_label_sensitivity_metrics.csv")
        _csv(alt["alt_label_prevalence"], "alternative_label_prevalence.csv")
        return out

    def _all_output_strings(self) -> List[str]:
        return self.summary.artifacts + [self.options.run_id, self.summary.split_run_id]

    def _update_latest(self) -> None:
        if self.options.output_root is not None:
            return
        cfg = self.cfg
        (cfg.path("models") / "LATEST_RUN.txt").write_text(
            self.options.run_id + "\n", encoding="utf-8")
        (cfg.path("results") / "LATEST_RUN.txt").write_text(
            self.options.run_id + "\n", encoding="utf-8")


__all__ = [
    "RobustnessContaminationError",
    "Stage9Options",
    "Stage9Summary",
    "Stage9Runner",
    "assert_not_contaminating_robustness",
]
