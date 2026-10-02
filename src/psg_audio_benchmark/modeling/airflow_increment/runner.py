"""Stage-8 airflow-incremental paired nested-CV orchestrator.

Drives the leak-proof paired comparison end-to-end:

1. input gate (:mod:`airflow_increment.data_gate`) -- accept ONLY the frozen
   approved v2 split AND the runtime-re-derived 34-patient airflow sub-cohort;
2. snapshot ``data/raw`` + protected prior-stage products (incl. Stage-7) BEFORE;
3. assemble the TWO membership-identical frames (core_restricted,
   airflow_enhanced) + ONE inherited fold adapter (frozen v2 assignments);
4. for each (feature set x model x outer fold): inner-select (tunable models) ->
   refit on full outer-train -> predict outer-test ONCE -> inner-OOF Youden
   threshold -> metrics. Each (feature set x model) selects INDEPENDENTLY on the
   SAME frozen inner folds;
5. pool each (feature set x model)'s outer-test OOF; build the paired same-window
   OOF; paired patient-cluster bootstrap the primary LR increment;
6. write all run-dir artifacts (models / results / reports), pseudonymized,
   path-pure;
7. snapshot AFTER -> prove ``data/raw`` + prior-stage products (incl. Stage-7)
   unchanged;
8. on success update ``models/LATEST_RUN.txt`` + ``results/LATEST_RUN.txt``.

Audio is BLOCKED throughout (no ``*.wav`` read, no audio product). Airflow enters
ONLY the airflow_enhanced matrix. A dry run does the gate + assembly +
fold-isolation check and trains nothing / writes nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...config import Config
from ...run_metadata import package_versions
from ...windowing.qc import (  # single path-purity implementation
    assert_paths_clean, relativize, snapshot_raw,
)
from . import schema as S
from .config import (
    ResolvedAirflowConfig,
    load_airflow_config,
    write_resolved_airflow_yaml,
)
from .data_gate import AirflowGateError, AirflowGateResult, verify_airflow_gate
from .feature_assembly import (
    AirflowFeatureAssemblyError,
    assemble_paired_frames,
)
from .paired_bootstrap import paired_patient_cluster_bootstrap
from .. import schema as _CORE_S
from ..folds import build_fold_adapter
from ..metrics import compute_metrics
from ..pipelines import build_model_specs, fit_predict
from ..selection import select_inner
from ..threshold import youden_threshold


class AirflowModelingContaminationError(RuntimeError):
    """Raised when a run would write test output into production paths, or when
    a production run looks like a test fixture."""


_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "\\tmp\\", "/tmp/")
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

#: protected prior-stage / sibling production roots that must stay untouched
#: (Stage-7 modeling roots are protected too -- Stage 8 must NOT alter them).
_PROTECTED_PATH_KEYS = (
    "data_raw", "splits", "features_physiology", "data_manifests",
    "annotations", "reports_data_audit", "reports_annotations",
    "reports_data_download", "reports_windowing", "reports_feature_extraction",
    "reports_evaluation", "reports_leakage_audit", "reports_model_audit",
    "reports_manuscript_outputs",
)
#: the roots Stage 8 writes into (production).
_STAGE8_OUTPUT_KEYS = ("models", "results", "reports_modeling")


@dataclass
class Stage8Options:
    dry_run: bool
    output_root: Optional[Path]
    config_path: Path
    run_id: str
    config_hash: str
    split_run_id: Optional[str] = None


@dataclass
class Stage8Summary:
    status: str  # OK | DRY_RUN | BLOCKED
    run_id: str
    split_run_id: str
    approval_status: str
    gate_checks: List[dict] = field(default_factory=list)
    airflow_cohort: Dict[str, Any] = field(default_factory=dict)
    core_cohort: Dict[str, Any] = field(default_factory=dict)
    pooled_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    paired_bootstrap_ci: List[dict] = field(default_factory=list)
    selected_hyperparameters: List[dict] = field(default_factory=list)
    raw_modified: bool = False
    historical_products_unchanged: bool = True
    production_paths_clean: bool = True
    audio_cohort_count: int = S.AUDIO_COHORT_COUNT
    read_wav: bool = False
    n_figures: int = 0
    artifacts: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# contamination guard
# ---------------------------------------------------------------------------

def assert_not_contaminating_airflow(
    *, cfg: Config, run_id: str, config_hash: str, output_root: Optional[Path],
) -> None:
    run_id_low = (run_id or "").lower()
    if output_root is None:
        hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
        if hit:
            raise AirflowModelingContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}."
            )
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise AirflowModelingContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value."
            )
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise AirflowModelingContaminationError(
                f"Production-mode run refused: data_raw {raw} is outside the "
                f"project root."
            ) from exc
    else:
        root = Path(output_root).resolve()
        for key in _PROTECTED_PATH_KEYS + _STAGE8_OUTPUT_KEYS:
            forbidden = cfg.path(key).resolve()
            try:
                root.relative_to(forbidden)
                inside = True
            except ValueError:
                inside = root == forbidden
            if inside:
                raise AirflowModelingContaminationError(
                    f"Isolated output_root {root} must not be (or be inside) the "
                    f"production path {forbidden} ({key})."
                )


# ---------------------------------------------------------------------------
# output-root resolution
# ---------------------------------------------------------------------------

def _run_dirs(
    *, cfg: Config, run_id: str, output_root: Optional[Path],
) -> Dict[str, Path]:
    base = {k: (Path(output_root) if output_root else cfg.project_root) for k in
            ("models", "results", "reports")}
    rel = {
        "models": Path("models") / "runs" / run_id,
        "results": Path("results") / "runs" / run_id,
        "reports": Path("reports") / "modeling" / "runs" / run_id,
    }
    return {k: base[k] / rel[k] for k in base}


# ---------------------------------------------------------------------------
# protected-product snapshot
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# nested CV per (feature set x model x outer fold)
# ---------------------------------------------------------------------------

def _run_feature_set_model_fold(
    *, spec, X: np.ndarray, y: np.ndarray, adapter, outer_fold: int,
    inner_n_folds: int,
) -> Dict[str, Any]:
    """Inner-select (if tunable), refit on full outer-train, predict outer-test
    once, choose threshold, compute metrics. Mirrors Stage-7's per-fold routine
    but is feature-set-agnostic (the caller supplies the feature set's X)."""
    train_idx = np.nonzero(adapter.outer_train_mask(outer_fold))[0]
    test_idx = np.nonzero(adapter.outer_test_mask(outer_fold))[0]

    selection = None
    inner_oof = None
    if spec.tunable:
        selection = select_inner(
            spec=spec, X=X, y=y, adapter=adapter, outer_fold=outer_fold,
            inner_n_folds=inner_n_folds,
        )
        cand = selection.selected
        inner_oof = (cand.oof_y_true, cand.oof_y_prob)
        proba = fit_predict(
            spec=spec, X=X, y=y, train_idx=train_idx, pred_idx=test_idx,
            candidate=cand.candidate,
        )
        selected_candidate = cand.candidate
        selected_mean_ap = cand.mean_ap
        selected_mean_auroc = cand.mean_auroc
        sel_rule = selection.selection_rule
        all_candidates = selection.candidates
    else:
        proba = fit_predict(spec=spec, X=X, y=y, train_idx=train_idx, pred_idx=test_idx)
        selected_candidate = spec.candidates[0]
        selected_mean_ap = None
        selected_mean_auroc = None
        sel_rule = "no tuning"
        all_candidates = []

    thr = youden_threshold(
        y_true=(inner_oof[0] if inner_oof is not None else y[train_idx]),
        y_prob=(inner_oof[1] if inner_oof is not None
                else np.full(train_idx.shape, 0.5)),
        model_name=spec.name,
    )
    y_test = y[test_idx]
    metrics = compute_metrics(y_true=y_test, y_prob=proba, threshold=thr.threshold)
    return {
        "spec": spec, "outer_fold": outer_fold, "test_idx": test_idx,
        "y_test": y_test, "proba": proba, "threshold": thr, "metrics": metrics,
        "selection": selection, "selected_candidate": selected_candidate,
        "selected_mean_ap": selected_mean_ap,
        "selected_mean_auroc": selected_mean_auroc, "sel_rule": sel_rule,
        "all_candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class Stage8Runner:
    def __init__(self, cfg: Config, options: Stage8Options) -> None:
        self.cfg = cfg
        self.options = options
        self.summary = Stage8Summary(
            status="BLOCKED", run_id=options.run_id, split_run_id="",
            approval_status="",
        )

    def _load_mc(self) -> ResolvedAirflowConfig:
        return load_airflow_config(self.options.config_path)

    # -- main --------------------------------------------------------------
    def run(self) -> Stage8Summary:
        t0 = time.perf_counter()
        cfg = self.cfg
        opts = self.options
        assert_not_contaminating_airflow(
            cfg=cfg, run_id=opts.run_id, config_hash=opts.config_hash,
            output_root=opts.output_root,
        )
        mc = self._load_mc()

        raw_before: Dict[str, str] = {}
        prot_before: Dict[str, str] = {}
        if opts.output_root is None:
            raw_before = snapshot_raw(cfg.path("data_raw"), cfg.project_root)
            prot_before = _snapshot_protected(cfg, skip_run_id=opts.run_id)

        # 1. gate
        gate: AirflowGateResult = verify_airflow_gate(
            cfg=cfg, config=mc, split_run_id_override=opts.split_run_id,
        )
        self.summary.split_run_id = gate.split_run_id
        self.summary.approval_status = gate.signature.get("approval_status", "")
        self.summary.gate_checks = [
            {"name": c.name, "passed": bool(c.passed), "detail": c.detail}
            for c in gate.checks
        ]
        self.summary.airflow_cohort = {
            "n_windows": int(len(gate.airflow_cohort_windows)),
            "n_patients": int(gate.airflow_cohort_windows["patient_id"].nunique()),
            "n_positive": int((gate.airflow_cohort_windows["binary_event_label"] == 1).sum()),
            "n_negative": int((gate.airflow_cohort_windows["binary_event_label"] == 0).sum()),
        }
        self.summary.core_cohort = {
            "n_windows": int(len(gate.core_cohort_windows)),
            "n_patients": int(gate.core_cohort_windows["patient_id"].nunique()),
        }

        # 2. assemble paired frames + inherited fold adapter
        fdir = cfg.path("features_physiology")
        core_frame, enhanced_frame = assemble_paired_frames(
            cohort_windows=gate.airflow_cohort_windows,
            outer_folds=gate.outer_folds,
            hr_features_path=fdir / "hr_window_features.parquet",
            spo2_features_path=fdir / "spo2_window_features.parquet",
            airflow_features_path=fdir / "airflow_window_features.parquet",
        )
        # re-assert identical membership at runner level (defence in depth)
        from .feature_assembly import assert_frames_identical_membership
        assert_frames_identical_membership(core_frame, enhanced_frame)
        adapter = build_fold_adapter(
            frame=core_frame, inner_folds=gate.inner_folds,
            outer_n_folds=mc.outer_n_folds, inner_n_folds=mc.inner_n_folds,
        )

        # dry run: no training, no writes
        if opts.dry_run:
            self.summary.status = "DRY_RUN"
            self.summary.raw_modified = False
            self.summary.historical_products_unchanged = True
            self.summary.elapsed_seconds = float(time.perf_counter() - t0)
            return self.summary

        # 3. nested CV: (feature set x model x outer fold)
        specs_all = build_model_specs()
        specs = {m: specs_all[m] for m in S.MODEL_FAMILIES}
        frames = {S.FEATURE_SET_CORE: core_frame, S.FEATURE_SET_ENHANCED: enhanced_frame}
        y = core_frame.y  # identical to enhanced.y

        fold_records: List[dict] = []
        oof_rows: List[dict] = []           # long format -> comparison table
        selected_records: List[dict] = []
        candidate_records: List[dict] = []
        # per (feature_set, model) pooled OOF keyed by window_id for pairing
        pooled_oof: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

        for feature_set in S.FEATURE_SETS:
            frame = frames[feature_set]
            X = frame.X
            for model_name in S.MODEL_FAMILIES:
                spec = specs[model_name]
                oof_y: List[int] = []
                oof_p: List[float] = []
                oof_pid: List[str] = []
                oof_wid: List[str] = []
                oof_fold: List[int] = []
                oof_thr: List[float] = []
                for outer_fold in range(mc.outer_n_folds):
                    rec = _run_feature_set_model_fold(
                        spec=spec, X=X, y=y, adapter=adapter,
                        outer_fold=outer_fold, inner_n_folds=mc.inner_n_folds,
                    )
                    m = rec["metrics"]
                    fold_records.append(self._fold_record(
                        feature_set, model_name, outer_fold, rec, m))
                    test_idx = rec["test_idx"]
                    thr = rec["threshold"]
                    y_pred = (rec["proba"] >= thr.threshold).astype(int)
                    for pos, i in enumerate(test_idx):
                        oof_rows.append({
                            "run_id": opts.run_id,
                            "feature_set": feature_set,
                            "model": model_name,
                            "model_role": S.MODEL_ROLE[model_name],
                            "outer_fold": int(outer_fold),
                            "window_id": str(frame.window_id[i]),
                            "patient_id": str(frame.patient_id[i]),
                            "y_true": int(rec["y_test"][pos]),
                            "y_prob": float(rec["proba"][pos]),
                            "threshold": float(thr.threshold),
                            "y_pred": int(y_pred[pos]),
                            "model_version": S.MODELING_VERSION,
                        })
                    oof_y.extend(int(v) for v in rec["y_test"])
                    oof_p.extend(float(v) for v in rec["proba"])
                    oof_pid.extend(str(frame.patient_id[i]) for i in test_idx)
                    oof_wid.extend(str(frame.window_id[i]) for i in test_idx)
                    oof_fold.extend([int(outer_fold)] * len(test_idx))
                    oof_thr.extend([float(thr.threshold)] * len(test_idx))
                    sel = rec.get("selection")
                    if sel is not None:
                        for cr in sel.candidates:
                            d = cr.as_record(run_id=opts.run_id,
                                             model_name=model_name,
                                             outer_fold=outer_fold)
                            d["feature_set"] = feature_set
                            d["selected"] = bool(cr is sel.selected)
                            candidate_records.append(d)
                    selected_records.append(self._selected_record(
                        feature_set, model_name, outer_fold, rec))
                pooled_oof[(feature_set, model_name)] = {
                    "y": np.asarray(oof_y), "p": np.asarray(oof_p),
                    "pid": np.asarray(oof_pid), "wid": np.asarray(oof_wid),
                    "fold": np.asarray(oof_fold), "thr": np.asarray(oof_thr),
                }

        # 4. pooled metrics per (feature_set, model)
        pooled: Dict[str, Dict[str, Any]] = {}
        for feature_set in S.FEATURE_SETS:
            for model_name in S.MODEL_FAMILIES:
                o = pooled_oof[(feature_set, model_name)]
                thr_pw = o["thr"]
                base = compute_metrics(y_true=o["y"], y_prob=o["p"], threshold=0.5)
                y_pred = (o["p"] >= thr_pw).astype(int)
                pm = self._pooled_with_per_window_threshold(
                    base, o["y"], o["p"], thr_pw, y_pred)
                pooled[f"{feature_set}__{model_name}"] = pm
        self.summary.pooled_metrics = pooled

        # 5. paired bootstrap on the PRIMARY LR increment (core vs enhanced)
        bootstrap_rows: List[dict] = []
        if mc.bootstrap_enabled and S.PRIMARY_MODEL in mc.bootstrap_apply_to:
            paired = self._align_paired_primary(pooled_oof)
            if paired is not None:
                cis = paired_patient_cluster_bootstrap(
                    y_true=paired["y"], prob_core=paired["p_core"],
                    prob_enhanced=paired["p_enh"],
                    patient_ids=paired["pid"], threshold_core=paired["thr_core"],
                    threshold_enhanced=paired["thr_enh"],
                    metrics=tuple(mc.bootstrap_threshold_free_delta_metrics)
                            + tuple(mc.bootstrap_threshold_based_delta_metrics),
                    n_resamples=mc.bootstrap_n_resamples, seed=mc.bootstrap_seed,
                    ci_level=mc.bootstrap_ci_level,
                )
                for ci in cis:
                    bootstrap_rows.append(ci.as_record(
                        run_id=opts.run_id, model=S.PRIMARY_MODEL))
        self.summary.paired_bootstrap_ci = bootstrap_rows

        # 6. write artifacts
        run_dirs = _run_dirs(cfg=cfg, run_id=opts.run_id, output_root=opts.output_root)
        artifacts = self._write_artifacts(
            run_dirs=run_dirs, mc=mc, gate=gate,
            core_frame=core_frame, enhanced_frame=enhanced_frame,
            fold_records=fold_records, oof_rows=oof_rows, pooled=pooled,
            bootstrap_rows=bootstrap_rows, selected_records=selected_records,
            candidate_records=candidate_records, pooled_oof=pooled_oof,
        )
        from .report import write_airflow_reports  # local import (matplotlib lazy)
        n_fig = write_airflow_reports(
            run_dirs=run_dirs, mc=mc, gate=gate,
            core_frame=core_frame, enhanced_frame=enhanced_frame,
            pooled=pooled, pooled_oof=pooled_oof, bootstrap_rows=bootstrap_rows,
            fold_records=fold_records, selected_records=selected_records,
            run_id=opts.run_id, split_run_id=gate.split_run_id,
            elapsed=(time.perf_counter() - t0),
        )
        self.summary.n_figures = int(n_fig)
        self.summary.artifacts = artifacts

        # 7. path purity
        try:
            assert_paths_clean(self._all_output_strings())
            self.summary.production_paths_clean = True
        except AssertionError as exc:
            self.summary.production_paths_clean = False
            self.summary.reasons.append(f"path_purity: {exc}")

        # 8. immutability after (production only)
        if opts.output_root is None:
            raw_after = snapshot_raw(cfg.path("data_raw"), cfg.project_root)
            self.summary.raw_modified = (raw_before != raw_after)
            prot_after = _snapshot_protected(cfg, skip_run_id=opts.run_id)
            self.summary.historical_products_unchanged = (prot_before == prot_after)
            if self.summary.raw_modified:
                self.summary.reasons.append("data/raw was modified during the run")
            if not self.summary.historical_products_unchanged:
                self.summary.reasons.append("prior-stage products were modified")
            if (not self.summary.raw_modified
                    and self.summary.historical_products_unchanged
                    and self.summary.production_paths_clean):
                self._update_latest()

        self.summary.status = "OK"
        self.summary.elapsed_seconds = float(time.perf_counter() - t0)
        return self.summary

    # -- paired alignment --------------------------------------------------
    def _align_paired_primary(
        self, pooled_oof: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Join core_restricted & airflow_enhanced PRIMARY-model OOF on window_id
        so the paired bootstrap evaluates both feature sets on the SAME windows."""
        core = pooled_oof[(S.FEATURE_SET_CORE, S.PRIMARY_MODEL)]
        enh = pooled_oof[(S.FEATURE_SET_ENHANCED, S.PRIMARY_MODEL)]
        dc = pd.DataFrame({
            "wid": core["wid"], "y": core["y"], "pid": core["pid"],
            "fold": core["fold"], "p_core": core["p"], "thr_core": core["thr"],
        })
        de = pd.DataFrame({
            "wid": enh["wid"], "p_enh": enh["p"], "thr_enh": enh["thr"],
        })
        m = dc.merge(de, on="wid", how="inner")
        if m.shape[0] == 0:
            return None
        # sort by wid for determinism
        m = m.sort_values("wid").reset_index(drop=True)
        return {
            "y": m["y"].to_numpy(dtype=int),
            "p_core": m["p_core"].to_numpy(dtype=float),
            "p_enh": m["p_enh"].to_numpy(dtype=float),
            "pid": m["pid"].to_numpy(),
            "thr_core": m["thr_core"].to_numpy(dtype=float),
            "thr_enh": m["thr_enh"].to_numpy(dtype=float),
            "fold": m["fold"].to_numpy(dtype=int),
        }

    # -- artifact writers --------------------------------------------------
    def _write_artifacts(
        self, *, run_dirs, mc, gate, core_frame, enhanced_frame, fold_records,
        oof_rows, pooled, bootstrap_rows, selected_records, candidate_records,
        pooled_oof,
    ) -> List[str]:
        opts = self.options
        mdir = run_dirs["models"]
        rdir = run_dirs["results"]
        mdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(selected_records).to_csv(
            mdir / "airflow_selected_hyperparameters.csv", index=False)
        pd.DataFrame(candidate_records).to_csv(
            mdir / "airflow_inner_cv_candidate_scores.csv", index=False)
        write_resolved_airflow_yaml(
            cfg=mc, run_id=opts.run_id, config_hash=opts.config_hash,
            path=mdir / "airflow_incremental_resolved.yaml",
            split_run_id=gate.split_run_id, split_signature=gate.signature,
            package_versions=package_versions(),
            airflow_cohort=self.summary.airflow_cohort,
        )
        pd.DataFrame(fold_records).to_csv(
            rdir / "airflow_outer_fold_metrics.csv", index=False)
        # long-format comparison table (feature_set x model x window)
        pd.DataFrame(oof_rows).to_csv(
            rdir / "airflow_comparison_table.csv", index=False)
        # wide paired OOF (per window: core & enhanced for each model)
        wide = self._build_wide_paired(pooled_oof)
        wide.to_parquet(rdir / "airflow_paired_oof_predictions.parquet",
                        index=False)
        (rdir / "airflow_pooled_metrics.json").write_text(
            json.dumps(pooled, indent=2, default=_json_default), encoding="utf-8")
        pd.DataFrame(bootstrap_rows).to_csv(
            rdir / "airflow_paired_patient_bootstrap_ci.csv", index=False)
        sig = self._model_input_signature(
            mc=mc, gate=gate, core_frame=core_frame, enhanced_frame=enhanced_frame)
        (rdir / "airflow_model_input_signature.json").write_text(
            json.dumps(sig, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8")
        return [
            "models/runs/<run-id>/airflow_selected_hyperparameters.csv",
            "models/runs/<run-id>/airflow_inner_cv_candidate_scores.csv",
            "models/runs/<run-id>/airflow_incremental_resolved.yaml",
            "results/runs/<run-id>/airflow_paired_oof_predictions.parquet",
            "results/runs/<run-id>/airflow_outer_fold_metrics.csv",
            "results/runs/<run-id>/airflow_pooled_metrics.json",
            "results/runs/<run-id>/airflow_paired_patient_bootstrap_ci.csv",
            "results/runs/<run-id>/airflow_comparison_table.csv",
            "results/runs/<run-id>/airflow_model_input_signature.json",
        ]

    def _build_wide_paired(
        self, pooled_oof: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    ) -> pd.DataFrame:
        """One row per window with core & enhanced probabilities per model."""
        # use the core PRIMARY OOF as the window spine
        spine = pooled_oof[(S.FEATURE_SET_CORE, S.PRIMARY_MODEL)]
        out = pd.DataFrame({
            "window_id": spine["wid"], "patient_id": spine["pid"],
            "outer_fold": spine["fold"], "y_true": spine["y"],
        }).sort_values("window_id").reset_index(drop=True)
        for model_name in S.MODEL_FAMILIES:
            core = pooled_oof[(S.FEATURE_SET_CORE, model_name)]
            enh = pooled_oof[(S.FEATURE_SET_ENHANCED, model_name)]
            dc = pd.DataFrame({"window_id": core["wid"],
                               f"{model_name}__core_restricted__prob": core["p"],
                               f"{model_name}__core_restricted__threshold": core["thr"]})
            de = pd.DataFrame({"window_id": enh["wid"],
                               f"{model_name}__airflow_enhanced__prob": enh["p"],
                               f"{model_name}__airflow_enhanced__threshold": enh["thr"]})
            out = out.merge(dc, on="window_id", how="left")
            out = out.merge(de, on="window_id", how="left")
            cprob = out[f"{model_name}__core_restricted__prob"]
            eprob = out[f"{model_name}__airflow_enhanced__prob"]
            out[f"{model_name}__delta_prob"] = eprob - cprob
        out["model_version"] = S.MODELING_VERSION
        return out

    def _model_input_signature(
        self, *, mc, gate, core_frame, enhanced_frame,
    ) -> Dict[str, Any]:
        return {
            "run_id": self.options.run_id,
            "modeling_version": S.MODELING_VERSION,
            "task_name": S.TASK_NAME,
            "cohort": S.COHORT,
            "consumed_split_run_id": gate.split_run_id,
            "split_approval_status": gate.signature.get("approval_status"),
            "split_assignment_sha256": (gate.signature.get("split", {}) or {}).get(
                "assignment_sha256"),
            "split_patient_set_sha256": (gate.signature.get("cohort", {}) or {}).get(
                "patient_set_sha256"),
            "runtime_core_cohort": self.summary.core_cohort,
            "runtime_airflow_cohort": self.summary.airflow_cohort,
            "feature_sets": {
                "core_restricted": {
                    "whitelist": list(core_frame.feature_names),
                    "feature_count": len(core_frame.feature_names),
                },
                "airflow_enhanced": {
                    "whitelist": list(enhanced_frame.feature_names),
                    "feature_count": len(enhanced_frame.feature_names),
                    "airflow_columns": list(S.AIRFLOW_FEATURE_WHITELIST),
                },
            },
            "membership_identical": True,
            "n_outer_folds": mc.outer_n_folds,
            "n_inner_folds": mc.inner_n_folds,
            "main_comparison": {
                "model": S.PRIMARY_MODEL,
                "delta": "airflow_enhanced_minus_core_restricted",
            },
            "audio_block": {
                "audio_cohort_count": S.AUDIO_COHORT_COUNT,
                "read_wav_forbidden": True,
                "audio_features_in_matrix": bool(
                    any("audio" in c for c in enhanced_frame.feature_names)),
                "airflow_in_core_matrix": bool(
                    any("airflow" in c for c in core_frame.feature_names)),
                "airflow_in_enhanced_matrix": bool(
                    any("airflow" in c for c in enhanced_frame.feature_names)),
            },
            "forbidden_categories_absent": True,
        }

    # -- record builders ---------------------------------------------------
    def _fold_record(self, feature_set, model_name, outer_fold, rec, m) -> dict:
        return {
            "run_id": self.options.run_id,
            "feature_set": feature_set,
            "model": model_name,
            "model_role": S.MODEL_ROLE[model_name],
            "outer_fold": int(outer_fold),
            "n": m["n"], "n_positive": m["n_positive"], "n_negative": m["n_negative"],
            "positive_rate": m["positive_rate"],
            "threshold": m["threshold"],
            "threshold_rule": rec["threshold"].rule,
            "auroc": m["auroc"], "auprc": m["auprc"],
            "brier": m["brier"], "log_loss": m["log_loss"],
            "sensitivity": m["sensitivity"], "specificity": m["specificity"],
            "precision": m["precision"], "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "tp": m["confusion_matrix"]["tp"], "fp": m["confusion_matrix"]["fp"],
            "tn": m["confusion_matrix"]["tn"], "fn": m["confusion_matrix"]["fn"],
            "single_class": m["single_class"],
        }

    def _selected_record(self, feature_set, model_name, outer_fold, rec) -> dict:
        cand = rec["selected_candidate"]
        return {
            "run_id": self.options.run_id,
            "feature_set": feature_set,
            "model": model_name,
            "model_role": S.MODEL_ROLE[model_name],
            "outer_fold": int(outer_fold),
            "selected_candidate_index": int(cand.index),
            "selected_params": cand.label,
            "selected_mean_average_precision": rec["selected_mean_ap"],
            "selected_mean_auroc": rec["selected_mean_auroc"],
            "selection_rule": rec["sel_rule"],
            "threshold": float(rec["threshold"].threshold),
            "threshold_rule": rec["threshold"].rule,
            "threshold_reason": rec["threshold"].reason,
        }

    def _pooled_with_per_window_threshold(
        self, base, y, p, thr_per_window, y_pred) -> Dict[str, Any]:
        tp = int(np.sum((y_pred == 1) & (y == 1)))
        fp = int(np.sum((y_pred == 1) & (y == 0)))
        tn = int(np.sum((y_pred == 0) & (y == 0)))
        fn = int(np.sum((y_pred == 0) & (y == 1)))
        sens = tp / (tp + fn) if (tp + fn) else None
        spec = tn / (tn + fp) if (tn + fp) else None
        prec = tp / (tp + fp) if (tp + fp) else None
        f1 = (2 * prec * sens / (prec + sens)) if (prec is not None and sens is not None
                                                   and (prec + sens) > 0) else None
        bal = (sens + spec) / 2.0 if (sens is not None and spec is not None) else None
        base = dict(base)
        base["threshold"] = "per_window_inner_oof_youden"
        base["confusion_matrix"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        base["sensitivity"] = sens
        base["specificity"] = spec
        base["precision"] = prec
        base["f1"] = f1
        base["balanced_accuracy"] = bal
        return base

    def _all_output_strings(self) -> List[str]:
        return self.summary.artifacts + [self.options.run_id, self.summary.split_run_id]

    def _update_latest(self) -> None:
        cfg = self.cfg
        if self.options.output_root is not None:
            return
        (cfg.path("models") / "LATEST_RUN.txt").write_text(
            self.options.run_id + "\n", encoding="utf-8")
        (cfg.path("results") / "LATEST_RUN.txt").write_text(
            self.options.run_id + "\n", encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


__all__ = [
    "AirflowModelingContaminationError",
    "Stage8Options",
    "Stage8Summary",
    "Stage8Runner",
    "assert_not_contaminating_airflow",
]
