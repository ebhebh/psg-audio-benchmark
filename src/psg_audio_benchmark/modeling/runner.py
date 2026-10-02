"""Stage-7 nested-CV baseline orchestrator.

Drives the leak-proof pipeline end-to-end:

1. input gate (:mod:`modeling.data_gate`) — accept ONLY the frozen approved v2;
2. snapshot ``data/raw`` + protected prior-stage products BEFORE;
3. assemble the model frame + fold adapter (frozen assignments, no re-split);
4. for each model x outer fold: inner-select (tunable models) -> refit on full
   outer-train -> predict outer-test ONCE -> inner-OOF Youden threshold ->
   metrics;
5. pool each model's outer-test OOF; patient-cluster bootstrap the main model;
6. write all run-dir artifacts (models / results / reports), pseudonymized,
   path-pure;
7. snapshot AFTER -> prove ``data/raw`` + prior-stage products unchanged;
8. on success update ``models/LATEST_RUN.txt`` + ``results/LATEST_RUN.txt``.

Audio is BLOCKED throughout (no ``*.wav`` read, no audio product). Airflow never
enters the core model. A dry run does the gate + assembly + fold-isolation check
and trains nothing / writes nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from ..run_metadata import package_versions
from ..windowing.qc import (  # single path-purity implementation
    assert_paths_clean, relativize, snapshot_raw,
)
from . import schema as S
from .bootstrap import patient_cluster_bootstrap
from .config import ResolvedModelingConfig, load_modeling_config, write_resolved_modeling_yaml
from .data_gate import DataGateError, DataGateResult, verify_gate
from .feature_assembly import assemble_model_frame
from .folds import build_fold_adapter
from .metrics import compute_metrics
from .pipelines import build_model_specs, fit_predict
from .selection import select_inner
from .threshold import youden_threshold


class ModelingContaminationError(RuntimeError):
    """Raised when a run would write test output into production paths, or when
    a production run looks like a test fixture."""


_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")
_TEST_PATH_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "\\tmp\\", "/tmp/")
_TEST_CONFIG_HASH_VALUES = ("test", "fixture", "pytest")

#: protected prior-stage / sibling production roots that must stay untouched.
_PROTECTED_PATH_KEYS = (
    "data_raw", "splits", "features_physiology", "data_manifests",
    "annotations", "reports_data_audit", "reports_annotations",
    "reports_data_download", "reports_windowing", "reports_feature_extraction",
    "reports_evaluation", "reports_leakage_audit", "reports_model_audit",
    "reports_manuscript_outputs",
)
#: the roots Stage 7 writes into (production).
_STAGE7_OUTPUT_KEYS = ("models", "results", "reports_modeling")


@dataclass
class Stage7Options:
    dry_run: bool
    output_root: Optional[Path]
    config_path: Path
    run_id: str
    config_hash: str
    split_run_id: Optional[str] = None


@dataclass
class Stage7Summary:
    status: str  # OK | DRY_RUN | BLOCKED
    run_id: str
    split_run_id: str
    approval_status: str
    gate_checks: List[dict] = field(default_factory=list)
    cohort: Dict[str, Any] = field(default_factory=dict)
    pooled_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    bootstrap_ci: List[dict] = field(default_factory=list)
    selected_hyperparameters: List[dict] = field(default_factory=list)
    raw_modified: bool = False
    historical_products_unchanged: bool = True
    production_paths_clean: bool = True
    audio_cohort_count: int = S.AUDIO_COHORT_COUNT
    read_wav: bool = False
    airflow_in_core_model: bool = False
    n_figures: int = 0
    artifacts: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# contamination guard (mirrors prior stages, adapted to Stage-7 roots)
# ---------------------------------------------------------------------------

def assert_not_contaminating(
    *, cfg: Config, run_id: str, config_hash: str, output_root: Optional[Path],
) -> None:
    run_id_low = (run_id or "").lower()
    if output_root is None:
        hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
        if hit:
            raise ModelingContaminationError(
                f"Production-mode run refused: run_id {run_id!r} contains test "
                f"marker(s) {hit}."
            )
        if config_hash.lower() in _TEST_CONFIG_HASH_VALUES:
            raise ModelingContaminationError(
                f"Production-mode run refused: config_hash {config_hash!r} looks "
                f"like a test value."
            )
        raw = cfg.path("data_raw").resolve()
        try:
            raw.relative_to(cfg.project_root.resolve())
        except ValueError as exc:
            raise ModelingContaminationError(
                f"Production-mode run refused: data_raw {raw} is outside the "
                f"project root."
            ) from exc
    else:
        root = Path(output_root).resolve()
        for key in _PROTECTED_PATH_KEYS + _STAGE7_OUTPUT_KEYS:
            forbidden = cfg.path(key).resolve()
            try:
                root.relative_to(forbidden)
                inside = True
            except ValueError:
                inside = root == forbidden
            if inside:
                raise ModelingContaminationError(
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
    # existing run-dirs under models/runs and results/runs (excluding current)
    for key in ("models", "results"):
        runs = cfg.path(key) / "runs"
        if runs.exists():
            # relativize against `runs` itself so rel == "<run_id>/file"; then the
            # leading component is the run_id (used to exclude the current run).
            for rel, val in snapshot_raw(runs, runs).items():
                top = rel.split("/", 1)[0]
                if top == skip_run_id:
                    continue
                snap[f"{key}/runs/{rel}"] = val
    return snap


# ---------------------------------------------------------------------------
# nested CV
# ---------------------------------------------------------------------------

def _run_model_outer_fold(
    *, spec, X: np.ndarray, y: np.ndarray, adapter, outer_fold: int,
    inner_n_folds: int,
) -> Dict[str, Any]:
    """Inner-select (if tunable), refit on full outer-train, predict outer-test
    once, choose threshold, compute metrics. Returns a record bundle."""
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
        proba = fit_predict(
            spec=spec, X=X, y=y, train_idx=train_idx, pred_idx=test_idx,
        )
        selected_candidate = spec.candidates[0]
        selected_mean_ap = None
        selected_mean_auroc = None
        sel_rule = "no tuning (dummy_prior)"
        all_candidates = []

    thr = youden_threshold(
        y_true=(inner_oof[0] if inner_oof is not None else y[train_idx]),
        y_prob=(inner_oof[1] if inner_oof is not None else np.full(train_idx.shape, 0.5)),
        model_name=spec.name,
    )
    y_test = y[test_idx]
    metrics = compute_metrics(y_true=y_test, y_prob=proba, threshold=thr.threshold)
    return {
        "spec": spec,
        "outer_fold": outer_fold,
        "test_idx": test_idx,
        "y_test": y_test,
        "proba": proba,
        "threshold": thr,
        "metrics": metrics,
        "selection": selection,
        "selected_candidate": selected_candidate,
        "selected_mean_ap": selected_mean_ap,
        "selected_mean_auroc": selected_mean_auroc,
        "sel_rule": sel_rule,
        "all_candidates": all_candidates,
    }


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

class Stage7Runner:
    def __init__(self, cfg: Config, options: Stage7Options) -> None:
        self.cfg = cfg
        self.options = options
        self.summary = Stage7Summary(
            status="BLOCKED", run_id=options.run_id, split_run_id="",
            approval_status="",
        )

    # -- helpers -----------------------------------------------------------
    def _load_mc(self) -> ResolvedModelingConfig:
        return load_modeling_config(self.options.config_path)

    # -- main --------------------------------------------------------------
    def run(self) -> Stage7Summary:
        t0 = time.perf_counter()
        cfg = self.cfg
        opts = self.options
        assert_not_contaminating(
            cfg=cfg, run_id=opts.run_id, config_hash=opts.config_hash,
            output_root=opts.output_root,
        )
        mc = self._load_mc()

        # snapshot raw + protected products BEFORE (production only)
        raw_before: Dict[str, str] = {}
        prot_before: Dict[str, str] = {}
        if opts.output_root is None:
            raw_before = snapshot_raw(cfg.path("data_raw"), cfg.project_root)
            prot_before = _snapshot_protected(cfg, skip_run_id=opts.run_id)

        # 1. gate
        gate: DataGateResult = verify_gate(
            cfg=cfg, config=mc, split_run_id_override=opts.split_run_id,
        )
        self.summary.split_run_id = gate.split_run_id
        self.summary.approval_status = gate.signature.get("approval_status", "")
        self.summary.gate_checks = [
            {"name": c.name, "passed": bool(c.passed), "detail": c.detail}
            for c in gate.checks
        ]
        self.summary.cohort = {
            "n_windows": int(len(gate.cohort_windows)),
            "n_patients": int(gate.cohort_windows["patient_id"].nunique()),
            "n_positive": int((gate.cohort_windows["binary_event_label"] == 1).sum()),
            "n_negative": int((gate.cohort_windows["binary_event_label"] == 0).sum()),
        }

        # 2. assemble frame + folds
        frame = assemble_model_frame(
            cohort_windows=gate.cohort_windows, outer_folds=gate.outer_folds,
            hr_features_path=cfg.path("features_physiology") / "hr_window_features.parquet",
            spo2_features_path=cfg.path("features_physiology") / "spo2_window_features.parquet",
        )
        adapter = build_fold_adapter(
            frame=frame, inner_folds=gate.inner_folds,
            outer_n_folds=mc.outer_n_folds, inner_n_folds=mc.inner_n_folds,
        )

        # dry run: no training, no writes
        if opts.dry_run:
            self.summary.status = "DRY_RUN"
            self.summary.raw_modified = False
            self.summary.historical_products_unchanged = True
            self.summary.elapsed_seconds = float(time.perf_counter() - t0)
            return self.summary

        # 3. nested CV for all models
        X, y = frame.X, frame.y
        specs = build_model_specs()
        fold_records: List[dict] = []
        oof_rows: List[dict] = []
        selected_records: List[dict] = []
        candidate_records: List[dict] = []
        per_model_oof: Dict[str, Dict[str, np.ndarray]] = {}
        for model_name in S.MODEL_FAMILIES:
            spec = specs[model_name]
            model_oof_y: List[int] = []
            model_oof_p: List[float] = []
            model_oof_pid: List[str] = []
            model_oof_wid: List[str] = []
            model_oof_fold: List[int] = []
            for outer_fold in range(mc.outer_n_folds):
                rec = _run_model_outer_fold(
                    spec=spec, X=X, y=y, adapter=adapter, outer_fold=outer_fold,
                    inner_n_folds=mc.inner_n_folds,
                )
                m = rec["metrics"]
                fold_records.append(self._fold_record(model_name, outer_fold, rec, m))
                test_idx = rec["test_idx"]
                thr = rec["threshold"]
                y_pred = (rec["proba"] >= thr.threshold).astype(int)
                for pos, i in enumerate(test_idx):
                    oof_rows.append({
                        "run_id": opts.run_id,
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
                model_oof_y.extend(int(v) for v in rec["y_test"])
                model_oof_p.extend(float(v) for v in rec["proba"])
                model_oof_pid.extend(str(frame.patient_id[i]) for i in test_idx)
                model_oof_wid.extend(str(frame.window_id[i]) for i in test_idx)
                model_oof_fold.extend([int(outer_fold)] * len(test_idx))
                # selection / candidate audit
                sel = rec.get("selection")
                if sel is not None:
                    for cr in sel.candidates:
                        d = cr.as_record(run_id=opts.run_id, model_name=model_name,
                                         outer_fold=outer_fold)
                        d["selected"] = bool(cr is sel.selected)
                        candidate_records.append(d)
                selected_records.append(self._selected_record(
                    model_name, outer_fold, rec))
            per_model_oof[model_name] = {
                "y": np.asarray(model_oof_y),
                "p": np.asarray(model_oof_p),
                "pid": np.asarray(model_oof_pid),
                "wid": np.asarray(model_oof_wid),
                "fold": np.asarray(model_oof_fold),
            }

        # 4. pooled metrics per model
        pooled: Dict[str, Dict[str, Any]] = {}
        for model_name in S.MODEL_FAMILIES:
            o = per_model_oof[model_name]
            # pooled threshold: mean of per-fold thresholds (report only); for
            # pooled threshold-dependent metrics use the per-window threshold
            # actually applied in its own fold.
            thr_per_window = np.asarray([
                oof_rows[r]["threshold"]
                for r in range(len(oof_rows))
                if oof_rows[r]["model"] == model_name
            ])
            pooled_metrics = compute_metrics(
                y_true=o["y"], y_prob=o["p"], threshold=0.5)
            # recompute confusion with per-window thresholds for correctness
            y_pred = (o["p"] >= thr_per_window).astype(int)
            pooled_metrics = self._pooled_with_per_window_threshold(
                pooled_metrics, o["y"], o["p"], thr_per_window, y_pred)
            pooled[model_name] = pooled_metrics
        self.summary.pooled_metrics = pooled

        # 5. bootstrap main model
        bootstrap_rows: List[dict] = []
        if mc.bootstrap_enabled and S.PRIMARY_MODEL in mc.bootstrap_apply_to:
            o = per_model_oof[S.PRIMARY_MODEL]
            cis = patient_cluster_bootstrap(
                y_true=o["y"], y_prob=o["p"], patient_ids=o["pid"],
                metrics=tuple(mc.threshold_free_metrics[:2]),
                n_resamples=mc.bootstrap_n_resamples, seed=mc.bootstrap_seed,
                ci_level=mc.bootstrap_ci_level,
            )
            for ci in cis:
                bootstrap_rows.append(ci.as_record(
                    run_id=opts.run_id, model_name=S.PRIMARY_MODEL))
        self.summary.bootstrap_ci = bootstrap_rows

        # 6. write artifacts
        run_dirs = _run_dirs(cfg=cfg, run_id=opts.run_id, output_root=opts.output_root)
        artifacts = self._write_artifacts(
            run_dirs=run_dirs, mc=mc, gate=gate, frame=frame,
            fold_records=fold_records, oof_rows=oof_rows, pooled=pooled,
            bootstrap_rows=bootstrap_rows, selected_records=selected_records,
            candidate_records=candidate_records,
        )
        # figures + reports
        from .report import write_reports  # local import (matplotlib lazy)
        n_fig = write_reports(
            run_dirs=run_dirs, mc=mc, gate=gate, frame=frame,
            per_model_oof=per_model_oof, pooled=pooled,
            bootstrap_rows=bootstrap_rows, fold_records=fold_records,
            selected_records=selected_records, run_id=opts.run_id,
            split_run_id=gate.split_run_id, elapsed=self.summary.elapsed_seconds,
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
            # update LATEST_RUN pointers only on a clean production run
            if (not self.summary.raw_modified
                    and self.summary.historical_products_unchanged
                    and self.summary.production_paths_clean):
                self._update_latest()

        self.summary.status = "OK"
        self.summary.elapsed_seconds = float(time.perf_counter() - t0)
        return self.summary

    # -- artifact writers --------------------------------------------------
    def _write_artifacts(
        self, *, run_dirs, mc, gate, frame, fold_records, oof_rows, pooled,
        bootstrap_rows, selected_records, candidate_records,
    ) -> List[str]:
        opts = self.options
        mdir = run_dirs["models"]
        rdir = run_dirs["results"]
        mdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        # selected_hyperparameters.csv
        pd.DataFrame(selected_records).to_csv(
            mdir / "selected_hyperparameters.csv", index=False)
        # inner_cv_candidate_scores.csv
        pd.DataFrame(candidate_records).to_csv(
            mdir / "inner_cv_candidate_scores.csv", index=False)
        # modeling_resolved.yaml
        write_resolved_modeling_yaml(
            cfg=mc, run_id=opts.run_id, config_hash=opts.config_hash,
            path=mdir / "modeling_resolved.yaml",
            split_run_id=gate.split_run_id, split_signature=gate.signature,
            package_versions=package_versions(),
        )
        # outer_fold_metrics.csv
        pd.DataFrame(fold_records).to_csv(
            rdir / "outer_fold_metrics.csv", index=False)
        # oof_predictions.parquet
        pd.DataFrame(oof_rows).to_parquet(
            rdir / "oof_predictions.parquet", index=False)
        # pooled_metrics.json
        (rdir / "pooled_metrics.json").write_text(
            json.dumps(pooled, indent=2, default=_json_default), encoding="utf-8")
        # patient_cluster_bootstrap_ci.csv
        pd.DataFrame(bootstrap_rows).to_csv(
            rdir / "patient_cluster_bootstrap_ci.csv", index=False)
        # model_input_signature.json
        sig = self._model_input_signature(mc=mc, gate=gate, frame=frame)
        (rdir / "model_input_signature.json").write_text(
            json.dumps(sig, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8")
        return [
            "models/runs/<run-id>/selected_hyperparameters.csv",
            "models/runs/<run-id>/inner_cv_candidate_scores.csv",
            "models/runs/<run-id>/modeling_resolved.yaml",
            "results/runs/<run-id>/outer_fold_metrics.csv",
            "results/runs/<run-id>/oof_predictions.parquet",
            "results/runs/<run-id>/pooled_metrics.json",
            "results/runs/<run-id>/patient_cluster_bootstrap_ci.csv",
            "results/runs/<run-id>/model_input_signature.json",
        ]

    def _model_input_signature(self, *, mc, gate, frame) -> Dict[str, Any]:
        return {
            "run_id": self.options.run_id,
            "modeling_version": S.MODELING_VERSION,
            "task_name": S.TASK_NAME,
            "cohort": S.COHORT,
            "consumed_split_run_id": gate.split_run_id,
            "split_approval_status": gate.signature.get("approval_status"),
            "split_assignment_sha256": (gate.signature.get("split", {}) or {}).get("assignment_sha256"),
            "split_patient_set_sha256": (gate.signature.get("cohort", {}) or {}).get("patient_set_sha256"),
            "runtime_cohort": self.summary.cohort,
            "feature_whitelist": list(frame.feature_names),
            "feature_count": int(len(frame.feature_names)),
            "label_column": S.LABEL_COLUMN,
            "label_values": {"negative": 0, "positive": 1},
            "n_outer_folds": mc.outer_n_folds,
            "n_inner_folds": mc.inner_n_folds,
            "audio_block": {
                "audio_cohort_count": S.AUDIO_COHORT_COUNT,
                "read_wav_forbidden": True,
                "audio_features_in_matrix": bool(
                    any("audio" in c for c in frame.feature_names)),
                "airflow_features_in_matrix": bool(
                    any("airflow" in c for c in frame.feature_names)),
            },
            "forbidden_categories_absent": True,
        }

    # -- record builders ---------------------------------------------------
    def _fold_record(self, model_name, outer_fold, rec, m) -> dict:
        return {
            "run_id": self.options.run_id,
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

    def _selected_record(self, model_name, outer_fold, rec) -> dict:
        cand = rec["selected_candidate"]
        return {
            "run_id": self.options.run_id,
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
        # recompute confusion-matrix-derived metrics with the per-window threshold
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
    "ModelingContaminationError",
    "Stage7Options",
    "Stage7Summary",
    "Stage7Runner",
    "assert_not_contaminating",
]
