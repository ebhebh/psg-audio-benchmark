"""Stage-9 frozen nested-CV re-run engine.

Re-runs the **exact** Stage 7 (core) and Stage 8 (paired) nested CV using the
frozen v2 split, the frozen feature whitelists and the frozen inner/outer folds,
capturing -- per (cohort, feature_set, model, outer fold) -- the selected
candidate, its inner out-of-fold predictions (the threshold / calibration source)
and the single outer-test out-of-fold probability vector. This is the shared
engine for the threshold / calibration / class-weight / alternative-label
sensitivities; it never re-selects the primary model or threshold and never reads
outer-test labels for fitting.

The regenerated outer-test OOF is later compared (in :mod:`reproduction`) to the
frozen Stage 7/8 OOF to confirm pipeline determinism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...modeling import (
    FoldAdapter, ModelFrame, build_fold_adapter, build_model_specs,
    fit_predict, select_inner, youden_threshold,
)
from ...modeling import schema as MS
from ...modeling.airflow_increment import schema as AS
from ...modeling.airflow_increment.feature_assembly import (
    AirflowModelFrame, assemble_paired_frames,
)
from ...modeling.feature_assembly import assemble_model_frame
from . import schema as S


@dataclass
class FoldRerun:
    cohort: str                       # S.COHORT_CORE | S.COHORT_AIRFLOW
    feature_set: str                  # core_restricted | airflow_enhanced
    model: str
    model_role: str
    outer_fold: int
    test_window_id: np.ndarray
    test_patient_id: np.ndarray
    y_test: np.ndarray
    proba: np.ndarray                 # outer-test OOF P(y=1)
    threshold_youden: float
    threshold_rule: str
    inner_oof_y: Optional[np.ndarray] = None   # selected candidate's inner OOF
    inner_oof_p: Optional[np.ndarray] = None
    selected_params: Optional[str] = None
    selected_candidate_index: Optional[int] = None


@dataclass
class PooledOof:
    cohort: str
    feature_set: str
    model: str
    model_role: str
    y: np.ndarray
    prob: np.ndarray
    window_id: np.ndarray
    patient_id: np.ndarray
    outer_fold: np.ndarray
    threshold: np.ndarray             # per-window youden threshold

    def key(self) -> Tuple[str, str, str]:
        return (self.cohort, self.feature_set, self.model)


@dataclass
class RerunBundle:
    fold_reRuns: List[FoldRerun] = field(default_factory=list)
    pooled: Dict[Tuple[str, str, str], PooledOof] = field(default_factory=dict)
    # In-memory handles for downstream re-runs (class-weight / alt-label) that
    # need to re-select per inner CV on a modified spec/label. NEVER serialized;
    # these are dropped before any table is written.
    core_frame: Any = None
    core_adapter: Any = None
    airflow_core_frame: Any = None
    airflow_enh_frame: Any = None
    airflow_adapter: Any = None
    outer_n_folds: int = 0
    inner_n_folds: int = 0

    def folds(self, *, cohort: str, feature_set: str, model: str) -> List[FoldRerun]:
        return [f for f in self.fold_reRuns
                if f.cohort == cohort and f.feature_set == feature_set
                and f.model == model]

    def pooled_for(self, *, cohort: str, feature_set: str, model: str) -> PooledOof:
        return self.pooled[(cohort, feature_set, model)]

    def frame_and_adapter(self, *, cohort: str, feature_set: str) -> Tuple[Any, Any]:
        """Resolve the (frame, adapter) for a (cohort, feature_set) re-run."""
        if cohort == S.COHORT_CORE:
            if self.core_frame is None or self.core_adapter is None:
                raise RuntimeError("core frame/adapter not attached to bundle")
            return self.core_frame, self.core_adapter
        if cohort == S.COHORT_AIRFLOW:
            if self.airflow_adapter is None:
                raise RuntimeError("airflow adapter not attached to bundle")
            if feature_set == S.FEATURE_SET_CORE:
                frame = self.airflow_core_frame
            else:
                frame = self.airflow_enh_frame
            if frame is None:
                raise RuntimeError(f"airflow {feature_set} frame not attached to bundle")
            return frame, self.airflow_adapter
        raise ValueError(f"unknown cohort {cohort!r}")


# ---------------------------------------------------------------------------
# per-fold execution (mirrors modeling.runner._run_model_outer_fold)
# ---------------------------------------------------------------------------

def _run_one_fold(
    *, spec, X: np.ndarray, y: np.ndarray, adapter: FoldAdapter,
    outer_fold: int, inner_n_folds: int, frame_windows: np.ndarray,
    frame_patients: np.ndarray,
) -> FoldRerun:
    train_idx = np.nonzero(adapter.outer_train_mask(outer_fold))[0]
    test_idx = np.nonzero(adapter.outer_test_mask(outer_fold))[0]

    inner_oof_y: Optional[np.ndarray] = None
    inner_oof_p: Optional[np.ndarray] = None
    sel_params: Optional[str] = None
    sel_idx: Optional[int] = None
    if spec.tunable:
        sel = select_inner(spec=spec, X=X, y=y, adapter=adapter,
                           outer_fold=outer_fold, inner_n_folds=inner_n_folds)
        cand = sel.selected
        inner_oof_y = np.asarray(cand.oof_y_true)
        inner_oof_p = np.asarray(cand.oof_y_prob)
        sel_params = cand.candidate.label
        sel_idx = int(cand.candidate.index)
        proba = fit_predict(spec=spec, X=X, y=y, train_idx=train_idx,
                            pred_idx=test_idx, candidate=cand.candidate)
    else:
        proba = fit_predict(spec=spec, X=X, y=y, train_idx=train_idx, pred_idx=test_idx)

    thr = youden_threshold(
        y_true=(inner_oof_y if inner_oof_y is not None else y[train_idx]),
        y_prob=(inner_oof_p if inner_oof_p is not None
                else np.full(train_idx.shape, 0.5)),
        model_name=spec.name,
    )
    return FoldRerun(
        cohort="", feature_set="", model=spec.name, model_role=spec.role,
        outer_fold=int(outer_fold),
        test_window_id=frame_windows[test_idx],
        test_patient_id=frame_patients[test_idx],
        y_test=y[test_idx].astype(int), proba=proba.astype(float),
        threshold_youden=float(thr.threshold), threshold_rule=thr.rule,
        inner_oof_y=inner_oof_y, inner_oof_p=inner_oof_p,
        selected_params=sel_params, selected_candidate_index=sel_idx,
    )


def _pool(folds: List[FoldRerun], *, cohort: str, feature_set: str) -> PooledOof:
    order = sorted(folds, key=lambda f: f.outer_fold)
    y = np.concatenate([f.y_test for f in order])
    p = np.concatenate([f.proba for f in order])
    wid = np.concatenate([f.test_window_id for f in order])
    pid = np.concatenate([f.test_patient_id for f in order])
    of = np.concatenate([np.full(f.y_test.size, f.outer_fold) for f in order])
    thr = np.concatenate([np.full(f.y_test.size, f.threshold_youden) for f in order])
    model = order[0].model
    return PooledOof(cohort=cohort, feature_set=feature_set, model=model,
                     model_role=order[0].model_role, y=y, prob=p, window_id=wid,
                     patient_id=pid, outer_fold=of, threshold=thr)


# ---------------------------------------------------------------------------
# public engine
# ---------------------------------------------------------------------------

def run_frozen_nested_cv(
    *, cfg, mc, rin, log=print,
) -> RerunBundle:
    """Re-run the frozen core (Stage 7) and paired (Stage 8) nested CV."""
    bundle = RerunBundle()
    specs = build_model_specs()

    # ---- core (Stage 7: 50-patient main cohort) -----------------------------
    core_gate = rin.core_gate
    core_frame: ModelFrame = assemble_model_frame(
        cohort_windows=core_gate.cohort_windows, outer_folds=core_gate.outer_folds,
        hr_features_path=cfg.path("features_physiology") / "hr_window_features.parquet",
        spo2_features_path=cfg.path("features_physiology") / "spo2_window_features.parquet",
    )
    core_adapter = build_fold_adapter(
        frame=core_frame, inner_folds=core_gate.inner_folds,
        outer_n_folds=mc.outer_n_folds, inner_n_folds=mc.inner_n_folds,
    )
    Xc, yc = core_frame.X, core_frame.y
    wc, pc = core_frame.window_id, core_frame.patient_id
    for model_name in MS.MODEL_FAMILIES:
        spec = specs[model_name]
        folds: List[FoldRerun] = []
        for k in range(mc.outer_n_folds):
            fr = _run_one_fold(spec=spec, X=Xc, y=yc, adapter=core_adapter,
                               outer_fold=k, inner_n_folds=mc.inner_n_folds,
                               frame_windows=wc, frame_patients=pc)
            fr.cohort = S.COHORT_CORE
            fr.feature_set = S.FEATURE_SET_CORE
            folds.append(fr)
            bundle.fold_reRuns.append(fr)
        pooled = _pool(folds, cohort=S.COHORT_CORE, feature_set=S.FEATURE_SET_CORE)
        bundle.pooled[pooled.key()] = pooled
    log(f"[rerun] core: {len(MS.MODEL_FAMILIES)} models x {mc.outer_n_folds} folds done")

    # ---- paired (Stage 8: 34-patient airflow sub-cohort) --------------------
    air_gate = rin.airflow_gate
    core_pair, enh_pair = assemble_paired_frames(
        cohort_windows=air_gate.airflow_cohort_windows,
        outer_folds=air_gate.outer_folds,
        hr_features_path=cfg.path("features_physiology") / "hr_window_features.parquet",
        spo2_features_path=cfg.path("features_physiology") / "spo2_window_features.parquet",
        airflow_features_path=cfg.path("features_physiology") / "airflow_window_features.parquet",
    )
    pair_adapter = build_fold_adapter(
        frame=core_pair, inner_folds=air_gate.inner_folds,
        outer_n_folds=mc.outer_n_folds, inner_n_folds=mc.inner_n_folds,
    )
    for feature_set, frame in ((S.FEATURE_SET_CORE, core_pair),
                               (S.FEATURE_SET_ENHANCED, enh_pair)):
        Xp, yp = frame.X, frame.y
        wp, pp = frame.window_id, frame.patient_id
        for model_name in AS.MODEL_FAMILIES:
            spec = specs[model_name]
            folds = []
            for k in range(mc.outer_n_folds):
                fr = _run_one_fold(spec=spec, X=Xp, y=yp, adapter=pair_adapter,
                                   outer_fold=k, inner_n_folds=mc.inner_n_folds,
                                   frame_windows=wp, frame_patients=pp)
                fr.cohort = S.COHORT_AIRFLOW
                fr.feature_set = feature_set
                folds.append(fr)
                bundle.fold_reRuns.append(fr)
            pooled = _pool(folds, cohort=S.COHORT_AIRFLOW, feature_set=feature_set)
            bundle.pooled[pooled.key()] = pooled
    log(f"[rerun] airflow: 2 feature sets x {len(AS.MODEL_FAMILIES)} models "
        f"x {mc.outer_n_folds} folds done")

    # attach in-memory frame/adapter handles for downstream re-runs (never
    # serialized): class-weight restriction + alternative label both need them.
    bundle.core_frame = core_frame
    bundle.core_adapter = core_adapter
    bundle.airflow_core_frame = core_pair
    bundle.airflow_enh_frame = enh_pair
    bundle.airflow_adapter = pair_adapter
    bundle.outer_n_folds = mc.outer_n_folds
    bundle.inner_n_folds = mc.inner_n_folds
    return bundle


__all__ = [
    "FoldRerun", "PooledOof", "RerunBundle", "run_frozen_nested_cv",
]
