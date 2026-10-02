"""Stage-9 3.0 -- alternative-label branch (structured event-interval overlap).

A pre-registered *alternative* binary label is derived from the structured scored
event intervals, on the SAME window membership as the primary label (same frozen
v2 split, same features, same folds -- only ``y`` changes):

* a window is **positive** if any **retained** scored respiratory event (non-awake)
  interval overlaps the window by ``>= min_event_overlap_seconds`` seconds,
* the awake set-aside is mirrored (awake-overlapping events are excluded),
* eligibility (core coverage complete, no awake overlap) is identical to primary.

The two tunable models (LR primary, HGB exploratory) plus the dummy baseline are
re-run on this alternative label; results are reported as a sensitivity and
**never** used to select a "better" main label. The branch is flagged
``not_interpretable`` when the alternative label is degenerate (single class) in
the pooled OOF or unstable across folds.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...config import Config
from ...modeling import schema as MS
from ...modeling.data_gate import _make_window_id
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import RerunBundle, _pool, _run_one_fold
from .reproduction import _pooled_with_per_window_threshold
from .signature import RobustnessInput

_METRICS_TF = ("auroc", "auprc", "brier", "log_loss")
_METRICS_TB = ("sensitivity", "specificity", "precision", "f1", "balanced_accuracy")


def _alt_labels(
    *, cfg: Config, window_id: np.ndarray, patient_id: np.ndarray,
    min_overlap: float,
) -> np.ndarray:
    """Alternative 0/1 label per window (retained event overlap >= min_overlap s)."""
    idx = pd.read_parquet(cfg.path("data_manifests") / "csv_window_index.parquet")
    idx["window_id"] = [_make_window_id(p, i)
                        for p, i in zip(idx["patient_id"], idx["window_index"])]
    bounds = idx.set_index("window_id")[
        ["start_relative_to_record_start", "end_relative_to_record_start"]]
    ev = pd.read_parquet(cfg.path("annotations") / "parsed_events.parquet")
    retained = ev[(ev.get("is_any_scored_respiratory_event", False) == True)
                  & (ev.get("overlaps_awake_interval", True) == False)]
    by_patient: Dict[str, np.ndarray] = {}
    for pid, g in retained.groupby("patient_id"):
        by_patient[str(pid)] = g[["event_start_relative_to_record_start",
                                  "event_end_relative_to_record_start"]].to_numpy(float)
    out = np.zeros(len(window_id), dtype=int)
    for i, (wid, pid) in enumerate(zip(window_id, patient_id)):
        if wid not in bounds.index:
            continue
        ws = float(bounds.loc[wid, "start_relative_to_record_start"])
        we = float(bounds.loc[wid, "end_relative_to_record_start"])
        evs = by_patient.get(str(pid))
        if evs is None or len(evs) == 0:
            continue
        es = evs[:, 0]; ee = evs[:, 1]
        overlap = np.minimum(ee, we) - np.maximum(es, ws)
        if np.any(overlap >= min_overlap):
            out[i] = 1
    return out


def _row(run_id, cohort, feature_set, model, role, kind, metric, value,
         frozen_value, interpretable, reason):
    v = None if value is None else float(value)
    fv = None if frozen_value is None else float(frozen_value)
    diff = None if (v is None or fv is None) else abs(v - fv)
    return {
        "run_id": run_id, "analysis": S.A_ALTERNATIVE_LABEL,
        "analysis_role": S.ROLE_EXPLORATORY, "cohort": cohort,
        "feature_set": feature_set, "model": model, "model_role": role,
        "metric_kind": kind, "metric": metric, "variant": "alternative_label",
        "outer_fold": "pooled", "value": v, "frozen_value": fv,
        "abs_diff": diff, "interpretable": bool(interpretable), "reason": reason,
    }


def alternative_label(
    *, cfg: Config, bundle: RerunBundle, rin: RobustnessInput, run_id: str,
    mc: ResolvedRobustnessConfig,
) -> Dict[str, pd.DataFrame]:
    min_overlap = float(mc.alternative_label.get("min_event_overlap_seconds", 1.0))
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    rows: List[Dict[str, Any]] = []
    prev_rows: List[Dict[str, Any]] = []
    for cohort, feature_set in targets:
        frame, adapter = bundle.frame_and_adapter(cohort=cohort, feature_set=feature_set)
        wid = np.asarray(frame.window_id)
        pid = np.asarray(frame.patient_id)
        alt_y = _alt_labels(cfg=cfg, window_id=wid, patient_id=pid,
                            min_overlap=min_overlap)
        prim_y = np.asarray(frame.y).astype(int)
        n = int(len(alt_y))
        n_pos_alt = int(alt_y.sum()); n_pos_prim = int(prim_y.sum())
        pooled_single = (n_pos_alt == 0 or n_pos_alt == n)
        # how many outer-test folds are single-class under the alt label
        n_single_folds = 0
        for k in range(bundle.outer_n_folds):
            test_idx = np.nonzero(adapter.outer_test_mask(k))[0]
            yk = alt_y[test_idx]
            if len(np.unique(yk)) < 2:
                n_single_folds += 1
        interpretable = (not pooled_single) and (n_single_folds < bundle.outer_n_folds)
        reason = ("ok" if interpretable else
                  f"unstable_pooled_single_class={pooled_single}_single_folds={n_single_folds}")
        # prevalence rows (label-prevelence sensitivity; frozen = primary prevalence)
        prim_prev = (n_pos_prim / n) if n else None
        alt_prev = (n_pos_alt / n) if n else None
        for label, val, ref in (("primary_prevalence", prim_prev, prim_prev),
                                ("alternative_prevalence", alt_prev, prim_prev)):
            rows.append(_row(run_id, cohort, feature_set, "label", "label",
                             "label_prevalence", label, val, ref,
                             True, "ok"))
        prev_rows.append({"run_id": run_id, "cohort": cohort,
                          "feature_set": feature_set, "n_windows": n,
                          "primary_positive": n_pos_prim,
                          "alternative_positive": n_pos_alt,
                          "primary_prevalence": prim_prev,
                          "alternative_prevalence": alt_prev,
                          "n_outer_folds_single_class_alt": n_single_folds,
                          "interpretable": interpretable, "reason": reason})
        if not interpretable:
            continue  # model metrics NA; prevalence still reported
        for model_name in MS.MODEL_FAMILIES:
            spec_role = MS.MODEL_ROLE[model_name]
            from ...modeling import build_model_specs
            spec = build_model_specs()[model_name]
            folds = []
            for k in range(bundle.outer_n_folds):
                fr = _run_one_fold(spec=spec, X=frame.X, y=alt_y, adapter=adapter,
                                   outer_fold=k, inner_n_folds=bundle.inner_n_folds,
                                   frame_windows=wid, frame_patients=pid)
                fr.cohort = cohort; fr.feature_set = feature_set
                folds.append(fr)
            pooled = _pool(folds, cohort=cohort, feature_set=feature_set)
            m = _pooled_with_per_window_threshold(pooled.y, pooled.prob, pooled.threshold)
            # primary-label reference (the frozen main result for this model)
            try:
                prim = bundle.pooled_for(cohort=cohort, feature_set=feature_set,
                                         model=model_name)
                prim_m = _pooled_with_per_window_threshold(
                    prim.y, prim.prob, prim.threshold)
            except KeyError:
                prim_m = {}
            for metric in _METRICS_TF:
                rows.append(_row(run_id, cohort, feature_set, model_name, spec_role,
                                 "threshold_free", metric, m.get(metric),
                                 prim_m.get(metric), interpretable, reason))
            for metric in _METRICS_TB:
                rows.append(_row(run_id, cohort, feature_set, model_name, spec_role,
                                 "threshold_based", metric, m.get(metric),
                                 prim_m.get(metric), interpretable, reason))
    metrics = pd.DataFrame(rows, columns=list(S.METRIC_ROW_COLUMNS))
    prevalence = pd.DataFrame(prev_rows)
    return {"alt_label_metrics": metrics, "alt_label_prevalence": prevalence}


__all__ = ["alternative_label"]
