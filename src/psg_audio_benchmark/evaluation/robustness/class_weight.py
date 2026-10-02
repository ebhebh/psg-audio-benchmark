"""Stage-9 2.5 -- logistic-regression class-weight sensitivity.

The primary LR grid spans ``C ∈ {0.1,1,10}`` × ``class_weight ∈ {None,"balanced"}``
and the inner-CV may legitimately pick either. This sensitivity *restricts* the
candidate grid to one ``class_weight`` value at a time and **re-selects per inner
CV** within that restriction, so the contribution of the class-weight axis can be
read off the outer-test OOF. Three variants are emitted side by side:

* ``cw_none``            -- ``class_weight=None`` only,
* ``cw_balanced``        -- ``class_weight="balanced"`` only,
* ``primary_unrestricted`` -- the frozen / primary LR (the reference; ``abs_diff=0``).

The threshold rule stays inner-OOF Youden J, fit on each variant's own inner OOF.
This is LR-only; it never re-selects the model family and never reads outer-test
labels for fitting.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ...modeling import schema as MS
from ...modeling.pipelines import Candidate, ModelSpec
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import RerunBundle, _pool, _run_one_fold
from .reproduction import _pooled_with_per_window_threshold

_METRICS_TF = ("auroc", "auprc", "brier", "log_loss")
_METRICS_TB = ("sensitivity", "specificity", "precision", "f1", "balanced_accuracy")


def _restricted_lr_spec(*, class_weight: str) -> ModelSpec:
    """LR spec whose candidate grid is restricted to a single class_weight."""
    cw_val = None if class_weight == S.CW_NONE else "balanced"
    candidates = tuple(
        Candidate(i, {"logisticregression__C": c,
                      "logisticregression__class_weight": cw_val})
        for i, c in enumerate((0.1, 1.0, 10.0))
    )
    return ModelSpec(
        name=MS.MODEL_LOGISTIC_REGRESSION,
        role=MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION],
        tunable=True, needs_sample_weight=False,
        threshold_rule=MS.THRESHOLD_RULE_DATA_DRIVEN,
        step_names=("imputer", "scaler", "logisticregression"),
        candidates=candidates, base_params={},
    )


def _run_variant(*, spec: ModelSpec, bundle: RerunBundle, cohort: str,
                 feature_set: str) -> Dict[str, object]:
    frame, adapter = bundle.frame_and_adapter(cohort=cohort, feature_set=feature_set)
    folds = []
    for k in range(bundle.outer_n_folds):
        fr = _run_one_fold(spec=spec, X=frame.X, y=frame.y, adapter=adapter,
                           outer_fold=k, inner_n_folds=bundle.inner_n_folds,
                           frame_windows=frame.window_id, frame_patients=frame.patient_id)
        fr.cohort = cohort
        fr.feature_set = feature_set
        folds.append(fr)
    pooled = _pool(folds, cohort=cohort, feature_set=feature_set)
    metrics = _pooled_with_per_window_threshold(pooled.y, pooled.prob, pooled.threshold)
    metrics["model"] = MS.MODEL_LOGISTIC_REGRESSION
    metrics["model_role"] = MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION]
    return metrics


def _row(run_id, cohort, feature_set, model, role, kind, metric, variant,
         value, frozen_value, interpretable, reason):
    v = None if value is None else float(value)
    fv = None if frozen_value is None else float(frozen_value)
    diff = None if (v is None or fv is None) else abs(v - fv)
    return {
        "run_id": run_id, "analysis": S.A_CLASS_WEIGHT,
        "analysis_role": S.ROLE_SENSITIVITY, "cohort": cohort,
        "feature_set": feature_set, "model": model, "model_role": role,
        "metric_kind": kind, "metric": metric, "variant": variant,
        "outer_fold": "pooled", "value": v, "frozen_value": fv,
        "abs_diff": diff, "interpretable": bool(interpretable), "reason": reason,
    }


def class_weight_sensitivity(
    *, bundle: RerunBundle, run_id: str, mc: ResolvedRobustnessConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    for cohort, feature_set in targets:
        # primary (unrestricted) reference -- straight from the bundle
        primary = bundle.pooled_for(cohort=cohort, feature_set=feature_set,
                                    model=MS.MODEL_LOGISTIC_REGRESSION)
        primary_m = _pooled_with_per_window_threshold(
            primary.y, primary.prob, primary.threshold)
        variants = {
            S.CW_NONE: _restricted_lr_spec(class_weight=S.CW_NONE),
            S.CW_BALANCED: _restricted_lr_spec(class_weight=S.CW_BALANCED),
        }
        results: Dict[str, Dict[str, object]] = {
            "primary_unrestricted": primary_m,
        }
        for vname, spec in variants.items():
            results[vname] = _run_variant(spec=spec, bundle=bundle,
                                          cohort=cohort, feature_set=feature_set)
        ref = results["primary_unrestricted"]
        for vname, m in results.items():
            for metric in _METRICS_TF:
                rows.append(_row(run_id, cohort, feature_set,
                                 MS.MODEL_LOGISTIC_REGRESSION,
                                 MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION],
                                 "threshold_free", metric, vname, m.get(metric),
                                 ref.get(metric), True, "ok"))
            for metric in _METRICS_TB:
                rows.append(_row(run_id, cohort, feature_set,
                                 MS.MODEL_LOGISTIC_REGRESSION,
                                 MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION],
                                 "threshold_based", metric, vname, m.get(metric),
                                 ref.get(metric), True, "ok"))
    return pd.DataFrame(rows, columns=list(S.METRIC_ROW_COLUMNS))


__all__ = ["class_weight_sensitivity"]
