"""Stage-9 2.1 -- primary reproduction from the FROZEN out-of-fold predictions.

Two complementary checks, neither of which re-selects a model or threshold:

1. **Frozen-OOF recomputation (authoritative).** Recompute AUROC / AUPRC / Brier /
   log loss and the per-window-threshold confusion-derived metrics directly from
   the frozen Stage 7 / Stage 8 OOF tables and compare to the frozen pooled
   reports. This is a pure function of the stored OOF, so it is deterministic.
2. **Re-run determinism.** Compare the regenerated outer-test OOF (from
   :mod:`rerun`) to the frozen OOF, window-for-window, reporting the max abs
   probability / threshold drift (LR near-exact; HGB within a tight tolerance).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ...modeling import schema as MS
from ...modeling.metrics import compute_metrics
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import RerunBundle
from .signature import RobustnessInput

_METRICS_THRESHOLD_FREE = ("auroc", "auprc", "brier", "log_loss")
_METRICS_THRESHOLD_BASED = ("sensitivity", "specificity", "precision", "f1",
                            "balanced_accuracy")


def _pooled_with_per_window_threshold(
    y: np.ndarray, p: np.ndarray, thr_per_window: np.ndarray,
) -> Dict[str, Any]:
    """Mirror modeling.runner pooled aggregation (threshold-free + per-window thr)."""
    base = compute_metrics(y_true=y, y_prob=p, threshold=0.5)
    y_pred = (p >= thr_per_window).astype(int)
    tp = int(np.sum((y_pred == 1) & (y == 1)))
    fp = int(np.sum((y_pred == 1) & (y == 0)))
    tn = int(np.sum((y_pred == 0) & (y == 0)))
    fn = int(np.sum((y_pred == 0) & (y == 1)))
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    prec = tp / (tp + fp) if (tp + fp) else None
    f1 = (2 * prec * sens / (prec + sens)) if (
        prec is not None and sens is not None and (prec + sens) > 0) else None
    bal = (sens + spec) / 2.0 if (sens is not None and spec is not None) else None
    out = dict(base)
    out["threshold"] = "per_window_inner_oof_youden"
    out["confusion_matrix"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    out["sensitivity"] = sens
    out["specificity"] = spec
    out["precision"] = prec
    out["f1"] = f1
    out["balanced_accuracy"] = bal
    return out


def _rows_for_block(
    *, run_id: str, cohort: str, feature_set: str, label: str,
    recomputed: Dict[str, Any], frozen: Dict[str, Any], tol: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metric in _METRICS_THRESHOLD_FREE + _METRICS_THRESHOLD_BASED:
        rv = recomputed.get(metric)
        fv = frozen.get(metric)
        rv_f = None if rv is None else float(rv)
        fv_f = None if fv is None else float(fv)
        if rv_f is None or fv_f is None:
            diff = None
            match = (rv_f is None and fv_f is None)
        else:
            diff = abs(rv_f - fv_f)
            match = bool(diff <= tol)
        kind = ("threshold_free" if metric in _METRICS_THRESHOLD_FREE
                else "threshold_based")
        rows.append({
            "run_id": run_id, "analysis": S.A_PRIMARY_REPRODUCTION,
            "analysis_role": S.ROLE_PRIMARY_REPLICATION, "cohort": cohort,
            "feature_set": feature_set, "model": recomputed.get("model", label),
            "model_role": recomputed.get("model_role", ""), "metric_kind": kind,
            "metric": metric, "variant": "frozen_oof_recompute", "outer_fold": "pooled",
            "value": rv_f, "frozen_value": fv_f, "abs_diff": diff,
            "interpretable": True, "reason": "ok" if match else "mismatch",
        })
    return rows


def reproduce_from_frozen(
    *, run_id: str, rin: RobustnessInput, mc: ResolvedRobustnessConfig,
) -> pd.DataFrame:
    """Recompute pooled metrics from the frozen OOF; compare to frozen reports."""
    tol = float(mc.primary_reproduction.get("pooled_metric_abs_tol", 1e-12))
    fr = rin.frozen
    rows: List[Dict[str, Any]] = []

    # ---- Stage 7 (core, 3 models) ------------------------------------------
    role_by_model = dict(zip(fr.stage7_oof["model"], fr.stage7_oof["model_role"]))
    for model_name in MS.MODEL_FAMILIES:
        sub = fr.stage7_oof[fr.stage7_oof["model"] == model_name]
        y = sub["y_true"].to_numpy(int)
        p = sub["y_prob"].to_numpy(float)
        thr = sub["threshold"].to_numpy(float)
        recomputed = _pooled_with_per_window_threshold(y, p, thr)
        recomputed["model"] = model_name
        recomputed["model_role"] = MS.MODEL_ROLE[model_name]
        frozen = fr.stage7_pooled[model_name]
        rows.extend(_rows_for_block(
            run_id=run_id, cohort=S.COHORT_CORE, feature_set=S.FEATURE_SET_CORE,
            label=model_name, recomputed=recomputed, frozen=frozen, tol=tol))

    # ---- Stage 8 (paired, 2 feature sets x 2 models) -----------------------
    for feature_set in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED):
        for model_name in MS.MODEL_FAMILIES:
            prob_col = f"{model_name}__{feature_set}__prob"
            thr_col = f"{model_name}__{feature_set}__threshold"
            if prob_col not in fr.stage8_paired_oof.columns:
                continue
            y = fr.stage8_paired_oof["y_true"].to_numpy(int)
            p = fr.stage8_paired_oof[prob_col].to_numpy(float)
            thr = fr.stage8_paired_oof[thr_col].to_numpy(float)
            recomputed = _pooled_with_per_window_threshold(y, p, thr)
            recomputed["model"] = model_name
            recomputed["model_role"] = MS.MODEL_ROLE[model_name]
            key = f"{feature_set}__{model_name}"
            frozen = fr.stage8_pooled[key]
            rows.extend(_rows_for_block(
                run_id=run_id, cohort=S.COHORT_AIRFLOW, feature_set=feature_set,
                label=key, recomputed=recomputed, frozen=frozen, tol=tol))

    return pd.DataFrame(rows, columns=list(S.METRIC_ROW_COLUMNS))


def _align_max_absdiff(
    *, frozen_by_wid: pd.DataFrame, rerun_prob: np.ndarray, rerun_thr: np.ndarray,
    rerun_wid: np.ndarray, prob_col: str, thr_col: str,
) -> Dict[str, float]:
    order = frozen_by_wid.set_index("window_id")
    rp = pd.Series(rerun_prob, index=rerun_wid)
    rt = pd.Series(rerun_thr, index=rerun_wid)
    common = order.index.intersection(rp.index)
    if len(common) == 0:
        return {"prob_max_abs_diff": float("nan"), "threshold_max_abs_diff": float("nan"),
                "n_compared": 0}
    fp = order.loc[common, prob_col].to_numpy(float)
    ft = order.loc[common, thr_col].to_numpy(float)
    rp_c = rp.loc[common].to_numpy(float)
    rt_c = rt.loc[common].to_numpy(float)
    return {
        "prob_max_abs_diff": float(np.max(np.abs(fp - rp_c))),
        "threshold_max_abs_diff": float(np.max(np.abs(ft - rt_c))),
        "n_compared": int(len(common)),
    }


def compare_rerun_to_frozen(
    *, bundle: RerunBundle, rin: RobustnessInput, mc: ResolvedRobustnessConfig,
) -> Dict[str, Any]:
    """Compare regenerated outer-test OOF to the frozen OOF (determinism check)."""
    fr = rin.frozen
    tol_lr = float(mc.primary_reproduction.get("rerun_oof_abs_tol_lr", 1e-9))
    tol_hgb = float(mc.primary_reproduction.get("rerun_oof_abs_tol_hgb", 1e-6))
    s7_frozen = fr.stage7_oof[["window_id", "y_prob", "threshold"]].copy()

    out: Dict[str, Any] = {"core": {}, "airflow": {}, "summary": {}}
    max_drift = 0.0

    # core (Stage 7)
    for model_name in MS.MODEL_FAMILIES:
        pooled = bundle.pooled_for(cohort=S.COHORT_CORE,
                                   feature_set=S.FEATURE_SET_CORE, model=model_name)
        sub = fr.stage7_oof[fr.stage7_oof["model"] == model_name]
        d = _align_max_absdiff(frozen_by_wid=sub[["window_id", "y_prob", "threshold"]],
                               rerun_prob=pooled.prob, rerun_thr=pooled.threshold,
                               rerun_wid=pooled.window_id,
                               prob_col="y_prob", thr_col="threshold")
        tol = tol_lr if model_name == MS.MODEL_LOGISTIC_REGRESSION else tol_hgb
        d["within_tolerance"] = bool(d["prob_max_abs_diff"] <= tol)
        d["tolerance"] = tol
        out["core"][model_name] = d
        max_drift = max(max_drift, d["prob_max_abs_diff"])

    # paired (Stage 8)
    for feature_set in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED):
        out["airflow"][feature_set] = {}
        for model_name in MS.MODEL_FAMILIES:
            prob_col = f"{model_name}__{feature_set}__prob"
            thr_col = f"{model_name}__{feature_set}__threshold"
            if prob_col not in fr.stage8_paired_oof.columns:
                continue  # Stage 8 stores only its own model families (LR + HGB)
            pooled = bundle.pooled_for(cohort=S.COHORT_AIRFLOW,
                                       feature_set=feature_set, model=model_name)
            d = _align_max_absdiff(
                frozen_by_wid=fr.stage8_paired_oof[["window_id", prob_col, thr_col]],
                rerun_prob=pooled.prob, rerun_thr=pooled.threshold,
                rerun_wid=pooled.window_id, prob_col=prob_col, thr_col=thr_col)
            tol = tol_lr if model_name == MS.MODEL_LOGISTIC_REGRESSION else tol_hgb
            d["within_tolerance"] = bool(d["prob_max_abs_diff"] <= tol)
            d["tolerance"] = tol
            out["airflow"][feature_set][model_name] = d
            max_drift = max(max_drift, d["prob_max_abs_diff"])

    out["summary"] = {
        "max_prob_abs_diff_overall": float(max_drift),
        "lr_tolerance": tol_lr,
        "hgb_tolerance": tol_hgb,
        "all_within_tolerance": all(
            d["within_tolerance"] for d in out["core"].values())
        and all(d["within_tolerance"]
                for fs in out["airflow"].values() for d in fs.values()),
    }
    return out


__all__ = ["reproduce_from_frozen", "compare_rerun_to_frozen"]
