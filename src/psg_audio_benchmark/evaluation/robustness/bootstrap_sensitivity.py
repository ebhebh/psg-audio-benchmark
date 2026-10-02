"""Stage-9 2.2 -- patient-cluster uncertainty + patient-macro descriptives.

Three pre-registered uncertainty artefacts, all resampling **patients** (the
independent cluster unit) and bringing every window of each drawn patient:

1. **patient-cluster bootstrap** AUROC/AUPRC CIs for every (cohort, feature_set,
   model), compared to the frozen Stage 7 / Stage 8 CIs where they exist.
2. **paired increment bootstrap** (airflow_enhanced - core_restricted) for the
   primary LR, compared to the frozen Stage 8 paired CI.
3. **per-fold paired delta** (enhanced - core, primary LR) + **patient-macro
   descriptives** (per-patient window counts / class balance) so a reader can see
   the cluster structure behind the uncertainty.

Window-IID resampling is structurally impossible here. No threshold or model is
re-selected; outer-test labels never enter any fit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ...modeling import schema as MS
from ...modeling.airflow_increment.paired_bootstrap import (
    paired_patient_cluster_bootstrap,
)
from ...modeling.bootstrap import patient_cluster_bootstrap
from ...modeling.metrics import safe_auroc
from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import FoldRerun, RerunBundle
from .signature import RobustnessInput

_METRICS_PC = ("auroc", "auprc")
_PAIRED_METRICS = ("auroc", "auprc", "brier", "sensitivity", "specificity",
                   "f1", "balanced_accuracy")
_FOLD_DELTA_METRICS = ("auroc", "auprc", "brier")


def _cfg(mc: ResolvedRobustnessConfig) -> Dict[str, Any]:
    p = mc.patient_weight_uncertainty
    return {
        "n": int(p.get("n_resamples", S.BOOTSTRAP_DEFAULT_N)),
        "seed": int(p.get("seed", S.BOOTSTRAP_DEFAULT_SEED)),
        "ci_level": float(p.get("ci_level", S.BOOTSTRAP_CI_LEVEL)),
    }


# ---------------------------------------------------------------------------
# patient-cluster bootstrap
# ---------------------------------------------------------------------------

def _s7_boot_index(rin: RobustnessInput) -> Dict[tuple, Dict[str, Any]]:
    df = rin.frozen.stage7_bootstrap
    return {(r["model"], r["metric"]): r for _, r in df.iterrows()}


def _patient_cluster_block(
    *, bundle: RerunBundle, rin: RobustnessInput, run_id: str, bcfg: Dict[str, Any],
) -> pd.DataFrame:
    s7 = _s7_boot_index(rin)
    rows: List[Dict[str, Any]] = []
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    for cohort, feature_set in targets:
        for model_name in MS.MODEL_FAMILIES:
            key = (cohort, feature_set, model_name)
            if key not in bundle.pooled:
                continue  # airflow cohort has only the Stage-8 model families
            pooled = bundle.pooled[key]
            cis = patient_cluster_bootstrap(
                y_true=pooled.y, y_prob=pooled.prob, patient_ids=pooled.patient_id,
                metrics=_METRICS_PC, n_resamples=bcfg["n"], seed=bcfg["seed"],
                ci_level=bcfg["ci_level"])
            for ci in cis:
                frozen = None
                if cohort == S.COHORT_CORE:
                    frozen = s7.get((model_name, ci.metric))
                rows.append(_pc_row(run_id, cohort, feature_set, model_name, ci, frozen))
    return pd.DataFrame(rows)


def _pc_row(run_id, cohort, feature_set, model_name, ci, frozen) -> Dict[str, Any]:
    fp = float(frozen["point_estimate"]) if frozen is not None else None
    fl = float(frozen["ci_low"]) if frozen is not None else None
    fh = float(frozen["ci_high"]) if frozen is not None else None
    pe = ci.point_estimate
    diff = None if (pe is None or fp is None) else abs(float(pe) - fp)
    return {
        "run_id": run_id, "analysis": S.A_PATIENT_WEIGHT,
        "analysis_role": S.ROLE_SENSITIVITY, "cohort": cohort,
        "feature_set": feature_set, "model": model_name,
        "model_role": MS.MODEL_ROLE[model_name], "block": "patient_cluster",
        "metric": ci.metric, "kind": "threshold_free",
        "direction": "na", "point_estimate": pe, "ci_low": ci.ci_low,
        "ci_high": ci.ci_high, "n_resamples": ci.n_resamples,
        "n_used": ci.n_used, "n_single_class_skipped": ci.n_single_class_skipped,
        "ci_level": ci.ci_level, "unit": ci.unit,
        "frozen_point_estimate": fp, "frozen_ci_low": fl, "frozen_ci_high": fh,
        "point_abs_diff": diff,
    }


# ---------------------------------------------------------------------------
# paired increment bootstrap (airflow sub-cohort, primary LR)
# ---------------------------------------------------------------------------

def _s8_paired_index(rin: RobustnessInput) -> Dict[tuple, Dict[str, Any]]:
    df = rin.frozen.stage8_bootstrap
    return {(r["model"], r["metric"]): r for _, r in df.iterrows()}


def _paired_block(
    *, bundle: RerunBundle, rin: RobustnessInput, run_id: str, bcfg: Dict[str, Any],
) -> pd.DataFrame:
    core = bundle.pooled_for(cohort=S.COHORT_AIRFLOW, feature_set=S.FEATURE_SET_CORE,
                             model=MS.MODEL_LOGISTIC_REGRESSION)
    enh = bundle.pooled_for(cohort=S.COHORT_AIRFLOW, feature_set=S.FEATURE_SET_ENHANCED,
                            model=MS.MODEL_LOGISTIC_REGRESSION)
    if not np.array_equal(core.window_id, enh.window_id):
        raise RuntimeError("paired bootstrap: core/enhanced window order mismatch")
    cis = paired_patient_cluster_bootstrap(
        y_true=core.y, prob_core=core.prob, prob_enhanced=enh.prob,
        patient_ids=core.patient_id, threshold_core=core.threshold,
        threshold_enhanced=enh.threshold, metrics=_PAIRED_METRICS,
        n_resamples=bcfg["n"], seed=bcfg["seed"], ci_level=bcfg["ci_level"])
    s8 = _s8_paired_index(rin)
    rows: List[Dict[str, Any]] = []
    for ci in cis:
        frozen = s8.get((MS.MODEL_LOGISTIC_REGRESSION, ci.metric))
        fp = float(frozen["delta_point_estimate"]) if frozen is not None else None
        fl = float(frozen["ci_low"]) if frozen is not None else None
        fh = float(frozen["ci_high"]) if frozen is not None else None
        pe = ci.delta_point_estimate
        diff = None if (pe is None or fp is None) else abs(float(pe) - fp)
        rows.append({
            "run_id": run_id, "analysis": S.A_PATIENT_WEIGHT,
            "analysis_role": S.ROLE_SENSITIVITY, "cohort": S.COHORT_AIRFLOW,
            "feature_set": S.FEATURE_SET_ENHANCED,
            "model": MS.MODEL_LOGISTIC_REGRESSION,
            "model_role": MS.MODEL_ROLE[MS.MODEL_LOGISTIC_REGRESSION],
            "block": "paired_increment", "metric": ci.metric, "kind": ci.kind,
            "direction": ci.direction, "point_estimate": pe,
            "ci_low": ci.ci_low, "ci_high": ci.ci_high,
            "n_resamples": ci.n_resamples, "n_used": ci.n_used,
            "n_single_class_skipped": ci.n_single_class_skipped,
            "ci_level": ci.ci_level, "unit": ci.unit,
            "frozen_point_estimate": fp, "frozen_ci_low": fl,
            "frozen_ci_high": fh, "point_abs_diff": diff,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# per-fold paired delta (airflow sub-cohort, primary LR)
# ---------------------------------------------------------------------------

def _fold_delta_block(
    *, bundle: RerunBundle, rin: RobustnessInput, run_id: str,
) -> pd.DataFrame:
    core_folds = bundle.folds(cohort=S.COHORT_AIRFLOW, feature_set=S.FEATURE_SET_CORE,
                              model=MS.MODEL_LOGISTIC_REGRESSION)
    enh_folds = bundle.folds(cohort=S.COHORT_AIRFLOW, feature_set=S.FEATURE_SET_ENHANCED,
                             model=MS.MODEL_LOGISTIC_REGRESSION)
    core_by = {f.outer_fold: f for f in core_folds}
    enh_by = {f.outer_fold: f for f in enh_folds}
    # frozen per-fold pivot (defensive: column set may vary)
    frozen_pivot = _s8_fold_pivot(rin)
    rows: List[Dict[str, Any]] = []
    for k in sorted(core_by.keys()):
        cf = core_by[k]; ef = enh_by[k]
        for metric in _FOLD_DELTA_METRICS:
            cv = _fold_metric(cf, metric)
            ev = _fold_metric(ef, metric)
            delta = None if (cv is None or ev is None) else float(ev) - float(cv)
            fr = frozen_pivot.get((k, metric))
            fcv = fr.get("core") if fr else None
            fev = fr.get("enhanced") if fr else None
            fdelta = None if (fcv is None or fev is None) else float(fev) - float(fcv)
            rows.append({
                "run_id": run_id, "analysis": S.A_PATIENT_WEIGHT,
                "analysis_role": S.ROLE_SENSITIVITY, "cohort": S.COHORT_AIRFLOW,
                "model": MS.MODEL_LOGISTIC_REGRESSION, "outer_fold": k,
                "metric": metric, "core_value": cv, "enhanced_value": ev,
                "delta": delta, "frozen_core_value": fcv,
                "frozen_enhanced_value": fev, "frozen_delta": fdelta,
            })
    return pd.DataFrame(rows)


def _fold_metric(f: FoldRerun, metric: str) -> Optional[float]:
    y, p = f.y_test, f.proba
    if metric in ("auroc", "auprc"):
        from ...modeling.metrics import safe_auroc, safe_auprc
        fn = safe_auroc if metric == "auroc" else safe_auprc
        v, _ = fn(y, p); return v
    if metric == "brier":
        from ...modeling.metrics import safe_brier
        return safe_brier(y, p)[0]
    return None


def _s8_fold_pivot(rin: RobustnessInput) -> Dict[tuple, Dict[str, Any]]:
    df = rin.frozen.stage8_fold_metrics
    lr = df[df["model"] == MS.MODEL_LOGISTIC_REGRESSION]
    out: Dict[tuple, Dict[str, Any]] = {}
    for _, r in lr.iterrows():
        fs = r["feature_set"]
        for metric in _FOLD_DELTA_METRICS:
            if metric not in r or pd.isna(r.get(metric)):
                continue
            key = (int(r["outer_fold"]), metric)
            out.setdefault(key, {})[("enhanced" if fs == S.FEATURE_SET_ENHANCED
                                     else "core")] = float(r[metric])
    return out


# ---------------------------------------------------------------------------
# patient-macro descriptives
# ---------------------------------------------------------------------------

def _patient_macro_block(*, bundle: RerunBundle, run_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    targets: List[tuple] = (
        [(S.COHORT_CORE, S.FEATURE_SET_CORE)]
        + [(S.COHORT_AIRFLOW, fs) for fs in (S.FEATURE_SET_CORE, S.FEATURE_SET_ENHANCED)]
    )
    for cohort, feature_set in targets:
        for model_name in MS.MODEL_FAMILIES:
            key = (cohort, feature_set, model_name)
            if key not in bundle.pooled:
                continue
            pooled = bundle.pooled[key]
            df = pd.DataFrame({"patient_id": pooled.patient_id,
                               "y": pooled.y, "p": pooled.prob})
            for pid, g in df.groupby("patient_id", sort=True):
                n = int(len(g)); npos = int(g["y"].sum()); nneg = n - npos
                au, _ = safe_auroc(g["y"].to_numpy(), g["p"].to_numpy())
                rows.append({
                    "run_id": run_id, "analysis": S.A_PATIENT_WEIGHT,
                    "analysis_role": S.ROLE_SENSITIVITY, "cohort": cohort,
                    "feature_set": feature_set, "model": model_name,
                    "model_role": MS.MODEL_ROLE[model_name],
                    "patient_id": pid, "n_windows": n, "n_positive": npos,
                    "n_negative": nneg,
                    "positive_rate": (npos / n if n else None),
                    "mean_prob": float(g["p"].mean()),
                    "per_patient_auroc": au,
                })
    return pd.DataFrame(rows)


def bootstrap_sensitivity(
    *, bundle: RerunBundle, rin: RobustnessInput, run_id: str,
    mc: ResolvedRobustnessConfig,
) -> Dict[str, pd.DataFrame]:
    bcfg = _cfg(mc)
    pc = _patient_cluster_block(bundle=bundle, rin=rin, run_id=run_id, bcfg=bcfg)
    paired = _paired_block(bundle=bundle, rin=rin, run_id=run_id, bcfg=bcfg)
    bootstrap_ci = pd.concat([pc, paired], ignore_index=True)
    fold_delta = _fold_delta_block(bundle=bundle, rin=rin, run_id=run_id)
    patient_macro = _patient_macro_block(bundle=bundle, run_id=run_id)
    return {
        "bootstrap_ci": bootstrap_ci,
        "patient_macro": patient_macro,
        "paired_fold_delta": fold_delta,
    }


__all__ = ["bootstrap_sensitivity"]
