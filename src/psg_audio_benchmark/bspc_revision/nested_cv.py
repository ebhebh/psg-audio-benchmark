"""Stage 15b — patient-level nested cross-validation engine.

Runs the pre-registered model battery on the INHERITED v2 split (5 outer x 4 inner),
with strict fold-internal leakage control:

* outer-train / outer-test formed from the frozen patient->outer_fold assignment;
* inner selection sweeps the 4 frozen inner folds of the outer-train (outer-test never
  participates); selection = max mean average_precision, tie -> higher mean AUROC ->
  ascending grid index;
* refit on the full outer-train, predict outer-test ONCE -> one OOF probability per window;
* threshold = inner-OOF max-Youden-J (lowest-probability tie-break), never fit on outer-test;
  dummy uses a fixed 0.5.

Reusable pure functions are imported from the existing ``modeling`` package
(metrics, patient-cluster bootstrap). No Stage 2-14 product is read for modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..modeling.metrics import safe_auroc, safe_auprc  # noqa: F401  (re-exported helpers)

SEED = 20250714
LR_C_GRID = [0.1, 1.0, 10.0]
LR_CW_GRID = [None, "balanced"]
HGB_LR_GRID = [0.05, 0.10]
HGB_LEAF_GRID = [15, 31]
_AP_TOL = 1e-9


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str          # lower_bound | simple_baseline | comparator | primary | exploratory
    kind: str          # dummy | lr | hgb
    feature_family: str


def model_battery() -> List[ModelSpec]:
    return [
        ModelSpec("dummy_prior", "lower_bound", "dummy", "none"),
        ModelSpec("simple_spo2_min_lr", "simple_baseline", "lr", "simple_spo2_min"),
        ModelSpec("hr_only_lr", "comparator", "lr", "hr_only"),
        ModelSpec("spo2_only_lr", "comparator", "lr", "spo2_only"),
        ModelSpec("hr_spo2_lr", "primary", "lr", "hr_spo2"),
        ModelSpec("hr_spo2_hgb", "exploratory", "hgb", "hr_spo2"),
    ]


def _candidate_grid(kind: str) -> List[dict]:
    grid: List[dict] = []
    if kind == "lr":
        for c in LR_C_GRID:
            for cw in LR_CW_GRID:
                grid.append({"C": c, "class_weight": cw})
    elif kind == "hgb":
        for lr in HGB_LR_GRID:
            for leaf in HGB_LEAF_GRID:
                grid.append({"learning_rate": lr, "max_leaf_nodes": leaf})
    return grid


def _build_pipeline(spec: ModelSpec, cand: dict) -> Pipeline:
    if spec.kind == "dummy":
        return Pipeline([("dummy", DummyClassifier(strategy="prior"))])
    if spec.kind == "lr":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(C=cand["C"], class_weight=cand["class_weight"],
                                       solver="lbfgs", max_iter=2000, random_state=SEED)),
        ])
    if spec.kind == "hgb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                learning_rate=cand["learning_rate"], max_leaf_nodes=cand["max_leaf_nodes"],
                random_state=SEED, early_stopping=False, loss="log_loss")),
        ])
    raise ValueError(f"unknown kind {spec.kind}")


def _fit_predict(spec, cand, X, y, train_idx, pred_idx) -> np.ndarray:
    pipe = _build_pipeline(spec, cand)
    Xtr, ytr = X[train_idx], y[train_idx]
    if spec.kind == "hgb":
        classes, counts = np.unique(ytr, return_counts=True)
        sw = np.ones(len(ytr), dtype=float)
        if len(classes) == 2:
            w = len(ytr) / (2.0 * counts)
            mapping = {int(c): float(w[i]) for i, c in enumerate(classes)}
            sw = np.array([mapping[int(v)] for v in ytr])
        pipe.fit(Xtr, ytr, clf__sample_weight=sw)
    else:
        pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(X[pred_idx])
    pos = list(pipe.classes_).index(1) if 1 in list(pipe.classes_) else (1 if len(pipe.classes_) == 1 else 1)
    # DummyClassifier with single seen class still exposes classes_ with the seen class
    if proba.shape[1] == 2:
        return proba[:, 1].astype(float)
    # single-column case (only one class seen in train): positive prob = prior of class 1 if seen
    seen = list(pipe.classes_)
    return np.full(pred_idx.shape[0], float(1.0 if seen == [1] else 0.0))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray, *, is_dummy: bool) -> tuple[float, str]:
    if is_dummy:
        return 0.5, "dummy_prior fixed 0.5"
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    if len(np.unique(y)) < 2:
        return 0.5, f"single true class {np.unique(y).tolist()}; default 0.5"
    if np.unique(p).size <= 1:
        return 0.5, "constant inner-OOF probability; default 0.5"
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    best_j = j.max()
    tied = np.where(j >= best_j - 1e-12)[0]
    chosen = tied[int(np.argmin(thr[tied]))]
    return float(thr[chosen]), f"max Youden J={best_j:.6f} at threshold={thr[chosen]:.6f}"


@dataclass
class OOFResult:
    spec: ModelSpec
    oof: pd.DataFrame   # window_id, patient_id, outer_fold, y_true, y_prob, threshold, y_pred
    selected: List[dict]  # per outer fold selection record
    candidates: List[dict]  # per (outer fold, candidate) inner-CV scores


def run_nested_cv(
    *, spec: ModelSpec, frame: pd.DataFrame, X: np.ndarray, y: np.ndarray,
    patient_ids: np.ndarray, outer_fold: np.ndarray,
    inner_map: Dict[tuple, int],
) -> OOFResult:
    """Run nested CV for one model spec on one (label, lag) frame.

    ``inner_map`` maps ``(outer_fold, patient_id) -> inner_validation_fold`` for the frozen
    v2 inner assignment. For outer fold k, each outer-TRAIN patient's inner validation fold is
    looked up under the key ``(k, patient_id)`` (a patient is outer-train in the 4 folds it is
    NOT the test patient of, each with its own frozen inner fold). Outer-test patients never
    enter inner selection. All fitting is fold-internal.
    """
    is_dummy = spec.kind == "dummy"
    grid = [{}] if is_dummy else _candidate_grid(spec.kind)
    n_outer = int(outer_fold.max()) + 1
    oof_rows: List[dict] = []
    selected_records: List[dict] = []
    candidate_records: List[dict] = []

    for k in range(n_outer):
        test_mask = outer_fold == k
        train_mask = ~test_mask
        test_idx = np.nonzero(test_mask)[0]
        train_idx = np.nonzero(train_mask)[0]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        tr_pids = patient_ids[train_idx]
        tr_inner = np.array([inner_map.get((k, str(p)), -1) for p in tr_pids], dtype=int)

        # ---- inner selection (tunable only) on outer-train ----
        if is_dummy:
            chosen_cand = grid[0]
            inner_oof_y, inner_oof_p = None, None
        else:
            cand_scores: List[tuple] = []   # (mean_ap, mean_auroc, idx, cand, oof_y, oof_p)
            for ci, cand in enumerate(grid):
                per_ap, per_au = [], []
                oo_y, oo_p, oo_row = [], [], []
                for j in range(4):
                    va_local = np.nonzero(tr_inner == j)[0]
                    tr_local = np.nonzero(tr_inner != j)[0]
                    if tr_local.size == 0 or va_local.size == 0:
                        per_ap.append(None); per_au.append(None)
                        continue
                    va_idx = train_idx[va_local]
                    tr_idx = train_idx[tr_local]
                    proba = _fit_predict(spec, cand, X, y, tr_idx, va_idx)
                    ap, _ = safe_auprc(y[va_idx], proba)
                    au, _ = safe_auroc(y[va_idx], proba)
                    per_ap.append(ap); per_au.append(au)
                    oo_row.extend(int(i) for i in va_idx)
                    oo_y.extend(int(v) for v in y[va_idx])
                    oo_p.extend(float(v) for v in proba)
                keep_ap = [v for v in per_ap if v is not None]
                mean_ap = float(np.mean(keep_ap)) if keep_ap else None
                keep_au = [v for v in per_au if v is not None]
                mean_au = float(np.mean(keep_au)) if keep_au else None
                candidate_records.append({
                    "model": spec.name, "outer_fold": k, "candidate_index": ci,
                    "candidate": str(cand), "mean_average_precision": mean_ap,
                    "mean_auroc": mean_au, "n_computable_inner_folds": len(keep_ap),
                    "per_fold_ap": ";".join("NA" if v is None else f"{v:.6f}" for v in per_ap),
                })
                ap_key = mean_ap if mean_ap is not None else -np.inf
                au_key = mean_au if mean_au is not None else -np.inf
                cand_scores.append((ap_key, au_key, ci, cand, oo_row, oo_y, oo_p))
            # selection: max mean_ap (tol); tie -> higher mean_auroc -> ascending idx
            best_ap = max(c[0] for c in cand_scores)
            eligible = [c for c in cand_scores if c[0] >= best_ap - _AP_TOL]
            eligible.sort(key=lambda c: (-c[1], c[2]))
            chosen = eligible[0]
            chosen_cand = chosen[3]
            # inner-OOF predictions of the selected candidate -> threshold
            order = np.argsort(chosen[4])
            inner_oof_y = np.asarray(chosen[5])[order]
            inner_oof_p = np.asarray(chosen[6])[order]
            selected_records.append({
                "model": spec.name, "outer_fold": k, "selected_candidate": str(chosen_cand),
                "selected_mean_ap": float(chosen[0]) if chosen[0] != -np.inf else None,
                "selected_mean_auroc": float(chosen[1]) if chosen[1] != -np.inf else None,
            })

        # ---- threshold (inner-OOF Youden; dummy 0.5) ----
        if is_dummy:
            threshold = 0.5
            thr_reason = "dummy_prior fixed 0.5"
        else:
            threshold, thr_reason = youden_threshold(inner_oof_y, inner_oof_p, is_dummy=False)

        # ---- refit on full outer-train, predict outer-test ONCE ----
        proba = _fit_predict(spec, chosen_cand, X, y, train_idx, test_idx)
        ypred = (proba >= threshold).astype(int)
        for pos, gi in enumerate(test_idx):
            oof_rows.append({
                "model": spec.name, "window_id": frame["window_id"].iloc[gi],
                "patient_id": frame["patient_id"].iloc[gi], "outer_fold": int(k),
                "y_true": int(y[gi]), "y_prob": float(proba[pos]),
                "threshold": float(threshold), "y_pred": int(ypred[pos]),
            })

    oof = pd.DataFrame(oof_rows)
    # deterministic column order
    oof = oof[["model", "window_id", "patient_id", "outer_fold", "y_true", "y_prob", "threshold", "y_pred"]]
    return OOFResult(spec=spec, oof=oof, selected=selected_records, candidates=candidate_records)


__all__ = [
    "SEED", "ModelSpec", "model_battery", "run_nested_cv", "youden_threshold", "OOFResult",
]
