"""Stage-7 estimator pipelines + pre-specified hyper-parameter grids.

Three pre-registered model families (roles fixed; no outer-test cherry-pick):

* ``dummy_prior``  — ``DummyClassifier(strategy="prior")``; not tunable; fixed
  0.5 threshold.
* ``logistic_regression`` (PRIMARY) — ``SimpleImputer(median)`` →
  ``StandardScaler`` → ``LogisticRegression``; grid ``C ∈ {0.1,1,10}`` ×
  ``class_weight ∈ {None,"balanced"}``; threshold = inner-OOF max Youden J.
* ``hist_gradient_boosting`` (EXPLORATORY) — ``SimpleImputer(median)`` →
  ``HistGradientBoostingClassifier``; grid ``learning_rate ∈ {0.05,0.10}`` ×
  ``max_leaf_nodes ∈ {15,31}``; balanced ``sample_weight`` is computed from the
  TRAIN labels only (never validation/test); threshold = inner-OOF Youden J.

Every preprocessor / weight / hyper-parameter is fitted on the TRAIN partition
supplied by the caller. This module never sees the outer-test or inner-val
labels except as the prediction target of a model fitted on a disjoint train
set. Fixed ``solver`` / ``max_iter`` / ``random_state`` / ``early_stopping``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import schema as S


# fixed seeds / solver constants
_SEED = 20250714


@dataclass(frozen=True)
class Candidate:
    """One hyper-parameter point with a deterministic grid index."""

    index: int
    params: Dict[str, Any]

    @property
    def label(self) -> str:
        return ";".join(f"{k}={v}" for k, v in sorted(self.params.items()))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    tunable: bool
    needs_sample_weight: bool
    threshold_rule: str
    step_names: Tuple[str, ...]
    candidates: Tuple[Candidate, ...]
    base_params: Dict[str, Any] = field(default_factory=dict)

    def make_pipeline(self) -> Pipeline:
        return _build_named_pipeline(self.name)


# ---------------------------------------------------------------------------
# pipeline construction
# ---------------------------------------------------------------------------

def _build_named_pipeline(name: str) -> Pipeline:
    if name == S.MODEL_DUMMY_PRIOR:
        return Pipeline([("dummy", DummyClassifier(strategy="prior"))])
    if name == S.MODEL_LOGISTIC_REGRESSION:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logisticregression", LogisticRegression(
                solver="lbfgs", max_iter=2000, random_state=_SEED)),
        ])
    if name == S.MODEL_HIST_GRADIENT_BOOSTING:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("histgradientboostingclassifier", HistGradientBoostingClassifier(
                random_state=_SEED, early_stopping=False, loss="log_loss",
                l2_regularization=0.0)),
        ])
    raise ValueError(f"unknown model family: {name!r}")


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of a grid, in a deterministic ascending order."""
    keys = list(grid.keys())
    if not keys:
        return [{}]
    # sort each axis to make the order deterministic
    def _sort_key(v: Any) -> Tuple[int, Any]:
        # None sorts first, then by (type-name, value)
        if v is None:
            return (0, "")
        return (1, v)
    values = [sorted(grid[k], key=_sort_key) for k in keys]
    out: List[Dict[str, Any]] = []
    # product
    idx = [0] * len(keys)
    total = 1
    for v in values:
        total *= len(v)
    for _ in range(total):
        out.append({keys[i]: values[i][idx[i]] for i in range(len(keys))})
        # increment with carry
        p = len(keys) - 1
        while p >= 0:
            idx[p] += 1
            if idx[p] < len(values[p]):
                break
            idx[p] = 0
            p -= 1
        if p < 0 and _ != total - 1:
            break
    return out


def _lr_candidates() -> Tuple[Candidate, ...]:
    grid = {
        "logisticregression__C": [0.1, 1.0, 10.0],
        "logisticregression__class_weight": [None, "balanced"],
    }
    return tuple(Candidate(i, p) for i, p in enumerate(_expand_grid(grid)))


def _hgb_candidates() -> Tuple[Candidate, ...]:
    grid = {
        "histgradientboostingclassifier__learning_rate": [0.05, 0.10],
        "histgradientboostingclassifier__max_leaf_nodes": [15, 31],
    }
    return tuple(Candidate(i, p) for i, p in enumerate(_expand_grid(grid)))


def build_model_specs() -> Dict[str, ModelSpec]:
    """Return the three pre-registered model specs."""
    specs: Dict[str, ModelSpec] = {
        S.MODEL_DUMMY_PRIOR: ModelSpec(
            name=S.MODEL_DUMMY_PRIOR, role=S.MODEL_ROLE[S.MODEL_DUMMY_PRIOR],
            tunable=False, needs_sample_weight=False,
            threshold_rule=S.THRESHOLD_RULE_DUMMY,
            step_names=("dummy",),
            candidates=(Candidate(0, {}),),
            base_params={},
        ),
        S.MODEL_LOGISTIC_REGRESSION: ModelSpec(
            name=S.MODEL_LOGISTIC_REGRESSION, role=S.MODEL_ROLE[S.MODEL_LOGISTIC_REGRESSION],
            tunable=True, needs_sample_weight=False,
            threshold_rule=S.THRESHOLD_RULE_DATA_DRIVEN,
            step_names=("imputer", "scaler", "logisticregression"),
            candidates=_lr_candidates(),
            base_params={},
        ),
        S.MODEL_HIST_GRADIENT_BOOSTING: ModelSpec(
            name=S.MODEL_HIST_GRADIENT_BOOSTING,
            role=S.MODEL_ROLE[S.MODEL_HIST_GRADIENT_BOOSTING],
            tunable=True, needs_sample_weight=True,
            threshold_rule=S.THRESHOLD_RULE_DATA_DRIVEN,
            step_names=("imputer", "histgradientboostingclassifier"),
            candidates=_hgb_candidates(),
            base_params={},
        ),
    }
    return specs


# ---------------------------------------------------------------------------
# balanced sample weight (TRAIN labels only)
# ---------------------------------------------------------------------------

def balanced_sample_weight(y_train: np.ndarray) -> np.ndarray:
    """``sample_weight`` for class balancing, computed from TRAIN labels only.

    weight_i = n / (n_classes * count(y_i)); identical to sklearn's
    ``class_weight='balanced'`` formula but as a per-sample vector so it can be
    passed to estimators (e.g. HGB) that have no ``class_weight`` argument.
    """
    y = np.asarray(y_train).astype(int)
    n = y.shape[0]
    classes = np.unique(y)
    counts = {int(c): int((y == c).sum()) for c in classes}
    w = np.empty(n, dtype=float)
    for i in range(n):
        c = int(y[i])
        w[i] = n / (len(classes) * counts[c])
    return w


# ---------------------------------------------------------------------------
# fit / predict helpers
# ---------------------------------------------------------------------------

def fit_predict(
    *,
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    pred_idx: np.ndarray,
    candidate: Optional[Candidate] = None,
) -> np.ndarray:
    """Fit ``spec``'s pipeline on ``train_idx`` and return P(y=1) for ``pred_idx``.

    Every transformer / estimator is fitted ONLY on the train partition. For
    HGB, ``sample_weight`` is computed from the TRAIN labels and passed to the
    final estimator's ``fit`` via the pipeline (the imputer is stateless w.r.t.
    weights). The pred partition's labels never influence the fit.
    """
    pipe = spec.make_pipeline()
    params = dict(candidate.params) if candidate is not None else {}
    if params:
        pipe.set_params(**params)
    Xtr = X[train_idx]
    ytr = y[train_idx]
    if spec.needs_sample_weight:
        sw = balanced_sample_weight(ytr)
        # sample_weight is only meaningful for the final supervised estimator;
        # the imputer fits identically with or without it.
        final_name = spec.step_names[-1]
        pipe.fit(Xtr, ytr, **{f"{final_name}__sample_weight": sw})
    else:
        pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(X[pred_idx])
    # locate the positive column (label 1)
    classes = pipe.classes_
    pos_col = int(np.where(classes == 1)[0][0])
    return proba[:, pos_col]


__all__ = [
    "Candidate",
    "ModelSpec",
    "build_model_specs",
    "balanced_sample_weight",
    "fit_predict",
]
