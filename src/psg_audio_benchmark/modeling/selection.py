"""Stage-7 inner-CV model selection.

For each tunable model and each outer fold, the frozen 4 inner folds are swept:
for every candidate hyper-point, fit on inner-train(j) and predict
inner-validation(j), for j in 0..3. The candidate's inner-selection score is the
mean ``average_precision`` across the computable inner folds; ties (within a
fixed tolerance) are broken by higher mean AUROC, then by ascending candidate
grid index (the simpler-candidate rule).

This module records **every** candidate's per-fold scores + mean/std (for the
``inner_cv_candidate_scores.csv`` audit), builds the selected candidate's inner
out-of-fold (OOF) predictions on the outer-train (used later by the threshold
rule), and returns the selected candidate. Outer-test never participates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import schema as S
from .folds import FoldAdapter
from .metrics import safe_auprc, safe_auroc
from .pipelines import Candidate, ModelSpec, fit_predict

_AP_TOL = 1e-9


@dataclass
class CandidateCVResult:
    candidate: Candidate
    per_fold_ap: List[Optional[float]] = field(default_factory=list)
    per_fold_auroc: List[Optional[float]] = field(default_factory=list)
    mean_ap: Optional[float] = None
    std_ap: Optional[float] = None
    mean_auroc: Optional[float] = None
    std_auroc: Optional[float] = None
    n_computable_folds: int = 0
    oof_y_true: Optional[np.ndarray] = None
    oof_y_prob: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    def as_record(self, *, run_id: str, model_name: str, outer_fold: int) -> dict:
        return {
            "run_id": run_id,
            "model": model_name,
            "outer_fold": outer_fold,
            "candidate_index": int(self.candidate.index),
            "candidate_params": self.candidate.label,
            "mean_average_precision": self.mean_ap,
            "std_average_precision": self.std_ap,
            "mean_auroc": self.mean_auroc,
            "std_auroc": self.std_auroc,
            "n_computable_inner_folds": int(self.n_computable_folds),
            "per_fold_average_precision": ";".join(
                "NA" if v is None else f"{v:.6f}" for v in self.per_fold_ap),
            "per_fold_auroc": ";".join(
                "NA" if v is None else f"{v:.6f}" for v in self.per_fold_auroc),
            "selected": False,  # set by the caller on the winner
        }


@dataclass
class InnerSelectionResult:
    model_name: str
    outer_fold: int
    candidates: List[CandidateCVResult]
    selected: CandidateCVResult
    selection_rule: str


def _mean_std(vals: List[Optional[float]]) -> tuple:
    """Mean/std over non-None values (NA folds excluded, not zero-filled)."""
    keep = [float(v) for v in vals if v is not None]
    if not keep:
        return None, None
    m = float(np.mean(keep))
    s = float(np.std(keep, ddof=1)) if len(keep) > 1 else 0.0
    return m, s


def _evaluate_candidate(
    *, spec: ModelSpec, candidate: Candidate, X: np.ndarray, y: np.ndarray,
    adapter: FoldAdapter, outer_fold: int, inner_n_folds: int,
) -> CandidateCVResult:
    res = CandidateCVResult(candidate=candidate)
    oof_rows: List[int] = []
    oof_y: List[int] = []
    oof_p: List[float] = []
    for j in range(inner_n_folds):
        tr = np.nonzero(adapter.inner_train_mask(outer_fold, j))[0]
        va = np.nonzero(adapter.inner_val_mask(outer_fold, j))[0]
        if tr.size == 0 or va.size == 0:
            res.per_fold_ap.append(None)
            res.per_fold_auroc.append(None)
            res.notes.append(f"inner_fold{j}: empty train/val partition")
            continue
        proba = fit_predict(
            spec=spec, X=X, y=y, train_idx=tr, pred_idx=va, candidate=candidate,
        )
        yva = y[va]
        ap, _ = safe_auprc(yva, proba)
        au, _ = safe_auroc(yva, proba)
        res.per_fold_ap.append(ap)
        res.per_fold_auroc.append(au)
        oof_rows.extend(int(i) for i in va)
        oof_y.extend(int(v) for v in yva)
        oof_p.extend(float(v) for v in proba)
    res.mean_ap, res.std_ap = _mean_std(res.per_fold_ap)
    res.mean_auroc, res.std_auroc = _mean_std(res.per_fold_auroc)
    res.n_computable_folds = sum(1 for v in res.per_fold_ap if v is not None)
    if oof_y:
        order = np.argsort(oof_rows)
        res.oof_y_true = np.asarray(oof_y)[order]
        res.oof_y_prob = np.asarray(oof_p)[order]
    return res


def select_inner(
    *, spec: ModelSpec, X: np.ndarray, y: np.ndarray, adapter: FoldAdapter,
    outer_fold: int, inner_n_folds: int,
) -> InnerSelectionResult:
    """Run the inner-CV search for one (model, outer fold)."""
    results: List[CandidateCVResult] = []
    for cand in spec.candidates:
        results.append(_evaluate_candidate(
            spec=spec, candidate=cand, X=X, y=y, adapter=adapter,
            outer_fold=outer_fold, inner_n_folds=inner_n_folds,
        ))

    # selection: max mean_ap (tol); tie -> higher mean_auroc; tie -> ascending idx
    def _sort_key(r: CandidateCVResult):
        ap = r.mean_ap if r.mean_ap is not None else float("-inf")
        au = r.mean_auroc if r.mean_auroc is not None else float("-inf")
        return (-ap, -au, r.candidate.index)

    # explicit tie handling within AP tolerance
    best_ap = max((r.mean_ap if r.mean_ap is not None else float("-inf")) for r in results)
    eligible = [r for r in results
                if r.mean_ap is not None and r.mean_ap >= best_ap - _AP_TOL]
    if not eligible:
        # no computable candidate at all (degenerate); pick lowest index
        eligible = list(results)
    eligible.sort(key=lambda r: (
        -(r.mean_auroc if r.mean_auroc is not None else float("-inf")),
        r.candidate.index,
    ))
    selected = eligible[0]
    rule = (f"primary=average_precision (tol {_AP_TOL}); tie-break=higher mean AUROC "
            f"then ascending grid index; best mean_ap={best_ap:.6f}")
    return InnerSelectionResult(
        model_name=spec.name, outer_fold=outer_fold, candidates=results,
        selected=selected, selection_rule=rule,
    )


__all__ = [
    "CandidateCVResult",
    "InnerSelectionResult",
    "select_inner",
]
