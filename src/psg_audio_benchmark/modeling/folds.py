"""Stage-7 fold CV adapter: derives index masks from the FROZEN v2 assignments.

This module NEVER splits, shuffles, re-balances or invents folds. It only
exposes the frozen outer (patient -> own outer-test fold) and inner
((outer_train, patient) -> inner-validation-fold) assignments as boolean masks
over a :class:`~modeling.feature_assembly.ModelFrame`.

For an outer fold ``k``:

* outer-test  = windows whose patient's OWN outer fold == k
* outer-train = all other windows
* within outer-train, each patient has a frozen ``inner_validation_fold`` in
  {0..3}; inner-validation(j) = outer-train & inner == j, inner-train(j) = the
  rest. The inner assignments come from the frozen
  ``inner_patient_folds_core_v2`` table filtered to ``outer_fold == k``, so an
  outer-test patient can never appear in any inner fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .feature_assembly import ModelFrame


class FoldError(RuntimeError):
    """Raised when the fold adapter detects a structural violation."""


@dataclass(frozen=True)
class FoldAdapter:
    """Boolean-mask adapter over a ModelFrame using the frozen assignments."""

    frame: ModelFrame
    #: (outer_fold_k) -> {patient_id -> inner_validation_fold}
    inner_map_by_outer: Dict[int, Dict[str, int]]
    outer_n_folds: int
    inner_n_folds: int

    # -- outer -------------------------------------------------------------
    def outer_test_mask(self, k: int) -> np.ndarray:
        return self.frame.outer_fold == k

    def outer_train_mask(self, k: int) -> np.ndarray:
        return self.frame.outer_fold != k

    def outer_test_indices(self, k: int) -> np.ndarray:
        return np.nonzero(self.outer_test_mask(k))[0]

    def outer_train_indices(self, k: int) -> np.ndarray:
        return np.nonzero(self.outer_train_mask(k))[0]

    # -- inner -------------------------------------------------------------
    def inner_fold_labels(self, k: int) -> np.ndarray:
        """Inner-validation-fold label per outer-train window of fold ``k``
        (``-1`` for outer-test windows)."""
        pid_map = self.inner_map_by_outer[k]
        labels = np.full(self.frame.outer_fold.shape[0], -1, dtype=int)
        ot = self.outer_train_mask(k)
        pids = self.frame.patient_id
        for i in np.nonzero(ot)[0]:
            labels[i] = int(pid_map[str(pids[i])])
        return labels

    def inner_val_mask(self, k: int, j: int) -> np.ndarray:
        ot = self.outer_train_mask(k)
        labels = self.inner_fold_labels(k)
        return ot & (labels == j)

    def inner_train_mask(self, k: int, j: int) -> np.ndarray:
        ot = self.outer_train_mask(k)
        labels = self.inner_fold_labels(k)
        return ot & (labels != j) & (labels >= 0)

    # -- integrity ---------------------------------------------------------
    def assert_patient_isolation(self) -> None:
        """No patient appears in two outer folds; no outer-test patient in any
        inner fold of the same outer_train."""
        seen: Dict[str, int] = {}
        of = self.frame.outer_fold
        pids = self.frame.patient_id
        for i in range(of.shape[0]):
            p = str(pids[i])
            f = int(of[i])
            if p in seen and seen[p] != f:
                raise FoldError(f"patient {p} in two outer folds ({seen[p]}, {f})")
            seen[p] = f
        for k in range(self.outer_n_folds):
            test_pids = set(str(p) for p in pids[of == k])
            inner_labels = self.inner_fold_labels(k)
            inner_idx = np.nonzero((self.outer_train_mask(k)) & (inner_labels >= 0))[0]
            inner_pids = set(str(pids[i]) for i in inner_idx)
            leak = test_pids & inner_pids
            if leak:
                raise FoldError(
                    f"outer fold {k}: outer-test patients {sorted(leak)} leaked into "
                    f"its inner folds."
                )


def build_fold_adapter(
    *, frame: ModelFrame, inner_folds: pd.DataFrame,
    outer_n_folds: int, inner_n_folds: int,
) -> FoldAdapter:
    """Build a :class:`FoldAdapter` from the frozen inner-fold table."""
    inner_map_by_outer: Dict[int, Dict[str, int]] = {}
    for k in range(outer_n_folds):
        sub = inner_folds[inner_folds["outer_fold"] == k]
        m: Dict[str, int] = {}
        for _, row in sub.iterrows():
            m[str(row["patient_id"])] = int(row["inner_validation_fold"])
        inner_map_by_outer[k] = m
        if len(m) == 0:
            raise FoldError(f"outer fold {k}: no inner-fold rows found.")
        folds_seen = sorted(set(m.values()))
        if folds_seen != list(range(inner_n_folds)):
            raise FoldError(
                f"outer fold {k}: inner folds {folds_seen} != "
                f"{list(range(inner_n_folds))}."
            )
    adapter = FoldAdapter(
        frame=frame,
        inner_map_by_outer=inner_map_by_outer,
        outer_n_folds=outer_n_folds,
        inner_n_folds=inner_n_folds,
    )
    adapter.assert_patient_isolation()
    return adapter


__all__ = ["FoldError", "FoldAdapter", "build_fold_adapter"]
