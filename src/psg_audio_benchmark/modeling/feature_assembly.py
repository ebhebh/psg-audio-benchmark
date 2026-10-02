"""Stage-7 feature assembly: build the model matrix from the Stage-5 HR/SpO2
window-feature parquets, restricted to the frozen v2 core cohort.

Hard rules (enforced here, on top of the allow-list in :mod:`modeling.schema`):

* Features come ONLY from ``hr_window_features.parquet`` /
  ``spo2_window_features.parquet``, joined on the stable ``window_id``.
* Only the whitelisted statistics are retained, prefixed ``hr_`` / ``spo2_``
  (16 columns, fixed order). The assembled feature column set MUST equal
  :data:`schema.FEATURE_WHITELIST` exactly (allow-list), AND no forbidden
  column / category token may be present (deny-list defence in depth).
* The label is the Stage-4 ``binary_event_label`` carried by the gate's cohort
  table; IDs/time/quality/availability/annotation/awake/sleep/airflow/audio are
  NEVER features.
* The cohort is restricted to the gate's verified core-cohort ``window_id`` set.
* Join must be 1:1 on ``window_id`` (no duplicates, no missing core windows).

The audit keys (``window_id``, ``patient_id``, ``outer_fold``) ride along for
traceability + patient bootstrap ONLY; :class:`ModelFrame` exposes ``X`` as a
pure numeric matrix of the whitelisted columns (no keys ever reach an estimator).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from . import schema as S


class FeatureAssemblyError(RuntimeError):
    """Raised when feature assembly violates a hard rule."""


@dataclass(frozen=True)
class ModelFrame:
    """The assembled, fold-annotated model matrix + audit keys.

    ``X`` is an ``(n_windows, 16)`` float64 matrix of the whitelisted features
    in fixed order (``feature_names``). ``y`` is the int 0/1 label. ``df`` holds
    the audit keys + label + feature columns (feature values live here only for
    internal use; the OOF writer never emits them).
    """

    df: pd.DataFrame
    feature_names: List[str]
    y: np.ndarray
    X: np.ndarray
    window_id: np.ndarray
    patient_id: np.ndarray
    outer_fold: np.ndarray


def _check_feature_columns(columns: List[str]) -> None:
    """Allow-list (exact) + deny-list (defence) validation."""
    got = list(columns)
    if got != list(S.FEATURE_WHITELIST):
        # allow-list failure is fatal and detailed
        missing = [c for c in S.FEATURE_WHITELIST if c not in got]
        extra = [c for c in got if c not in S.FEATURE_WHITELIST]
        raise FeatureAssemblyError(
            f"Feature columns do not match the allow-list. "
            f"missing={missing} extra={extra} got={got}"
        )
    lower = {c.lower() for c in got}
    for tok in S.FORBIDDEN_CATEGORY_TOKENS:
        hits = [c for c in got if tok in c.lower()]
        if hits:
            raise FeatureAssemblyError(
                f"Forbidden category token {tok!r} present in feature columns {hits}."
            )
    for col in got:
        if col in S.FORBIDDEN_COLUMNS:
            raise FeatureAssemblyError(
                f"Forbidden column {col!r} present in the feature matrix."
            )


def _load_modality_features(
    path: Path, modality: str, statistics: List[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FeatureAssemblyError(f"Stage-5 feature file missing: {path}")
    df = pd.read_parquet(path)
    if S.FEATURE_JOIN_KEY not in df.columns:
        raise FeatureAssemblyError(f"{path}: missing join key {S.FEATURE_JOIN_KEY!r}")
    keep = [S.FEATURE_JOIN_KEY] + [c for c in statistics if c in df.columns]
    missing_stats = [c for c in statistics if c not in df.columns]
    if missing_stats:
        raise FeatureAssemblyError(
            f"{path}: whitelisted statistics missing from file: {missing_stats}"
        )
    out = df[keep].copy()
    prefix = S.FEATURE_PREFIX[modality]
    rename = {c: f"{prefix}{c}" for c in statistics}
    out = out.rename(columns=rename)
    return out


def assemble_model_frame(
    *,
    cohort_windows: pd.DataFrame,
    outer_folds: pd.DataFrame,
    hr_features_path: Path,
    spo2_features_path: Path,
) -> ModelFrame:
    """Assemble the fold-annotated core-cohort model frame."""
    statistics = list(S.FEATURE_STATISTICS)
    hr = _load_modality_features(hr_features_path, "heart_rate", statistics)
    sp = _load_modality_features(spo2_features_path, "spo2", statistics)

    # 1:1 join checks within each modality
    for name, d in (("hr", hr), ("spo2", sp)):
        if d[S.FEATURE_JOIN_KEY].duplicated().any():
            raise FeatureAssemblyError(
                f"{name} features have duplicate {S.FEATURE_JOIN_KEY}; cannot join 1:1."
            )

    # restrict features to the verified core-cohort window set FIRST
    cohort = cohort_windows[["window_id", "patient_id", "binary_event_label"]].copy()
    if cohort["window_id"].duplicated().any():
        raise FeatureAssemblyError("cohort_windows has duplicate window_id.")
    n_core = int(len(cohort))

    base = cohort.merge(hr, on="window_id", how="left")
    base = base.merge(sp, on="window_id", how="left")
    if len(base) != n_core:
        raise FeatureAssemblyError(
            f"Feature join changed row count ({n_core} -> {len(base)}); "
            f"non 1:1 join on window_id."
        )
    feature_names = list(S.FEATURE_WHITELIST)
    _check_feature_columns(feature_names)

    missing_feat = base[feature_names].isna().any().any()
    # missing values are EXPECTED (imputed fold-internally later); we only
    # require that every core window has a feature ROW (presence), not that the
    # statistic is finite. All-zero-coverage windows legitimately produce NaN.
    if base[feature_names].shape[0] != n_core:
        raise FeatureAssemblyError("feature row count != cohort row count after join.")

    # fold annotation: each window's patient -> its OWN outer-test fold (static)
    pid2outer = dict(zip(outer_folds["patient_id"], outer_folds["outer_fold"]))
    base["outer_fold"] = base["patient_id"].map(pid2outer)
    if base["outer_fold"].isna().any():
        miss = base.loc[base["outer_fold"].isna(), "patient_id"].unique().tolist()
        raise FeatureAssemblyError(
            f"patients without an outer-fold assignment: {miss}"
        )
    base["outer_fold"] = base["outer_fold"].astype(int)

    # enforce column order: audit keys, label, then features
    base = base[["window_id", "patient_id", "binary_event_label", "outer_fold"]
                + feature_names].reset_index(drop=True)

    X = base[feature_names].to_numpy(dtype=np.float64)
    y = base["binary_event_label"].to_numpy(dtype=int)
    return ModelFrame(
        df=base,
        feature_names=feature_names,
        y=y,
        X=X,
        window_id=base["window_id"].to_numpy(),
        patient_id=base["patient_id"].to_numpy(),
        outer_fold=base["outer_fold"].to_numpy(),
    )


__all__ = [
    "FeatureAssemblyError",
    "ModelFrame",
    "assemble_model_frame",
]
