"""Stage-8 paired feature assembly.

Builds TWO estimator matrices on the IDENTICAL airflow-eligible window set:

* ``core_restricted``  -- the 16 HR+SpO2 statistics (Stage-7 allow-list);
* ``airflow_enhanced`` -- those 16 + the 11 airflow statistics (27 columns).

Hard rules (enforced here, on top of the per-set allow-list in
:mod:`airflow_increment.schema`):

* Both frames are restricted to the gate's verified airflow-eligible cohort
  (HR & SpO2 & airflow all available, positive/negative windows).
* Each frame's feature column set MUST exactly equal its allow-list (primary
  guard), AND no forbidden column / category token may be present (deny-list
  defence in depth). Airflow statistics are admissible ONLY in
  ``airflow_enhanced``; they MUST be absent from ``core_restricted``.
* The two frames' ``window_id`` / ``patient_id`` / label / ``outer_fold`` are
  asserted IDENTICAL, in the SAME row order -- so the core/enhanced OOF
  predictions are one-to-one pairable per window.
* Join is 1:1 on ``window_id`` (no duplicates, no missing cohort windows).

The audit keys (``window_id``, ``patient_id``, ``outer_fold``) ride along for
traceability + paired patient bootstrap ONLY; :class:`AirflowModelFrame.X` is a
pure numeric matrix (no keys ever reach an estimator).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import schema as S


class AirflowFeatureAssemblyError(RuntimeError):
    """Raised when paired feature assembly violates a hard Stage-8 rule."""


@dataclass(frozen=True)
class AirflowModelFrame:
    """One feature set's fold-annotated matrix + audit keys.

    ``X`` is an ``(n_windows, p)`` float64 matrix of the feature set's
    whitelisted columns in fixed order (``feature_names``). ``y`` is the int 0/1
    label. The two frames share identical ``window_id``/``patient_id``/``y``/
    ``outer_fold`` arrays (asserted by :func:`assemble_paired_frames`).
    """

    feature_set: str
    df: pd.DataFrame
    feature_names: List[str]
    y: np.ndarray
    X: np.ndarray
    window_id: np.ndarray
    patient_id: np.ndarray
    outer_fold: np.ndarray


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _check_feature_columns(columns: Sequence[str], feature_set: str) -> None:
    """Exact allow-list (per feature set) + deny-list defence in depth."""
    allow = list(S.FEATURE_SET_WHITELIST[feature_set])
    got = list(columns)
    if got != allow:
        missing = [c for c in allow if c not in got]
        extra = [c for c in got if c not in allow]
        raise AirflowFeatureAssemblyError(
            f"[{feature_set}] feature columns do not match the allow-list. "
            f"missing={missing} extra={extra} got={got}"
        )
    for tok in S.FORBIDDEN_CATEGORY_TOKENS:
        hits = [c for c in got if tok in c.lower()]
        if hits:
            raise AirflowFeatureAssemblyError(
                f"[{feature_set}] forbidden category token {tok!r} present in "
                f"feature columns {hits}."
            )
    for col in got:
        if col in S.FORBIDDEN_COLUMNS:
            raise AirflowFeatureAssemblyError(
                f"[{feature_set}] forbidden column {col!r} present in the "
                f"feature matrix."
            )


def assert_airflow_set_isolation() -> None:
    """Airflow columns appear ONLY in the enhanced allow-list, never in core."""
    air = set(S.AIRFLOW_FEATURE_WHITELIST)
    core = set(S.CORE_FEATURE_WHITELIST)
    leak = air & core
    if leak:
        raise AirflowFeatureAssemblyError(
            f"airflow columns leaked into the core allow-list: {sorted(leak)}"
        )
    if not air.issubset(set(S.ENHANCED_FEATURE_WHITELIST)):
        raise AirflowFeatureAssemblyError(
            "enhanced allow-list is missing airflow statistics."
        )
    # core must contain NO airflow-prefixed column
    bad = [c for c in S.CORE_FEATURE_WHITELIST if c.startswith(S.AIRFLOW_PREFIX)]
    if bad:
        raise AirflowFeatureAssemblyError(
            f"core_restricted must contain no airflow column; found {bad}"
        )


# ---------------------------------------------------------------------------
# modality loaders
# ---------------------------------------------------------------------------

def _load_modality_features(
    path: Path, modality: str, statistics: Sequence[str], prefix: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise AirflowFeatureAssemblyError(f"Stage-5 feature file missing: {path}")
    df = pd.read_parquet(path)
    if S.FEATURE_JOIN_KEY not in df.columns:
        raise AirflowFeatureAssemblyError(
            f"{path}: missing join key {S.FEATURE_JOIN_KEY!r}")
    missing_stats = [c for c in statistics if c not in df.columns]
    if missing_stats:
        raise AirflowFeatureAssemblyError(
            f"{path} [{modality}]: whitelisted statistics missing from file: "
            f"{missing_stats}")
    keep = [S.FEATURE_JOIN_KEY] + list(statistics)
    out = df[keep].copy()
    out = out.rename(columns={c: f"{prefix}{c}" for c in statistics})
    return out


# ---------------------------------------------------------------------------
# single-frame assembly
# ---------------------------------------------------------------------------

def _assemble_one(
    *, feature_set: str, cohort_windows: pd.DataFrame,
    outer_folds: pd.DataFrame, hr_features_path: Path, spo2_features_path: Path,
    airflow_features_path: Path,
) -> AirflowModelFrame:
    from ..schema import FEATURE_STATISTICS as _HR_SPO2_STATS

    statistics = list(_HR_SPO2_STATS)  # 8 HR/SpO2 stats
    hr = _load_modality_features(hr_features_path, "heart_rate", statistics, "hr_")
    sp = _load_modality_features(spo2_features_path, "spo2", statistics, "spo2_")
    mods = [("hr", hr), ("spo2", sp)]
    if feature_set == S.FEATURE_SET_ENHANCED:
        af = _load_modality_features(
            airflow_features_path, "airflow", S.AIRFLOW_STATISTICS, S.AIRFLOW_PREFIX)
        mods.append(("airflow", af))

    for name, d in mods:
        if d[S.FEATURE_JOIN_KEY].duplicated().any():
            raise AirflowFeatureAssemblyError(
                f"{name} features have duplicate {S.FEATURE_JOIN_KEY}; cannot "
                f"join 1:1."
            )

    cohort = cohort_windows[["window_id", "patient_id", "binary_event_label"]].copy()
    if cohort["window_id"].duplicated().any():
        raise AirflowFeatureAssemblyError("cohort_windows has duplicate window_id.")
    n_cohort = int(len(cohort))

    base = cohort.merge(hr, on="window_id", how="left")
    base = base.merge(sp, on="window_id", how="left")
    if feature_set == S.FEATURE_SET_ENHANCED:
        base = base.merge(af, on="window_id", how="left")
    if len(base) != n_cohort:
        raise AirflowFeatureAssemblyError(
            f"[{feature_set}] feature join changed row count ({n_cohort} -> "
            f"{len(base)}); non 1:1 join on window_id."
        )

    feature_names = list(S.FEATURE_SET_WHITELIST[feature_set])
    _check_feature_columns(feature_names, feature_set)
    if base[feature_names].shape[0] != n_cohort:
        raise AirflowFeatureAssemblyError(
            f"[{feature_set}] feature row count != cohort row count after join.")

    # fold annotation: each window's patient -> its OWN outer-test fold (static,
    # inherited from the v2 assignment).
    pid2outer = dict(zip(outer_folds["patient_id"], outer_folds["outer_fold"]))
    base["outer_fold"] = base["patient_id"].map(pid2outer)
    if base["outer_fold"].isna().any():
        miss = base.loc[base["outer_fold"].isna(), "patient_id"].unique().tolist()
        raise AirflowFeatureAssemblyError(
            f"[{feature_set}] patients without an outer-fold assignment: {miss}"
        )
    base["outer_fold"] = base["outer_fold"].astype(int)

    base = base[["window_id", "patient_id", "binary_event_label", "outer_fold"]
                + feature_names].reset_index(drop=True)

    X = base[feature_names].to_numpy(dtype=np.float64)
    y = base["binary_event_label"].to_numpy(dtype=int)
    return AirflowModelFrame(
        feature_set=feature_set, df=base, feature_names=feature_names, y=y, X=X,
        window_id=base["window_id"].to_numpy(),
        patient_id=base["patient_id"].to_numpy(),
        outer_fold=base["outer_fold"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# membership assertion
# ---------------------------------------------------------------------------

def assert_frames_identical_membership(
    core: AirflowModelFrame, enhanced: AirflowModelFrame,
) -> None:
    """The two frames MUST share identical window/patient/label/fold membership
    in the SAME row order, else the paired comparison is invalid -> FAIL."""
    checks: List[Tuple[str, str, str]] = [
        ("window_id", "window_id arrays differ", "core.window_id != enhanced.window_id"),
        ("patient_id", "patient_id arrays differ", "core.patient_id != enhanced.patient_id"),
        ("label", "labels differ", "core.y != enhanced.y"),
        ("outer_fold", "outer_fold arrays differ", "core.outer_fold != enhanced.outer_fold"),
    ]
    if core.window_id.shape[0] != enhanced.window_id.shape[0]:
        raise AirflowFeatureAssemblyError(
            f"row count differs: core={core.window_id.shape[0]} "
            f"enhanced={enhanced.window_id.shape[0]}")
    pairs = [
        ("window_id", core.window_id, enhanced.window_id),
        ("patient_id", core.patient_id, enhanced.patient_id),
        ("label", core.y, enhanced.y),
        ("outer_fold", core.outer_fold, enhanced.outer_fold),
    ]
    for name, a, b in pairs:
        if not np.array_equal(a, b):
            diff_idx = int(np.argmax(a != b)) if a.shape == b.shape else -1
            raise AirflowFeatureAssemblyError(
                f"core_restricted vs airflow_enhanced {name} mismatch "
                f"(first diff at row {diff_idx}); paired comparison invalid.")
    # explicit set equality on windows (defence even if order were swapped)
    if set(core.window_id.tolist()) != set(enhanced.window_id.tolist()):
        raise AirflowFeatureAssemblyError(
            "core_restricted and airflow_enhanced window SETS differ.")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def assemble_paired_frames(
    *, cohort_windows: pd.DataFrame, outer_folds: pd.DataFrame,
    hr_features_path: Path, spo2_features_path: Path,
    airflow_features_path: Path,
) -> Tuple[AirflowModelFrame, AirflowModelFrame]:
    """Assemble the two membership-identical paired frames."""
    assert_airflow_set_isolation()
    if airflow_features_path is None or not Path(airflow_features_path).is_file():
        raise AirflowFeatureAssemblyError(
            f"airflow feature file required for Stage 8: {airflow_features_path}")
    core = _assemble_one(
        feature_set=S.FEATURE_SET_CORE, cohort_windows=cohort_windows,
        outer_folds=outer_folds, hr_features_path=hr_features_path,
        spo2_features_path=spo2_features_path, airflow_features_path=airflow_features_path,
    )
    enhanced = _assemble_one(
        feature_set=S.FEATURE_SET_ENHANCED, cohort_windows=cohort_windows,
        outer_folds=outer_folds, hr_features_path=hr_features_path,
        spo2_features_path=spo2_features_path,
        airflow_features_path=airflow_features_path,
    )
    assert_frames_identical_membership(core, enhanced)
    return core, enhanced


__all__ = [
    "AirflowFeatureAssemblyError",
    "AirflowModelFrame",
    "assert_airflow_set_isolation",
    "assert_frames_identical_membership",
    "assemble_paired_frames",
]
