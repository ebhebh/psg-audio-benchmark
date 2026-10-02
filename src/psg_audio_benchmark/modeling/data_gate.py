"""Stage-7 input data gate: accept ONLY the frozen, approved v2 split.

This is the single chokepoint that decides whether modeling may proceed. It
verifies, in order, and refuses with a loud ``DataGateError`` (BLOCKED) on any
failure — **never a silent fallback** to v1 or to a self-built split:

1. the license gate is PASSED;
2. ``splits/LATEST_RUN.txt`` resolves to the v2 run (or an explicit
   ``--split-run-id`` points to an equivalent approved v2 run);
3. the split-balance signature ``approval_status == approved_for_modeling`` and
   its ``run_id`` equals the resolved split run id;
4. the v2 fold files' bytes-SHA-256, row counts and schema fingerprints match
   the signature records;
5. the lineage run ids (Stage 4 window index, Stage 5 feature parquets) match
   the signature lineage;
6. the assignment hash and patient-set hash, **recomputed at runtime** from the
   fold files and the independently re-derived cohort, match the signature;
7. the cohort (Stage-4 label in {positive,negative} AND Stage-5
   ``core_hr_spo2_available``) is **re-derived at runtime** — counts are
   compared to the signature fingerprint but **never hard-coded**;
8. every patient has exactly one outer-test fold, and each outer-train's 4
   inner folds contain no outer-test patient.

The gate reads NO raw data and NO ``*.wav`` (audio BLOCKED). It returns the
verified artifacts (signature, fold tables, cohort window table) for the runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from . import schema as S


class DataGateError(RuntimeError):
    """Raised when the input gate refuses the split (BLOCKED)."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class DataGateResult:
    approved: bool
    split_run_id: str
    signature: Dict[str, Any]
    outer_folds: pd.DataFrame
    inner_folds: pd.DataFrame
    cohort_windows: pd.DataFrame  # window_id, patient_id, binary_event_label
    checks: List[Check] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# hashing helpers (must match split_balance_signature.py exactly)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_fingerprint(df: pd.DataFrame) -> str:
    pairs: List[str] = []
    for c in df.columns:
        dt = pd.api.types.pandas_dtype(df[c].dtype)
        pairs.append(f"{c}:{getattr(dt, 'char', str(dt))}")
    return hashlib.sha256("\n".join(sorted(pairs)).encode("utf-8")).hexdigest()


def _sha256_lines(lines: List[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _make_window_id(patient_id: Any, window_index: Any) -> str:
    return f"{str(patient_id).zfill(2)}-{int(window_index):05d}"


def _canonical_pid(value: Any) -> str:
    """Canonical 2-digit zero-padded patient id (``1``/``"01"``/``1.0`` -> ``"01"``).

    ``read_csv`` strips the leading zero from ``"01"`` -> integer ``1``; the
    cohort/feature parquets keep the string ``"01"``. Normalize everything to
    the cohort's canonical form so set/hash comparisons are consistent.
    """
    return f"{int(float(str(value).strip())):02d}"


def _canonicalize_pid_column(df: pd.DataFrame, col: str = "patient_id") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = df[col].map(_canonical_pid)
    return df


# ---------------------------------------------------------------------------
# license gate
# ---------------------------------------------------------------------------

def _verify_license(cfg: Config, relpath: str) -> Check:
    p = cfg.project_root / relpath
    ok = p.is_file()
    detail = f"{relpath} present={ok}"
    if ok:
        text = p.read_text(encoding="utf-8")
        ok = ("PASSED" in text) or ("confirmed_for_noncommercial_research" in text)
        detail += "; license_status marker found" if ok else "; PASSED/license marker MISSING"
    return Check("license_gate_passed", bool(ok), detail)


# ---------------------------------------------------------------------------
# split-run resolution
# ---------------------------------------------------------------------------

def _resolve_split_run_id(
    *, splits_dir: Path, latest_run_path: Path, override: Optional[str],
) -> Tuple[str, Check]:
    if override:
        return override, Check(
            "split_run_id_resolved", True,
            f"resolved from explicit --split-run-id={override}",
        )
    if not latest_run_path.is_file():
        raise DataGateError(
            f"splits/LATEST_RUN.txt not found at {latest_run_path}; cannot resolve "
            f"the split run."
        )
    run_id = latest_run_path.read_text(encoding="utf-8").strip()
    return run_id, Check(
        "split_run_id_resolved", True,
        f"resolved from splits/LATEST_RUN.txt -> {run_id}",
    )


def _load_signature(*, splits_dir: Path, split_run_id: str) -> Tuple[Dict[str, Any], Path]:
    """Load the signature for ``split_run_id`` from its run-dir (preferred) or
    the fixed mirror. The signature's ``run_id`` must equal ``split_run_id``."""
    candidates = [
        splits_dir / "runs" / split_run_id / "split_balance_version_signature.json",
        splits_dir / "split_balance_version_signature.json",
    ]
    for c in candidates:
        if c.is_file():
            try:
                sig = json.loads(c.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise DataGateError(f"Signature {c} is not valid JSON: {exc}") from exc
            return sig, c
    raise DataGateError(
        f"No split-balance signature found for split run {split_run_id!r} "
        f"(looked in run-dir and fixed mirror)."
    )


# ---------------------------------------------------------------------------
# per-file record verification (sha256 + rows + schema)
# ---------------------------------------------------------------------------

def _locate_v2_file(*, splits_dir: Path, split_run_id: str, relpath: str) -> Path:
    run_dir = splits_dir / "runs" / split_run_id
    cands = [run_dir / relpath, splits_dir / relpath]
    for c in cands:
        if c.is_file():
            return c
    raise DataGateError(
        f"v2 split file {relpath!r} not found for run {split_run_id!r} "
        f"(looked in run-dir and fixed mirror)."
    )


def _verify_file_record(
    *, splits_dir: Path, split_run_id: str, key: str, record: Dict[str, Any],
) -> Tuple[Check, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    relpath = record.get("relpath", "")
    if not relpath:
        return Check(key, False, "missing relpath"), None, None
    path = _locate_v2_file(splits_dir=splits_dir, split_run_id=split_run_id, relpath=relpath)
    exp_sha = record.get("sha256")
    got_sha = _sha256_file(path)
    ok = (exp_sha == got_sha)
    sig_tail = str(exp_sha)[:12] if exp_sha else "NA"
    detail = f"{relpath}: sha256 {got_sha[:12]}.. vs sig {sig_tail}.."
    df = None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    if df is not None:
        exp_rows = record.get("rows")
        got_rows = int(df.shape[0])
        rows_ok = (exp_rows == got_rows)
        exp_schema = record.get("schema_fingerprint")
        got_schema = _schema_fingerprint(df)
        schema_ok = (exp_schema == got_schema)
        ok = ok and rows_ok and schema_ok
        detail += f" | rows {got_rows}/{exp_rows} schema {'ok' if schema_ok else 'MISMATCH'}"
    return Check(key, bool(ok), detail), df, (df if df is not None else None)


# ---------------------------------------------------------------------------
# cohort re-derivation
# ---------------------------------------------------------------------------

def _rederive_cohort(
    *, stage4_index: pd.DataFrame, stage5_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Re-derive the core cohort from Stage-4 labels + Stage-5 availability.

    core = Stage-4 windows with label_status in {positive, negative} AND Stage-5
    ``core_hr_spo2_available`` True. Returns window_id, patient_id,
    binary_event_label (int). Independent of the Stage-6 membership table.
    """
    idx = stage4_index[["patient_id", "window_index", "label_status",
                        "binary_event_label"]].copy()
    idx["window_id"] = [
        _make_window_id(p, i) for p, i in zip(idx["patient_id"], idx["window_index"])
    ]
    av = stage5_availability[["window_id", "core_hr_spo2_available"]].copy()
    merged = idx.merge(av, on="window_id", how="left")
    # NaN after the left-join means "not core-available"; coerce to bool without
    # pandas' fillna-downcast path (which raises a FutureWarning on object dtype).
    col = merged["core_hr_spo2_available"]
    merged["core_hr_spo2_available"] = np.where(col.isna(), False, col).astype(bool)
    core = merged[
        merged["label_status"].isin(("positive", "negative"))
        & merged["core_hr_spo2_available"]
    ].copy()
    core["binary_event_label"] = core["binary_event_label"].astype(int)
    core["patient_id"] = core["patient_id"].map(_canonical_pid)
    return core[["window_id", "patient_id", "binary_event_label"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# main gate
# ---------------------------------------------------------------------------

def verify_gate(
    *,
    cfg: Config,
    config,
    split_run_id_override: Optional[str] = None,
) -> DataGateResult:
    """Run the full input gate. Raises :class:`DataGateError` if refused."""
    splits_dir = cfg.path("splits")
    latest_run_path = splits_dir / "LATEST_RUN.txt"
    checks: List[Check] = []

    # 1. license gate (fixed convention; the license gate writes this mirror)
    checks.append(_verify_license(cfg, "reports/data_audit/license_gate_passed.md"))

    # 2. resolve split run id
    split_run_id, ck = _resolve_split_run_id(
        splits_dir=splits_dir, latest_run_path=latest_run_path,
        override=split_run_id_override,
    )
    checks.append(ck)
    if config.reject_v1 and "stage6-patient-splits-" in split_run_id:
        # v1 run ids are 'stage6-patient-splits-...'; v2 are 'stage6b-split-balance-v2-...'
        raise DataGateError(
            f"BLOCKED: split run {split_run_id!r} is a v1 patient-split run; the "
            f"core-CSV baseline requires the approved v2 split-balance run."
        )

    # 3. signature load + approval
    signature, sig_path = _load_signature(splits_dir=splits_dir, split_run_id=split_run_id)
    checks.append(Check(
        "signature_run_id_matches", signature.get("run_id") == split_run_id,
        f"signature.run_id={signature.get('run_id')!r} vs resolved={split_run_id!r}",
    ))
    approval = signature.get("approval_status")
    checks.append(Check(
        "signature_approved", approval == S.APPROVAL_REQUIRED,
        f"approval_status={approval!r} (required {S.APPROVAL_REQUIRED!r})",
    ))

    # 4. v2 fold-file records (sha256 + rows + schema) -> also loads the tables
    v2_files = signature.get("v2_files", {}) or {}
    outer_folds: Optional[pd.DataFrame] = None
    inner_folds: Optional[pd.DataFrame] = None
    for key, record in v2_files.items():
        relpath = record.get("relpath", "")
        ckf, df, _ = _verify_file_record(
            splits_dir=splits_dir, split_run_id=split_run_id, key=key, record=record,
        )
        checks.append(ckf)
        if relpath == "outer_patient_folds_core_v2.csv" and df is not None:
            outer_folds = df
        elif relpath == "inner_patient_folds_core_v2.parquet" and df is not None:
            inner_folds = df
    checks.append(Check(
        "v2_fold_tables_loaded",
        outer_folds is not None and inner_folds is not None,
        f"outer={0 if outer_folds is None else len(outer_folds)} "
        f"inner={0 if inner_folds is None else len(inner_folds)} rows",
    ))
    # canonicalize patient_id on the fold tables (read_csv strips leading zeros)
    if outer_folds is not None:
        outer_folds = _canonicalize_pid_column(outer_folds)
    if inner_folds is not None:
        inner_folds = _canonicalize_pid_column(inner_folds)

    # 5. lineage run ids vs actual file run_id columns
    lineage = signature.get("lineage", {}) or {}
    s4_expected = lineage.get("stage4_run_id")
    s5_expected = lineage.get("stage5_run_id")
    stage4_path = cfg.path("data_manifests") / "csv_window_index.parquet"
    hr_path = cfg.path("features_physiology") / "hr_window_features.parquet"
    spo2_path = cfg.path("features_physiology") / "spo2_window_features.parquet"
    av_path = cfg.path("features_physiology") / "physiology_feature_availability.parquet"
    for p in (stage4_path, hr_path, spo2_path, av_path):
        if not p.is_file():
            raise DataGateError(f"Required Stage-4/5 input missing: {p}")
    stage4_index = pd.read_parquet(stage4_path)
    hr_feat = pd.read_parquet(hr_path)
    spo2_feat = pd.read_parquet(spo2_path)
    stage5_av = pd.read_parquet(av_path)
    s4_got = str(stage4_index["run_id"].iloc[0]) if "run_id" in stage4_index.columns else None
    s5_got = str(hr_feat["run_id"].iloc[0]) if "run_id" in hr_feat.columns else None
    s5_got2 = str(spo2_feat["run_id"].iloc[0]) if "run_id" in spo2_feat.columns else None
    checks.append(Check(
        "lineage_stage4_run_id_matches",
        s4_got == s4_expected,
        f"stage4 run_id file={s4_got!r} sig={s4_expected!r}",
    ))
    checks.append(Check(
        "lineage_stage5_run_id_matches",
        s5_got == s5_expected and s5_got2 == s5_expected,
        f"stage5 run_id hr={s5_got!r} spo2={s5_got2!r} sig={s5_expected!r}",
    ))

    # 6. runtime cohort re-derivation (independent of Stage-6 membership table)
    cohort_windows = _rederive_cohort(
        stage4_index=stage4_index, stage5_availability=stage5_av,
    )
    sig_cohort = signature.get("cohort", {}) or {}
    n_windows = int(len(cohort_windows))
    n_patients = int(cohort_windows["patient_id"].nunique())
    n_positive = int((cohort_windows["binary_event_label"] == 1).sum())
    n_negative = int((cohort_windows["binary_event_label"] == 0).sum())
    sorted_pids = sorted(str(p) for p in cohort_windows["patient_id"].unique())
    patient_set_sha = _sha256_lines(sorted_pids)
    checks.append(Check(
        "cohort_patient_set_sha256_matches",
        patient_set_sha == sig_cohort.get("patient_set_sha256"),
        f"runtime {patient_set_sha[:12]}.. vs sig {str(sig_cohort.get('patient_set_sha256'))[:12]}..",
    ))
    checks.append(Check(
        "cohort_window_count_matches",
        n_windows == int(sig_cohort.get("n_windows", -1)),
        f"runtime n_windows={n_windows} vs sig={sig_cohort.get('n_windows')}",
    ))
    checks.append(Check(
        "cohort_patient_count_matches",
        n_patients == int(sig_cohort.get("n_patients", -1)),
        f"runtime n_patients={n_patients} vs sig={sig_cohort.get('n_patients')}",
    ))
    checks.append(Check(
        "cohort_label_counts_match",
        n_positive == int(sig_cohort.get("n_positive", -1))
        and n_negative == int(sig_cohort.get("n_negative", -1)),
        f"runtime pos={n_positive} neg={n_negative} vs sig "
        f"pos={sig_cohort.get('n_positive')} neg={sig_cohort.get('n_negative')}",
    ))

    # 7. assignment hash recomputed from outer folds
    if outer_folds is None:
        raise DataGateError("outer_patient_folds_core_v2.csv could not be loaded.")
    of = outer_folds.sort_values(["patient_id"]).reset_index(drop=True)
    assign_lines = [
        f"{str(pid)}:{int(fold)}" for pid, fold in zip(of["patient_id"], of["outer_fold"])
    ]
    assign_sha = _sha256_lines(assign_lines)
    sig_split = signature.get("split", {}) or {}
    checks.append(Check(
        "assignment_sha256_matches",
        assign_sha == sig_split.get("assignment_sha256"),
        f"runtime {assign_sha[:12]}.. vs sig {str(sig_split.get('assignment_sha256'))[:12]}..",
    ))

    # 8. fold-structure checks (each patient exactly one outer fold; inner
    #    folds never contain an outer-test patient)
    dup = of["patient_id"].duplicated().any()
    checks.append(Check(
        "each_patient_one_outer_fold", not dup,
        f"duplicated_patient_in_outer={bool(dup)}; "
        f"folds_used={sorted(of['outer_fold'].unique().tolist())}",
    ))
    union_ok = set(of["patient_id"]) == set(cohort_windows["patient_id"])
    checks.append(Check(
        "outer_patient_union_equals_cohort", union_ok,
        f"outer_minus_cohort={sorted(set(of['patient_id']) - set(cohort_windows['patient_id']))} "
        f"cohort_minus_outer={sorted(set(cohort_windows['patient_id']) - set(of['patient_id']))}",
    ))
    # inner leakage: for each outer fold k, inner rows of that outer_train must
    # not include any patient whose outer fold == k.
    if inner_folds is None:
        raise DataGateError("inner_patient_folds_core_v2.parquet could not be loaded.")
    pid_to_outer = dict(zip(of["patient_id"], of["outer_fold"]))
    leaks: List[str] = []
    inner_folds_checked = inner_folds.copy()
    inner_folds_checked["_patient_outer"] = inner_folds_checked["patient_id"].map(pid_to_outer)
    for outer_k in sorted(of["outer_fold"].unique()):
        rows = inner_folds_checked[inner_folds_checked["outer_fold"] == outer_k]
        # the patient's OWN outer fold must differ from this outer_train's k
        bad = rows[rows["_patient_outer"] == outer_k]
        if len(bad):
            leaks.append(f"outer_fold={outer_k}: {len(bad)} inner rows belong to outer-test patients")
    checks.append(Check(
        "inner_folds_exclude_outer_test_patients", len(leaks) == 0,
        "; ".join(leaks) if leaks else "no outer-test patient in any inner fold",
    ))
    # inner folds per outer_train must be exactly {0,1,2,3}
    inner_folds_ok = True
    inner_detail_parts: List[str] = []
    for outer_k in sorted(of["outer_fold"].unique()):
        iv = sorted(inner_folds_checked.loc[
            inner_folds_checked["outer_fold"] == outer_k, "inner_validation_fold"
        ].unique().tolist())
        if iv != list(range(S.INNER_N_FOLDS)):
            inner_folds_ok = False
        inner_detail_parts.append(f"outer{outer_k}={iv}")
    checks.append(Check(
        "inner_has_four_folds_per_outer_train", inner_folds_ok,
        "; ".join(inner_detail_parts),
    ))

    # ---- decision ----
    reasons = [f"{c.name}: {c.detail}" for c in checks if not c.passed]
    approved = all(c.passed for c in checks)
    if not approved:
        raise DataGateError(
            "STAGE-7 INPUT GATE BLOCKED. The frozen v2 split did not verify. "
            "No fallback to v1, no self-built/modified split. Failures:\n  - "
            + "\n  - ".join(reasons)
        )
    return DataGateResult(
        approved=True,
        split_run_id=split_run_id,
        signature=signature,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        cohort_windows=cohort_windows,
        checks=checks,
        reasons=[],
    )


__all__ = [
    "DataGateError",
    "Check",
    "DataGateResult",
    "verify_gate",
]
