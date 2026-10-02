"""Immutable v2 cohort/split version signature (Stage 6B).

Pins the full provenance of a v2 balance split:

* the transitive Stage 4/5/6-v1 lineage (copied from the v1
  ``input_version_signature.json`` — run ids, config hashes, Stage4/5 file SHAs);
* the v1 comparator files Stage 6B read (v1 outer folds + airflow inheritance +
  the v1 signature itself), with rows/schema/SHA-256;
* the v2 product files (rows/schema/SHA-256);
* the cohort fingerprint (sorted patient set + counts), the deterministic
  assignment fingerprint, the seed/algorithm/selection and the resolved config
  hash, plus the gate approval status.

A downstream model stage may accept ONLY a signature with
``approval_status == approved_for_modeling``; an unapproved or old v1 signature
must be rejected as the default modeling input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..run_metadata import sha256_text
from .split_balance_config import ResolvedSplitBalanceConfig


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_fingerprint(df: pd.DataFrame) -> str:
    pairs: List[str] = []
    for c in df.columns:
        dt = pd.api.types.pandas_dtype(df[c].dtype)
        pairs.append(f"{c}:{getattr(dt, 'char', str(dt))}")
    return sha256_text("\n".join(sorted(pairs)))


def _file_record(path: Path) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"relpath": path.name, "sha256": _sha256_file(path)}
    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
            rec["rows"] = int(df.shape[0]); rec["schema_fingerprint"] = _schema_fingerprint(df)
        except Exception:
            rec["rows"] = -1; rec["schema_fingerprint"] = "unreadable_parquet"
    elif path.suffix == ".csv":
        try:
            df = pd.read_csv(path)
            rec["rows"] = int(df.shape[0]); rec["schema_fingerprint"] = _schema_fingerprint(df)
        except Exception:
            rec["rows"] = -1; rec["schema_fingerprint"] = "unreadable_csv"
    elif path.suffix in (".yaml", ".yml"):
        rec["rows"] = int(sum(1 for _ in path.open("r", encoding="utf-8")))
        rec["schema_fingerprint"] = "text"
    elif path.suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            rec["rows"] = int(len(obj)) if isinstance(obj, dict) else int(len(obj))
        except Exception:
            rec["rows"] = -1
        rec["schema_fingerprint"] = "json"
    else:
        rec["rows"] = int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")))
        rec["schema_fingerprint"] = "text"
    return rec


def _load_v1_signature(v1_run_dir: Path) -> Dict[str, Any]:
    p = v1_run_dir / "input_version_signature.json"
    if not p.is_file():
        return {"_v1_signature_file": str(p), "present": False}
    return json.loads(p.read_text(encoding="utf-8"))


def build_split_balance_signature(
    *,
    v1_run_dir: Path,
    v2_product_dir: Path,
    v2_filenames: List[str],
    config: ResolvedSplitBalanceConfig,
    run_id: str,
    config_hash: str,
    patient_ids: List[str],
    patient_to_fold: Dict[str, int],
    n_windows: int,
    n_positive: int,
    n_negative: int,
    positive_rate_spread: float,
    window_count_cv: float,
    approval_status: str,
) -> Dict[str, Any]:
    """Assemble the immutable v2 split-balance version signature dict."""
    try:
        import sklearn  # type: ignore
        sklearn_version = str(getattr(sklearn, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        sklearn_version = "not_installed"

    v1_sig = _load_v1_signature(v1_run_dir)

    # v1 comparator files actually read by Stage 6B
    v1_comparator: Dict[str, Any] = {}
    for name in ("outer_patient_folds_core.csv", "airflow_outer_fold_inheritance.csv",
                 "input_version_signature.json"):
        p = v1_run_dir / name
        if p.is_file():
            v1_comparator[f"v1/{name}"] = _file_record(p)

    # v2 product files
    v2_files: Dict[str, Any] = {}
    for name in v2_filenames:
        p = v2_product_dir / name
        if p.is_file():
            v2_files[f"v2/{name}"] = _file_record(p)

    sorted_pids = sorted(str(p) for p in patient_ids)
    assignment_lines = [f"{pid}:{int(patient_to_fold[pid])}" for pid in sorted_pids]

    sig = {
        "run_id": run_id,
        "patient_split_stage": "stage6b",
        "split_balance_version": config.split_balance_version,
        "approval_status": approval_status,
        "config_hash": config_hash,
        "sklearn_version": sklearn_version,
        # transitive lineage copied from the v1 signature (Stage 4/5 provenance)
        "lineage": {
            "v1_comparator_run_id": config.v1_comparator_run_id,
            "stage4_run_id": v1_sig.get("stage4_run_id", ""),
            "stage5_run_id": v1_sig.get("stage5_run_id", ""),
            "stage3_run_id": v1_sig.get("stage3_run_id", ""),
            "stage4_config_hash": v1_sig.get("stage4_config_hash", ""),
            "stage5_config_hash": v1_sig.get("stage5_config_hash", ""),
            "lineage_stage5_matches_stage4": v1_sig.get("lineage_stage5_matches_stage4"),
            "v1_signature_present": v1_sig.get("present", True),
        },
        "v1_comparator_files": v1_comparator,
        "v2_files": v2_files,
        "cohort": {
            "n_patients": len(sorted_pids),
            "n_windows": int(n_windows),
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "overall_positive_rate": float(n_positive / n_windows) if n_windows else 0.0,
            "patient_set_sha256": sha256_text("\n".join(sorted_pids)),
        },
        "split": {
            "algorithm": config.algorithm,
            "base_seed": config.base_seed,
            "n_candidate_seeds": config.n_candidate_seeds,
            "local_swap_iterations": config.local_swap_iterations,
            "greedy_seed_sort_key": config.greedy_seed_sort_key,
            "selection_rule": config.selection_rule,
            "outer_n_folds": config.outer_n_folds,
            "patients_per_outer_fold": config.patients_per_outer_fold,
            "assignment_sha256": sha256_text("\n".join(assignment_lines)),
            "positive_rate_spread": float(positive_rate_spread),
            "window_count_cv": float(window_count_cv),
        },
        "objective": {
            "components": ["positive_rate_spread", "window_count_cv"],
            "weights": {
                "positive_rate_spread": config.weight_positive_rate_spread,
                "window_count_cv": config.weight_window_count_cv,
            },
            "aggregate": config.objective_aggregate,
            "primary_gate_metric": config.primary_gate_metric,
        },
        "audio_block": {
            "audio_cohort_count": config.audio_cohort_count,
            "audio_block_reason": config.audio_block_reason,
            "read_wav_forbidden": config.read_wav_forbidden,
        },
        "deferred_transforms": {
            "no_model_matrix": config.no_model_matrix,
            "no_imputation": config.no_imputation,
            "no_normalization": config.no_normalization,
            "no_feature_selection": config.no_feature_selection,
            "no_class_resampling": config.no_class_resampling,
            "no_threshold_optimization": config.no_threshold_optimization,
            "no_performance_metrics": config.no_performance_metrics,
        },
    }
    return sig


def write_signature(path: Path, sig: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sig, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["build_split_balance_signature", "write_signature"]
