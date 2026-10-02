"""Stage-9 frozen-input verification (the strict BLOCKED gate) + input signature.

This verifies that the frozen v2 split AND the fixed Stage 7 / Stage 8 runs are
intact and mutually consistent, by RE-RUNNING the Stage 7 / Stage 8 input gates
(re-derived cohorts, hashes, audio=0) and cross-checking their stored
signatures. Any mismatch -> :class:`RobustnessInputError` (BLOCKED). Nothing is
trusted blindly; nothing in Stage 1-8 history is modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ...config import Config
from ...modeling import schema as MS
from ...modeling.data_gate import DataGateError, DataGateResult, verify_gate
from ...modeling.airflow_increment.data_gate import (
    AirflowGateError,
    AirflowGateResult,
    verify_airflow_gate,
)
from . import schema as S
from .config import ResolvedRobustnessConfig


class RobustnessInputError(RuntimeError):
    """Raised when a frozen input is missing, tampered or inconsistent (BLOCKED)."""


@dataclass
class FrozenArtifacts:
    """Handles to the read-only Stage 7 / Stage 8 run products."""

    stage7_run_dir: Path
    stage8_run_dir: Path
    stage7_oof: pd.DataFrame
    stage7_fold_metrics: pd.DataFrame
    stage7_pooled: Dict[str, Any]
    stage7_bootstrap: pd.DataFrame
    stage7_signature: Dict[str, Any]
    stage8_paired_oof: pd.DataFrame
    stage8_fold_metrics: pd.DataFrame
    stage8_pooled: Dict[str, Any]
    stage8_bootstrap: pd.DataFrame
    stage8_signature: Dict[str, Any]


@dataclass
class RobustnessInput:
    approved: bool
    split_run_id: str
    split_signature: Dict[str, Any]
    core_gate: DataGateResult
    airflow_gate: AirflowGateResult
    frozen: FrozenArtifacts
    checks: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RobustnessInputError(f"missing frozen artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(ok), "detail": detail}


def _verify_signature_block(
    *, label: str, sig: Dict[str, Any], mc: ResolvedRobustnessConfig,
    checks: List[Dict[str, Any]],
) -> None:
    """Cross-check a frozen Stage-7/8 model-input signature against the rules."""
    consumed = sig.get("consumed_split_run_id")
    ok = consumed == mc.expected_consumed_split_run_id
    checks.append(_check(f"{label}_consumed_split",
                         ok, f"consumed_split_run_id={consumed}"))
    if not ok:
        raise RobustnessInputError(
            f"{label} did not consume the expected frozen split "
            f"{mc.expected_consumed_split_run_id!r} (got {consumed!r})")

    approval = sig.get("split_approval_status")
    ok = approval == S.APPROVAL_REQUIRED
    checks.append(_check(f"{label}_approval", ok, f"approval={approval}"))
    if not ok:
        raise RobustnessInputError(f"{label} split not approved_for_modeling")

    ab = sig.get("audio_block", {})
    ok = (ab.get("audio_cohort_count") == S.AUDIO_COHORT_COUNT
          and ab.get("read_wav_forbidden") is True)
    checks.append(_check(f"{label}_audio_block", ok, f"audio_block={ab}"))
    if not ok:
        raise RobustnessInputError(f"{label} audio block not enforced in signature")

    assign = sig.get("split_assignment_sha256")
    pset = sig.get("split_patient_set_sha256")
    ok = (assign == mc.expected_assignment_sha256
          and pset == mc.expected_patient_set_sha256)
    checks.append(_check(f"{label}_split_hashes", ok,
                         f"assign={assign} patient_set={pset}"))
    if not ok:
        raise RobustnessInputError(
            f"{label} split hashes do not match the frozen v2 fingerprints")


def _load_stage7(*, results_root: Path, run_id: str) -> FrozenArtifacts_stage7:
    rdir = results_root / "runs" / run_id
    if not rdir.is_dir():
        raise RobustnessInputError(f"missing Stage 7 run dir: {rdir}")
    sig = _read_json(rdir / "model_input_signature.json")
    try:
        oof = pd.read_parquet(rdir / "oof_predictions.parquet")
        fold = pd.read_csv(rdir / "outer_fold_metrics.csv")
        pooled = _read_json(rdir / "pooled_metrics.json")
        boot = pd.read_csv(rdir / "patient_cluster_bootstrap_ci.csv")
    except Exception as exc:
        raise RobustnessInputError(f"cannot load Stage 7 frozen artifacts: {exc}") from exc
    # schema sanity (Stage-7 OOF carries no cohort/feature_set/threshold_rule
    # columns -- those are Stage-8 concepts; analysis_role is added by Stage 9)
    required = set(S.OOF_ALLOWED_COLUMNS) - {
        "analysis_role", "cohort", "feature_set", "threshold_rule"}
    missing = required - set(oof.columns)
    if missing:
        raise RobustnessInputError(f"Stage 7 OOF missing columns {sorted(missing)}")
    return rdir, sig, oof, fold, pooled, boot


# placeholder type alias to keep the loader readable (returned tuple unpacked)
FrozenArtifacts_stage7 = Any


def _load_stage8(*, results_root: Path, run_id: str):
    rdir = results_root / "runs" / run_id
    if not rdir.is_dir():
        raise RobustnessInputError(f"missing Stage 8 run dir: {rdir}")
    sig = _read_json(rdir / "airflow_model_input_signature.json")
    try:
        paired = pd.read_parquet(rdir / "airflow_paired_oof_predictions.parquet")
        fold = pd.read_csv(rdir / "airflow_outer_fold_metrics.csv")
        pooled = _read_json(rdir / "airflow_pooled_metrics.json")
        boot = pd.read_csv(rdir / "airflow_paired_patient_bootstrap_ci.csv")
    except Exception as exc:
        raise RobustnessInputError(f"cannot load Stage 8 frozen artifacts: {exc}") from exc
    return rdir, sig, paired, fold, pooled, boot


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def verify_robustness_inputs(
    *, cfg: Config, mc: ResolvedRobustnessConfig,
    split_run_id_override: str | None = None,
) -> RobustnessInput:
    """Verify the frozen split + Stage 7/8 runs; return the loaded inputs."""
    checks: List[Dict[str, Any]] = []

    # Layer 1+2: re-run the Stage 7 / Stage 8 gates (re-derive cohorts, hashes)
    try:
        core_gate = verify_gate(cfg=cfg, config=mc,
                                split_run_id_override=split_run_id_override)
    except DataGateError as exc:
        checks.append(_check("core_gate", False, str(exc)))
        raise RobustnessInputError(f"core (Stage 7) gate failed: {exc}") from exc
    try:
        airflow_gate = verify_airflow_gate(cfg=cfg, config=mc,
                                           split_run_id_override=split_run_id_override)
    except AirflowGateError as exc:
        checks.append(_check("airflow_gate", False, str(exc)))
        raise RobustnessInputError(f"airflow (Stage 8) gate failed: {exc}") from exc

    split_run_id = core_gate.split_run_id
    ok = airflow_gate.split_run_id == split_run_id
    checks.append(_check("gates_agree_on_split", ok,
                         f"core={split_run_id} airflow={airflow_gate.split_run_id}"))
    if not ok:
        raise RobustnessInputError("core and airflow gates disagree on the split run id")

    ok = split_run_id == mc.expected_consumed_split_run_id
    checks.append(_check("split_matches_expected", ok, f"split={split_run_id}"))
    if not ok:
        raise RobustnessInputError(
            f"split {split_run_id!r} != expected {mc.expected_consumed_split_run_id!r}")

    # frozen run products
    results_root = cfg.path("results")
    s7_dir, s7_sig, s7_oof, s7_fold, s7_pooled, s7_boot = _load_stage7(
        results_root=results_root, run_id=mc.stage7_run_id)
    s8_dir, s8_sig, s8_paired, s8_fold, s8_pooled, s8_boot = _load_stage8(
        results_root=results_root, run_id=mc.stage8_run_id)

    _verify_signature_block(label="stage7", sig=s7_sig, mc=mc, checks=checks)
    _verify_signature_block(label="stage8", sig=s8_sig, mc=mc, checks=checks)

    # row-count sanity (read at runtime, never hard-coded as the truth)
    n_s7_models = s7_oof["model"].nunique()
    n_s7_windows = int(s7_oof["window_id"].nunique())
    checks.append(_check("stage7_oof_rows",
                         s7_oof.shape[0] == n_s7_models * n_s7_windows,
                         f"rows={s7_oof.shape[0]} models={n_s7_models} windows={n_s7_windows}"))
    checks.append(_check("stage8_paired_rows",
                         s8_paired.shape[0] == int(airflow_gate.airflow_cohort["n_windows"]),
                         f"rows={s8_paired.shape[0]} "
                         f"airflow_windows={airflow_gate.airflow_cohort['n_windows']}"))

    frozen = FrozenArtifacts(
        stage7_run_dir=s7_dir, stage8_run_dir=s8_dir,
        stage7_oof=s7_oof, stage7_fold_metrics=s7_fold,
        stage7_pooled=s7_pooled, stage7_bootstrap=s7_boot,
        stage7_signature=s7_sig,
        stage8_paired_oof=s8_paired, stage8_fold_metrics=s8_fold,
        stage8_pooled=s8_pooled, stage8_bootstrap=s8_boot,
        stage8_signature=s8_sig,
    )

    failed = [c for c in checks if not c["passed"]]
    return RobustnessInput(
        approved=not failed, split_run_id=split_run_id,
        split_signature=core_gate.signature, core_gate=core_gate,
        airflow_gate=airflow_gate, frozen=frozen, checks=checks,
    )


def build_robustness_input_signature(
    *, run_id: str, mc: ResolvedRobustnessConfig, rin: RobustnessInput,
    determinism: Dict[str, Any], analyses_executed: Dict[str, str],
) -> Dict[str, Any]:
    """Build the ``robustness_input_signature.json`` content."""
    cg = rin.core_gate
    ag = rin.airflow_gate
    return {
        "run_id": run_id,
        "modeling_version": S.MODELING_VERSION,
        "stage": S.ROBUSTNESS_STAGE,
        "task_name": S.TASK_NAME,
        "cohort": S.COHORT,
        "consumed_split_run_id": rin.split_run_id,
        "split_approval_status": rin.split_signature.get("approval_status"),
        "split_assignment_sha256": (rin.split_signature.get("split", {}) or {}).get("assignment_sha256"),
        "split_patient_set_sha256": (rin.split_signature.get("cohort", {}) or {}).get("patient_set_sha256"),
        "frozen_inputs": {
            "stage7_run_id": mc.stage7_run_id,
            "stage8_run_id": mc.stage8_run_id,
            "stage7_signature_verified": True,
            "stage8_signature_verified": True,
        },
        "runtime_cohorts": {
            "core": {
                "n_windows": int(len(cg.cohort_windows)),
                "n_patients": int(cg.cohort_windows["patient_id"].nunique()),
                "n_positive": int((cg.cohort_windows["binary_event_label"] == 1).sum()),
                "n_negative": int((cg.cohort_windows["binary_event_label"] == 0).sum()),
            },
            "airflow": {
                "n_windows": int(ag.airflow_cohort["n_windows"]),
                "n_patients": int(ag.airflow_cohort["n_patients"]),
                "n_positive": int(ag.airflow_cohort["n_positive"]),
                "n_negative": int(ag.airflow_cohort["n_negative"]),
            },
        },
        "rerun_determinism": determinism,
        "analyses_executed": analyses_executed,
        "analysis_roles": {
            "primary_reproduction": S.ROLE_PRIMARY_REPLICATION,
            "all_others": S.ROLE_SENSITIVITY,
        },
        "audio_block": {
            "audio_cohort_count": S.AUDIO_COHORT_COUNT,
            "read_wav_forbidden": True,
            "audio_features_in_matrix": False,
            "airflow_in_core_matrix": False,
        },
        "input_checks": rin.checks,
    }


__all__ = [
    "RobustnessInputError",
    "FrozenArtifacts",
    "RobustnessInput",
    "verify_robustness_inputs",
    "build_robustness_input_signature",
]
