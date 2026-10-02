"""Shared synthetic + real-data fixtures for Stage-6B (split balance) tests.

Stage 6B consumes ONLY the Stage-6 v1 patient-level burden summary
(``outer_patient_folds_core.csv``), the v1 airflow inheritance, and the v1
``input_version_signature.json``. It reads no raw CSV/WAV and no feature value.

These fixtures provide:

* a synthetic 50-patient balanceable burden (optimizer unit tests);
* a synthetic 50-patient UN-balanceable burden (one dominant positive patient ->
  forces ``not_approved_balance_target_not_met``);
* a writer for a synthetic v1 run-dir (outer folds + airflow + signature stub)
  under an isolated Config, for production-mode mirror/LATEST tests;
* a harness to construct + run ``Stage6bRunner`` (isolated or production mode).

No real patient data is used in the synthetic fixtures (all content is fabricated).
The real-data production audit lives in ``test_stage6b_production_audit.py`` and
reads the genuine v1 run-dir (read-only).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import pandas as pd

from psg_audio_benchmark.config import Config
from psg_audio_benchmark.evaluation import Stage6bOptions, Stage6bRunner
from psg_audio_benchmark.evaluation.balance_optimizer import PatientBurden
from psg_audio_benchmark.evaluation.split_balance_config import (
    ResolvedSplitBalanceConfig,
    load_split_balance_config,
)

REAL_V1_RUN_ID = "stage6-patient-splits-20260808T002136Z"
S6B_CONFIG_HASH = "abcd" * 16  # not a test token for the contamination guard


# ---------------------------------------------------------------------------
# Synthetic burdens
# ---------------------------------------------------------------------------

def synthetic_balanceable_burden(n: int = 50) -> List[PatientBurden]:
    """50 fabricated patients, varied window counts + positive rates, balanceable.

    Deterministic (no RNG): patient i gets a window count from a fixed formula and
    a positive rate from a fixed cycle, so the optimizer has rich structure to
    balance and tests are reproducible.
    """
    rates = [0.05, 0.12, 0.22, 0.31, 0.40, 0.48, 0.55, 0.63, 0.70, 0.78]
    out: List[PatientBurden] = []
    for i in range(n):
        pid = f"{i+1:02d}"
        n_windows = 60 + (i * 37 % 900) + (i % 7)
        rate = rates[i % len(rates)]
        rate = min(0.95, max(0.0, rate + ((i % 5) - 2) * 0.03))
        n_pos = int(round(n_windows * rate))
        out.append(PatientBurden(
            patient_id=pid, n_windows=n_windows, n_positive=n_pos,
            n_negative=n_windows - n_pos, positive_rate=n_pos / n_windows,
        ))
    return out


def synthetic_unbalanceable_burden(n: int = 50) -> List[PatientBurden]:
    """50 fabricated patients where ONE patient carries almost all the positive
    burden -> any fold holding it has a far higher positive rate, so the
    pre-registered spread <= 0.10 gate cannot be met (forces not_approved)."""
    out: List[PatientBurden] = []
    for i in range(n):
        pid = f"{i+1:02d}"
        if i == 0:
            n_windows, n_pos = 12000, 12000  # dominant all-positive patient
        else:
            n_windows = 500 + (i * 17 % 400)
            n_pos = int(round(n_windows * 0.20))
        out.append(PatientBurden(
            patient_id=pid, n_windows=n_windows, n_positive=n_pos,
            n_negative=n_windows - n_pos, positive_rate=n_pos / n_windows,
        ))
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_6b_config(project_root: Path) -> ResolvedSplitBalanceConfig:
    return load_split_balance_config(project_root / "config" / "split_balance_optimization.yaml")


def config_with_expected(
    config: ResolvedSplitBalanceConfig, *, patients: int, windows: int, positive: int,
) -> ResolvedSplitBalanceConfig:
    """Return a copy of the config with the cohort-identity expectations overridden
    (used by synthetic-world tests so the cohort-identity hard check reflects the
    fabricated cohort, isolating the balance gate under test)."""
    return replace(
        config, expected_core_patients=patients,
        expected_core_windows=windows, expected_core_positive=positive,
        expected_core_negative=windows - positive,
    )


def fast_config(config: ResolvedSplitBalanceConfig) -> ResolvedSplitBalanceConfig:
    """Smaller search budget for fast unit/runner tests (still deterministic)."""
    return replace(config, n_candidate_seeds=4, local_swap_iterations=6000)


# ---------------------------------------------------------------------------
# Synthetic v1 run-dir writer
# ---------------------------------------------------------------------------

def write_synthetic_v1_run(
    cfg: Config, run_id: str, burden: List[PatientBurden],
    *, airflow_pids: Optional[List[str]] = None,
) -> Path:
    """Write a fabricated v1 run-dir (outer folds + airflow inheritance + signature
    stub) under ``<cfg splits>/runs/<run_id>``. The v1 fold assignment is a trivial
    round-robin (it is only the immutable comparator; 6B recomputes v2)."""
    v1_dir = cfg.path("splits") / "runs" / run_id
    v1_dir.mkdir(parents=True, exist_ok=True)

    sorted_burden = sorted(burden, key=lambda x: x.patient_id)
    by_pid = {b.patient_id: b for b in burden}

    # round-robin v1 outer folds
    rows = []
    pid_to_v1fold = {}
    for i, b in enumerate(sorted_burden):
        pid_to_v1fold[b.patient_id] = i % 5
        rows.append({
            "run_id": run_id, "patient_id": b.patient_id, "outer_fold": i % 5,
            "n_windows": b.n_windows, "n_positive": b.n_positive,
            "n_negative": b.n_negative, "positive_rate": b.positive_rate,
        })
    pd.DataFrame(rows).to_csv(v1_dir / "outer_patient_folds_core.csv", index=False)

    # airflow inheritance (subset of core patients)
    if airflow_pids is None:
        airflow_pids = [b.patient_id for b in sorted_burden][: max(5, len(sorted_burden) // 5)]
    af_rows = []
    for pid in airflow_pids:
        b = by_pid[pid]
        aw = max(1, int(b.n_windows * 0.8)); ap = int(round(b.n_positive * 0.8))
        af_rows.append({
            "run_id": run_id, "patient_id": pid, "outer_fold": pid_to_v1fold[pid],
            "n_airflow_windows": aw, "n_airflow_positive": ap,
            "n_airflow_negative": aw - ap, "airflow_positive_rate": ap / aw,
        })
    pd.DataFrame(af_rows).to_csv(v1_dir / "airflow_outer_fold_inheritance.csv", index=False)

    # v1 signature stub (lineage provenance only)
    stub = {
        "run_id": run_id, "patient_split_stage": "stage6",
        "stage4_run_id": "stage4-synth-6b", "stage5_run_id": "stage5-synth-6b",
        "stage3_run_id": "stage3-synth-6b",
        "stage4_config_hash": "4" * 64, "stage5_config_hash": "5" * 64,
        "lineage_stage5_input_stage4_run_id": "stage4-synth-6b",
        "lineage_stage5_matches_stage4": True,
        "sklearn_version": "synthetic",
        "inputs": {
            "stage4/csv_window_index.parquet": {"relpath": "synthetic", "rows": 1,
                                                "schema_fingerprint": "x", "sha256": "a" * 64},
            "stage5/physiology_feature_availability.parquet": {"relpath": "synthetic", "rows": 1,
                                                               "schema_fingerprint": "y", "sha256": "b" * 64},
        },
    }
    (v1_dir / "input_version_signature.json").write_text(
        json.dumps(stub, indent=2, sort_keys=True), encoding="utf-8")
    return v1_dir


# ---------------------------------------------------------------------------
# Runner harness
# ---------------------------------------------------------------------------

def run_6b(
    cfg: Config, *, run_id: str, config: ResolvedSplitBalanceConfig,
    output_root: Optional[Path] = None, relpath_base: Optional[Path] = None,
    dry_run: bool = False, n_cand: Optional[int] = None, iters: Optional[int] = None,
):
    options = Stage6bOptions(
        dry_run=dry_run, output_root=output_root, relpath_base=relpath_base,
        config_path=cfg.project_root / "config" / "split_balance_optimization.yaml",
        run_id=run_id, config_hash=S6B_CONFIG_HASH,
        n_candidate_seeds=n_cand, local_swap_iterations=iters,
    )
    runner = Stage6bRunner(cfg=cfg, options=options)
    runner.config = config  # inject the (possibly overridden) resolved config
    runner.summary.n_candidate_seeds = config.n_candidate_seeds
    runner.summary.local_swap_iterations = config.local_swap_iterations
    summary = runner.run()
    return summary, runner


__all__ = [
    "REAL_V1_RUN_ID", "S6B_CONFIG_HASH",
    "synthetic_balanceable_burden", "synthetic_unbalanceable_burden",
    "load_6b_config", "config_with_expected", "fast_config",
    "write_synthetic_v1_run", "run_6b",
]
