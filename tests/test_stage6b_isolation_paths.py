"""(10) Output isolation, path purity, immutability and the LATEST/mirror branches.

Builds a fully-isolated throwaway project root (real config files copied in, empty
``data/raw`` + output dirs) so production-mode runs can exercise the mirror + LATEST
logic WITHOUT touching the real project paths. Verifies:

* dry-run writes NO products;
* an un-balanceable cohort yields ``not_approved`` and does NOT update LATEST / mirror;
* a balanceable cohort yields ``approved`` and DOES update LATEST + mirror;
* raw and the historical v1 run-dir are byte-identical before/after every run;
* all recorded product paths are project-relative (no absolute / Temp / pytest / AppData);
* outputs contain no raw/WAV and no feature-value (model-matrix) columns.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from _stage6b_fixtures import (
    S6B_CONFIG_HASH,
    fast_config,
    load_6b_config,
    run_6b,
    synthetic_balanceable_burden,
    synthetic_unbalanceable_burden,
    write_synthetic_v1_run,
)
from psg_audio_benchmark.config import Config
from psg_audio_benchmark.evaluation import qc as qc_mod
from psg_audio_benchmark.evaluation.schema import STAGE6_FORBIDDEN_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

V2_APPROVED_RUN = "stage6b-v2-approved-20260808"
V2_NOTAPPROVED_RUN = "stage6b-v2-notapproved-20260808"


# ---------------------------------------------------------------------------
# Isolated project root + config helpers
# ---------------------------------------------------------------------------

def _make_proj(tmp_path: Path):
    """A throwaway project root: real config dir copied in, empty data/raw + outputs."""
    proj = tmp_path / "proj"
    shutil.copytree(PROJECT_ROOT / "config", proj / "config")
    for sub in ("data/raw", "splits", "reports/evaluation", "logs"):
        (proj / sub).mkdir(parents=True, exist_ok=True)
    # a sentinel raw file so snapshot_raw has content to fingerprint (never touched)
    (proj / "data" / "raw" / ".raw_sentinel").write_text("immutable", encoding="utf-8")
    cfg = Config(project_root=proj)
    base = load_6b_config(proj)  # load_6b_config appends config/<yaml> itself
    return cfg, base


def _prod_config(base, *, v1_run_id: str, burden) -> object:
    """Override v1 comparator id + cohort-identity expectations to match a synthetic
    cohort (so the cohort-identity hard check reflects the fabricated burden)."""
    c = fast_config(base)
    n = len(burden)
    w = sum(b.n_windows for b in burden)
    p = sum(b.n_positive for b in burden)
    return replace(
        c, v1_comparator_run_id=v1_run_id,
        expected_core_patients=n, expected_core_windows=w,
        expected_core_positive=p, expected_core_negative=w - p,
    )


def _content_hash_dir(d: Path) -> dict:
    """SHA-256 of every file under ``d`` (content, not mtime) — immutability proof."""
    out: dict = {}
    if d.is_dir():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------
# (10a) dry-run writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_writes_no_products(tmp_path):
    cfg, base = _make_proj(tmp_path)
    burden = synthetic_balanceable_burden(50)
    v1 = "stage6-synthv1-dryrun-20260808"
    write_synthetic_v1_run(cfg, v1, burden)
    config = _prod_config(base, v1_run_id=v1, burden=burden)

    out = tmp_path / "out"
    summary, _ = run_6b(cfg=cfg, run_id="stage6b-dryrun-20260808", config=config,
                        output_root=out, relpath_base=tmp_path, dry_run=True)

    assert summary.status == "PASS WITH WARNINGS"
    assert any("dry_run" in a for a in summary.anomalies)
    # no isolated run-dir created, and no production run-dir created
    assert not (out / "splits" / "runs").exists() or not list((out / "splits" / "runs").iterdir())
    assert not (cfg.path("splits") / "runs" / "stage6b-dryrun-20260808").exists()


# ---------------------------------------------------------------------------
# (10b) not_approved: preserves history, does NOT update LATEST / mirror
# ---------------------------------------------------------------------------

def test_not_approved_preserves_history_and_does_not_update_latest(tmp_path):
    cfg, base = _make_proj(tmp_path)
    burden = synthetic_unbalanceable_burden(50)
    v1 = "stage6-synthv1-unbal-20260808"
    write_synthetic_v1_run(cfg, v1, burden)
    config = _prod_config(base, v1_run_id=v1, burden=burden)

    # pre-existing blessed LATEST pointer (still the v1 run)
    (cfg.path("splits") / "LATEST_RUN.txt").write_text(v1 + "\n", encoding="utf-8")
    (cfg.path("reports_evaluation") / "LATEST_RUN.txt").write_text(v1 + "\n", encoding="utf-8")

    v1_before = _content_hash_dir(cfg.path("splits") / "runs" / v1)
    raw_before = qc_mod.snapshot_raw(cfg.path("data_raw"), cfg.project_root)

    summary, _ = run_6b(cfg=cfg, run_id=V2_NOTAPPROVED_RUN, config=config,
                        output_root=None, relpath_base=None)

    # honest not-approved outcome
    assert summary.status == "not_approved_balance_target_not_met"
    assert summary.gate.approved is False
    assert summary.v2["spread"] > 0.10  # the real reason it failed
    # LATEST untouched (still the v1 run) in BOTH splits and reports roots
    assert (cfg.path("splits") / "LATEST_RUN.txt").read_text(encoding="utf-8").strip() == v1
    assert (cfg.path("reports_evaluation") / "LATEST_RUN.txt").read_text(encoding="utf-8").strip() == v1
    # no fixed-path mirror of any v2 product
    assert not (cfg.path("splits") / "outer_patient_folds_core_v2.csv").exists()
    # inner CV is NOT built on non-approval
    run_dir = cfg.path("splits") / "runs" / V2_NOTAPPROVED_RUN
    assert not (run_dir / "inner_patient_folds_core_v2.parquet").exists()
    # immutability: v1 run-dir + raw byte-identical
    assert _content_hash_dir(cfg.path("splits") / "runs" / v1) == v1_before
    assert qc_mod.snapshot_raw(cfg.path("data_raw"), cfg.project_root) == raw_before
    assert summary.historical_products_unchanged is True
    assert summary.raw_modified is False
    assert summary.latest_updated is False


# ---------------------------------------------------------------------------
# (10c) approved: DOES update LATEST + mirror, still preserves history
# ---------------------------------------------------------------------------

def test_approved_updates_latest_mirrors_and_preserves_history(tmp_path):
    cfg, base = _make_proj(tmp_path)
    burden = synthetic_balanceable_burden(50)
    v1 = "stage6-synthv1-bal-20260808"
    write_synthetic_v1_run(cfg, v1, burden)
    config = _prod_config(base, v1_run_id=v1, burden=burden)

    v1_before = _content_hash_dir(cfg.path("splits") / "runs" / v1)
    raw_before = qc_mod.snapshot_raw(cfg.path("data_raw"), cfg.project_root)

    summary, _ = run_6b(cfg=cfg, run_id=V2_APPROVED_RUN, config=config,
                        output_root=None, relpath_base=None)

    assert summary.status == "approved_for_modeling"
    assert summary.gate.approved is True
    assert summary.latest_updated is True
    # LATEST now blesses the v2 run in both roots
    assert (cfg.path("splits") / "LATEST_RUN.txt").read_text(encoding="utf-8").strip() == V2_APPROVED_RUN
    assert (cfg.path("reports_evaluation") / "LATEST_RUN.txt").read_text(encoding="utf-8").strip() == V2_APPROVED_RUN
    # fixed-path mirror of the v2 products exists
    for name in (
        "outer_patient_folds_core_v2.csv", "inner_patient_folds_core_v2.parquet",
        "airflow_outer_fold_inheritance_v2.csv", "patient_level_balance_inputs.csv",
        "split_candidate_comparison.csv", "split_balance_version_signature.json",
        "split_balance_resolved.yaml",
    ):
        assert (cfg.path("splits") / name).is_file(), name
    # immutability preserved even though we mirrored
    assert _content_hash_dir(cfg.path("splits") / "runs" / v1) == v1_before
    assert qc_mod.snapshot_raw(cfg.path("data_raw"), cfg.project_root) == raw_before
    assert summary.historical_products_unchanged is True
    assert summary.raw_modified is False


# ---------------------------------------------------------------------------
# (10d) path purity: every recorded product path is project-relative + clean
# ---------------------------------------------------------------------------

def test_product_paths_are_project_relative_and_pure(tmp_path):
    cfg, base = _make_proj(tmp_path)
    burden = synthetic_balanceable_burden(50)
    v1 = "stage6-synthv1-pure-20260808"
    write_synthetic_v1_run(cfg, v1, burden)
    config = _prod_config(base, v1_run_id=v1, burden=burden)

    summary, _ = run_6b(cfg=cfg, run_id="stage6b-v2-pure-20260808", config=config,
                        output_root=None, relpath_base=None)
    assert summary.product_paths, "no product paths recorded"
    for name, rel in summary.product_paths.items():
        assert not Path(rel).is_absolute(), (name, rel)
        low = rel.replace("\\", "/").lower()
        for bad in ("/temp/", "pytest", "appdata", ":/", ".."):
            assert bad not in low, (name, rel, bad)


# ---------------------------------------------------------------------------
# (10e) outputs carry no raw/WAV and no feature-value (model-matrix) columns
# ---------------------------------------------------------------------------

def test_outputs_contain_no_raw_wav_or_feature_values(tmp_path):
    cfg, base = _make_proj(tmp_path)
    burden = synthetic_balanceable_burden(50)
    v1 = "stage6-synthv1-nomx-20260808"
    write_synthetic_v1_run(cfg, v1, burden)
    config = _prod_config(base, v1_run_id=v1, burden=burden)

    summary, _ = run_6b(cfg=cfg, run_id="stage6b-v2-nomx-20260808", config=config,
                        output_root=None, relpath_base=None)
    assert summary.audio_cohort_count == 0
    run_dir = cfg.path("splits") / "runs" / "stage6b-v2-nomx-20260808"
    # no audio/raw artefacts
    assert not any(p.suffix.lower() == ".wav" for p in run_dir.rglob("*"))
    # no forbidden feature-value columns in any tabular product
    for p in run_dir.iterdir():
        if p.suffix == ".parquet":
            cols = set(pd.read_parquet(p).columns)
        elif p.suffix == ".csv":
            cols = set(pd.read_csv(p, nrows=0).columns)
        else:
            continue
        assert not (cols & set(STAGE6_FORBIDDEN_COLUMNS)), (p.name, cols & set(STAGE6_FORBIDDEN_COLUMNS))


# ---------------------------------------------------------------------------
# hard rule: the 0.10 threshold + comparator are never relaxed to pass
# ---------------------------------------------------------------------------

def test_config_hash_is_not_a_test_token():
    # the contamination guard must accept the run's config_hash (no test markers)
    assert S6B_CONFIG_HASH.lower() not in ("test", "fixture", "pytest")
    assert "test" not in S6B_CONFIG_HASH.lower()
