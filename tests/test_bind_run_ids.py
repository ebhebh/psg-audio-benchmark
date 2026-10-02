"""Synthetic tests for the Mode-B run-ID binding tool (Stage-25 release fix;
fixture extended in Stage 26).

``tools/bind_run_ids.py`` chains a reproducer's OWN regenerated run ids to the
next stage while enforcing the license / lineage / approval / signature gates.
These tests build a fully synthetic mini-project under ``tmp_path`` (parquet
fixtures included) and pin:

* happy path: complete consistent chain -> no problems, correct commands;
* license gate record missing -> refused;
* unapproved split signature (status != approved_for_modeling) -> refused;
* tampered split product (byte changed after signature) -> refused;
* wrong-stage run id at a pointer (prefix mismatch) -> refused;
* lineage mismatch (stage5 cites a different stage4) -> refused;
* cohort patient-set drift vs the approved signature -> refused.

Stage-26 note: the tool additionally re-checks the CURRENT license evidence
file (the same gate as scripts/02_data_audit.py), so the fixture now also
writes a complete gate-passing evidence file — the assertions above are
unchanged (nothing was weakened; the gate got stricter).

No real data is read; everything is synthetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from _stage2_fixtures import write_complete_license_evidence

_REPO = Path(__file__).resolve().parents[1]


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "bind_run_ids", _REPO / "tools" / "bind_run_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _sig_cohort(pids: list[str]) -> dict:
    return {
        "n_patients": len(pids), "n_windows": 100 * len(pids),
        "n_positive": 40 * len(pids), "n_negative": 60 * len(pids),
        "overall_positive_rate": 0.4,
        "patient_set_sha256": hashlib.sha256(
            "\n".join(sorted(pids)).encode("utf-8")).hexdigest(),
    }


def build_synthetic_chain(root: Path, *, license_passed: bool = True,
                          approval: str = "approved_for_modeling",
                          stage5_cites_stage4: str | None = None,
                          stage4_prefix_ok: bool = True) -> dict:
    """Write a synthetic Mode-B chain under root; returns fixed run ids."""
    ids = {
        "stage3": "stage3-annotation-sync-20990101T000000Z",
        "stage4": "stage4-csv-window-index-20990101T000001Z",
        "stage5": "stage5-physiology-features-20990101T000002Z",
        "stage6b": "stage6b-split-balance-v2-20990101T000003Z",
    }
    if not stage4_prefix_ok:
        ids["stage4"] = "stage9-unrelated-20990101T000001Z"

    # stage 3 (products = what Stage 4/5 actually require from a Stage-3 run;
    # Stage-26: aligned to windowing.runner._REQUIRED_INPUT_ARTIFACTS)
    (root / "annotations/runs" / ids["stage3"]).mkdir(parents=True)
    _write_parquet(root / "annotations/runs" / ids["stage3"] /
                   "parsed_events.parquet", pd.DataFrame({"event": [1]}))
    _write_parquet(root / "annotations/runs" / ids["stage3"] /
                   "awake_intervals_canonical.parquet",
                   pd.DataFrame({"patient_id": ["01"]}))
    _write_parquet(root / "annotations/runs" / ids["stage3"] /
                   "signal_time_ranges.parquet", pd.DataFrame({"m": ["hr"]}))
    _write_parquet(root / "annotations/runs" / ids["stage3"] /
                   "record_time_anchors.parquet",
                   pd.DataFrame({"patient_id": ["01"]}))
    (root / "annotations/runs" / ids["stage3"] /
     "audio_time_alignment_inventory.csv").write_text(
         f"run_id\n{ids['stage3']}\n", encoding="utf-8")
    (root / "annotations").joinpath("LATEST_RUN.txt").write_text(
        ids["stage3"] + "\n", encoding="utf-8")

    # stage 4 (incl. the resolved config citing its Stage-3 input = lineage)
    d4 = root / "data/manifests/runs" / ids["stage4"]
    _write_parquet(d4 / "csv_window_index.parquet",
                   pd.DataFrame({"window_id": ["w1"]}))
    (d4 / "patient_window_summary.csv").parent.mkdir(parents=True,
                                                     exist_ok=True)
    (d4 / "patient_window_summary.csv").write_text("patient_id\n01\n",
                                                   encoding="utf-8")
    (d4 / "windowing_config_resolved.yaml").write_text(
        f"run_id: {ids['stage4']}\n"
        f"input_stage3_run_id: {ids['stage3']}\n", encoding="utf-8")
    (root / "reports/windowing").mkdir(parents=True, exist_ok=True)
    (root / "reports/windowing/LATEST_RUN.txt").write_text(
        ids["stage4"] + "\n", encoding="utf-8")

    # stage 5 (+ resolved config citing its inputs = lineage record)
    d5 = root / "features/physiology/runs" / ids["stage5"]
    _write_parquet(d5 / "hr_window_features.parquet",
                   pd.DataFrame({"hr_mean": [1.0]}))
    _write_parquet(d5 / "spo2_window_features.parquet",
                   pd.DataFrame({"spo2_min": [0.9]}))
    (d5 / "physiology_features_resolved.yaml").write_text(
        f"input_stage3_run_id: {ids['stage3']}\n"
        f"input_stage4_run_id: "
        f"{stage5_cites_stage4 or ids['stage4']}\n"
        f"run_id: {ids['stage5']}\n",
        encoding="utf-8",
    )
    (root / "reports/feature_extraction").mkdir(parents=True, exist_ok=True)
    (root / "reports/feature_extraction/LATEST_RUN.txt").write_text(
        ids["stage5"] + "\n", encoding="utf-8")

    # stage 6b: split products + signature (hashes recorded over real bytes)
    pids = ["01", "02", "03", "04", "05"]
    splits_dir = root / "splits"
    v2_files = {}
    for name, df in (
        ("outer_patient_folds_core_v2.csv",
         pd.DataFrame({"patient_id": pids, "outer_fold": [0, 1, 2, 3, 4]})),
        ("inner_patient_folds_core_v2.parquet",
         pd.DataFrame({"patient_id": pids * 4, "outer_fold": [0] * 5 + [1] * 5
                      + [2] * 5 + [3] * 5,
                      "inner_validation_fold": [1, 2, 3, 4, 1] * 4})),
    ):
        p = splits_dir / name
        if name.endswith(".csv"):
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
        else:
            _write_parquet(p, df)
        v2_files[f"v2/{name}"] = {"relpath": name, "sha256": _sha(p)}
    _write_parquet(splits_dir / "core_cohort_window_membership.parquet",
                   pd.DataFrame({"patient_id": [p for p in pids for _ in range(4)],
                                 "window_id": [f"{p}-w{i}" for p in pids
                                               for i in range(4)]}))
    signature = {
        "run_id": ids["stage6b"],
        "approval_status": approval,
        "cohort": _sig_cohort(pids),
        "lineage": {
            "stage3_run_id": ids["stage3"],
            "stage4_run_id": ids["stage4"],
            "stage5_run_id": ids["stage5"],
        },
        "v2_files": v2_files,
    }
    d6b = splits_dir / "runs" / ids["stage6b"]
    d6b.mkdir(parents=True)
    sig_text = json.dumps(signature)
    # production 6b writes the signature in the run dir AND mirrors it to the
    # fixed splits/ location that downstream stages read — do both here too.
    (d6b / "split_balance_version_signature.json").write_text(
        sig_text, encoding="utf-8")
    (splits_dir / "split_balance_version_signature.json").write_text(
        sig_text, encoding="utf-8")
    (splits_dir / "LATEST_RUN.txt").write_text(ids["stage6b"] + "\n",
                                               encoding="utf-8")

    # license gate record + CURRENT gate-passing evidence file (Stage-26
    # stricter gate: the tool re-runs the real Stage-2 license check)
    if license_passed:
        lg = root / "reports/data_audit"
        lg.mkdir(parents=True, exist_ok=True)
        (lg / "license_gate_passed.md").write_text(
            "synthetic passed record\n", encoding="utf-8")
    write_complete_license_evidence(root / "docs")
    return ids


def test_happy_path_full_chain_passes(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    ids = build_synthetic_chain(root)
    bindings, problems = tool.check(root)
    assert problems == []
    assert bindings["run_ids"] == ids
    assert bindings["license_gate_passed_record"] is True
    assert bindings["split_approval_status"] == "approved_for_modeling"
    assert bindings["split_signature_run_id"] == ids["stage6b"]
    cmds = tool._commands(ids)
    assert any(ids["stage4"] in c and "--input-stage4-run-id" in c for c in cmds)
    assert any(ids["stage5"] in c and "--input-stage5-run-id" in c for c in cmds)


def test_license_gate_missing_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root, license_passed=False)
    _, problems = tool.check(root)
    assert any("license_gate_passed.md" in p for p in problems)


def test_unapproved_split_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root, approval="not_approved_balance_target_not_met")
    _, problems = tool.check(root)
    assert any("approval gate" in p and "NOT approved" in p for p in problems)


def test_tampered_split_product_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root)
    # flip bytes of a signed split product AFTER the signature was written
    target = root / "splits/outer_patient_folds_core_v2.csv"
    target.write_text(target.read_text(encoding="utf-8").replace(
        "patient_id", "patient_id_tampered"), encoding="utf-8")
    _, problems = tool.check(root)
    assert any("tamper gate" in p and "SHA-256 mismatch" in p for p in problems)


def test_wrong_stage_run_id_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root, stage4_prefix_ok=False)
    _, problems = tool.check(root)
    assert any("does not start with the expected prefix" in p for p in problems)


def test_lineage_mismatch_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root,
                          stage5_cites_stage4="stage4-csv-window-index-11111111T111111Z")
    _, problems = tool.check(root)
    assert any("lineage" in p and "cites stage4" in p for p in problems)


def test_cohort_drift_refuses_binding(tool, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    build_synthetic_chain(root)
    # add a patient to the live cohort membership (post-signature drift)
    p = root / "splits/core_cohort_window_membership.parquet"
    df = pd.read_parquet(p)
    df = pd.concat([df, pd.DataFrame({"patient_id": ["99"],
                                      "window_id": ["99-w0"]})],
                   ignore_index=True)
    _write_parquet(p, df)
    _, problems = tool.check(root)
    assert any("patient-split gate" in p for p in problems)
