"""(8) Input version signature: pinning + lineage + the version gate.

* ``build_input_signature`` records both run ids, the per-file row/schema/sha,
  and the Stage5->Stage4 lineage match;
* ``compare_signatures`` PASSES on identical signatures and RAISES on run-id,
  row-count or schema-fingerprint disagreement (byte-hash drift is only a
  warning);
* the runner's written ``input_version_signature.json`` carries
  ``lineage_stage5_matches_stage4 == True``.
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from _stage6_fixtures import build_stage6_world, default_windows, make_isolated_cfg, run_stage6, splits_out
from psg_audio_benchmark.evaluation.signature import (
    SignatureMismatchError,
    build_input_signature,
    compare_signatures,
)


def _run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return cfg, ids


def test_signature_lineage_matches_and_inputs_pinned(tmp_path):
    cfg, ids = _run(tmp_path)
    sig, anomalies = build_input_signature(
        cfg=cfg, stage4_run_id=ids["stage4_run_id"], stage5_run_id=ids["stage5_run_id"],
        stage3_run_id=ids["stage3_run_id"], run_id="sig-probe",
        project_root=cfg.project_root,
    )
    assert anomalies == []
    assert sig["lineage_stage5_matches_stage4"] is True
    assert sig["stage4_run_id"] == ids["stage4_run_id"]
    assert sig["stage5_run_id"] == ids["stage5_run_id"]
    # both consumed parquets are pinned with row + schema + sha
    assert "stage4/csv_window_index.parquet" in sig["inputs"]
    assert "stage5/physiology_feature_availability.parquet" in sig["inputs"]
    for key, rec in sig["inputs"].items():
        assert rec["rows"] > 0
        assert len(rec["sha256"]) == 64
        # parquet/csv inputs carry a real content fingerprint; yaml inputs are "text"
        if key.endswith((".parquet", ".csv")):
            assert rec["schema_fingerprint"] not in ("text", "", "unreadable_parquet", "unreadable_csv")


def test_compare_passes_on_identical_signature(tmp_path):
    cfg, ids = _run(tmp_path)
    sig, _ = build_input_signature(
        cfg=cfg, stage4_run_id=ids["stage4_run_id"], stage5_run_id=ids["stage5_run_id"],
        stage3_run_id=ids["stage3_run_id"], run_id="sig-probe",
        project_root=cfg.project_root,
    )
    # identical -> no raise; byte-hash warnings (if any) are returned, not raised
    warnings = compare_signatures(copy.deepcopy(sig), copy.deepcopy(sig))
    assert warnings == []


def test_compare_raises_on_run_id_mismatch(tmp_path):
    cfg, ids = _run(tmp_path)
    sig, _ = build_input_signature(
        cfg=cfg, stage4_run_id=ids["stage4_run_id"], stage5_run_id=ids["stage5_run_id"],
        stage3_run_id=ids["stage3_run_id"], run_id="sig-probe",
        project_root=cfg.project_root,
    )
    bad = copy.deepcopy(sig)
    bad["stage5_run_id"] = "stage5-different-run-id"
    with pytest.raises(SignatureMismatchError):
        compare_signatures(sig, bad)


def test_compare_raises_on_row_count_mismatch(tmp_path):
    cfg, ids = _run(tmp_path)
    sig, _ = build_input_signature(
        cfg=cfg, stage4_run_id=ids["stage4_run_id"], stage5_run_id=ids["stage5_run_id"],
        stage3_run_id=ids["stage3_run_id"], run_id="sig-probe",
        project_root=cfg.project_root,
    )
    bad = copy.deepcopy(sig)
    key = "stage4/csv_window_index.parquet"
    bad["inputs"][key]["rows"] = sig["inputs"][key]["rows"] + 999
    with pytest.raises(SignatureMismatchError):
        compare_signatures(sig, bad)


def test_compare_raises_on_schema_mismatch_but_only_warns_on_sha_drift(tmp_path):
    cfg, ids = _run(tmp_path)
    sig, _ = build_input_signature(
        cfg=cfg, stage4_run_id=ids["stage4_run_id"], stage5_run_id=ids["stage5_run_id"],
        stage3_run_id=ids["stage3_run_id"], run_id="sig-probe",
        project_root=cfg.project_root,
    )
    # schema change -> raise
    bad_schema = copy.deepcopy(sig)
    key = "stage5/physiology_feature_availability.parquet"
    bad_schema["inputs"][key]["schema_fingerprint"] = "different"
    with pytest.raises(SignatureMismatchError):
        compare_signatures(sig, bad_schema)
    # pure byte-hash drift -> warning only, no raise
    bad_sha = copy.deepcopy(sig)
    bad_sha["inputs"][key]["sha256"] = "0" * 64
    warnings = compare_signatures(sig, bad_sha)
    assert any("sha256_drift" in w for w in warnings)


def test_written_signature_has_lineage_match(tmp_path):
    _cfg, _ids = _run(tmp_path)
    sig = json.loads((splits_out(tmp_path) / "input_version_signature.json").read_text("utf-8"))
    assert sig["lineage_stage5_matches_stage4"] is True
    assert sig["sklearn_version"] != "not_installed"
