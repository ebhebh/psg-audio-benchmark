"""License-gate tests (prompt section 1).

The gate is the decisive Stage-2 control. These tests pin its behaviour without
touching real patient data:

* evidence absent -> blocked;
* evidence present but fields TBD -> blocked;
* DOI mismatch -> blocked;
* non-commercial basis not affirmative -> blocked;
* complete evidence with matching DOI -> passed;
* a BLOCKED run never scans raw.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_audit import license_gate
from psg_audio_benchmark.data_audit.runner import Stage2Options, Stage2Runner

from _stage2_fixtures import (
    build_synthetic_raw,
    write_complete_license_evidence,
)


def _cfg_with_tmp_docs(tmp_path: Path) -> Config:
    cfg = Config(project_root=default_project_root())
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    cfg._resolved_paths["docs"] = docs
    return cfg


def test_gate_blocks_when_evidence_absent(tmp_path: Path) -> None:
    cfg = _cfg_with_tmp_docs(tmp_path)
    g = license_gate.check_license_gate(cfg)
    assert g.status == "blocked"
    assert "evidence_file_absent" in g.missing_items


def test_gate_blocks_when_fields_are_tbd(tmp_path: Path) -> None:
    cfg = _cfg_with_tmp_docs(tmp_path)
    # Write the same template shape the project ships (all TBD placeholders).
    ev = cfg.path("docs") / license_gate.EVIDENCE_REL_SUBPATH
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(
        "- data_doi: TBD_待人工填写\n- access_date: TBD_待人工填写\n"
        "- license_terms: TBD_待人工填写\n"
        "- noncommercial_research_basis: TBD_待人工填写\n"
        "- deriv_features_policy: TBD_待人工填写\n"
        "- redistribute_audio_policy: TBD_待人工填写\n"
        "- confirmer_name: TBD_待人工填写\n"
        "- confirmation_date: TBD_待人工填写\n",
        encoding="utf-8",
    )
    g = license_gate.check_license_gate(cfg)
    assert g.status == "blocked"
    assert "data_doi" in g.missing_items


def test_gate_blocks_on_doi_mismatch(tmp_path: Path) -> None:
    cfg = _cfg_with_tmp_docs(tmp_path)
    ev = cfg.path("docs") / license_gate.EVIDENCE_REL_SUBPATH
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(
        "- data_doi: 10.11922/sciencedb.00345\n"  # the BACKGROUND dataset DOI
        "- access_date: 2026-07-14\n- license_terms: some terms\n"
        "- noncommercial_research_basis: permitted for non-commercial research\n"
        "- deriv_features_policy: allowed\n- redistribute_audio_policy: not allowed\n"
        "- confirmer_name: Jane Doe\n- confirmation_date: 2026-07-14\n",
        encoding="utf-8",
    )
    g = license_gate.check_license_gate(cfg)
    assert g.status == "blocked"
    assert "data_doi_mismatch" in g.missing_items


def test_gate_blocks_when_basis_not_affirmative(tmp_path: Path) -> None:
    cfg = _cfg_with_tmp_docs(tmp_path)
    ev = cfg.path("docs") / license_gate.EVIDENCE_REL_SUBPATH
    ev.parent.mkdir(parents=True, exist_ok=True)
    ev.write_text(
        "- data_doi: 10.57760/sciencedb.19070\n- access_date: 2026-07-14\n"
        "- license_terms: some terms\n"
        "- noncommercial_research_basis: unclear; needs review\n"
        "- deriv_features_policy: allowed\n- redistribute_audio_policy: not allowed\n"
        "- confirmer_name: Jane Doe\n- confirmation_date: 2026-07-14\n",
        encoding="utf-8",
    )
    g = license_gate.check_license_gate(cfg)
    assert g.status == "blocked"


def test_gate_passes_with_complete_matching_evidence(tmp_path: Path) -> None:
    cfg = _cfg_with_tmp_docs(tmp_path)
    write_complete_license_evidence(cfg.path("docs"))
    g = license_gate.check_license_gate(cfg)
    assert g.status == "passed"
    assert g.license_status != "TBD_AFTER_MANUAL_LICENSE_REVIEW"
    assert g.dataset_version == "V5"


def test_blocked_run_does_not_scan_raw(tmp_path: Path) -> None:
    """When the gate is blocked the runner must NOT scan data/raw."""
    # Use an EMPTY tmp docs/ (no license evidence) so the gate deterministically
    # blocks on 'evidence_file_absent', regardless of the real project's evidence
    # state. The gate logic itself is unchanged; this only isolates the test from
    # the real evidence file (which is now complete and passing).
    cfg = _cfg_with_tmp_docs(tmp_path)
    raw = tmp_path / "raw"
    build_synthetic_raw(raw)
    cfg._resolved_paths["data_raw"] = raw
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-blocked-1", "config_hash": "realhash1234"},
        options=Stage2Options(
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    summary = runner.run()
    assert summary.overall_status == "BLOCKED"
    assert summary.license_gate_passed is False
    assert summary.raw_scanned is False
    # raw was never snapshotted/scanned.
    assert summary.raw_before == {}
    assert summary.raw_after == {}
    # Only the blocked + completion reports exist; no manifests.
    out = tmp_path / "outputs"
    assert (out / "reports" / "data_audit" / "license_gate_blocked.md").is_file()
    assert not (out / "manifests").exists()
