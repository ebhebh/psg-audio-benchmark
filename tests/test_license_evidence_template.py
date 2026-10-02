"""Shipped license-evidence TEMPLATE tests (Stage-25 release fix).

The release ships ``docs/license_evidence/primary_dataset_license_confirmation.md``
as an UNFILLED template so reproducers have the exact field-complete form.
These tests pin BOTH documented branches of the gate against the actual
shipped file (not a hand-written lookalike):

* the shipped template (all ``<...>`` placeholders) -> ``blocked``;
* a synthetic COMPLETE evidence file (test fixture, clearly marked synthetic)
  -> ``passed``.

The template must never drift into a pre-confirmed state, and the shipped
``docs/license_evidence/README.md`` verification steps must stay present.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_audit import license_gate

from _stage2_fixtures import write_complete_license_evidence


def _shipped_docs_dir() -> Path:
    """Real shipped docs/ dir of this release checkout."""
    return default_project_root() / "docs"


def _cfg_with_docs(tmp_path: Path, docs_dir: Path) -> Config:
    cfg = Config(project_root=default_project_root())
    cfg._resolved_paths["docs"] = docs_dir
    return cfg


def test_shipped_template_exists_with_all_fields() -> None:
    """The template ships and declares every field the gate parses."""
    p = _shipped_docs_dir() / license_gate.EVIDENCE_REL_SUBPATH
    assert p.is_file(), "shipped template missing from docs/license_evidence/"
    ev = license_gate.parse_evidence_md(p.read_text(encoding="utf-8"))
    for key in license_gate.LicenseEvidence().as_dict():
        # field label present in the template (value may be a placeholder)
        assert getattr(ev, key) != "" or key in ("open_items",), (
            f"template lacks a value line for {key}"
        )


def test_shipped_template_parses_to_blocked(tmp_path: Path) -> None:
    """The ACTUAL shipped template must gate to blocked (nothing pre-confirmed)."""
    src = _shipped_docs_dir() / license_gate.EVIDENCE_REL_SUBPATH
    docs = tmp_path / "docs" / "license_evidence"
    docs.mkdir(parents=True)
    (docs / src.name).write_text(src.read_text(encoding="utf-8"),
                                 encoding="utf-8")
    g = license_gate.check_license_gate(_cfg_with_docs(tmp_path, docs.parent))
    assert g.status == "blocked"
    assert g.license_status == "TBD_AFTER_MANUAL_LICENSE_REVIEW"
    # the required fields are the blockers (placeholders, not absent file)
    assert "evidence_file_absent" not in g.missing_items
    blockers = set(g.missing_items)
    assert blockers & {
        "data_doi", "access_date", "license_terms",
        "noncommercial_research_basis", "deriv_features_policy",
        "redistribute_audio_policy", "confirmer_name", "confirmation_date",
    }, f"expected required-field blockers, got {blockers}"


def test_shipped_template_blocks_stage2_before_raw_scan(tmp_path: Path) -> None:
    """End-to-end: with the shipped template in place the Stage-2 runner
    writes license_gate_blocked.md and never scans raw."""
    from psg_audio_benchmark.data_audit.runner import (
        Stage2Options,
        Stage2Runner,
    )
    from _stage2_fixtures import build_synthetic_raw

    src = _shipped_docs_dir() / license_gate.EVIDENCE_REL_SUBPATH
    docs = tmp_path / "docs" / "license_evidence"
    docs.mkdir(parents=True)
    (docs / src.name).write_text(src.read_text(encoding="utf-8"),
                                 encoding="utf-8")
    cfg = _cfg_with_docs(tmp_path, docs.parent)
    raw = tmp_path / "raw"
    build_synthetic_raw(raw)
    cfg._resolved_paths["data_raw"] = raw
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-template-1", "config_hash": "realhash1234"},
        options=Stage2Options(output_root=tmp_path / "outputs",
                              relpath_base=tmp_path),
    )
    summary = runner.run()
    assert summary.overall_status == "BLOCKED"
    assert summary.license_gate_passed is False
    assert summary.raw_scanned is False
    assert (tmp_path / "outputs" / "reports" / "data_audit" /
            "license_gate_blocked.md").is_file()


def test_synthetic_complete_evidence_passes(tmp_path: Path) -> None:
    """Mirror branch: a COMPLETE evidence file (synthetic fixture) passes."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    write_complete_license_evidence(docs)  # synthetic, clearly marked fixture
    g = license_gate.check_license_gate(_cfg_with_docs(tmp_path, docs))
    assert g.status == "passed"
    assert g.license_status == "confirmed_for_noncommercial_research"


def test_verification_steps_doc_shipped() -> None:
    """The official-source verification steps must ship next to the template."""
    p = _shipped_docs_dir() / "license_evidence" / "README.md"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "10.57760/sciencedb.19070" in text
    assert "blocked" in text and "passed" in text
