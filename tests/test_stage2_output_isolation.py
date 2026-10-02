"""Stage-2 output-isolation regression tests (prompt section 2.3 & 7).

Guarantees:
  1. A fixture Stage-2 run (gate passed, output_root=tmp) leaves the Stage-1 AND
     Stage-2 production directories byte-for-byte unchanged.
  2. Fixture artifacts live strictly under tmp_path; path fields are
     fixture-relative (never absolute, never pytest/AppData/Temp); no real
     patient-identifiable content.
  3. The contamination guard refuses a production run that looks like a test
     fixture, and an isolated run whose output_root targets a production dir.
  4. The real production Stage-2 fixed-path outputs (when absent) are not
     created by tests.
  5. Full pipeline: with a passing gate the six CSVs + reports are produced and
     contain the expected structure (varied recorder counts, preliminary event
     note, classification + audio metadata joined).
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pytest

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_audit import manifests as manifests_mod
from psg_audio_benchmark.data_audit.runner import (
    Stage2ContaminationError,
    Stage2Options,
    Stage2Runner,
)

from _stage2_fixtures import build_synthetic_raw, write_complete_license_evidence

#: Production directories tests must NEVER write into (Stage 1 + Stage 2).
_PROD_DIRS = (
    "data/manifests",
    "reports/data_download",
    "reports/data_audit",
)

_BAD_TOKENS = ("pytest", "AppData", "/Temp/", "\\Temp\\")
_TEST_RUN_ID_TOKENS = ("test", "pytest", "fixture", "immutability")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prod_snapshot(project_root: Path) -> dict:
    snap: dict = {}
    for sub in _PROD_DIRS:
        d = project_root / sub
        if not d.exists():
            continue
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                p = Path(dp) / fn
                try:
                    rel = str(p.relative_to(project_root)).replace("\\", "/")
                    snap[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
                except Exception as exc:  # pragma: no cover - defensive
                    snap[str(p)] = f"ERR:{exc}"
    return snap


def _make_isolated_cfg(tmp_path: Path) -> Config:
    """Real Config whose raw + docs point at small tmp dirs under tmp_path."""
    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    build_synthetic_raw(raw)
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    write_complete_license_evidence(docs)  # gate passes
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["docs"] = docs
    return cfg


# ---------------------------------------------------------------------------
# Check 1 + 5: full fixture run leaves production unchanged AND emits products
# ---------------------------------------------------------------------------

def test_full_fixture_run_leaves_production_unchanged(tmp_path: Path) -> None:
    project_root = default_project_root()
    before = _prod_snapshot(project_root)

    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-1", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    assert summary.raw_modified is False
    assert summary.n_patients == 2

    after = _prod_snapshot(project_root)
    assert before == after, "A fixture Stage-2 run modified production outputs."

    # All six products + gate-passed report produced under tmp.
    mdir = tmp_path / "outputs" / "manifests"
    for name in manifests_mod.all_product_names():
        assert (mdir / f"{name}.csv").is_file(), f"missing product {name}"
    rdir = tmp_path / "outputs" / "reports" / "data_audit"
    assert (rdir / "license_gate_passed.md").is_file()
    assert (rdir / "data_structure_exploration_report.md").is_file()
    assert (rdir / "phase_02_completion_report.md").is_file()


# ---------------------------------------------------------------------------
# Check 2: fixture artifacts isolated + fixture-relative + no bad tokens
# ---------------------------------------------------------------------------

def test_fixture_products_are_isolated_and_clean(tmp_path: Path) -> None:
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-2", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    runner.run()
    mdir = tmp_path / "outputs" / "manifests"

    # patient_manifest fixture-relative paths, no absolute / pytest tokens.
    with open(mdir / "patient_manifest.csv", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert len(rows) == 2
    for row in rows:
        rel = row["source_patient_dir_relative"]
        assert not Path(rel).is_absolute()
        assert rel.startswith("raw/V5/Data/")
        for tok in _BAD_TOKENS:
            assert tok.lower() not in rel.lower()

    # enriched manifest path fields clean + classification populated.
    with open(mdir / "file_manifest_enriched.csv", encoding="utf-8") as h:
        erows = list(csv.DictReader(h))
    assert erows
    for row in erows:
        rel = row["relative_path"]
        assert not Path(rel).is_absolute()
        assert "data/raw" not in rel  # must not reference the real raw tree
        assert row["modality_candidate"]  # classified
    # audio rows carry header metadata
    audio_rows = [r for r in erows if r["audio_container"] == "wav"]
    assert audio_rows and all(r["audio_sample_rate_hz"] == "8000" for r in audio_rows)


def test_event_distribution_carries_preliminary_note(tmp_path: Path) -> None:
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-3", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    runner.run()
    mdir = tmp_path / "outputs" / "manifests"
    with open(mdir / "event_distribution_preliminary.csv", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert rows, "expected preliminary event rows"
    for row in rows:
        assert "PRELIMINARY" in row["preliminary_note"]
        assert row["count"].isdigit()
    # README sidecar stamped.
    assert (mdir / "event_distribution_preliminary.README.txt").is_file()


def test_varied_recorder_counts_captured(tmp_path: Path) -> None:
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-4", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    runner.run()
    with open(tmp_path / "outputs" / "manifests" / "signal_availability_matrix.csv",
              encoding="utf-8") as h:
        rows = {r["patient_id_canonical"]: r for r in csv.DictReader(h)}
    # 01 -> 2 recorders, 02 -> 6 recorders (never assumed exactly two).
    assert rows["01"]["recorder_file_count"] == "2"
    assert rows["02"]["recorder_file_count"] == "6"
    assert rows["02"]["has_airflow"] == "True"
    assert rows["01"]["has_airflow"] == "False"


def test_annotation_inventory_reports_real_top_level_structure(tmp_path: Path) -> None:
    """The annotation inventory must report the REAL record_start /
    awake_intervals / events presence from the files (prompt 4.3, 5.4) -- never
    a hardcoded placeholder, since the next-stage parser relies on it."""
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-5", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    mdir = tmp_path / "outputs" / "manifests"
    with open(mdir / "annotation_structure_inventory.csv", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert rows, "expected annotation inventory rows"
    for row in rows:
        keys = row["annotation_top_level_keys"]
        assert keys, "top_level_keys must be populated from the real file"
        assert "record_start" in keys
        assert "awake_intervals" in keys
        assert "events" in keys
        assert row["annotation_record_start_present"] == "True"
        assert row["annotation_awake_intervals_present"] == "True"
    # Aggregate presence flags also measured from the probes, not hardcoded False.
    assert summary.annotation_summary["has_record_start"] is True
    assert summary.annotation_summary["has_awake"] is True
    assert summary.annotation_summary["has_events"] is True


def test_completion_report_status_matches_overall_when_passed(tmp_path: Path) -> None:
    """A passing fixture run must render overall_status (not the 'BLOCKED'
    dataclass default) and a non-contradictory admission section in the
    completion report."""
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage2-prodlike-6", "config_hash": "deadbeefcafebabe"},
        options=Stage2Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    text = (tmp_path / "outputs" / "reports" / "data_audit" /
            "phase_02_completion_report.md").read_text(encoding="utf-8")
    assert summary.overall_status in ("PASS", "PASS WITH WARNINGS")
    assert f"阶段 2 状态：**{summary.overall_status}**" in text
    # Passing run must not carry the blocked-path admission contradiction.
    assert "未满足" not in text


# ---------------------------------------------------------------------------
# Check 3: contamination guards
# ---------------------------------------------------------------------------

def test_production_mode_rejects_test_run_id(tmp_path: Path) -> None:
    # isolated cfg under the pytest-managed tmp dir (portable: a literal
    # "/tmp/..." maps to the drive root on Windows and may be unwritable);
    # raw is unused in the guard under test.
    cfg = _make_isolated_cfg(tmp_path / "nonexistent-for-guard-only")
    # Reset raw to project raw so the only failure is the run_id token.
    cfg2 = Config(project_root=default_project_root())
    cfg2._resolved_paths["docs"] = cfg._resolved_paths["docs"]
    runner = Stage2Runner(
        cfg=cfg2,
        run_metadata={"run_id": "stage2-test-run", "config_hash": "realhash1234"},
        options=Stage2Options(),  # no output_root => production
    )
    with pytest.raises(Stage2ContaminationError):
        runner.run()


def test_production_mode_rejects_test_config_hash(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "test"},
        options=Stage2Options(),
    )
    with pytest.raises(Stage2ContaminationError):
        runner.run()


def test_production_mode_rejects_tmp_raw(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())
    cfg._resolved_paths["data_raw"] = tmp_path / "raw"  # outside project root
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage2Options(),
    )
    with pytest.raises(Stage2ContaminationError):
        runner.run()


@pytest.mark.parametrize("target_key", ["data_manifests", "reports_data_audit"])
def test_isolated_mode_rejects_production_output_root(target_key: str) -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage2Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage2Options(output_root=cfg.path(target_key)),
    )
    with pytest.raises(Stage2ContaminationError):
        runner.run()
