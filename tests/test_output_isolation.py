"""Anti-contamination regression tests (remediation prompt section 2.3).

These tests lock in the guarantee that **test runs can never write into
production paths**, and that the real production manifest is free of test
artifacts. They implement the five checks from section 2.3:

  1. Running a fixture full run leaves the production ``data/manifests/`` and
     ``reports/data_download/`` directories byte-for-byte unchanged.
  2. Fixture-generated manifest/receipt/report all live under ``tmp_path`` and
     their path *fields* are fixture-relative (never absolute, never production
     project paths).
  3. The production runner refuses a test ``run_id`` / ``config_hash`` / tmp
     raw path, and the isolated runner refuses an ``output_root`` that points
     at (or into) a production manifest/report directory.
  4. The real production manifest contains no absolute paths, no pytest/tmp
     tokens, and no test ``run_id`` (e.g. ``full-immutability-test``).
  5. The real ``data/raw`` tree is stable across two lightweight
     (size + mtime) snapshots.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pytest

from psg_audio_benchmark.config import Config, default_project_root
from psg_audio_benchmark.data_download.runner import (
    Stage1ContaminationError,
    Stage1Options,
    Stage1Runner,
)

#: Production directories that tests must NEVER write into.
_PROD_DIRS = ("data/manifests", "reports/data_download")

#: Tokens that must never appear in a real production manifest path/run_id.
_BAD_TOKENS = (
    "full-immutability-test",
    "immutability-test",
    "pytest",
    "AppData",
    "Temp/pytest",
)

#: run_id markers that identify a test fixture (must not be a production run).
_TEST_RUN_ID_TOKENS = ("test", "immutability", "pytest", "fixture")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prod_snapshot(project_root: Path) -> dict:
    """{project-relative-path: sha256} over every file in the production dirs."""
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
    """A real Config whose raw + staging point at small tmp dirs under tmp_path."""
    cfg = Config(project_root=default_project_root())
    raw = tmp_path / "raw"
    (raw / "01").mkdir(parents=True)
    (raw / "01" / "01_phone.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    (raw / "01" / "01_SpO2.csv").write_bytes(b"time,spo2\n0,98\n")
    staging = tmp_path / "incoming"
    staging.mkdir()
    cfg._resolved_paths["data_raw"] = raw
    cfg._resolved_paths["data_external_incoming"] = staging
    return cfg


def _lightweight_raw_snapshot(raw: Path) -> dict:
    """{relative_path: 'size:mtime_ns'} mirroring the runner's snapshot."""
    snap: dict = {}
    if not raw.exists():
        return snap
    for dp, _dn, fns in os.walk(raw):
        for fn in fns:
            p = Path(dp) / fn
            rel = str(p.relative_to(raw)).replace("\\", "/")
            st = p.stat()
            snap[rel] = f"{st.st_size}:{st.st_mtime_ns}"
    return snap


# ---------------------------------------------------------------------------
# Check 1 — production byte-identical before/after a fixture full run
# ---------------------------------------------------------------------------

def test_full_test_run_leaves_production_unchanged(tmp_path: Path) -> None:
    project_root = default_project_root()
    before = _prod_snapshot(project_root)

    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-prod-like-runid", "config_hash": "deadbeefcafebabe"},
        options=Stage1Options(
            verify_existing=True,
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    summary = runner.run()
    assert summary.raw_modified is False

    after = _prod_snapshot(project_root)
    assert before == after, (
        "A fixture run modified production outputs — isolation is broken."
    )


# ---------------------------------------------------------------------------
# Check 2 — fixture artifacts under tmp_path + fixture-relative path fields
# ---------------------------------------------------------------------------

def test_fixture_artifacts_are_isolated_and_fixture_relative(tmp_path: Path) -> None:
    cfg = _make_isolated_cfg(tmp_path)
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-prod-like-runid", "config_hash": "deadbeefcafebabe"},
        options=Stage1Options(
            verify_existing=True,
            output_root=tmp_path / "outputs",
            relpath_base=tmp_path,
        ),
    )
    runner.run()
    paths = runner.artifact_paths()

    # Every artifact lives strictly under the tmp fixture, never in production.
    for kind, p in paths.items():
        assert str(p).startswith(str(tmp_path)), (
            f"{kind} artifact {p} is not under the tmp fixture root."
        )

    # The manifest's relative_path field must be fixture-relative, not absolute
    # and not a production project path.
    manifest = paths["manifest"]
    with open(manifest, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "fixture manifest should have data rows"
    project_root = default_project_root().resolve()
    for row in rows:
        rel = row["relative_path"]
        assert not Path(rel).is_absolute(), f"absolute path leaked: {rel}"
        assert rel.startswith("raw/"), f"expected fixture-relative 'raw/...', got {rel}"
        for tok in _BAD_TOKENS:
            assert tok.lower() not in rel.lower(), f"bad token {tok!r} in {rel}"
        # Must not accidentally reference the real production raw tree.
        assert "data/raw" not in rel


# ---------------------------------------------------------------------------
# Check 3 — contamination guards
# ---------------------------------------------------------------------------

def test_production_mode_rejects_test_run_id() -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "full-immutability-test", "config_hash": "realhash1234"},
        options=Stage1Options(verify_existing=True),  # no output_root => production
    )
    with pytest.raises(Stage1ContaminationError):
        runner.run()


def test_production_mode_rejects_test_config_hash() -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "test"},
        options=Stage1Options(verify_existing=True),
    )
    with pytest.raises(Stage1ContaminationError):
        runner.run()


def test_production_mode_rejects_tmp_raw(tmp_path: Path) -> None:
    cfg = Config(project_root=default_project_root())
    cfg._resolved_paths["data_raw"] = tmp_path / "raw"  # outside project root
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage1Options(verify_existing=True),
    )
    with pytest.raises(Stage1ContaminationError):
        runner.run()


@pytest.mark.parametrize(
    "target_key", ["data_manifests", "reports_data_download"]
)
def test_isolated_mode_rejects_production_output_root(target_key: str) -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage1Options(verify_existing=True, output_root=cfg.path(target_key)),
    )
    with pytest.raises(Stage1ContaminationError):
        runner.run()


def test_isolated_mode_rejects_project_root_as_output_root() -> None:
    cfg = Config(project_root=default_project_root())
    runner = Stage1Runner(
        cfg=cfg,
        run_metadata={"run_id": "20260714T120000Z-abcdef12", "config_hash": "realhash1234"},
        options=Stage1Options(verify_existing=True, output_root=cfg.project_root),
    )
    with pytest.raises(Stage1ContaminationError):
        runner.run()


# ---------------------------------------------------------------------------
# Check 4 — real production manifest is free of test artifacts
# ---------------------------------------------------------------------------

def test_real_production_manifest_is_pure() -> None:
    """The fixed-path production manifest must never carry test contamination."""
    project_root = default_project_root()
    manifest = project_root / "data" / "manifests" / "file_manifest_stage1.csv"
    if not manifest.is_file():
        pytest.skip("no production manifest present to audit")
    with open(manifest, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        pytest.skip("production manifest is header-only (no run yet)")
    for row in rows:
        rel = row.get("relative_path", "")
        run_id = row.get("run_id", "")
        sha = row.get("sha256", "")
        # No absolute / pytest-temp paths.
        assert not Path(rel).is_absolute(), f"absolute path in production manifest: {rel}"
        low_rel = rel.lower()
        for tok in ("pytest", "appdata", "/temp/", "\\temp\\"):
            assert tok not in low_rel, f"tmp/pytest token {tok!r} in production path {rel}"
        # No test run_id leaked into production.
        low_id = run_id.lower()
        for tok in _TEST_RUN_ID_TOKENS:
            assert tok not in low_id, f"test run_id marker {tok!r} in production manifest ({run_id})"
        # SHA-256, when present, must be 64 hex chars.
        if sha:
            assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
                f"bad sha256 in production manifest: {sha!r}"
            )


def test_real_production_receipt_has_no_test_run_id() -> None:
    import json

    project_root = default_project_root()
    receipt = project_root / "data" / "manifests" / "download_receipt.json"
    if not receipt.is_file():
        pytest.skip("no production receipt present to audit")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    run_id = str(data.get("run_id", ""))
    config_hash = str(data.get("config_hash", ""))
    for tok in _TEST_RUN_ID_TOKENS:
        assert tok not in run_id.lower(), f"test run_id {run_id!r} in production receipt"
    assert config_hash.lower() not in ("test", "fixture", "pytest"), (
        f"test config_hash {config_hash!r} in production receipt"
    )
    assert data.get("raw_modified") is False, "production receipt claims raw was modified"


# ---------------------------------------------------------------------------
# Check 5 — real raw tree is stable across two lightweight snapshots
# ---------------------------------------------------------------------------

def test_real_raw_lightweight_snapshot_is_stable() -> None:
    cfg = Config(project_root=default_project_root())
    raw = cfg.path("data_raw")
    if not raw.exists() or not any(raw.rglob("*")):
        pytest.skip("real data/raw is absent/empty on this machine")
    first = _lightweight_raw_snapshot(raw)
    second = _lightweight_raw_snapshot(raw)
    assert first == second, (
        "real data/raw changed between two read-only snapshots (should be stable)"
    )
