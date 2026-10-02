"""Stage-3 boundary case 10: machine-readable products carry no absolute / test
/ pytest / AppData / Temp path tokens.

Guarantees (prompt section 6.10):
  * every string field in every CSV/parquet product is relative + clean;
  * the empty ``event_parse_rejections.csv`` is a well-formed CSV with a header
    (not a 0-byte file) when there are zero rejections;
  * the path-purity guard itself rejects bad tokens (guard is not weakened).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from psg_audio_benchmark.annotation_parser import Stage3Options, Stage3Runner
from psg_audio_benchmark.annotation_parser.qc import assert_paths_clean
from psg_audio_benchmark.config import default_project_root

from _stage3_fixtures import build_stage3_patient, make_isolated_cfg

_BAD_TOKENS = ("pytest", "appdata", "/temp/", "\\temp\\", "/tmp/")


def _run(tmp_path: Path, run_id: str) -> Path:
    cfg = make_isolated_cfg(tmp_path)
    # relpath_base under tmp so recorded paths are tmp-rooted (clean of real paths)
    build_stage3_patient(cfg.path("data_raw"), "01", cross_midnight_csv=True)
    runner = Stage3Runner(
        cfg=cfg,
        run_metadata={"run_id": run_id, "config_hash": "realhash1234"},
        options=Stage3Options(output_root=tmp_path / "outputs", relpath_base=tmp_path),
    )
    summary = runner.run()
    assert summary.license_gate_passed is True
    return tmp_path / "outputs" / "annotations" / "runs" / run_id


def _all_string_values(adir: Path) -> list:
    values = []
    for p in sorted(adir.iterdir()):
        if p.suffix == ".csv":
            with open(p, "r", encoding="utf-8", newline="") as h:
                for row in csv.DictReader(h):
                    values.extend(v for v in row.values() if isinstance(v, str))
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
            for col in df.columns:
                if df[col].dtype == object:
                    values.extend(str(v) for v in df[col].dropna().tolist())
    return values


def test_no_absolute_or_test_tokens_in_products(tmp_path: Path) -> None:
    adir = _run(tmp_path, "iso-stage3-clean-1")
    values = _all_string_values(adir)
    assert values, "expected some string values to scan"
    for v in values:
        as_posix = v.replace("\\", "/")
        assert not Path(v).is_absolute(), f"absolute path in product: {v!r}"
        assert not as_posix.startswith("/"), f"absolute-ish path in product: {v!r}"
        low = v.lower()
        for tok in _BAD_TOKENS:
            assert tok not in low, f"test-path token {tok!r} in product: {v!r}"


def test_empty_rejections_csv_has_header(tmp_path: Path) -> None:
    adir = _run(tmp_path, "iso-stage3-clean-2")
    p = adir / "event_parse_rejections.csv"
    assert p.is_file()
    # not a 0-byte mystery file
    assert p.stat().st_size > 0
    with open(p, "r", encoding="utf-8", newline="") as h:
        reader = csv.DictReader(h)
        header = reader.fieldnames
        rows = list(reader)
    # header present even with zero rejections
    assert header is not None
    assert "reason" in header and "patient_id" in header and "severity" in header
    # and zero data rows (the fixture patient parses cleanly)
    assert rows == []


def test_audio_inventory_uses_relative_paths(tmp_path: Path) -> None:
    adir = _run(tmp_path, "iso-stage3-clean-3")
    with open(adir / "audio_time_alignment_inventory.csv", encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    assert rows, "expected audio rows"
    for r in rows:
        rel = r["source_relpath"]
        assert not Path(rel).is_absolute()
        assert rel.startswith("raw/V5/Data/")


# ---------------------------------------------------------------------------
# The guard itself must not be weakened
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "/abs/path/x",
    "C:\\Users\\someone\\AppData\\Local\\x",
    "x/pytest_tmp/y",
    "c:/temp/secret",
])
def test_assert_paths_clean_rejects_bad_tokens(bad: str) -> None:
    with pytest.raises(AssertionError):
        assert_paths_clean([bad])


def test_assert_paths_clean_accepts_clean_relative() -> None:
    # must NOT raise on a clean relative path
    assert_paths_clean(["raw/V5/Data/01/01_phone.wav", "raw/V5/Data/02/x.csv"])
