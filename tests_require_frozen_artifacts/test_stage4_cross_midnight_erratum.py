"""Stage-4 boundary case 8: Stage-3 cross-midnight aggregate erratum regression.

Independent re-audit of the REAL input run's ``awake_intervals_canonical.parquet``
confirms ``cross_midnight=true`` canonical intervals = 10, involving 10 patients
(prompt section 2). The Stage-3 narrative count of "46 cross-midnight patients"
was inflated by also counting CSV-signal cross-midnight patients; this test pins
the correct value so it cannot regress.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from psg_audio_benchmark.config import default_project_root


def _input_run_dir() -> Path:
    root = default_project_root()
    latest = root / "annotations" / "LATEST_RUN.txt"
    assert latest.is_file(), "annotations/LATEST_RUN.txt must exist"
    run_id = latest.read_text(encoding="utf-8").strip()
    assert run_id, "LATEST_RUN.txt is empty"
    run_dir = root / "annotations" / "runs" / run_id
    assert run_dir.is_dir(), f"input run dir missing: {run_dir}"
    return run_dir


def test_case8_cross_midnight_canonical_is_10_intervals_10_patients() -> None:
    run_dir = _input_run_dir()
    aw = pd.read_parquet(run_dir / "awake_intervals_canonical.parquet")
    xmid = aw[aw["cross_midnight"] == True]  # noqa: E712
    assert len(xmid) == 10, f"expected 10 cross-midnight intervals, got {len(xmid)}"
    assert xmid["patient_id"].nunique() == 10, (
        f"expected 10 cross-midnight patients, got {xmid['patient_id'].nunique()}"
    )


def test_case8_cross_midnight_patients_are_the_expected_set() -> None:
    run_dir = _input_run_dir()
    aw = pd.read_parquet(run_dir / "awake_intervals_canonical.parquet")
    xmid = aw[aw["cross_midnight"] == True]  # noqa: E712
    assert sorted(xmid["patient_id"].unique().tolist()) == [
        "05", "10", "11", "12", "14", "19", "30", "37", "44", "50"
    ]


def test_case8_erratum_file_records_correct_value() -> None:
    """The erratum written by a Stage-4 run records the correct 10/10 value.

    Uses an isolated fixture run so the production erratum path is not depended
    upon here; the production erratum is verified separately after the real run.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from psg_audio_benchmark.windowing.report import render_phase03_cross_midnight_erratum

    text = render_phase03_cross_midnight_erratum(
        correct_intervals=10, correct_patients=10, reported_patients=46,
        correct_patient_ids=["05", "10", "11", "12", "14", "19", "30", "37", "44", "50"],
        input_run_id="stage3-annotation-sync-20260807T154558Z",
        erratum_date="2026-08-08",
    )
    assert "10 条，涉及 10 位患者" in text
    assert "误记为 46" in text
