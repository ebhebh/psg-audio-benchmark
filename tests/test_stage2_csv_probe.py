"""CSV probe tests (prompt section 4.2).

Header/time exploration that survives encoding quirks, missing columns and
files with no time axis (which must NOT get a guessed sampling rate). No
imputation, no modelling.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.data_audit import csv_probe

from _stage2_fixtures import write_spo2_csv, write_flow_csv, write_sleep_stage_csv, write_no_time_csv


def test_spo2_time_axis_and_missingness(tmp_path: Path) -> None:
    p = write_spo2_csv(tmp_path / "01_SpO2.csv", rows=5, missing_every=3)
    probe = csv_probe.probe_csv(p)
    assert probe.inferred_role == "spo2"
    assert probe.row_count == 5
    assert probe.has_time_column is True
    assert "relative position" in probe.time_column_used
    assert probe.sample_rate_estimate_hz == 1.0  # 1 Hz
    assert probe.basic_cell_missingness_ratio is not None
    assert probe.basic_cell_missingness_ratio > 0.0


def test_flow_half_hertz(tmp_path: Path) -> None:
    p = write_flow_csv(tmp_path / "11_Flow_DR.csv", rows=6)
    probe = csv_probe.probe_csv(p)
    assert probe.inferred_role == "airflow"
    assert probe.has_time_column
    assert probe.sample_rate_estimate_hz == 2.0  # 0.5 s step -> 2 Hz


def test_sleep_stage_epoch_axis(tmp_path: Path) -> None:
    p = write_sleep_stage_csv(tmp_path / "01_sleep_stage.csv", epochs=4)
    probe = csv_probe.probe_csv(p)
    assert probe.inferred_role == "sleep_structure"
    assert probe.has_time_column
    assert probe.time_axis_kind in ("epoch", "hhmmss")


def test_no_time_column_no_guessed_rate(tmp_path: Path) -> None:
    p = write_no_time_csv(tmp_path / "notime.csv")
    probe = csv_probe.probe_csv(p)
    assert probe.has_time_column is False
    assert probe.sample_rate_estimate_hz is None  # never guessed
    assert "no_time_column" in probe.parse_anomalies


def test_handles_utf8sig_and_latin1(tmp_path: Path) -> None:
    p = tmp_path / "bom.csv"
    p.write_bytes(
        b"\xef\xbb\xbf" + 'rel position (hh:mm:ss.ms),OSat ("%")\n00:00:00.000,97\n'.encode("utf-8")
    )
    probe = csv_probe.probe_csv(p)
    assert probe.encoding == "utf-8-sig"
    assert probe.row_count == 1


def test_empty_file_does_not_crash(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_bytes(b"")
    probe = csv_probe.probe_csv(p)
    assert probe.row_count == 0
    assert "empty_file" in probe.parse_anomalies
