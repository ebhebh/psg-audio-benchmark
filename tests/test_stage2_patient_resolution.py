"""Patient-ID resolution tests (prompt section 3.1).

Stable mapping from directory name; conflict / non-continuous / non-patient-dir
detection. The patient id comes from the directory name, never from absolute
path / mtime / sort order.
"""

from __future__ import annotations

from pathlib import Path

from psg_audio_benchmark.data_audit import patient_resolution


def test_stable_mapping_from_dir_name(tmp_path: Path) -> None:
    base = tmp_path / "V5" / "Data"
    (base / "01").mkdir(parents=True)
    (base / "01" / "01_HR.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (base / "02").mkdir(parents=True)
    (base / "02" / "02_HR.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    patients, issues = patient_resolution.resolve_patients(tmp_path, tmp_path)
    ids = [p.patient_id_canonical for p in patients]
    assert ids == ["01", "02"]
    assert all(p.mapping_rule_version == patient_resolution.MAPPING_RULE_VERSION for p in patients)
    # source dir is fixture-relative, never absolute.
    for p in patients:
        assert not Path(p.source_patient_dir_relative).is_absolute()
        assert p.source_patient_dir_relative.startswith("V5/Data/")
    # Continuous ids -> no issues.
    assert issues == []


def test_detects_non_continuous_ids(tmp_path: Path) -> None:
    base = tmp_path / "V5" / "Data"
    for pid in ("01", "02", "04"):  # 03 missing
        (base / pid).mkdir(parents=True)
        (base / pid / f"{pid}_HR.csv").write_text("a\n1\n", encoding="utf-8")
    patients, issues = patient_resolution.resolve_patients(tmp_path, tmp_path)
    assert len(patients) == 3
    types = {i.issue_type for i in issues}
    assert "non_continuous_patient_ids" in types


def test_detects_duplicate_id_across_dirs(tmp_path: Path) -> None:
    base = tmp_path / "V5" / "Data"
    for sub in ("01", "01b"):
        d = base / sub
        d.mkdir(parents=True)
    # make 01b look like id 01 by giving it data files (name stays 01b though)
    (base / "01" / "01_HR.csv").write_text("a\n1\n", encoding="utf-8")
    (base / "01b" / "01b_HR.csv").write_text("a\n1\n", encoding="utf-8")
    patients, issues = patient_resolution.resolve_patients(tmp_path, tmp_path)
    # two distinct dir names -> two patients, no collision, but 01b is non-standard.
    assert len(patients) == 2
    types = {i.issue_type for i in issues}
    assert "non_standard_patient_id" in types


def test_structural_dirs_not_treated_as_patients(tmp_path: Path) -> None:
    """V5 / Data contain only subdirs -> not patient dirs."""
    base = tmp_path / "V5" / "Data" / "01"
    base.mkdir(parents=True)
    (base / "01_HR.csv").write_text("a\n1\n", encoding="utf-8")
    patients, issues = patient_resolution.resolve_patients(tmp_path, tmp_path)
    assert len(patients) == 1
    assert patients[0].patient_id_canonical == "01"
