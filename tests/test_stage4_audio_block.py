"""Stage-4 boundary case 7: audio_window_eligible is always False, and the runner
rejects any attempt to enable audio eligibility.

Guarantees (prompt section 6.7):
  * every window in csv_window_index has audio_window_eligible=False with a
    block reason;
  * loading a windowing.yaml with audio_window_eligible:true (or allowing
    substitute anchors) raises WindowingConfigError, so Stage4Runner construction
    refuses it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from psg_audio_benchmark.windowing import (
    Stage4Options,
    Stage4Runner,
    WindowingConfigError,
)

from _stage4_fixtures import (
    DEFAULT_INPUT_RUN_ID,
    _patient_spec,
    build_input_run,
    make_isolated_cfg,
)


def test_case7_every_window_audio_eligible_false(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    runner = Stage4Runner(
        cfg=cfg,
        run_metadata={"run_id": "iso-stage4-case7a", "config_hash": "realhash1234"},
        options=Stage4Options(
            output_root=tmp_path / "outputs", relpath_base=tmp_path,
            input_run_id=DEFAULT_INPUT_RUN_ID,
        ),
    )
    summary = runner.run()
    assert summary.audio_window_eligible_count == 0
    mdir = tmp_path / "outputs" / "data" / "manifests" / "runs" / "iso-stage4-case7a"
    idx = pd.read_parquet(mdir / "csv_window_index.parquet")
    assert (idx["audio_window_eligible"] == False).all()
    assert (idx["audio_block_reason"] == "no_trustworthy_audio_time_anchor_unresolved").all()
    # the synthetic input carried only unresolved audio
    assert summary.audio_window_eligible_count == 0


def _write_bad_windowing_yaml(tmp_path: Path, **overrides) -> Path:
    import yaml as _yaml

    cfg_dict = {
        "windowing": {
            "window_length_s": 30.0,
            "hop_step_s": 30.0,
            "boundary": "half_open_start_inclusive_end_exclusive",
            "coordinate": "relative_to_record_start",
            "candidate_time_domain": "core_signal_intersection",
            "core_signal_modalities": ["heart_rate", "spo2"],
            "optional_signal_modalities": ["airflow"],
            "max_awake_overlap_fraction": 0.0,
            "audio": {
                "audio_window_eligible": False,
                "block_reason": "no_trustworthy_audio_time_anchor_unresolved",
                "allow_substitute_anchors": False,
            },
        }
    }
    for k, v in overrides.items():
        if k in ("audio_window_eligible", "allow_substitute_anchors"):
            cfg_dict["windowing"]["audio"][k] = v
        else:
            cfg_dict["windowing"][k] = v
    p = tmp_path / "bad_windowing.yaml"
    p.write_text(_yaml.safe_dump(cfg_dict), encoding="utf-8")
    return p


def test_case7_runner_rejects_enabling_audio_eligibility(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    bad = _write_bad_windowing_yaml(tmp_path, audio_window_eligible=True)
    import pytest

    with pytest.raises(WindowingConfigError):
        Stage4Runner(
            cfg=cfg,
            run_metadata={"run_id": "iso-stage4-case7b", "config_hash": "realhash1234"},
            options=Stage4Options(
                output_root=tmp_path / "outputs", relpath_base=tmp_path,
                input_run_id=DEFAULT_INPUT_RUN_ID, config_path=bad,
            ),
        )


def test_case7_runner_rejects_substitute_anchors(tmp_path: Path) -> None:
    cfg = make_isolated_cfg(tmp_path)
    build_input_run(
        cfg, DEFAULT_INPUT_RUN_ID,
        [_patient_spec(patient_id="01", hr=(0.0, 120.0), spo2=(0.0, 120.0))],
    )
    bad = _write_bad_windowing_yaml(tmp_path, allow_substitute_anchors=True)
    import pytest

    with pytest.raises(WindowingConfigError):
        Stage4Runner(
            cfg=cfg,
            run_metadata={"run_id": "iso-stage4-case7c", "config_hash": "realhash1234"},
            options=Stage4Options(
                output_root=tmp_path / "outputs", relpath_base=tmp_path,
                input_run_id=DEFAULT_INPUT_RUN_ID, config_path=bad,
            ),
        )
