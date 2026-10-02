"""(9) Audio is BLOCKED: cohort count 0, no audio access, no .wav anywhere.

* ``audio_cohort_count`` is hard-coded 0 and the block reason is recorded;
* ``audio_features_present`` is False across the membership;
* no ``.wav`` path appears in any product (Stage 6 never reads audio);
* the completion report states the audio route is BLOCKED.
"""

from __future__ import annotations

import pandas as pd

from _stage6_fixtures import (
    build_stage6_world,
    default_windows,
    make_isolated_cfg,
    reports_out,
    run_stage6,
    splits_out,
)
from psg_audio_benchmark.evaluation import AUDIO_BLOCK_REASON, AUDIO_COHORT_COUNT


def _run(tmp_path):
    cfg = make_isolated_cfg(tmp_path)
    ids = build_stage6_world(cfg, windows=default_windows())
    summary, _r, _o = run_stage6(cfg, tmp_path, ids=ids)
    assert summary.overall_status == "PASS", summary.anomalies
    return summary


def test_audio_constants_and_summary(tmp_path):
    assert AUDIO_COHORT_COUNT == 0
    summary = _run(tmp_path)
    assert summary.audio_cohort_count == 0
    assert summary.audio_block_reason == AUDIO_BLOCK_REASON


def test_no_audio_membership_or_wav_paths(tmp_path):
    _summary = _run(tmp_path)
    core = pd.read_parquet(splits_out(tmp_path) / "core_cohort_window_membership.parquet")
    assert bool((core["audio_features_present"] == False).all())
    # no .wav referenced as DATA: scan machine-readable products only. (Markdown
    # reports legitimately say "does not read *.wav" as prose, so they are excluded.)
    hits = []
    for p in splits_out(tmp_path).iterdir():
        if not p.is_file():
            continue
        if ".wav" in p.name or "audio_feature" in p.name:
            hits.append(p.name)
        if p.suffix in (".csv", ".yaml", ".yml", ".json"):
            if ".wav" in p.read_text(encoding="utf-8", errors="ignore"):
                hits.append(p.name + ":wav-in-text")
    assert hits == [], hits


def test_completion_report_states_audio_blocked(tmp_path):
    _summary = _run(tmp_path)
    report = (reports_out(tmp_path) / "phase_06_completion_report.md").read_text(encoding="utf-8")
    assert "BLOCKED" in report
    assert "audio cohort count = **0**" in report
