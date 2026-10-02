"""File classification tests (prompt section 3.2).

Covers: smartphone vs recorder audio, multiple recorder segments, a single
bare ``_recorder.wav``, CSV role from header, JSON vs TXT, unknown/low-confidence
fallback. Recorder count is never assumed to be exactly two.
"""

from __future__ import annotations

from psg_audio_benchmark.data_audit import classifier


def test_smartphone_and_recorder_audio() -> None:
    c = classifier.classify_file("p/01_phone.wav", "01_phone.wav", "wav")
    assert c.modality_candidate == classifier.MODALITY_SMARTPHONE_AUDIO
    assert c.classification_confidence == "high"
    assert c.recorder_index is None


def test_recorder_segment_indices() -> None:
    c1 = classifier.classify_file("p/01_recorder_1.wav", "01_recorder_1.wav", "wav")
    c2 = classifier.classify_file("p/01_recorder_2.wav", "01_recorder_2.wav", "wav")
    assert c1.modality_candidate == classifier.MODALITY_RECORDER_AUDIO
    assert c1.recorder_index == 1
    assert c2.recorder_index == 2
    assert c1.classification_confidence == "high"


def test_single_bare_recorder_indexed_as_one() -> None:
    c = classifier.classify_file("p/03_recorder.wav", "03_recorder.wav", "wav")
    assert c.modality_candidate == classifier.MODALITY_RECORDER_AUDIO
    assert c.recorder_index == 1  # not None, not zero


def test_csv_roles_from_header() -> None:
    spo2 = classifier.classify_file(
        "p/01_SpO2.csv", "01_SpO2.csv", "csv",
        csv_header=['relative position (hh:mm:ss.ms)', 'absolute position (hh:mm:ss.ms)', 'OSat ("%")'],
    )
    hr = classifier.classify_file(
        "p/01_HR.csv", "01_HR.csv", "csv",
        csv_header=['relative position (hh:mm:ss.ms)', 'absolute position (hh:mm:ss.ms)', 'Heart Rate ("bpm")'],
    )
    flow = classifier.classify_file(
        "p/11_Flow_DR.csv", "11_Flow_DR.csv", "csv",
        csv_header=['relative position (hh:mm:ss.ms)', 'absolute position (hh:mm:ss.ms)', 'Flow_DR'],
    )
    sleep = classifier.classify_file(
        "p/01_sleep_stage.csv", "01_sleep_stage.csv", "csv",
        csv_header=['position (epoch)', 'absolute position (hh:mm:ss.ms)', 'Default Staging Set ("stage")'],
    )
    assert spo2.modality_candidate == classifier.MODALITY_SPO2
    assert hr.modality_candidate == classifier.MODALITY_HEART_RATE
    assert flow.modality_candidate == classifier.MODALITY_AIRFLOW
    assert sleep.modality_candidate == classifier.MODALITY_SLEEP_STRUCTURE
    assert all(c.classification_confidence == "high" for c in (spo2, hr, flow, sleep))


def test_json_and_txt_annotation() -> None:
    j = classifier.classify_file(
        "p/01_annotation.json", "01_annotation.json", "json",
        json_keys=["record_start", "awake_intervals", "events"],
    )
    t = classifier.classify_file("p/notes.txt", "notes.txt", "txt")
    assert j.modality_candidate == classifier.MODALITY_ANNOTATION_JSON
    assert j.classification_confidence == "high"
    assert t.modality_candidate == classifier.MODALITY_ANNOTATION_TXT


def test_unknown_and_low_confidence() -> None:
    # Audio with no device token in the name -> unknown audio, low confidence.
    u = classifier.classify_file("p/clip.wav", "clip.wav", "wav")
    assert u.modality_candidate == classifier.MODALITY_UNKNOWN
    assert u.classification_confidence == "low"
    # Unhandled extension -> unknown.
    u2 = classifier.classify_file("p/x.bin", "x.bin", "bin")
    assert u2.modality_candidate == classifier.MODALITY_UNKNOWN


def test_count_recorder_segments_handles_zero_one_many() -> None:
    cs = [
        classifier.classify_file("a/x_phone.wav", "x_phone.wav", "wav"),
        classifier.classify_file("a/x_recorder_1.wav", "x_recorder_1.wav", "wav"),
        classifier.classify_file("a/x_recorder_2.wav", "x_recorder_2.wav", "wav"),
        classifier.classify_file("a/x_recorder_3.wav", "x_recorder_3.wav", "wav"),
    ]
    assert classifier.count_recorder_segments(cs) == 3
    assert classifier.count_recorder_segments(cs[:1]) == 0
