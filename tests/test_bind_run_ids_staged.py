"""Stage-26 staged reproduction-chain tests: the per-stage binder entry.

``tools/bind_run_ids.py --next-stage {3,4,5,6,6b,15}`` must need ONLY the
upstream products the target stage genuinely requires, so a FIRST
reproduction can advance one stage at a time (the Stage-25 whole-chain check
created a dependency cycle: Stage-5 binding demanded Stage 5/6b outputs).

These tests build a synthetic chain INCREMENTALLY from a blank licensed root
and pin, at every boundary:

* per-stage happy path: exactly the needed upstream exists -> no problems,
  correct bound command (the Stage-5 entry needs NO Stage-5/6b products);
* earlier entries stay refused until their upstream exists (ordering);
* printed commands are accepted by the REAL script CLI parsers (argparse
  round-trip of every printed flag — not string containment);
* the printed bound ids are the synthetic NEW ids, never the frozen
  historical pins, and the real Stage-4/5 CLIs BLOCK on their historical
  defaults in this synthetic root (no silent old-id fallback);
* ``scripts/06b --v1-comparator-run-id`` really redirects the Stage-6B
  runner to the bound v1 run (real runner, dry-run) and refuses wrong-stage
  ids; the frozen config pin stays untouched and the override is recorded;
* Stage-15 binding is fail-closed: unapproved, mirror/run signature
  mismatch, tampered mirror products, product missing from the run dir,
  empty v2_files, missing cohort field, cohort (queue) drift, empty lineage,
  missing Stage-15 mirrors, traversal/absolute signature relpaths and unsafe
  signature run ids are all refused;
* stale license: a passed record without a currently-valid evidence file
  (reverted to the unfilled template) is refused;
* ``provenance_gate.split_run`` in the frozen Stage-15 config is DECLARATIVE:
  the real Stage-15b runner completes with a bogus value (honest
  non-participation proof — the executable gate is the mirror/signature
  verification the binder performs).

No real data is read; everything is synthetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from _stage2_fixtures import write_complete_license_evidence

_REPO = Path(__file__).resolve().parents[1]

HISTORICAL_PINS = {
    "stage3-annotation-sync-20260807T154558Z",
    "stage4-csv-window-index-20260807T170636Z",
    "stage5-physiology-features-20260807T232931Z",
    "stage6-patient-splits-20260808T002136Z",
    "stage6b-split-balance-v2-20260808T020039Z",
}


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "bind_run_ids_staged", _REPO / "tools" / "bind_run_ids.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), _REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _license_ok(root: Path) -> None:
    write_complete_license_evidence(root / "docs")
    lg = root / "reports/data_audit"
    lg.mkdir(parents=True, exist_ok=True)
    (lg / "license_gate_passed.md").write_text(
        "synthetic passed record\n", encoding="utf-8")


def _add_stage3(root: Path, rid: str) -> None:
    d = root / "annotations/runs" / rid
    _parquet(d / "parsed_events.parquet", pd.DataFrame({"event": [1]}))
    _parquet(d / "awake_intervals_canonical.parquet",
             pd.DataFrame({"patient_id": ["01"]}))
    _parquet(d / "signal_time_ranges.parquet", pd.DataFrame({"m": ["hr"]}))
    _parquet(d / "record_time_anchors.parquet", pd.DataFrame({"patient_id": ["01"]}))
    (d / "audio_time_alignment_inventory.csv").write_text(
        f"run_id\n{rid}\n", encoding="utf-8")
    (root / "annotations/LATEST_RUN.txt").write_text(rid + "\n", encoding="utf-8")


def _add_stage4(root: Path, rid: str, s3: str) -> None:
    d = root / "data/manifests/runs" / rid
    _parquet(d / "csv_window_index.parquet", pd.DataFrame({"window_id": ["w1"]}))
    (d / "patient_window_summary.csv").write_text("patient_id\n01\n", encoding="utf-8")
    (d / "windowing_config_resolved.yaml").write_text(
        f"run_id: {rid}\ninput_stage3_run_id: {s3}\n", encoding="utf-8")
    (root / "reports/windowing").mkdir(parents=True, exist_ok=True)
    (root / "reports/windowing/LATEST_RUN.txt").write_text(rid + "\n", encoding="utf-8")


def _add_stage5(root: Path, rid: str, s3: str, s4: str, *, lineage: str = "ok") -> None:
    d = root / "features/physiology/runs" / rid
    _parquet(d / "hr_window_features.parquet", pd.DataFrame({"hr_mean": [1.0]}))
    _parquet(d / "spo2_window_features.parquet", pd.DataFrame({"spo2_min": [0.9]}))
    if lineage == "ok":
        txt = (f"input_stage3_run_id: {s3}\ninput_stage4_run_id: {s4}\n"
               f"run_id: {rid}\n")
    elif lineage == "empty":
        txt = f"run_id: {rid}\n"  # required lineage fields ABSENT (fail closed)
    else:  # mismatch
        txt = (f"input_stage3_run_id: {s3}\n"
               f"input_stage4_run_id: stage4-csv-window-index-11111111T111111Z\n"
               f"run_id: {rid}\n")
    (d / "physiology_features_resolved.yaml").write_text(txt, encoding="utf-8")
    (root / "reports/feature_extraction").mkdir(parents=True, exist_ok=True)
    (root / "reports/feature_extraction/LATEST_RUN.txt").write_text(
        rid + "\n", encoding="utf-8")


def _add_stage6(root: Path, rid: str, s3: str, s4: str, s5: str, *,
                lineage: str = "ok", products: bool = True) -> None:
    d = root / "splits/runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    if products:
        pd.DataFrame({"patient_id": ["01"], "outer_fold": [0]}).to_csv(
            d / "outer_patient_folds_core.csv", index=False)
        pd.DataFrame({"patient_id": ["01"]}).to_csv(
            d / "airflow_outer_fold_inheritance.csv", index=False)
    sig = {"run_id": rid, "patient_split_stage": "stage6",
           "stage3_run_id": s3 if lineage == "ok" else "",
           "stage4_run_id": s4 if lineage == "ok" else "",
           "stage5_run_id": s5 if lineage == "ok" else ""}
    (d / "input_version_signature.json").write_text(
        json.dumps(sig), encoding="utf-8")
    (root / "reports/evaluation").mkdir(parents=True, exist_ok=True)
    (root / "reports/evaluation/LATEST_RUN.txt").write_text(rid + "\n", encoding="utf-8")


def _add_stage15_mirrors(root: Path, pids: list[str]) -> None:
    _parquet(root / "annotations/parsed_events.parquet", pd.DataFrame({"e": [1]}))
    _parquet(root / "data/manifests/csv_window_index.parquet",
             pd.DataFrame({"window_id": ["w1"]}))
    for mod in ("hr", "spo2", "airflow"):
        _parquet(root / f"features/physiology/{mod}_window_features.parquet",
                 pd.DataFrame({"v": [1.0]}))
    _parquet(root / "splits/core_cohort_window_membership.parquet",
             pd.DataFrame({"patient_id": [p for p in pids for _ in range(4)],
                           "window_id": [f"{p}-w{i}" for p in pids
                                         for i in range(4)]}))


def _add_stage6b(root: Path, rid: str, s3: str, s4: str, s5: str, *,
                 approval: str = "approved_for_modeling",
                 mirror_sig_same: bool = True,
                 run_dir_products: bool = True,
                 v2_files: dict | None = None,
                 drop_cohort_field: str = "",
                 lineage: str = "ok",
                 cohort_drift: bool = False) -> dict:
    """Write a synthetic Stage-6b run dir + fixed mirrors the way production
    6b does (run-dir products, then mirrored copies + LATEST)."""
    pids = ["01", "02", "03", "04", "05"]
    _add_stage15_mirrors(root, pids)
    d = root / "splits/runs" / rid
    d.mkdir(parents=True, exist_ok=True)
    splits = root / "splits"

    v2_local = {}
    frames = {
        "outer_patient_folds_core_v2.csv":
            pd.DataFrame({"patient_id": pids, "outer_fold": [0, 1, 2, 3, 4]}),
        "inner_patient_folds_core_v2.parquet":
            pd.DataFrame({"patient_id": pids * 4,
                          "outer_fold": [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
                          "inner_validation_fold": [1, 2, 3, 4, 1] * 4}),
        "airflow_outer_fold_inheritance_v2.csv":
            pd.DataFrame({"patient_id": pids, "outer_fold": [0, 1, 2, 3, 4]}),
        "patient_level_balance_inputs.csv":
            pd.DataFrame({"patient_id": pids}),
        "split_candidate_comparison.csv":
            pd.DataFrame({"scope": ["outer"]}),
    }
    for name, df in frames.items():
        p = d / name
        if name.endswith(".csv"):
            df.to_csv(p, index=False)
        else:
            df.to_parquet(p, index=False)
        v2_local[name] = _sha(p)
        shutil.copyfile(p, splits / name)
    if not run_dir_products:
        (d / "outer_patient_folds_core_v2.csv").unlink()

    sig_v2 = v2_files if v2_files is not None else {
        f"v2/{n}": {"relpath": n, "sha256": h} for n, h in v2_local.items()}
    live_pids = pids + (["99"] if cohort_drift else [])
    cohort = {
        "n_patients": len(pids), "n_windows": 100 * len(pids),
        "n_positive": 40 * len(pids), "n_negative": 60 * len(pids),
        "overall_positive_rate": 0.4,
        "patient_set_sha256": hashlib.sha256(
            "\n".join(sorted(pids)).encode("utf-8")).hexdigest(),
    }
    if drop_cohort_field:
        cohort.pop(drop_cohort_field, None)
    sig = {
        "run_id": rid,
        "approval_status": approval,
        "v2_files": sig_v2,
        "cohort": cohort,
        "lineage": {
            "stage3_run_id": s3 if lineage == "ok" else "",
            "stage4_run_id": s4 if lineage == "ok" else "",
            "stage5_run_id": s5 if lineage == "ok" else "",
            "v1_comparator_run_id": "stage6-patient-splits-20990101T000004Z",
        },
    }
    sig_text = json.dumps(sig, indent=2, sort_keys=True)
    (d / "split_balance_version_signature.json").write_text(sig_text, encoding="utf-8")
    (splits / "split_balance_version_signature.json").write_text(
        sig_text if mirror_sig_same else json.dumps(
            {**sig, "approval_status": "approved_for_modeling_tampered"},
            indent=2, sort_keys=True), encoding="utf-8")
    (splits / "LATEST_RUN.txt").write_text(rid + "\n", encoding="utf-8")
    if cohort_drift:  # add a patient to the LIVE mirror after signing
        p = splits / "core_cohort_window_membership.parquet"
        df = pd.read_parquet(p)
        df = pd.concat([df, pd.DataFrame({"patient_id": ["99"],
                                          "window_id": ["99-w0"]})],
                       ignore_index=True)
        _parquet(p, df)
        assert live_pids  # (documentation only)
    return {"ids": pids}


def _full_chain_ids() -> dict:
    return {
        "stage3": "stage3-annotation-sync-20990101T000000Z",
        "stage4": "stage4-csv-window-index-20990101T000001Z",
        "stage5": "stage5-physiology-features-20990101T000002Z",
        "stage6": "stage6-patient-splits-20990101T000004Z",
        "stage6b": "stage6b-split-balance-v2-20990101T000003Z",
    }


def _build_chain(root: Path, upto: str, **kw) -> dict:
    """Build the synthetic chain from scratch up to and including ``upto``."""
    ids = _full_chain_ids()
    order = ["stage3", "stage4", "stage5", "stage6", "stage6b"]
    _license_ok(root)
    for name in order[: order.index(upto) + 1]:
        if name == "stage3":
            _add_stage3(root, ids["stage3"])
        elif name == "stage4":
            _add_stage4(root, ids["stage4"], ids["stage3"])
        elif name == "stage5":
            _add_stage5(root, ids["stage5"], ids["stage3"], ids["stage4"],
                        lineage=kw.get("stage5_lineage", "ok"))
        elif name == "stage6":
            _add_stage6(root, ids["stage6"], ids["stage3"], ids["stage4"],
                        ids["stage5"], lineage=kw.get("stage6_lineage", "ok"),
                        products=kw.get("stage6_products", True))
        elif name == "stage6b":
            _add_stage6b(root, ids["stage6b"], ids["stage3"], ids["stage4"],
                         ids["stage5"], **kw.get("stage6b_kw", {}))
    return ids


# ---------------------------------------------------------------------------
# 1. The dependency cycle is gone: per-stage entries need only their upstream
# ---------------------------------------------------------------------------

def test_blank_root_binds_stage3_and_refuses_everything_later(tool, tmp_path):
    root = tmp_path / "repo"
    _license_ok(root)
    b, probs = tool.check_stage(root, "3")
    assert probs == []
    assert b["commands"] == ["python scripts/03_annotation_sync.py"]
    for stage in ("4", "5", "6", "6b", "15"):
        _, probs = tool.check_stage(root, stage)
        assert probs, f"--next-stage {stage} must be refused on a blank root"
        assert any("pointer missing" in p for p in probs)


def test_stage5_entry_needs_only_stage3_and_4(tool, tmp_path):
    """THE Stage-25 defect: binding Stage 5 must NOT require Stage-5/6b."""
    root = tmp_path / "repo"
    ids = _build_chain(root, "stage4")
    assert not (root / "reports/feature_extraction").exists()
    assert not (root / "splits").exists()
    b, probs = tool.check_stage(root, "5")
    assert probs == []
    cmd = b["commands"][0]
    assert f'--input-stage4-run-id "{ids["stage4"]}"' in cmd
    assert f'--input-stage3-run-id "{ids["stage3"]}"' in cmd
    # later entries still refused (ordering is enforced, not skipped)
    for stage in ("6", "6b", "15"):
        _, probs = tool.check_stage(root, stage)
        assert any("pointer missing" in p for p in probs)


def test_stage6_entry_requires_complete_stage5_lineage(tool, tmp_path):
    root = tmp_path / "repo"
    _build_chain(root, "stage5")
    b, probs = tool.check_stage(root, "6")
    assert probs == []
    assert '--input-stage5-run-id' in b["commands"][0]

    for mode, needle in (("empty", "is empty"), ("mismatch", "cites stage4")):
        root2 = root.parent / f"repo6_{mode}"
        _build_chain(root2, "stage5", stage5_lineage=mode)
        _, probs = tool.check_stage(root2, "6")
        assert any(needle in p for p in probs), (mode, probs)


def test_stage6b_entry_binds_v1_comparator(tool, tmp_path):
    root = tmp_path / "repo"
    ids = _build_chain(root, "stage6")
    b, probs = tool.check_stage(root, "6b")
    assert probs == []
    assert b["commands"] == [
        f'python scripts/06b_split_balance_optimize.py '
        f'--v1-comparator-run-id "{ids["stage6"]}"']

    # v1 products missing / empty v1 lineage -> refused
    r2 = root.parent / "repo6b_noprod"
    _build_chain(r2, "stage6", stage6_products=False)
    _, probs = tool.check_stage(r2, "6b")
    assert any("product missing" in p for p in probs)
    r3 = root.parent / "repo6b_nolineage"
    _build_chain(r3, "stage6", stage6_lineage="empty")
    _, probs = tool.check_stage(r3, "6b")
    assert any("stage6 v1 signature stage4_run_id is empty" in p for p in probs)

    # pointer already tracks a 6b run -> clear refusal, no guessing
    (root / "reports/evaluation/LATEST_RUN.txt").write_text(
        ids["stage6b"] + "\n", encoding="utf-8")
    _, probs = tool.check_stage(root, "6b")
    assert any("a 6b already published the mirrors" in p for p in probs)


def test_stage15_entry_requires_approved_consistent_6b(tool, tmp_path):
    root = tmp_path / "repo"
    ids = _build_chain(root, "stage6b")
    b, probs = tool.check_stage(root, "15")
    assert probs == []
    assert b["commands"][0] == "python scripts/15_bspc_reanalysis.py"
    assert b["run_ids"]["stage6b"] == ids["stage6b"]
    assert any("declarative" in c for c in b["commands"])

    cases = {  # name -> (stage6b kwargs, expected problem needle)
        "unapproved": ({"approval": "not_approved_balance_target_not_met"},
                       "NOT approved"),
        "mirror_sig_differs": ({"mirror_sig_same": False},
                               "mirror gate: fixed-mirror signature differs"),
        "product_missing_from_run_dir": ({"run_dir_products": False},
                                         "absent from the 6b run dir"),
        "empty_v2_files": ({"v2_files": {}}, "v2_files missing/empty"),
        "missing_cohort_field": ({"drop_cohort_field": "n_windows"},
                                 "cohort.n_windows missing/empty"),
        "cohort_drift": ({"cohort_drift": True}, "patient-split gate"),
        "lineage_empty": ({"lineage": "empty"}, "signature stage4_run_id is empty"),
    }
    for name, (kw, needle) in cases.items():
        r = root.parent / f"repo15_{name}"
        _build_chain(r, "stage6b", stage6b_kw=kw)
        _, probs = tool.check_stage(r, "15")
        assert any(needle in p for p in probs), (name, probs)

    # tampered MIRROR product (run-dir copy intact) -> refused
    r = root.parent / "repo15_mirror_tamper"
    _build_chain(r, "stage6b")
    tgt = r / "splits/outer_patient_folds_core_v2.csv"
    tgt.write_text(tgt.read_text(encoding="utf-8").replace(
        "patient_id", "patient_id_tampered"), encoding="utf-8")
    _, probs = tool.check_stage(r, "15")
    assert any("SHA-256 mismatch" in p for p in probs)

    # missing Stage-15 mirror input -> refused
    r = root.parent / "repo15_no_mirror"
    _build_chain(r, "stage6b")
    (r / "features/physiology/hr_window_features.parquet").unlink()
    _, probs = tool.check_stage(r, "15")
    assert any("stage-15 mirror missing/empty" in p for p in probs)


def test_traversal_and_injection_refused(tool, tmp_path):
    root = tmp_path / "repo"
    ids = _build_chain(root, "stage6b")

    # pointer-level traversal / shell metacharacters -> refused, nothing bound
    for evil in ("../escape", "C:\\abs\\path", "ok; rm -rf /", "a b c", ".."):
        (root / "annotations/LATEST_RUN.txt").write_text(evil, encoding="utf-8")
        _, probs = tool.check_stage(root, "4")
        assert any("unsafe run id" in p for p in probs), evil

    (root / "annotations/LATEST_RUN.txt").write_text(
        ids["stage3"] + "\n", encoding="utf-8")

    # signature-level traversal / absolute / too-deep relpaths -> refused
    r = root.parent / "repo_trav"
    _build_chain(r, "stage6b")
    sig_path = r / "splits/split_balance_version_signature.json"
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    for evil_rel in ("../evil.csv", "C:/evil.csv", "a/b/c.csv"):
        sig["v2_files"] = {"v2/x.csv": {"relpath": evil_rel, "sha256": "0" * 64}}
        sig_path.write_text(json.dumps(sig), encoding="utf-8")
        (r / "splits/runs" / ids["stage6b"] /
         "split_balance_version_signature.json").write_text(
            json.dumps(sig), encoding="utf-8")
        _, probs = tool.check_stage(r, "15")
        assert any("not a safe relative path" in p for p in probs), evil_rel

    # unsafe signature run_id -> refused
    sig["v2_files"] = None
    sig["run_id"] = "../../evil"
    for p_ in (sig_path, r / "splits/runs" / ids["stage6b"] /
               "split_balance_version_signature.json"):
        p_.write_text(json.dumps(sig), encoding="utf-8")
    _, probs = tool.check_stage(r, "15")
    assert any("run_id missing/unsafe" in p for p in probs)


def test_stale_license_record_refused(tool, tmp_path):
    """Passed record present but the CURRENT evidence is the unfilled shipped
    template (or absent) -> refused (expired/stale license report)."""
    root = tmp_path / "repo"
    _build_chain(root, "stage4")
    ev = root / "docs/license_evidence/primary_dataset_license_confirmation.md"
    template = (_REPO / "docs/license_evidence" /
                "primary_dataset_license_confirmation.md").read_text(
                    encoding="utf-8")
    ev.write_text(template, encoding="utf-8")
    _, probs = tool.check_stage(root, "5")
    assert any("CURRENT evidence re-check" in p for p in probs)
    ev.unlink()
    _, probs = tool.check_stage(root, "5")
    assert any("CURRENT evidence re-check" in p for p in probs)
    assert any("license_gate_passed.md" not in p for p in probs)  # record intact


def test_bound_ids_are_new_not_historical_pins(tool, tmp_path):
    root = tmp_path / "repo"
    ids = _build_chain(root, "stage6")
    _, probs = tool.check_stage(root, "6b")
    assert probs == []
    for value in ids.values():
        assert value not in HISTORICAL_PINS
    cmd = tool.check_stage(root, "6")[0]
    assert ids["stage6"] not in cmd["commands"][0]  # 6 cmd binds 4/5, not 6
    assert ids["stage4"] in cmd["commands"][0] and ids["stage5"] in cmd["commands"][0]
    for pin in HISTORICAL_PINS:
        assert pin not in cmd["commands"][0]


# ---------------------------------------------------------------------------
# 2. Printed commands are accepted by the REAL CLI parsers (round-trip)
# ---------------------------------------------------------------------------

def _parse_with_real_cli(tool, tmp_path, upto: str, target: str) -> dict:
    root = tmp_path / "cli"
    _build_chain(root, upto)
    b, probs = tool.check_stage(root, target)
    assert probs == []
    out = {}
    import shlex
    for line in b["commands"]:
        if line.startswith("#"):
            continue
        toks = shlex.split(line)
        assert toks[0] == "python"
        script = toks[1]
        mod = _load_script(Path(script).name)
        args = mod._build_parser().parse_args(toks[2:])
        out[script] = args
    return out


def test_printed_commands_round_trip_through_real_cli_parsers(tool, tmp_path):
    parsed = {}
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage3", "3"))
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage4", "4"))
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage4", "5"))
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage5", "6"))
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage6", "6b"))
    parsed.update(_parse_with_real_cli(tool, tmp_path, "stage6b", "15"))

    ids = _full_chain_ids()
    a4 = parsed["scripts/04_window_index.py"]
    assert a4.input_run_id == ids["stage3"]  # 04's real flag binds the Stage-3 run
    a5 = parsed["scripts/05_physiology_features.py"]
    assert a5.input_stage4_run_id == ids["stage4"]
    assert a5.input_stage3_run_id == ids["stage3"]
    a6 = parsed["scripts/06_patient_splits.py"]
    assert a6.input_stage4_run_id == ids["stage4"]
    assert a6.input_stage5_run_id == ids["stage5"]
    a6b = parsed["scripts/06b_split_balance_optimize.py"]
    assert a6b.v1_comparator_run_id == ids["stage6"]
    assert "scripts/15_bspc_reanalysis.py" in parsed  # bare command parses


# ---------------------------------------------------------------------------
# 3. Real CLI execution: historical defaults BLOCK; bound ids run (dry-run)
# ---------------------------------------------------------------------------

def _make_cli_proj(tmp_path: Path, upto: str) -> Path:
    proj = tmp_path / "proj"
    shutil.copytree(_REPO / "config", proj / "config")
    _license_ok(proj)
    for sub in ("data/raw", "logs", "reports/windowing", "reports/evaluation",
                "reports/feature_extraction", "annotations", "splits"):
        (proj / sub).mkdir(parents=True, exist_ok=True)
    return proj


def test_real_stage5_cli_blocks_on_historical_default_and_runs_bound(tool, tmp_path,
                                                                     monkeypatch):
    proj = _make_cli_proj(tmp_path, "stage4")
    ids = _full_chain_ids()
    _add_stage3(proj, ids["stage3"])
    _add_stage4(proj, ids["stage4"], ids["stage3"])
    s05 = _load_script("05_physiology_features.py")
    monkeypatch.setenv("PSG_BENCHMARK_ROOT", str(proj))

    # (a) defaults = frozen historical pins: absent in a fresh reproduction ->
    #     BLOCKED (exit 1). No silent fallback to the authors' runs.
    rc = s05.main(["--dry-run"])
    assert rc == 1

    # (b) the command the binder printed for THIS chain executes (exit 0)
    b, probs = tool.check_stage(proj, "5")
    assert probs == []
    import shlex
    toks = shlex.split(b["commands"][0])[2:]  # drop 'python scripts/05_...'
    rc = s05.main(["--dry-run"] + toks)
    assert rc == 0


def test_real_stage4_cli_blocks_on_historical_default_and_runs_bound(tool, tmp_path,
                                                                     monkeypatch):
    proj = _make_cli_proj(tmp_path, "stage3")
    ids = _full_chain_ids()
    _add_stage3(proj, ids["stage3"])
    s04 = _load_script("04_window_index.py")
    monkeypatch.setenv("PSG_BENCHMARK_ROOT", str(proj))

    assert s04.main(["--dry-run"]) == 1  # historical default pin absent
    b, probs = tool.check_stage(proj, "4")
    assert probs == []
    import shlex
    toks = shlex.split(b["commands"][0])[2:]
    assert s04.main(["--dry-run"] + toks) == 0


# ---------------------------------------------------------------------------
# 4. Stage-6B v1 comparator override (real runner, frozen config untouched)
# ---------------------------------------------------------------------------

def test_stage6b_v1_comparator_override_redirects_real_runner(tmp_path):
    from psg_audio_benchmark.config import Config
    from psg_audio_benchmark.evaluation import Stage6bOptions, Stage6bRunner
    from psg_audio_benchmark.evaluation.split_balance_config import (
        SplitBalanceConfigError,
    )
    from _stage6b_fixtures import (
        S6B_CONFIG_HASH, synthetic_balanceable_burden, write_synthetic_v1_run,
    )

    proj = tmp_path / "proj"
    shutil.copytree(_REPO / "config", proj / "config")
    for sub in ("data/raw", "splits", "reports/evaluation", "logs"):
        (proj / sub).mkdir(parents=True, exist_ok=True)
    cfg = Config(project_root=proj)

    v1 = "stage6-patient-splits-20990101T000004Z"
    write_synthetic_v1_run(cfg, v1, synthetic_balanceable_burden(50))

    # wrong-stage id refused (must be a stage6-patient-splits- v1 run)
    with pytest.raises(SplitBalanceConfigError):
        Stage6bRunner(cfg=cfg, options=Stage6bOptions(
            run_id="stage6b-split-balance-v2-20990101T000005Z",
            config_hash=S6B_CONFIG_HASH,
            config_path=proj / "config/split_balance_optimization.yaml",
            v1_comparator_run_id="stage6b-split-balance-v2-20990101T000003Z"))

    # real override: redirects the v1-product gate to the bound run dir and
    # records provenance; the frozen pin stays untouched.
    runner = Stage6bRunner(cfg=cfg, options=Stage6bOptions(
        run_id="stage6b-split-balance-v2-20990101T000005Z",
        config_hash=S6B_CONFIG_HASH,
        config_path=proj / "config/split_balance_optimization.yaml",
        v1_comparator_run_id=v1))
    assert runner.config.v1_comparator_run_id == v1
    assert runner.v1_comparator_source == "cli_override"
    assert runner.frozen_v1_comparator_pin == "stage6-patient-splits-20260808T002136Z"
    import yaml as _yaml
    frozen = _yaml.safe_load(
        (proj / "config/split_balance_optimization.yaml").read_text(
            encoding="utf-8"))["split_balance_optimization"]
    assert frozen["v1_comparator_run_id"] == runner.frozen_v1_comparator_pin

    summary = runner.run()  # dry_run=False but v1 gate + optimize on synthetics
    assert summary.status in ("PASS", "PASS WITH WARNINGS",
                              "approved_for_modeling",
                              "not_approved_balance_target_not_met")
    # the run-dir signature lineage records the OVERRIDDEN comparator
    sig = json.loads((cfg.path("splits") / "runs" /
                      "stage6b-split-balance-v2-20990101T000005Z" /
                      "split_balance_version_signature.json").read_text(
                          encoding="utf-8"))
    assert sig["lineage"]["v1_comparator_run_id"] == v1

    # a NON-existent override id fails closed at the v1-product gate
    runner2 = Stage6bRunner(cfg=cfg, options=Stage6bOptions(
        run_id="stage6b-split-balance-v2-20990101T000006Z",
        config_hash=S6B_CONFIG_HASH,
        config_path=proj / "config/split_balance_optimization.yaml",
        v1_comparator_run_id="stage6-patient-splits-20990101T000099Z",
        dry_run=True))
    s2 = runner2.run()
    assert s2.status == "BLOCKED"
    assert any("v1_product_missing" in a for a in s2.anomalies)


# ---------------------------------------------------------------------------
# 5. Stage-15 provenance_gate.split_run is declarative (honest verification)
# ---------------------------------------------------------------------------

def test_stage15_provenance_gate_is_declarative(tmp_path):
    """The frozen Stage-15 config's provenance_gate.split_run is NOT a runtime
    gate: the real Stage-15b runner completes even with a bogus value (it
    reads the fixed mirrors directly). The EXECUTABLE gate is the
    mirror/signature verification performed by bind_run_ids --next-stage 15."""
    from test_release_synthetic_e2e import _build_synthetic_root
    from psg_audio_benchmark.bspc_revision import runner as R

    root = tmp_path / "synthroot"
    _build_synthetic_root(root)
    cfg_yaml = root / "config/bspc_revision_analysis.yaml"
    txt = cfg_yaml.read_text(encoding="utf-8")
    assert "split_run: stage6b-split-balance-v2-20260808T020039Z" in txt
    cfg_yaml.write_text(
        txt.replace("split_run: stage6b-split-balance-v2-20260808T020039Z",
                    "split_run: BOGUS-NOT-A-RUN-ID"), encoding="utf-8")
    result = R.run(project_root=root, run_id="synthetic-declarative-proof")
    assert result["onset_matches_stage4"] is True  # ran to completion
