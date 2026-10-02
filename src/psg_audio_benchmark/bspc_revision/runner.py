"""Stage 15b — re-analysis runner.

Orchestrates the duration-overlap label revision + model/feature/lag battery on the
inherited frozen v2 split, computes window / patient-macro / event-level / calibration
metrics with patient-cluster and paired bootstrap uncertainty, runs the airflow
shared-source secondary, and writes everything to isolated run dirs. Stage 2-14 products
are never modified; audio stays BLOCKED.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from ..modeling.metrics import compute_metrics
from ..modeling.bootstrap import patient_cluster_bootstrap
from ..modeling.airflow_increment.paired_bootstrap import paired_patient_cluster_bootstrap
from . import labels as L
from . import features as F
from . import nested_cv as N
from . import eval_metrics as E

SEED = N.SEED


# ----------------------------------------------------------------- frame building
def build_frame(*, label_result, fm, outer_map) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the analyzable frame for one (label variant, lag): lag-kept, non-indeterminate
    windows with features, label, patient, outer fold, window times. Returns (frame, X, y,
    patient_ids, outer_fold)."""
    lab = label_result.table[["window_id", "patient_id", "window_start_relative_to_record_start",
                              "window_end_relative_to_record_start", "label", "max_overlap_s"]].copy()
    fm_df = pd.DataFrame({"window_id": fm.window_id})
    keep = set(fm.window_id.tolist())
    lab = lab[lab["window_id"].isin(keep) & (lab["label"] != L.INDETERMINATE)].copy()
    lab = lab.merge(fm_df, on="window_id", how="inner")
    # reorder to fm order
    order = {w: i for i, w in enumerate(fm.window_id.tolist())}
    lab["__o"] = lab["window_id"].map(order)
    lab = lab.sort_values("__o").drop(columns="__o").reset_index(drop=True)
    # X aligned to lab order
    pos = {w: i for i, w in enumerate(fm.window_id.tolist())}
    src = np.array([pos[w] for w in lab["window_id"].tolist()])
    X = fm.X[src]
    y = lab["label"].to_numpy(dtype=int)
    pids = lab["patient_id"].astype(str).to_numpy()
    ofold = np.array([outer_map[str(p)] for p in pids], dtype=int)
    lab = lab.rename(columns={"window_start_relative_to_record_start": "window_start",
                              "window_end_relative_to_record_start": "window_end"})
    return lab, X, y, pids, ofold


def evaluate_oof(*, oof: pd.DataFrame, retained_events: pd.DataFrame,
                 window_starts: pd.DataFrame) -> Dict[str, object]:
    """Window pooled + per-fold + patient-macro + event-level + calibration for one OOF."""
    oof = oof.merge(window_starts[["window_id", "window_start", "window_end"]], on="window_id", how="left")
    y = oof["y_true"].to_numpy()
    p = oof["y_prob"].to_numpy()
    thr = float(oof["threshold"].iloc[0]) if len(oof) else 0.5
    pooled = compute_metrics(y_true=y, y_prob=p, threshold=thr)
    per_fold = {}
    for k, g in oof.groupby("outer_fold"):
        gg = compute_metrics(y_true=g["y_true"].to_numpy(), y_prob=g["y_prob"].to_numpy(),
                             threshold=float(g["threshold"].iloc[0]))
        per_fold[int(k)] = {m: gg[m] for m in ["auroc", "auprc", "brier", "log_loss",
                                               "sensitivity", "specificity", "precision", "f1"]}
    pm = E.patient_macro(oof=oof[["patient_id", "y_true", "y_prob"]])
    hours = float((window_starts["window_end"] - window_starts["window_start"]).sum() / 3600.0)
    ev = E.event_level(oof=oof, retained_events=retained_events, analyzable_hours=hours)
    cal = E.calibration(y_true=y, y_prob=p)
    return {"pooled": pooled, "per_fold": per_fold, "patient_macro": pm,
            "event_level": ev, "calibration": cal, "n_windows": int(len(oof)),
            "n_patients": int(oof["patient_id"].nunique()),
            "positive_rate": float(np.mean(y)) if len(y) else None}


# ----------------------------------------------------------------- main run
def run(*, project_root, run_id: str) -> Dict[str, object]:
    import os
    pr = str(project_root)
    R = lambda *a: os.path.join(pr, *a)  # noqa
    res_dir = R("results", "runs", run_id); os.makedirs(res_dir, exist_ok=True)
    mdl_dir = R("models", "runs", run_id); os.makedirs(mdl_dir, exist_ok=True)
    rep_dir = R("reports", "modeling", "runs", run_id); os.makedirs(rep_dir, exist_ok=True)

    # ---- load frozen inputs ----
    cohort = pd.read_parquet(R("splits", "core_cohort_window_membership.parquet"))
    events = pd.read_parquet(R("annotations", "parsed_events.parquet"))
    hr = pd.read_parquet(R("features", "physiology", "hr_window_features.parquet"))
    spo2 = pd.read_parquet(R("features", "physiology", "spo2_window_features.parquet"))
    airflow = pd.read_parquet(R("features", "physiology", "airflow_window_features.parquet"))
    outer = pd.read_csv(R("splits", "outer_patient_folds_core_v2.csv"))
    inner = pd.read_parquet(R("splits", "inner_patient_folds_core_v2.parquet"))

    # Patient ids are stored inconsistently across the frozen split files: the cohort
    # and inner parquet use zero-padded STRINGS ('01'), the outer CSV is parsed as int64
    # (-> '1'). Normalize every patient id to the cohort's canonical zero-padded string
    # form so that window_id prefixes, frame patient_ids and all split lookups key
    # identically. Cohort ids are numeric codes; this never changes which patient is which.
    _pid_width = max(len(str(x)) for x in cohort["patient_id"].astype(str).unique())

    def _npid(p) -> str:
        return f"{int(float(p)):0{_pid_width}d}"

    outer_map = {_npid(r.patient_id): int(r.outer_fold) for r in outer.itertuples()}
    inner_map = {(int(r.outer_fold), _npid(r.patient_id)): int(r.inner_validation_fold)
                 for r in inner.itertuples()}

    base_all = F.build_base(hr=hr, spo2=spo2, airflow=airflow)

    # ---- labels ----
    overlaps = L.derive_overlaps(cohort=cohort, events=events)
    tables = L.assign_labels(overlaps=overlaps)
    onset_ok = L.onset_matches_stage4(onset_table=tables[L.ONSET].table, stage4_cohort=cohort)

    label_rows = []
    for v in L.VARIANT_ORDER:
        c = L.variant_counts(label_result=tables[v])
        c.update({"variant": v, "name": L.VARIANT_NAMES[v]})
        label_rows.append(c)
    pd.DataFrame(label_rows)[["variant", "name", "n_windows", "n_positive", "n_negative",
                              "n_indeterminate", "positive_rate"]].to_csv(
        R(res_dir, "label_versions.csv"), index=False)

    # onset vs main confusion (label drift)
    m = tables[L.MAIN].table[["window_id", "label"]].rename(columns={"label": "main_label"})
    o = tables[L.ONSET].table[["window_id", "label"]].rename(columns={"label": "onset_label"})
    cm = m.merge(o, on="window_id")
    cm_dis = cm.groupby(["onset_label", "main_label"]).size().reset_index(name="n")
    cm_dis.to_csv(R(res_dir, "label_onset_vs_main_confusion.csv"), index=False)

    battery = N.model_battery()
    summary_rows: List[dict] = []
    patient_macro_rows: List[dict] = []
    event_rows: List[dict] = []
    cal_rows: List[dict] = []
    oof_store: Dict[str, pd.DataFrame] = {}
    selected_rows: List[dict] = []
    cand_rows: List[dict] = []
    lag_rows: List[dict] = []

    def _run_task(variant, lag, model_specs, task, lag_windows):
        lab = tables[variant]
        rows_out = []
        for spec in model_specs:
            fam = spec.feature_family if spec.kind != "dummy" else "hr_spo2"
            fm = F.assemble(cohort=cohort, base=base_all, family=fam, lag_windows=lag_windows)
            frame, X, y, pids, ofold = build_frame(label_result=lab, fm=fm, outer_map=outer_map)
            res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                                  outer_fold=ofold, inner_map=inner_map)
            starts = frame[["window_id", "window_start", "window_end"]]
            ev = evaluate_oof(oof=res.oof, retained_events=events, window_starts=starts)
            pooled = ev["pooled"]
            row = {"task": task, "variant": variant, "variant_name": L.VARIANT_NAMES[variant],
                   "lag_windows": lag_windows, "lag_label": lag, "model": spec.name,
                   "model_role": spec.role, "n_windows": ev["n_windows"],
                   "n_patients": ev["n_patients"], "positive_rate": ev["positive_rate"],
                   "auroc": pooled["auroc"], "auprc": pooled["auprc"], "brier": pooled["brier"],
                   "log_loss": pooled["log_loss"], "sensitivity": pooled["sensitivity"],
                   "specificity": pooled["specificity"], "precision": pooled["precision"],
                   "f1": pooled["f1"], "balanced_accuracy": pooled["balanced_accuracy"],
                   "n_dropped_lag": int(fm.n_dropped_missing_shift)}
            rows_out.append((spec, res, ev, row))
            summary_rows.append(row)
            pm = ev["patient_macro"]
            for key in ["patient_macro_auroc", "patient_macro_auprc"]:
                patient_macro_rows.append({"task": task, "variant": variant, "lag_label": lag,
                                           "model": spec.name, "metric": key, "value": pm[key],
                                           "n": pm.get(key + "_n"), "n_included": pm["n_patients_included"],
                                           "n_excluded": pm["n_patients_excluded"]})
            el = ev["event_level"]
            event_rows.append({"task": task, "variant": variant, "lag_label": lag, "model": spec.name,
                               **{k: el[k] for k in ["n_gt_events_in_scope", "n_gt_events_matched",
                                "n_candidate_events", "event_sensitivity", "event_precision",
                                "event_f1", "false_positives_per_hour", "analyzable_hours"]}})
            cc = ev["calibration"]
            cal_rows.append({"task": task, "variant": variant, "lag_label": lag, "model": spec.name,
                             "brier": cc["brier"], "calibration_intercept": cc["calibration_intercept"],
                             "calibration_slope": cc["calibration_slope"], "ece": cc["ece"]})
            selected_rows.extend([{**s, "variant": variant, "lag_label": lag} for s in res.selected])
            cand_rows.extend([{**s, "variant": variant, "lag_label": lag} for s in res.candidates])
            oof_store[f"{task}|{variant}|{lag}|{spec.name}"] = res.oof.assign(variant=variant, lag_label=lag, task=task)
        return rows_out

    # ---- Task A: primary/comparator battery, main label, lag 0 ----
    A = _run_task(L.MAIN, "lag_0", [s for s in battery], "primary_battery", 0)

    # ---- Task B: label sensitivity (onset/any/cov50), lag 0, dummy + primary LR ----
    for v in [L.ONSET, L.ANY, L.COV50]:
        _run_task(v, "lag_0", [s for s in battery if s.name in ("dummy_prior", "hr_spo2_lr")],
                  "label_sensitivity", 0)

    # ---- Task C: lag sensitivity, main label, +30/+60, dummy + simple + primary LR ----
    for lag_w, lag_lbl in [(1, "lag_plus_30"), (2, "lag_plus_60")]:
        _run_task(L.MAIN, lag_lbl,
                  [s for s in battery if s.name in ("dummy_prior", "simple_spo2_min_lr", "hr_spo2_lr")],
                  "lag_sensitivity", lag_w)

    # lag sensitivity summary (primary LR across lags)
    for r in summary_rows:
        if r["model"] == "hr_spo2_lr" and r["variant"] == L.MAIN:
            lag_rows.append({"lag_label": r["lag_label"], "auroc": r["auroc"], "auprc": r["auprc"],
                             "brier": r["brier"], "n_windows": r["n_windows"]})

    # ---- write core metric tables ----
    pd.DataFrame(summary_rows).to_csv(R(res_dir, "summary_metrics.csv"), index=False)
    pd.DataFrame(patient_macro_rows).to_csv(R(res_dir, "patient_macro_metrics.csv"), index=False)
    pd.DataFrame(event_rows).to_csv(R(res_dir, "event_level_metrics.csv"), index=False)
    pd.DataFrame(cal_rows).to_csv(R(res_dir, "calibration_metrics.csv"), index=False)
    pd.DataFrame(lag_rows).to_csv(R(res_dir, "lag_sensitivity.csv"), index=False)
    pd.DataFrame(selected_rows).to_csv(R(res_dir, "selected_hyperparameters.csv"), index=False)
    pd.DataFrame(cand_rows).to_csv(R(res_dir, "inner_cv_candidate_scores.csv"), index=False)

    # ---- bootstrap CI on primary (main, lag0) ----
    primary_oof = oof_store["primary_battery|main|lag_0|hr_spo2_lr"]
    bcis = patient_cluster_bootstrap(
        y_true=primary_oof["y_true"].to_numpy(), y_prob=primary_oof["y_prob"].to_numpy(),
        patient_ids=primary_oof["patient_id"].to_numpy(), metrics=["auroc", "auprc"],
        n_resamples=1000, seed=SEED)
    pd.DataFrame([b.as_record(run_id=run_id, model_name="hr_spo2_lr") for b in bcis]).to_csv(
        R(res_dir, "bootstrap_ci_primary.csv"), index=False)

    # ---- paired bootstrap (main, lag0): primary LR vs comparators ----
    pair_rows = []
    base_oof = primary_oof[["window_id", "patient_id", "y_true", "y_prob", "threshold"]].rename(
        columns={"y_prob": "prob_enhanced", "threshold": "threshold_enhanced"})
    for cmp_name in ["hr_only_lr", "spo2_only_lr", "simple_spo2_min_lr"]:
        co = oof_store[f"primary_battery|main|lag_0|{cmp_name}"][["window_id", "y_prob", "threshold"]].rename(
            columns={"y_prob": "prob_core", "threshold": "threshold_core"})
        mrg = base_oof.merge(co, on="window_id", how="inner")
        if mrg.empty:
            continue
        pcis = paired_patient_cluster_bootstrap(
            y_true=mrg["y_true"].to_numpy(), prob_core=mrg["prob_core"].to_numpy(),
            prob_enhanced=mrg["prob_enhanced"].to_numpy(), patient_ids=mrg["patient_id"].to_numpy(),
            threshold_core=mrg["threshold_core"].to_numpy(), threshold_enhanced=mrg["threshold_enhanced"].to_numpy(),
            metrics=["auroc", "auprc", "brier"], n_resamples=1000, seed=SEED)
        for pci in pcis:
            d = pci.as_record(run_id=run_id, model=f"hr_spo2_lr_vs_{cmp_name}")
            d["comparator"] = cmp_name
            pair_rows.append(d)
    pd.DataFrame(pair_rows).to_csv(R(res_dir, "paired_bootstrap_ci.csv"), index=False)

    # ---- HGB vs LR (exploratory) ----
    hgb_oof = oof_store["primary_battery|main|lag_0|hr_spo2_hgb"][["window_id", "y_prob", "threshold"]]
    hpair = []
    mrg = primary_oof[["window_id", "patient_id", "y_true", "y_prob", "threshold"]].rename(
        columns={"y_prob": "prob_core", "threshold": "threshold_core"}).merge(
        hgb_oof.rename(columns={"y_prob": "prob_enhanced", "threshold": "threshold_enhanced"}), on="window_id")
    if not mrg.empty:
        for pci in paired_patient_cluster_bootstrap(
            y_true=mrg["y_true"].to_numpy(), prob_core=mrg["prob_core"].to_numpy(),
            prob_enhanced=mrg["prob_enhanced"].to_numpy(), patient_ids=mrg["patient_id"].to_numpy(),
            threshold_core=mrg["threshold_core"].to_numpy(), threshold_enhanced=mrg["threshold_enhanced"].to_numpy(),
            metrics=["auroc", "auprc", "brier"], n_resamples=1000, seed=SEED):
            d = pci.as_record(run_id=run_id, model="hgb_vs_lr_exploratory"); d["comparator"] = "hr_spo2_hgb"; hpair.append(d)
    pd.DataFrame(hpair).to_csv(R(res_dir, "paired_bootstrap_hgb_vs_lr.csv"), index=False)

    # ---- Task D: airflow shared-source secondary (34-pt subcohort, main, lag0) ----
    af_rows = []
    af_sub = cohort[cohort["cohort_airflow"] == True].copy()  # noqa: E712
    airflow_cohort_labels = tables[L.MAIN].table[tables[L.MAIN].table["window_id"].isin(set(af_sub["window_id"].tolist()))].copy()
    for role, fam in [("core_restricted", "hr_spo2"), ("airflow_enhanced", None)]:
        if fam is not None:
            fm = F.assemble(cohort=af_sub, base=base_all, family="hr_spo2", lag_windows=0)
        else:
            fm = F.assemble_airflow_enhanced(cohort_airflow=af_sub, base=base_all, lag_windows=0)
        lab_res = L.LabelResult(L.MAIN, L.VARIANT_NAMES[L.MAIN], airflow_cohort_labels)
        frame, X, y, pids, ofold = build_frame(label_result=lab_res, fm=fm, outer_map=outer_map)
        spec = N.ModelSpec(f"airflow_{role}", "secondary", "lr", fam or "hr_spo2+airflow")
        res = N.run_nested_cv(spec=spec, frame=frame, X=X, y=y, patient_ids=pids,
                              outer_fold=ofold, inner_map=inner_map)
        starts = frame[["window_id", "window_start", "window_end"]]
        ev = evaluate_oof(oof=res.oof, retained_events=events, window_starts=starts)
        p = ev["pooled"]
        af_rows.append({"feature_set": role, "n_windows": ev["n_windows"], "n_patients": ev["n_patients"],
                        "auroc": p["auroc"], "auprc": p["auprc"], "brier": p["brier"],
                        "log_loss": p["log_loss"], "sensitivity": p["sensitivity"], "specificity": p["specificity"]})
        oof_store[f"airflow|{role}"] = res.oof.assign(variant=L.MAIN, lag_label="lag_0", task="airflow_secondary")
    # paired airflow increment
    core_o = oof_store["airflow|core_restricted"][["window_id", "patient_id", "y_true", "y_prob", "threshold"]].rename(
        columns={"y_prob": "prob_core", "threshold": "threshold_core"})
    enh_o = oof_store["airflow|airflow_enhanced"][["window_id", "y_prob", "threshold"]].rename(
        columns={"y_prob": "prob_enhanced", "threshold": "threshold_enhanced"})
    amrg = core_o.merge(enh_o, on="window_id", how="inner")
    af_ci_rows = []
    if not amrg.empty:
        for pci in paired_patient_cluster_bootstrap(
            y_true=amrg["y_true"].to_numpy(), prob_core=amrg["prob_core"].to_numpy(),
            prob_enhanced=amrg["prob_enhanced"].to_numpy(), patient_ids=amrg["patient_id"].to_numpy(),
            threshold_core=amrg["threshold_core"].to_numpy(), threshold_enhanced=amrg["threshold_enhanced"].to_numpy(),
            metrics=["auroc", "auprc", "brier"], n_resamples=1000, seed=SEED):
            d = pci.as_record(run_id=run_id, model="airflow_enhanced_minus_core_restricted")
            af_ci_rows.append(d)
    pd.DataFrame(af_rows).to_csv(R(res_dir, "airflow_paired_summary.csv"), index=False)
    pd.DataFrame(af_ci_rows).to_csv(R(res_dir, "airflow_paired_ci.csv"), index=False)

    # ---- persist OOF (main tasks) ----
    keep_keys = [k for k in oof_store if k.split("|")[0] in ("primary_battery", "airflow")]
    pd.concat([oof_store[k] for k in keep_keys], ignore_index=True).to_parquet(
        R(res_dir, "oof_predictions.parquet"), index=False)

    # ---- signature ----
    sig = {
        "run_id": run_id, "seed": SEED, "onset_matches_stage4": bool(onset_ok),
        "n_patients": int(cohort["patient_id"].nunique()), "n_cohort_windows": int(len(cohort)),
        "label_variants": {v: L.variant_counts(label_result=tables[v]) for v in L.VARIANT_ORDER},
        "main_overlap_threshold_s": L.MAIN_OVERLAP_S,
        "airflow_subcohort_patients": int(af_sub["patient_id"].nunique()),
        "airflow_subcohort_windows": int(len(af_sub)),
        "audio_blocked": True, "audio_eligible_windows": 0,
    }
    json.dump(sig, open(R(res_dir, "model_input_signature.json"), "w"), indent=2, default=str)

    # ---- resolved config snapshot ----
    import shutil
    shutil.copy(R("config", "bspc_revision_analysis.yaml"), R(mdl_dir, "reanalysis_resolved.yaml"))

    # ---- completion report ----
    _write_report(
        rep_dir=rep_dir, run_id=run_id, onset_ok=onset_ok, sig=sig,
        summary_rows=summary_rows, patient_macro_rows=patient_macro_rows,
        event_rows=event_rows, cal_rows=cal_rows, lag_rows=lag_rows,
        bcis=bcis, pair_rows=pair_rows, af_rows=af_rows, af_ci_rows=af_ci_rows,
        label_rows=label_rows)

    return {"run_id": run_id, "onset_matches_stage4": onset_ok, "sig": sig,
            "res_dir": res_dir, "rep_dir": rep_dir, "mdl_dir": mdl_dir,
            "summary": pd.DataFrame(summary_rows)}


def _f(v, nd=4):
    return "NA" if v is None else f"{float(v):.{nd}f}"


def _write_report(*, rep_dir, run_id, onset_ok, sig, summary_rows, patient_macro_rows,
                  event_rows, cal_rows, lag_rows, bcis, pair_rows, af_rows, af_ci_rows,
                  label_rows) -> None:
    import os
    sm = {r["model"]: r for r in summary_rows if r["task"] == "primary_battery"}
    lines: List[str] = []
    A = lines.append
    A("# Phase 15b — BSPC strict re-analysis completion report\n")
    A(f"- **run_id**: `{run_id}`")
    A("- **round**: 2 (strict re-analysis); stage 15a evidence/registration audit precedes; "
      "stage 15c (journalization + references) follows.")
    A(f"- **frozen registration**: `config/bspc_revision_analysis.yaml` + "
      f"`docs/bspc_revision_analysis_protocol.md` (seed {SEED}).")
    A("- **split**: INHERITED frozen v2 patient split (5 outer x 4 inner); NOT re-split. "
      "Stage 2-14 products untouched.")
    A("- **audio**: BLOCKED (count 0); no audio features read; `audio_blocked=true`.")
    A("- **outputs**: isolated under this run dir; no external download; no submission/upload.\n")

    A("## 1. Label revision (duration-overlap)\n")
    A(f"- onset variant **reproduces Stage-4 exactly**: `{onset_ok}`.")
    A(f"- main label = overlap >= {L.MAIN_OVERLAP_S:.0f}s; 0 < overlap < {L.MAIN_OVERLAP_S:.0f}s "
      f"-> INDETERMINATE (excluded from main train/eval).")
    A("\n| variant | name | n_windows | n_pos | n_neg | n_indet | pos_rate |")
    A("|---|---|--:|--:|--:|--:|--:|")
    for r in label_rows:
        A(f"| {r['variant']} | {r['name']} | {r['n_windows']} | {r['n_positive']} | "
          f"{r['n_negative']} | {r['n_indeterminate']} | {_f(r['positive_rate'])} |")
    A("\n## 2. Primary battery (main label, lag 0 / causal)\n")
    A("| model | role | AUROC | AUPRC | Brier | sens | spec | F1 | n_win |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for name in ["dummy_prior", "simple_spo2_min_lr", "hr_only_lr", "spo2_only_lr",
                 "hr_spo2_lr", "hr_spo2_hgb"]:
        r = sm.get(name)
        if r:
            A(f"| {name} | {r['model_role']} | {_f(r['auroc'])} | {_f(r['auprc'])} | "
              f"{_f(r['brier'])} | {_f(r['sensitivity'])} | {_f(r['specificity'])} | "
              f"{_f(r['f1'])} | {r['n_windows']} |")
    A("")
    ci = {b.metric: b for b in bcis}
    ba = ci.get("auroc"); bp = ci.get("auprc")
    prim = sm.get("hr_spo2_lr", {})
    A(f"- Primary **hr_spo2_lr** window AUROC = {_f(prim.get('auroc'))} "
      f"[{_f(ba.ci_low if ba else None)}, {_f(ba.ci_high if ba else None)}] "
      f"(patient-cluster bootstrap, n_used={ba.n_used if ba else 'NA'}).")
    A(f"- Primary window AUPRC = {_f(prim.get('auprc'))} "
      f"[{_f(bp.ci_low if bp else None)}, {_f(bp.ci_high if bp else None)}].")
    pm = next((r for r in patient_macro_rows if r["model"] == "hr_spo2_lr"
               and r["metric"] == "patient_macro_auroc"), {})
    A(f"- Patient-macro AUROC = {_f(pm.get('value'))} (n_included={pm.get('n_included')}, "
      f"n_excluded={pm.get('n_excluded')}).")

    A("\n## 3. Label sensitivity (lag 0)\n")
    A("| variant | model | AUROC | AUPRC | Brier | n_win |")
    A("|---|---|--:|--:|--:|--:|")
    for r in summary_rows:
        if r["task"] == "label_sensitivity":
            A(f"| {r['variant']} | {r['model']} | {_f(r['auroc'])} | {_f(r['auprc'])} | "
              f"{_f(r['brier'])} | {r['n_windows']} |")

    A("\n## 4. Lag (context) sensitivity (main label)\n")
    A("| lag | AUROC | AUPRC | Brier | n_win |")
    A("|---|--:|--:|--:|--:|")
    for r in lag_rows:
        A(f"| {r['lag_label']} | {_f(r['auroc'])} | {_f(r['auprc'])} | {_f(r['brier'])} | "
          f"{r['n_windows']} |")
    A("\n_(lag_0 = causal/online; lag_+30/+60 use future signal -> OFFLINE context, not a "
      "deployable online model.)_")

    A("\n## 5. Airflow shared-source secondary (34-pt subcohort)\n")
    A("| feature_set | AUROC | AUPRC | Brier | n_win |")
    A("|---|--:|--:|--:|--:|")
    for r in af_rows:
        A(f"| {r['feature_set']} | {_f(r['auroc'])} | {_f(r['auprc'])} | {_f(r['brier'])} | "
          f"{r['n_windows']} |")
    d_auroc = next((r for r in af_ci_rows if r["metric"] == "auroc"), {})
    A(f"\n- Paired increment (airflow_enhanced - core_restricted) AUROC delta = "
      f"{_f(d_auroc.get('delta_point_estimate'))} "
      f"[{_f(d_auroc.get('ci_low'))}, {_f(d_auroc.get('ci_high'))}] "
      f"(paired patient-cluster bootstrap, n_used={d_auroc.get('n_used')}).")
    A("- **Shared-source / incorporation caveat**: airflow participates in PSG event scoring, "
      "so its incremental value is not independent of the reference standard. Reported as a "
      "secondary, construct-limited analysis — not a clean incremental claim.")

    A("\n## 6. Calibration (primary, main, lag 0)\n")
    prim_cal = next((r for r in cal_rows if r["model"] == "hr_spo2_lr"), {})
    A(f"- Brier = {_f(prim_cal.get('brier'))}; Cox intercept = "
      f"{_f(prim_cal.get('calibration_intercept'))}; slope = "
      f"{_f(prim_cal.get('calibration_slope'))}; ECE = {_f(prim_cal.get('ece'))}.")

    A("\n## 7. Event-level (primary, main, lag 0)\n")
    prim_ev = next((r for r in event_rows if r["model"] == "hr_spo2_lr"), {})
    A(f"- n_gt_events_in_scope = {prim_ev.get('n_gt_events_in_scope')}; "
      f"event_sensitivity = {_f(prim_ev.get('event_sensitivity'))}; "
      f"event_precision = {_f(prim_ev.get('event_precision'))}; "
      f"event_F1 = {_f(prim_ev.get('event_f1'))}; "
      f"false_positives/hour = {_f(prim_ev.get('false_positives_per_hour'), 3)}.")
    A("- Event-level metrics describe DETECTION at the window-aggregated candidate level, "
      "NOT patient diagnosis (AHI).")

    A("\n## 8. State\n")
    A("- This is a reproducible internal re-analysis candidate (Round 2 of the BSPC revision).")
    A("- Local ethics is a PLACEHOLDER (HYEC-Y-PJ-20260XX) and is NOT approved -> emits "
      "`ETHICS_AUTHOR_DECISION_REQUIRED`.")
    A("- Next: Round 3 (BSPC IMRaD manuscript + legal OA reference archiving). Final state can "
      "ONLY be `READY_FOR_AUTHOR_ETHICS_AND_SUBMISSION_DECISION` (never `READY_FOR_SUBMISSION`).")
    A("\n## Artifacts\n")
    A("- metrics: `summary_metrics.csv`, `patient_macro_metrics.csv`, `event_level_metrics.csv`, "
      "`calibration_metrics.csv`, `lag_sensitivity.csv`")
    A("- uncertainty: `bootstrap_ci_primary.csv`, `paired_bootstrap_ci.csv`, "
      "`paired_bootstrap_hgb_vs_lr.csv`, `airflow_paired_ci.csv`")
    A("- selection: `selected_hyperparameters.csv`, `inner_cv_candidate_scores.csv`")
    A("- labels: `label_versions.csv`, `label_onset_vs_main_confusion.csv`")
    A("- predictions: `oof_predictions.parquet`; provenance: `model_input_signature.json`, "
      "`reanalysis_resolved.yaml`")
    with open(os.path.join(rep_dir, "phase_15b_completion_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


__all__ = ["run", "build_frame", "evaluate_oof"]
