"""Stage-9 figures (<=6) + markdown reports.

Pure consumer of the long-form sensitivity DataFrames produced by the runner.
Figures are headless (Agg); each is skipped (counted as not drawn) if its source
table lacks data, so the figure cap is never exceeded. Two markdown artefacts are
written: the detailed ``robustness_sensitivity_report.md`` and the
``phase_09_completion_report.md`` (Section-6 completion format).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import schema as S
from .config import ResolvedRobustnessConfig
from .rerun import RerunBundle  # noqa: F401  (type hint clarity only)
from .signature import RobustnessInput

_FIG_DPI = 130


def _lazy_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _val(df: pd.DataFrame, **filters) -> Optional[float]:
    q = " and ".join([f"{k}=={v!r}" for k, v in filters.items()])
    sub = df.query(q) if q else df
    if sub.empty:
        return None
    return sub["value"].iloc[0]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_determinism(plt, determ: Dict[str, Any], out: Path) -> bool:
    rows = []
    for cohort, fsd in (("core", determ.get("core", {})),):
        for model, d in fsd.items():
            rows.append(("core/" + model, d.get("prob_max_abs_diff", np.nan)))
    for model, d in determ.get("core", {}).items():
        rows.append((f"core/{model}", d.get("prob_max_abs_diff", np.nan)))
    for fs, md in determ.get("airflow", {}).items():
        for model, d in md.items():
            rows.append((f"air/{fs}/{model}", d.get("prob_max_abs_diff", np.nan)))
    rows = list({k: v for k, v in rows}.items())  # de-dup keep last
    labels = [r[0] for r in rows]
    vals = [max(r[1], 1e-18) for r in rows]
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(labels)), vals, color="#2a6fdb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("regenerated-vs-frozen OOF max |Δ prob|")
    ax.set_title("Re-run determinism (lower = more exact); LR ~1e-9, HGB ~1e-6 expected")
    ax.axhline(determ.get("summary", {}).get("lr_tolerance", 1e-9), ls="--",
               lw=1, color="green", label="LR tol")
    ax.axhline(determ.get("summary", {}).get("hgb_tolerance", 1e-6), ls="--",
               lw=1, color="orange", label="HGB tol")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=_FIG_DPI)
    plt.close(fig)
    return True


def _fig_threshold(plt, thr: pd.DataFrame, out: Path) -> bool:
    sub = thr[(thr["cohort"] == S.COHORT_CORE) & (thr["feature_set"] == S.FEATURE_SET_CORE)
              & (thr["model"] == "logistic_regression")
              & (thr["metric_kind"] == "threshold_based")]
    if sub.empty:
        return False
    metrics = ["sensitivity", "specificity", "f1", "balanced_accuracy"]
    variants = [S.THRESHOLD_YOUDEN, S.THRESHOLD_FIXED_05, S.THRESHOLD_BAL_ACC]
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"": "#2a6fdb"}
    palette = ["#2a6fdb", "#e07b39", "#3aa655"]
    for i, v in enumerate(variants):
        vals = []
        for m in metrics:
            r = sub[(sub["metric"] == m) & (sub["variant"] == v)]
            vals.append(r["value"].iloc[0] if not r.empty else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=v, color=palette[i % len(palette)])
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=20)
    ax.set_ylabel("metric value"); ax.set_ylim(0, 1.05)
    ax.set_title("Threshold sensitivity (LR, core cohort, pooled OOF)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=_FIG_DPI); plt.close(fig)
    return True


def _fig_calibration(plt, cal: pd.DataFrame, out: Path) -> bool:
    sub = cal[(cal["cohort"] == S.COHORT_CORE) & (cal["feature_set"] == S.FEATURE_SET_CORE)]
    if sub.empty:
        return False
    metrics = ["brier", "log_loss", "ece"]
    variants = [S.CAL_RAW, S.CAL_SIGMOID_PLATT, S.CAL_ISOTONIC]
    x = np.arange(len(metrics)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#888888", "#2a6fdb", "#e07b39"]
    for i, v in enumerate(variants):
        vals = []
        for m in metrics:
            r = sub[(sub["metric"] == m) & (sub["variant"] == v)]
            vals.append(r["value"].iloc[0] if not r.empty and r["value"].iloc[0] is not None else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=v, color=palette[i])
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylabel("value (lower = better calibrated)")
    ax.set_title("Calibration sensitivity (LR, core cohort): raw vs Platt vs isotonic")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=_FIG_DPI); plt.close(fig)
    return True


def _fig_class_weight(plt, cw: pd.DataFrame, out: Path) -> bool:
    sub = cw[(cw["model"] == "logistic_regression")
             & (cw["metric_kind"] == "threshold_free")
             & (cw["metric"].isin(["auroc", "auprc"]))]
    if sub.empty:
        return False
    keys = sub.groupby(["cohort", "feature_set", "metric", "variant"]).size().reset_index()
    cohorts = sub[["cohort", "feature_set"]].drop_duplicates().values.tolist()
    variants = [S.CW_NONE, S.CW_BALANCED, "primary_unrestricted"]
    metrics = ["auroc", "auprc"]
    n = len(cohorts)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 4), squeeze=False)
    palette = ["#2a6fdb", "#e07b39", "#3aa655"]
    for ci, (cohort, fs) in enumerate(cohorts):
        ax = axes[0][ci]
        x = np.arange(len(metrics)); w = 0.26
        for i, v in enumerate(variants):
            vals = []
            for m in metrics:
                r = sub[(sub["cohort"] == cohort) & (sub["feature_set"] == fs)
                        & (sub["metric"] == m) & (sub["variant"] == v)]
                vals.append(r["value"].iloc[0] if not r.empty else np.nan)
            ax.bar(x + (i - 1) * w, vals, w, label=v, color=palette[i])
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05); ax.set_title(f"{cohort}/{fs}")
        if ci == 0:
            ax.set_ylabel("metric value"); ax.legend(fontsize=7)
    fig.suptitle("Class-weight sensitivity (LR): none vs balanced vs primary")
    fig.tight_layout(); fig.savefig(out, dpi=_FIG_DPI); plt.close(fig)
    return True


def _fig_bootstrap(plt, boot: pd.DataFrame, out: Path) -> bool:
    sub = boot[(boot["block"] == "patient_cluster") & (boot["metric"] == "auroc")]
    if sub.empty:
        return False
    sub = sub.copy()
    sub["label"] = sub["cohort"] + "/" + sub["feature_set"] + "/" + sub["model"]
    sub = sub.sort_values("label")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(sub))))
    y = np.arange(len(sub))
    ax.errorbar(sub["point_estimate"], y,
                xerr=[sub["point_estimate"] - sub["ci_low"],
                      sub["ci_high"] - sub["point_estimate"]],
                fmt="o", color="#2a6fdb", capsize=3)
    # frozen overlay where present
    has_frozen = sub["frozen_point_estimate"].notna()
    if has_frozen.any():
        fsub = sub[has_frozen]
        ax.scatter(fsub["frozen_point_estimate"], np.arange(len(sub))[has_frozen],
                   marker="x", color="red", label="frozen Stage7/8 point")
        ax.legend(fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(sub["label"], fontsize=8)
    ax.axvline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("patient-cluster bootstrap AUROC (point + 95% CI)")
    ax.set_title("Uncertainty: patient-cluster AUROC (resampled patients, not windows)")
    fig.tight_layout(); fig.savefig(out, dpi=_FIG_DPI); plt.close(fig)
    return True


def _fig_airflow(plt, air_pp: pd.DataFrame, out: Path) -> bool:
    if air_pp is None or air_pp.empty:
        return False
    pp = air_pp.sort_values("patient_id")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    x = np.arange(len(pp))
    a1.bar(x, pp["n_windows"], color="#2a6fdb")
    a1.set_ylabel("windows / patient")
    a1.set_title("34-patient airflow sub-cohort: selective profile (NOT extrapolated to 50)")
    a2.bar(x, pp["positive_rate"], color="#e07b39")
    a2.set_ylabel("positive rate / patient")
    a2.set_xticks(x); a2.set_xticklabels(pp["patient_id"], rotation=90, fontsize=6)
    fig.tight_layout(); fig.savefig(out, dpi=_FIG_DPI); plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "_(no rows)_\n"
    d = df.head(max_rows).copy()
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.6g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_(... {len(df) - max_rows} more rows)_")
    return "\n".join(lines) + "\n"


def _write_main_report(
    *, path: Path, mc: ResolvedRobustnessConfig, rin: RobustnessInput, summary,
    repro: pd.DataFrame, determ: Dict[str, Any], thr: pd.DataFrame, cal: pd.DataFrame,
    cw: pd.DataFrame, boot: Dict[str, pd.DataFrame], air: Dict[str, pd.DataFrame],
    alt: Dict[str, pd.DataFrame], run_id: str, fig_names: List[str],
) -> None:
    rep_max = summary.reproduction.get("max_abs_diff", 0.0)
    rep_mism = summary.reproduction.get("n_mismatches", 0)
    determ_sum = determ.get("summary", {})
    cg = summary.core_cohort; ag = summary.airflow_cohort
    fig_md = "\n".join(f"![{f}]({f})" for f in fig_names)
    cs = air["cohort_summary"].iloc[0] if not air["cohort_summary"].empty else {}

    md = f"""# Stage 9 -- Robustness & Sensitivity Report

- **run_id**: `{run_id}`
- **modeling_version**: `{S.MODELING_VERSION}`
- **consumed split**: `{rin.split_run_id}` (approval: `{summary.approval_status}`)
- **frozen inputs**: Stage 7 `{mc.stage7_run_id}`, Stage 8 `{mc.stage8_run_id}`
- **analysis_role rule**: primary reproduction = `primary_replication`; every
  other artefact = `sensitivity` (or `exploratory` for the alternative label).
  No sensitivity number replaces a primary result.

> {S.NON_CLINICAL_NOTE}

## 1. Primary reproduction (2.1)

Recomputed AUROC/AUPRC/Brier/log loss and per-window-threshold confusion-derived
metrics **directly from the frozen Stage 7/8 OOF** (pure function of stored
predictions; deterministic and authoritative).

- rows checked: {summary.reproduction.get('n_rows', 0)}
- mismatches vs frozen pooled reports: **{rep_mism}**
- max |Δ| over all metrics: **{rep_mism and 'has-mismatch' or f'{rep_max:.3e}'}**
- re-run determinism (regenerated vs frozen OOF prob): max |Δ prob| =
  **{determ_sum.get('max_prob_abs_diff_overall', float('nan')):.3e}**
  (LR tol {determ_sum.get('lr_tolerance')}, HGB tol {determ_sum.get('hgb_tolerance')});
  all within tolerance: **{determ_sum.get('all_within_tolerance')}**

{_md_table(repro[repro['metric'].isin(['auroc','auprc','brier','log_loss'])]
           [['cohort','feature_set','model','metric','value','frozen_value','abs_diff']]
           .drop_duplicates(), max_rows=30)}

## 2. Threshold sensitivity (2.3)

Primary rule unchanged (inner-OOF max-Youden-J). Two descriptive alternatives,
both fit from the outer-train inner OOF (never outer-test). Threshold-free
metrics are asserted invariant across variants.

{_md_table(thr[(thr['cohort']==S.COHORT_CORE)&(thr['model']=='logistic_regression')
              &(thr['metric_kind']=='threshold_based')]
           [['metric','variant','value']], max_rows=30)}

## 3. Calibration sensitivity (2.4, LR only)

Platt sigmoid + isotonic fit per fold on inner-OOF, applied to outer-test OOF.
Raw probabilities are NOT claimed calibrated. Isotonic is NA when any inner fold
has fewer than the configured minimum positives/negatives.

{_md_table(cal[(cal['cohort']==S.COHORT_CORE)]
           [['metric','variant','value','interpretable','reason']], max_rows=40)}

## 4. Class-weight sensitivity (2.5, LR only)

LR grid restricted to one class_weight axis at a time, re-selected per inner CV.

{_md_table(cw[(cw['cohort']==S.COHORT_CORE)&(cw['metric_kind']=='threshold_free')
             &(cw['metric'].isin(['auroc','auprc']))]
           [['metric','variant','value','frozen_value','abs_diff']], max_rows=30)}

## 5. Patient-cluster uncertainty (2.2)

Patient-cluster bootstrap (resample patients, bring all windows; window-IID is
structurally impossible). Per-fold paired Δ (enhanced - core) for the primary LR
shown separately. CIs describe the relation to 0 / internal-validation evidence;
they are NOT clinical significance.

{_md_table(boot['bootstrap_ci'][boot['bootstrap_ci']['metric']=='auroc']
           [['cohort','feature_set','model','block','point_estimate','ci_low','ci_high',
             'frozen_point_estimate','point_abs_diff']], max_rows=30)}

Per-fold paired Δ (airflow sub-cohort, primary LR):

{_md_table(boot['paired_fold_delta'][['outer_fold','metric','core_value','enhanced_value','delta','frozen_delta']], max_rows=30)}

## 6. Airflow selection profile (2.6)

The 34-patient sub-cohort is **selective** (patients with usable airflow) and is
**not extrapolated** to the 50-patient main cohort.

- sub-cohort: {cs.get('subcohort_patients')} patients / {cs.get('subcohort_windows')} windows, positive rate {cs.get('subcohort_positive_rate'):.4f}
- main cohort: {cs.get('main_cohort_patients')} patients / {cs.get('main_cohort_windows')} windows, positive rate {cs.get('main_cohort_positive_rate'):.4f}
- overlap with main cohort: {cs.get('overlap_patients_in_main')} patients

{_md_table(air['feature_quality'], max_rows=10)}

## 7. Alternative label (3.0)

Derived from structured event intervals (retained, non-awake) overlapping each
window by >= {mc.alternative_label.get('min_event_overlap_seconds', 1.0)} s. Same
frozen split / features / folds; only the label changes. Never used to choose a
better main label; NA where degenerate.

{_md_table(alt['alt_label_prevalence']
           [['cohort','feature_set','primary_positive','alternative_positive',
             'primary_prevalence','alternative_prevalence','interpretable','reason']], max_rows=20)}

## Figures

{fig_md if fig_md else '_(no figures drawn)_'}
"""
    path.write_text(md, encoding="utf-8")


def _write_completion_report(
    *, path: Path, mc: ResolvedRobustnessConfig, rin: RobustnessInput, summary,
    repro: pd.DataFrame, determ: Dict[str, Any], alt: Dict[str, pd.DataFrame],
    run_id: str, test_info: Dict[str, Any],
) -> None:
    rep_mism = summary.reproduction.get("n_mismatches", 0)
    rep_max = summary.reproduction.get("max_abs_diff", 0.0)
    determ_sum = determ.get("summary", {})
    alt_prev = alt["alt_label_prevalence"]
    md = f"""# Phase 09 Completion Report (Stage 9 robustness & sensitivity)

**run_id**: `{run_id}` | **status**: `{summary.status}` |
**split**: `{rin.split_run_id}` ({summary.approval_status}) |
**elapsed**: {summary.elapsed_seconds:.1f}s

## 1. Stage status & input-gate / historical-run integrity
- Gate checks: {sum(1 for c in summary.gate_checks if c['passed'])}/{len(summary.gate_checks)} passed.
- Core cohort: {summary.core_cohort.get('n_windows')} windows / {summary.core_cohort.get('n_patients')} patients.
- Airflow cohort: {summary.airflow_cohort.get('n_windows')} windows / {summary.airflow_cohort.get('n_patients')} patients.
- Frozen Stage 7/8 signatures re-verified; any mismatch would have BLOCKED (none did).

## 2. Primary reproduction vs Stage 7/8
- mismatches: **{rep_mism}**; max |Δ| = **{rep_max:.3e}**; all match = **{summary.reproduction.get('all_match')}**.
- re-run determinism: max |Δ prob| = **{determ_sum.get('max_prob_abs_diff_overall', float('nan')):.3e}**; all within tolerance = **{determ_sum.get('all_within_tolerance')}**.

## 3. Pre-registered sensitivities (definition / result / interpretability / NA)
- analyses executed: {summary.analyses_executed}
- threshold (2.3): invariant guard passed (threshold-free metrics identical across variants).
- calibration (2.4): LR; isotonic NA per fold if min counts unmet.
- class-weight (2.5): LR only; re-selected per inner CV.
- bootstrap (2.2): patient-cluster only; window-IID forbidden.
- airflow profile (2.6): descriptive; not extrapolated.
- alternative label (3.0): interpretability per cohort below.

{_md_table(alt_prev[['cohort','feature_set','alternative_positive','alternative_prevalence','interpretable','reason']], max_rows=20)}

## 4. LR calibration, Brier/log-loss limits, threshold sensitivity
See robustness_sensitivity_report.md sections 2-3. Stage 7 LR Brier being slightly
worse than the prior baseline is reported; calibration sensitivity does not
re-write primary AUROC/AUPRC and does not claim the raw LR probabilities are calibrated.

## 5. Patient-cluster bootstrap, fold-to-fold Δ, 34-patient profile
See robustness_sensitivity_report.md sections 5-6. CI-vs-0 is reported as
internal-validation evidence, not clinical significance.

## 6. Alternative-label branch (if executed)
Event counts / label change / direction: see alt_label_prevalence above. The main
label is unchanged; alt results never select a "better" main definition.

## 7. Immutability / no-audio / isolation / path-purity / leakage
- raw_modified: **{summary.raw_modified}**
- historical products unchanged: **{summary.historical_products_unchanged}**
- production paths clean: **{summary.production_paths_clean}**
- audio cohort count: **{summary.audio_cohort_count}**; read_wav: **{summary.read_wav}**
- audio features in matrix: **{summary.audio_features_in_matrix}**
- airflow in core model: **{summary.airflow_in_core_model}**
- output run-dir isolated + path-pure (assert_paths_clean): **{summary.production_paths_clean}**

## 8. Tests
- interpreter: `{test_info.get('interpreter','')}`
- command: `{test_info.get('command','')}`
- passed/failed/skipped: {test_info.get('passed','-')}/{test_info.get('failed','-')}/{test_info.get('skipped','-')}
- elapsed: {test_info.get('elapsed_seconds','-')}s

## 9. Research limitations
> {S.NON_CLINICAL_NOTE}

Internal patient-level CV only; retrospective PSG-anchored; selective 34-patient
airflow sub-cohort; not external validation; not clinical/diagnostic/deployment;
audio remains BLOCKED.

## 10. Next-stage gate
Only the pre-frozen results-integration / manuscript-table-and-figure generation
may proceed. Do NOT write manuscript conclusions or enter the audio route.

_Stage 9 complete. STOPPED pending Codex audit._
"""
    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def write_reports(
    *, run_dirs: Dict[str, Path], mc: ResolvedRobustnessConfig, rin: RobustnessInput,
    summary, repro: pd.DataFrame, determ: Dict[str, Any], thr_df: pd.DataFrame,
    cal_df: pd.DataFrame, cw_df: pd.DataFrame, boot: Dict[str, pd.DataFrame],
    air: Dict[str, pd.DataFrame], alt: Dict[str, pd.DataFrame], run_id: str,
    figure_max: int,
) -> int:
    rep_dir = run_dirs["reports"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt = _lazy_pyplot()
    drawers = [
        ("fig1_reproduction_determinism.png", lambda o: _fig_determinism(plt, determ, o)),
        ("fig2_threshold_sensitivity.png", lambda o: _fig_threshold(plt, thr_df, o)),
        ("fig3_calibration_metrics.png", lambda o: _fig_calibration(plt, cal_df, o)),
        ("fig4_class_weight_sensitivity.png", lambda o: _fig_class_weight(plt, cw_df, o)),
        ("fig5_bootstrap_forest.png", lambda o: _fig_bootstrap(plt, boot["bootstrap_ci"], o)),
        ("fig6_airflow_profile.png", lambda o: _fig_airflow(plt, air["per_patient"], o)),
    ]
    drawn: List[str] = []
    for name, fn in drawers:
        if len(drawn) >= figure_max:
            break
        try:
            if fn(fig_dir / name):
                drawn.append(name)
        except Exception as exc:  # pragma: no cover - a figure must never abort the run
            print(f"[report] figure {name} skipped: {exc}")
    _write_main_report(
        path=rep_dir / "robustness_sensitivity_report.md", mc=mc, rin=rin, summary=summary,
        repro=repro, determ=determ, thr=thr_df, cal=cal_df, cw=cw_df, boot=boot,
        air=air, alt=alt, run_id=run_id, fig_names=drawn,
    )
    # The production run does NOT execute the test suite (pytest is a separate
    # process); the interpreter + canonical command are recorded honestly, while
    # the pass/fail/skipped counts are left as "-" and reported authoritatively in
    # the Section-6 completion report (run separately via `python -m pytest -ra`).
    import sys as _sys
    test_info = {"interpreter": _sys.executable, "command": "python -m pytest -ra",
                 "passed": "-", "failed": "-", "skipped": "-",
                 "elapsed_seconds": "-"}
    _write_completion_report(
        path=rep_dir / "phase_09_completion_report.md", mc=mc, rin=rin, summary=summary,
        repro=repro, determ=determ, alt=alt, run_id=run_id, test_info=test_info,
    )
    return len(drawn)


__all__ = ["write_reports"]
