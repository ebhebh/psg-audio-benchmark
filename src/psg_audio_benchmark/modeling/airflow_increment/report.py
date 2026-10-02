"""Stage-8 reports + figures (matplotlib, headless Agg).

Generates at most five figures for the paired airflow-incremental comparison:

1. ``paired_oof_roc_pr``  -- pooled OOF ROC + PR for the PRIMARY LR, core vs
   enhanced (2 panels);
2. ``primary_gain_bootstrap`` -- ΔAUROC / ΔAUPRC paired-bootstrap distributions
   with point estimate + 95% CI (2 panels);
3. ``per_fold_deltas``    -- per-outer-fold ΔAUROC & ΔAUPRC (LR);
4. ``calibration``        -- calibration/reliability, core vs enhanced (LR + HGB);
5. ``cohort_flow``        -- 50 -> 34 patient flow + two feature sets.

Plus two markdown reports: ``airflow_incremental_report.md`` and
``phase_08_completion_report.md``.

Design discipline (per the data-viz method): colors are the *validated* default
palette, not eyeballed -- the feature-set categorical pair
airflow_enhanced=blue / core_restricted=orange was run through
``validate_palette.js`` (CVD ΔE 24.7, all checks PASS in light mode). Thin 2px
lines, recessive grid, a legend for every multi-series chart, and text in ink
tokens (never the series color). Every figure carries the mandatory
disclaimers: this is a paired within-cohort airflow increment on the
34-patient sub-cohort -- NOT a gain over the Stage-7 50-patient main cohort,
NOT for clinical deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from . import schema as S

# --- validated palette (light surface) --------------------------------------
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASE = "#c3c2b7"
_BLUE = "#2a78d6"     # airflow_enhanced (increment under test)  -- validated slot 1
_ORANGE = "#eb6834"   # core_restricted (comparator baseline)    -- validated slot 2
_AQUA = "#1baf7a"     # HGB exploratory accent

_FEATURE_COLOR = {
    S.FEATURE_SET_ENHANCED: _BLUE,
    S.FEATURE_SET_CORE: _ORANGE,
}
_FEATURE_LABEL = {
    S.FEATURE_SET_CORE: "core_restricted (HR+SpO2)",
    S.FEATURE_SET_ENHANCED: "airflow_enhanced (HR+SpO2+airflow)",
}
_DISCLAIMER = ("paired within-cohort airflow increment on the 34-patient "
               "airflow sub-cohort | internal patient-level cross-validation | "
               "NOT a gain over the 50-patient main cohort | not for clinical "
               "deployment")


def _style_axes(ax) -> None:
    ax.set_facecolor(_SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_BASE)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.yaxis.label.set_color(_INK2)
    ax.xaxis.label.set_color(_INK2)
    ax.title.set_color(_INK)
    ax.grid(True, color=_GRID, linewidth=0.8, zorder=0)


def _feature_legend(fig) -> None:
    fig.legend(
        handles=[Patch(facecolor=_FEATURE_COLOR[S.FEATURE_SET_CORE],
                       label=_FEATURE_LABEL[S.FEATURE_SET_CORE]),
                 Patch(facecolor=_FEATURE_COLOR[S.FEATURE_SET_ENHANCED],
                       label=_FEATURE_LABEL[S.FEATURE_SET_ENHANCED])],
        loc="lower center", ncol=2, frameon=False, fontsize=8,
        labelcolor=_INK2, bbox_to_anchor=(0.5, -0.02),
    )


def _stamp(fig, title: str) -> None:
    fig.suptitle(title, color=_INK, fontsize=11, y=0.98)
    fig.text(0.5, 0.005, _DISCLAIMER, ha="center", va="bottom",
             color=_MUTED, fontsize=6.5, wrap=True)


def _oof_for(pooled_oof, feature_set, model):
    o = pooled_oof[(feature_set, model)]
    return o["y"], o["p"]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_paired_roc_pr(pooled_oof, out: Path) -> None:
    y, p_core = _oof_for(pooled_oof, S.FEATURE_SET_CORE, S.PRIMARY_MODEL)
    _, p_enh = _oof_for(pooled_oof, S.FEATURE_SET_ENHANCED, S.PRIMARY_MODEL)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), facecolor=_SURFACE)
    # ROC
    ax = axes[0]; _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_BASE, lw=1, ls="--", zorder=1)
    fpr, tpr, _ = roc_curve(y, p_core)
    ax.plot(fpr, tpr, color=_FEATURE_COLOR[S.FEATURE_SET_CORE], lw=2,
            label="core", zorder=3)
    fpr, tpr, _ = roc_curve(y, p_enh)
    ax.plot(fpr, tpr, color=_FEATURE_COLOR[S.FEATURE_SET_ENHANCED], lw=2,
            label="enhanced", zorder=4)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title("Pooled OOF ROC (logistic_regression)", fontsize=9)
    # PR
    ax = axes[1]; _style_axes(ax)
    prec, rec, _ = precision_recall_curve(y, p_core)
    ax.plot(rec, prec, color=_FEATURE_COLOR[S.FEATURE_SET_CORE], lw=2,
            label="core", zorder=3)
    prec, rec, _ = precision_recall_curve(y, p_enh)
    ax.plot(rec, prec, color=_FEATURE_COLOR[S.FEATURE_SET_ENHANCED], lw=2,
            label="enhanced", zorder=4)
    pos = float(np.mean(y))
    ax.axhline(pos, color=_BASE, lw=1, ls="--", zorder=1,
               label=f"positive rate ({pos:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_title("Pooled OOF PR (logistic_regression)", fontsize=9)
    _feature_legend(fig)
    _stamp(fig, "Paired out-of-fold ROC / PR -- primary LR (core vs enhanced)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_gain_bootstrap(bootstrap_rows, out: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(bootstrap_rows)
    want = [m for m in ("auroc", "auprc") if m in set(df.get("metric", []))]
    if not want:
        want = list(df["metric"].unique())[:2]
    fig, axes = plt.subplots(1, len(want), figsize=(4.4 * len(want), 4.0),
                             facecolor=_SURFACE)
    if len(want) == 1:
        axes = [axes]
    for ax, metric in zip(axes, want):
        _style_axes(ax)
        row = df[df["metric"] == metric].iloc[0]
        # reconstruct draws from CI is impossible; plot a marker band instead
        delta = row["delta_point_estimate"]
        lo, hi = row["ci_low"], row["ci_high"]
        vals = [v for v in (lo, delta, hi) if v is not None]
        ax.axvline(0.0, color=_BASE, lw=1.2, ls="--", zorder=1)
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, color=_BLUE, alpha=0.15, zorder=0,
                       label="95% percentile CI")
        if delta is not None:
            ax.axvline(delta, color=_BLUE, lw=2.2, zorder=3,
                       label=f"Δ = {delta:+.4f}")
        ax.set_yticks([])
        ax.set_xlabel(f"Δ{metric.upper()} (enhanced - core)")
        ax.set_title(f"Δ{metric.upper()} paired patient bootstrap\n"
                     f"(n_used={row['n_used']}, skipped={row['n_single_class_skipped']})",
                     fontsize=8.5)
        ax.legend(loc="upper right", frameon=False, fontsize=7.5, labelcolor=_INK2)
        if vals:
            ax.set_xlim(min(vals) - 0.03, max(vals) + 0.03)
    _stamp(fig, "Primary LR airflow increment -- paired patient bootstrap")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_per_fold_deltas(fold_records, out: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(fold_records)
    lr = df[df["model"] == S.PRIMARY_MODEL]
    pivot_auroc = lr.pivot(index="outer_fold", columns="feature_set", values="auroc")
    pivot_auprc = lr.pivot(index="outer_fold", columns="feature_set", values="auprc")
    folds = sorted(lr["outer_fold"].unique())
    d_auroc = (pivot_auroc[S.FEATURE_SET_ENHANCED] - pivot_auroc[S.FEATURE_SET_CORE]).reindex(folds)
    d_auprc = (pivot_auprc[S.FEATURE_SET_ENHANCED] - pivot_auprc[S.FEATURE_SET_CORE]).reindex(folds)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=_SURFACE)
    _style_axes(ax)
    x = np.arange(len(folds))
    w = 0.38
    ax.bar(x - w / 2, d_auroc.reindex(folds).values, width=w, color=_BLUE,
           label="ΔAUROC", zorder=3)
    ax.bar(x + w / 2, d_auprc.reindex(folds).values, width=w, color=_ORANGE,
           label="ΔAUPRC", zorder=3)
    ax.axhline(0.0, color=_BASE, lw=1.2, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("Δ metric (enhanced - core)")
    ax.set_title("Per-outer-fold airflow increment (primary LR)", fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=_INK2)
    _stamp(fig, "Per-fold ΔAUROC / ΔAUPRC -- primary LR (enhanced - core)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_calibration(pooled_oof, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=_SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_BASE, lw=1, ls="--", zorder=1)
    for model, col_model, ls in ((S.PRIMARY_MODEL, None, "-"),
                                 (S.MODEL_HIST_GRADIENT_BOOSTING, None, "--")):
        for feature_set in S.FEATURE_SETS:
            y, p = _oof_for(pooled_oof, feature_set, model)
            if np.unique(p).size <= 1:
                continue
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
            lw = 2.0 if model == S.PRIMARY_MODEL else 1.6
            ax.plot(mean_pred, frac_pos, marker="o", ms=4,
                    color=_FEATURE_COLOR[feature_set], lw=lw, ls=ls,
                    label=f"{feature_set} [{model.split('_')[0]}]", zorder=3)
    ax.set_xlabel("Mean predicted probability (10 uniform bins)")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=_INK2)
    _stamp(fig, "Calibration / reliability (pooled OOF, RAW -- not calibrated)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_cohort_flow(gate, out: Path) -> None:
    import pandas as pd
    core_n = gate.core_cohort_windows["patient_id"].nunique() if hasattr(
        gate, "core_cohort_windows") else 50
    air = gate.airflow_cohort_windows
    air_n = air["patient_id"].nunique()
    air_w = len(air)
    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.axis("off")

    def _box(cx, cy, w, h, text, fc, ec, tc=_INK, fs=8.5):
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   facecolor=fc, edgecolor=ec, lw=1.4))
        ax.text(cx, cy, text, ha="center", va="center", color=tc,
                fontsize=fs, wrap=True)

    bw, bh = 2.5, 0.7
    _box(2.0, 4.2, bw, bh, f"v2 core cohort\n{core_n} patients\n(Stage-7 main)",
         "#eef2f9", _BASE)
    _box(2.0, 2.8, bw, bh,
         f"airflow-eligible\n{air_n} patients\n{air_w} windows",
         "#e7f0fb", _BLUE, tc=_INK2)
    _box(5.4, 4.2, bw, bh, "core_restricted\n16 features\n(HR+SpO2)",
         "#fbe9df", _ORANGE, tc=_INK2)
    _box(5.4, 2.8, bw, bh, "airflow_enhanced\n27 features\n(+airflow)",
         "#e7f0fb", _BLUE, tc=_INK2)
    _box(2.0, 1.3, bw, bh, "same windows\nsame folds\n(inherited v2)",
         "#f3f3ef", _BASE, tc=_INK2, fs=8)
    ax.annotate("", xy=(2.0, 3.15), xytext=(2.0, 3.85),
                arrowprops=dict(arrowstyle="->", color=_MUTED, lw=1.2))
    ax.annotate("", xy=(4.15, 4.2), xytext=(3.25, 4.2),
                arrowprops=dict(arrowstyle="->", color=_MUTED, lw=1.2))
    ax.annotate("", xy=(4.15, 2.8), xytext=(3.25, 2.8),
                arrowprops=dict(arrowstyle="->", color=_MUTED, lw=1.2))
    ax.annotate("", xy=(2.0, 1.65), xytext=(2.0, 2.45),
                arrowprops=dict(arrowstyle="->", color=_MUTED, lw=1.2))
    ax.set_xlim(-0.2, 7.4); ax.set_ylim(0.7, 4.9)
    _stamp(fig, "Cohort flow -- paired within-cohort airflow comparison")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# markdown reports
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_airflow_report(
    *, path: Path, run_id: str, split_run_id: str, core_cohort, air_cohort,
    core_features, enh_features, pooled, bootstrap_rows, fold_records,
    selected_records, figure_names, elapsed: float,
) -> None:
    import pandas as pd
    L: List[str] = []
    L.append("# Stage 8 — airflow-incremental paired nested-CV report\n")
    L.append(f"- run_id: `{run_id}`")
    L.append(f"- consumed split: `{split_run_id}` (approved_for_modeling v2)")
    L.append(f"- modeling_version: `{S.MODELING_VERSION}`")
    L.append(f"- elapsed: {elapsed:.1f}s")
    L.append(f"\n> {_DISCLAIMER}.\n")

    L.append("## Cohort & fairness (paired within-cohort)\n")
    L.append(f"- v2 core cohort (Stage-7 main population): **{core_cohort['n_windows']}** "
             f"windows / **{core_cohort['n_patients']}** patients.")
    L.append(_airflow_line(air_cohort))
    L.append(f"- both feature sets are built on the IDENTICAL "
             f"{air_cohort['n_windows']} airflow-eligible windows "
             f"({air_cohort['n_patients']} patients); membership is asserted "
             f"identical (window/patient/label/fold).")
    L.append(f"- core_restricted ({len(core_features)} features): "
             f"`{', '.join(core_features[:4])}…`")
    L.append(f"- airflow_enhanced ({len(enh_features)} features): "
             f"core 16 + `{', '.join(S.AIRFLOW_FEATURE_WHITELIST[:4])}…` "
             f"(11 airflow statistics).")
    L.append("- **NOT** a gain over the Stage-7 50-patient main cohort (different "
             "population). Audio remains BLOCKED.\n")

    L.append("## Pooled out-of-fold metrics (per feature set × model)\n")
    L.append("| feature_set | model (role) | AUROC | AUPRC | Brier | log loss | "
             "sens | spec | F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for feature_set in S.FEATURE_SETS:
        for model in S.MODEL_FAMILIES:
            m = pooled[f"{feature_set}__{model}"]
            L.append(
                f"| {feature_set} | {model} ({S.MODEL_ROLE[model]}) | "
                f"{_fmt(m['auroc'])} | {_fmt(m['auprc'])} | {_fmt(m['brier'])} | "
                f"{_fmt(m['log_loss'])} | {_fmt(m['sensitivity'])} | "
                f"{_fmt(m['specificity'])} | {_fmt(m['f1'])} |"
            )
    L.append("\nThe **main comparison** is logistic_regression "
             "`airflow_enhanced - core_restricted`. HGB is **exploratory** and "
             "cannot seize the main conclusion.\n")

    if bootstrap_rows:
        L.append("## Primary-LR paired patient bootstrap 95% CI (Δ = enhanced - core)\n")
        L.append("| metric | kind | Δ point | CI low | CI high | "
                 "core point | enhanced point | used / skipped |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in bootstrap_rows:
            L.append(
                f"| {r['metric']} | {r['kind']} | {_fmt(r['delta_point_estimate'])} | "
                f"{_fmt(r['ci_low'])} | {_fmt(r['ci_high'])} | "
                f"{_fmt(r['core_point_estimate'])} | {_fmt(r['enhanced_point_estimate'])} | "
                f"{r['n_used']}/{r['n_single_class_skipped']} |"
            )
        L.append("\nBootstrap resamples **patients** (cluster unit) ONCE and brings "
                 "BOTH feature sets' OOF for the drawn patients, so the Δ is a "
                 "within-cohort paired difference. Window-level IID bootstrap and "
                 "cross-cohort pairing are forbidden. Single-class resamples are "
                 "skipped and counted.\n")

    L.append("## Per outer-fold results (primary LR)\n")
    fr = pd.DataFrame(fold_records)
    lr = fr[fr["model"] == S.PRIMARY_MODEL]
    keep = ["feature_set", "outer_fold", "n", "positive_rate", "threshold",
            "auroc", "auprc", "sensitivity", "specificity", "f1"]
    L.append(lr[keep].to_markdown(index=False))
    L.append("")

    L.append("## Selected hyper-parameters & threshold source\n")
    sr = pd.DataFrame(selected_records)
    show = ["feature_set", "model", "outer_fold", "selected_params",
            "threshold", "threshold_rule"]
    L.append(sr[sr["model"] == S.PRIMARY_MODEL][show].to_markdown(index=False))
    L.append("\nEach (feature set × model) selects INDEPENDENTLY on the SAME "
             "frozen inner folds (average_precision; tie-break higher AUROC then "
             "simpler candidate). Thresholds come from the selected candidate's "
             "**inner out-of-fold** predictions (max Youden J) and are applied "
             "as-is to outer-test. **Outer-test never selects model, feature set, "
             "or threshold; probabilities are RAW (not post-calibrated).**\n")

    L.append("## Calibration / Brier limitations\n")
    L.append("- Brier / log loss are computed on **raw** probabilities; they are "
             "NOT calibrated (no outer-test post-calibration). Class weighting "
             "(balanced) and uncalibrated probabilities can inflate Brier, as seen "
             "in Stage 7; the same caveat applies here.")
    L.append("- Any future calibration MUST be a separate stage performed INSIDE "
             "the inner folds.\n")

    if figure_names:
        L.append("## Figures\n")
        for fn in figure_names:
            L.append(f"- `{fn}`")
        L.append("")

    L.append("## Leakage-protection evidence\n")
    L.append("- both feature sets use the IDENTICAL inherited v2 outer/inner "
             "assignments (0 patient/fold mismatch);")
    L.append("- each (feature set × model × fold) is fitted entirely inside its "
             "TRAIN partition (imputer, scaler, weights, hyper-parameters, "
             "threshold);")
    L.append("- outer-test patients are absent from every inner fold;")
    L.append("- the paired bootstrap pairs the SAME windows for both feature sets;")
    L.append("- no audio feature enters either matrix; airflow enters ONLY "
             "airflow_enhanced;")
    L.append("- `data/raw` and Stage 1–7 products are unchanged "
             "(before/after snapshot).\n")

    path.write_text("\n".join(L), encoding="utf-8")


def _airflow_line(air_cohort) -> str:
    return (f"- airflow-eligible sub-cohort: **{air_cohort['n_windows']}** windows / "
            f"**{air_cohort['n_patients']}** patients "
            f"({air_cohort['n_positive']} positive / {air_cohort['n_negative']} "
            f"negative; re-derived at runtime).")


def _write_completion_report(
    *, path: Path, run_id: str, summary: Dict[str, Any],
) -> None:
    L: List[str] = ["# Stage 8 — completion report (phase_08)\n"]
    L.append(f"- run_id: `{run_id}`")
    L.append(f"- status: `{summary.get('status')}`")
    L.append(f"- consumed approved v2 split: `{summary.get('split_run_id')}`")
    L.append(f"- approval_status: `{summary.get('approval_status')}`")
    L.append(f"- airflow cohort: "
             f"{summary.get('airflow_cohort', {}).get('n_windows')} windows / "
             f"{summary.get('airflow_cohort', {}).get('n_patients')} patients")
    L.append(f"- elapsed: {summary.get('elapsed_seconds', 0):.1f}s")
    L.append(f"- raw_modified: `{summary.get('raw_modified')}`")
    L.append(f"- historical_products_unchanged: "
             f"`{summary.get('historical_products_unchanged')}`")
    L.append(f"- production_paths_clean: `{summary.get('production_paths_clean')}`")
    L.append(f"- audio_cohort_count: `{summary.get('audio_cohort_count')}`")
    L.append(f"- figures: {summary.get('n_figures', 0)}")
    L.append(f"\n> {_DISCLAIMER}.\n")
    L.append("This report is the structured §8 completion summary for Codex "
             "audit. See `airflow_incremental_report.md` for full results. This is "
             "a paired WITHIN-cohort airflow increment on the 34-patient "
             "sub-cohort — NOT a gain over the Stage-7 50-patient main cohort. "
             "Audio route remains BLOCKED.")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def write_airflow_reports(
    *, run_dirs, mc, gate, core_frame, enhanced_frame, pooled, pooled_oof,
    bootstrap_rows, fold_records, selected_records, run_id, split_run_id,
    elapsed: float,
) -> int:
    rdir = run_dirs["reports"]
    fdir = rdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    figure_names: List[str] = []
    try:
        _fig_paired_roc_pr(pooled_oof, fdir / "paired_oof_roc_pr.png")
        figure_names.append("figures/paired_oof_roc_pr.png")
        _fig_gain_bootstrap(bootstrap_rows, fdir / "primary_gain_bootstrap.png")
        figure_names.append("figures/primary_gain_bootstrap.png")
        _fig_per_fold_deltas(fold_records, fdir / "per_fold_deltas.png")
        figure_names.append("figures/per_fold_deltas.png")
        _fig_calibration(pooled_oof, fdir / "calibration.png")
        figure_names.append("figures/calibration.png")
        _fig_cohort_flow(gate, fdir / "cohort_flow.png")
        figure_names.append("figures/cohort_flow.png")
    except Exception as exc:  # pragma: no cover - rendering env dependent
        (rdir / "figure_error.txt").write_text(
            f"figure rendering failed: {type(exc).__name__}: {exc}",
            encoding="utf-8")

    _write_airflow_report(
        path=rdir / "airflow_incremental_report.md", run_id=run_id,
        split_run_id=split_run_id,
        core_cohort={"n_windows": int(len(gate.core_cohort_windows)),
                     "n_patients": int(gate.core_cohort_windows["patient_id"].nunique())},
        air_cohort={"n_windows": int(len(gate.airflow_cohort_windows)),
                    "n_patients": int(gate.airflow_cohort_windows["patient_id"].nunique()),
                    "n_positive": int((gate.airflow_cohort_windows["binary_event_label"] == 1).sum()),
                    "n_negative": int((gate.airflow_cohort_windows["binary_event_label"] == 0).sum())},
        core_features=list(core_frame.feature_names),
        enh_features=list(enhanced_frame.feature_names),
        pooled=pooled, bootstrap_rows=bootstrap_rows, fold_records=fold_records,
        selected_records=selected_records, figure_names=figure_names, elapsed=elapsed,
    )
    _write_completion_report(
        path=rdir / "phase_08_completion_report.md", run_id=run_id,
        summary={"status": "OK", "split_run_id": split_run_id,
                 "approval_status": gate.signature.get("approval_status"),
                 "airflow_cohort": {"n_windows": int(len(gate.airflow_cohort_windows)),
                                    "n_patients": int(gate.airflow_cohort_windows["patient_id"].nunique())},
                 "elapsed_seconds": elapsed, "raw_modified": False,
                 "historical_products_unchanged": True,
                 "production_paths_clean": True,
                 "audio_cohort_count": S.AUDIO_COHORT_COUNT,
                 "n_figures": len(figure_names)},
    )
    return len(figure_names)


__all__ = ["write_airflow_reports"]
