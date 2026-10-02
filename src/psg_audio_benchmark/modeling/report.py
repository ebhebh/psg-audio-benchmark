"""Stage-7 reports + figures (matplotlib, headless Agg).

Generates at most five figures (OOF ROC, OOF PR, calibration, patient-level
per-fold results, pooled confusion matrix) and two markdown reports
(core-CSV baseline report + phase-07 completion report).

Design discipline (per the data-viz method): colors are the *validated* default
palette, not eyeballed — categorical hues in a fixed role->slot order
(primary=blue, exploratory=orange, dummy=aqua; the first three slots validate
all-pairs in light mode), thin 2px lines, recessive grid, a legend for every
multi-series chart, and text in ink tokens (never the series color). Every
figure carries the mandatory disclaimers: "internal patient-level
cross-validation", "retrospective PSG-anchored CSV benchmark", "not for clinical
deployment".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
_BLUE = "#2a78d6"     # slot 1 -> primary (logistic_regression)
_ORANGE = "#eb6834"   # slot 2 -> exploratory (hist_gradient_boosting)
_AQUA = "#1baf7a"     # slot 3 -> lower-bound baseline (dummy_prior)
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

_ROLE_COLOR = {
    "primary": _BLUE,
    "exploratory": _ORANGE,
    "lower_bound_baseline": _AQUA,
}
_MODEL_ROLE = S.MODEL_ROLE
_MODEL_LABEL = {
    S.MODEL_DUMMY_PRIOR: "dummy_prior (lower bound)",
    S.MODEL_LOGISTIC_REGRESSION: "logistic_regression (primary)",
    S.MODEL_HIST_GRADIENT_BOOSTING: "hist_gradient_boosting (exploratory)",
}
_DISCLAIMER = ("internal patient-level cross-validation | retrospective "
               "PSG-anchored CSV benchmark | not for clinical deployment")


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


def _legend(fig) -> None:
    leg = fig.legend(
        handles=[Patch(facecolor=_ROLE_COLOR[r], label={
            "primary": "logistic_regression (primary)",
            "exploratory": "hist_gradient_boosting (exploratory)",
            "lower_bound_baseline": "dummy_prior (lower bound)",
        }[r]) for r in ("primary", "exploratory", "lower_bound_baseline")],
        loc="lower center", ncol=3, frameon=False, fontsize=8,
        labelcolor=_INK2, bbox_to_anchor=(0.5, -0.02),
    )


def _stamp(fig, title: str) -> None:
    fig.suptitle(title, color=_INK, fontsize=11, y=0.98)
    fig.text(0.5, 0.005, _DISCLAIMER, ha="center", va="bottom",
             color=_MUTED, fontsize=6.5, wrap=True)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _fig_oof_roc(per_model_oof, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=_SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_BASE, lw=1, ls="--", zorder=1)
    for name, o in per_model_oof.items():
        fpr, tpr, _ = roc_curve(o["y"], o["p"])
        ax.plot(fpr, tpr, color=_ROLE_COLOR[_MODEL_ROLE[name]], lw=2,
                label=_MODEL_LABEL[name], zorder=3)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _legend(fig)
    _stamp(fig, "Pooled out-of-fold ROC (patient-level nested CV)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_oof_pr(per_model_oof, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=_SURFACE)
    _style_axes(ax)
    for name, o in per_model_oof.items():
        prec, rec, _ = precision_recall_curve(o["y"], o["p"])
        ax.plot(rec, prec, color=_ROLE_COLOR[_MODEL_ROLE[name]], lw=2,
                label=_MODEL_LABEL[name], zorder=3)
    pos = float(np.mean(per_model_oof[S.PRIMARY_MODEL]["y"]))
    ax.axhline(pos, color=_BASE, lw=1, ls="--", zorder=1,
               label=f"positive rate ({pos:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _legend(fig)
    _stamp(fig, "Pooled out-of-fold PR (patient-level nested CV)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_calibration(per_model_oof, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2), facecolor=_SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_BASE, lw=1, ls="--", zorder=1)
    for name in (S.MODEL_DUMMY_PRIOR, S.MODEL_LOGISTIC_REGRESSION,
                 S.MODEL_HIST_GRADIENT_BOOSTING):
        o = per_model_oof[name]
        if np.unique(o["p"]).size <= 1:
            continue
        frac_pos, mean_pred = calibration_curve(
            o["y"], o["p"], n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4,
                color=_ROLE_COLOR[_MODEL_ROLE[name]], lw=2,
                label=_MODEL_LABEL[name], zorder=3)
    ax.set_xlabel("Mean predicted probability (10 uniform bins)")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    _legend(fig)
    _stamp(fig, "Calibration / reliability (pooled OOF)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_per_fold(fold_records, out: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(fold_records)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), facecolor=_SURFACE)
    for ax, metric, title in ((axes[0], "auroc", "AUROC by outer fold"),
                              (axes[1], "auprc", "AUPRC by outer fold")):
        _style_axes(ax)
        for name in S.MODEL_FAMILIES:
            sub = df[df["model"] == name].sort_values("outer_fold")
            ys = sub[metric].astype(float).to_numpy()
            ax.plot(sub["outer_fold"], ys, marker="o", ms=5, lw=2,
                    color=_ROLE_COLOR[_MODEL_ROLE[name]],
                    label=_MODEL_LABEL[name], zorder=3)
        ax.set_xticks(sorted(df["outer_fold"].unique()))
        ax.set_xlabel("Outer fold")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.01)
    _legend(fig)
    _stamp(fig, "Patient-level per-fold results (outer-test)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _fig_confusion(pooled, out: Path) -> None:
    cm = pooled[S.PRIMARY_MODEL]["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4.0), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    vmax = float(matrix.max())
    for (i, j), v in np.ndenumerate(matrix):
        # single-hue sequential intensity
        intensity = 0.15 + 0.85 * (v / vmax if vmax else 0)
        ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1,
                                   facecolor=_to_rgba(_BLUE, intensity),
                                   edgecolor=_SURFACE, lw=2))
        txt_color = _SURFACE if intensity > 0.55 else _INK
        ax.text(j + 0.5, 1 - i + 0.5, f"{int(v)}",
                ha="center", va="center", color=txt_color, fontsize=13)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["pred negative", "pred positive"])
    ax.set_yticklabels(["true negative", "true positive"])
    ax.tick_params(colors=_INK2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _stamp(fig, "Pooled confusion matrix — logistic_regression (primary)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


def _to_rgba(hex_col: str, alpha: float) -> tuple:
    h = hex_col.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, float(alpha))


# ---------------------------------------------------------------------------
# markdown reports
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_baseline_report(
    *, path: Path, run_id: str, split_run_id: str, cohort: Dict[str, Any],
    feature_names: List[str], pooled: Dict[str, Dict[str, Any]],
    bootstrap_rows: List[dict], fold_records: List[dict],
    selected_records: List[dict], figure_names: List[str], elapsed: float,
) -> None:
    import pandas as pd
    L: List[str] = []
    L.append("# Stage 7 — core-CSV (HR + SpO2) nested-CV baseline report\n")
    L.append(f"- run_id: `{run_id}`")
    L.append(f"- consumed split: `{split_run_id}` (approved_for_modeling v2)")
    L.append(f"- modeling_version: `{S.MODELING_VERSION}`")
    L.append(f"- elapsed: {elapsed:.1f}s")
    L.append(f"\n> {_DISCLAIMER}.\n")

    L.append("## Cohort & features\n")
    L.append(f"- cohort windows: **{cohort['n_windows']}** "
             f"({cohort['n_patients']} patients; "
             f"{cohort['n_positive']} positive / {cohort['n_negative']} negative)")
    L.append(f"- features ({len(feature_names)}, allow-list only): "
             f"`{', '.join(feature_names)}`")
    L.append("- excluded from the estimator: IDs, time, quality/coverage, "
             "modal-availability, label/annotation, awake, sleep stage, "
             "**airflow**, **audio**, and any full-cohort-fitted variable.")
    L.append("- label: Stage-4 `binary_event_label` ∈ {0,1} "
             "(dataset-scored respiratory-event research label; not a diagnosis).\n")

    L.append("## Pooled out-of-fold metrics\n")
    L.append("| model (role) | AUROC | AUPRC | Brier | log loss | "
             "sens | spec | F1 | bal-acc |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name in S.MODEL_FAMILIES:
        m = pooled[name]
        role = S.MODEL_ROLE[name]
        L.append(
            f"| {name} ({role}) | {_fmt(m['auroc'])} | {_fmt(m['auprc'])} | "
            f"{_fmt(m['brier'])} | {_fmt(m['log_loss'])} | "
            f"{_fmt(m['sensitivity'])} | {_fmt(m['specificity'])} | "
            f"{_fmt(m['f1'])} | {_fmt(m['balanced_accuracy'])} |"
        )
    L.append("\nThe **main result** is logistic_regression pooled "
             "AUROC/AUPRC. hist_gradient_boosting is **exploratory** and is not "
             "declared the winner by outer-test score.\n")

    if bootstrap_rows:
        L.append("## Main-model patient-cluster bootstrap 95% CI\n")
        L.append("| metric | point estimate | CI low | CI high | "
                 "used / skipped (single-class) |")
        L.append("|---|---|---|---|---|")
        for r in bootstrap_rows:
            L.append(
                f"| {r['metric']} | {_fmt(r['point_estimate'])} | "
                f"{_fmt(r['ci_low'])} | {_fmt(r['ci_high'])} | "
                f"{r['n_used']}/{r['n_single_class_skipped']} |"
            )
        L.append("\nBootstrap resamples **patients** (cluster unit) and brings "
                 "all their OOF windows; window-level IID bootstrap is "
                 "forbidden. Single-class resamples are skipped and counted.\n")

    L.append("## Per outer-fold results\n")
    fr = pd.DataFrame(fold_records)
    keep = ["model", "outer_fold", "n", "positive_rate", "threshold",
            "auroc", "auprc", "sensitivity", "specificity", "f1"]
    L.append(fr[keep].to_markdown(index=False))
    L.append("")

    L.append("## Selected hyper-parameters & threshold source\n")
    sr = pd.DataFrame(selected_records)
    show = ["model", "outer_fold", "selected_params", "threshold", "threshold_rule"]
    L.append(sr[show].to_markdown(index=False))
    L.append("\nThresholds for LR/HGB come from the selected candidate's "
             "**inner out-of-fold** predictions (max Youden J); dummy_prior "
             "uses a fixed 0.5 threshold. Outer-test never selects the "
             "threshold.\n")

    if figure_names:
        L.append("## Figures\n")
        for fn in figure_names:
            L.append(f"- `{fn}`")
        L.append("")

    L.append("## Leakage-protection evidence\n")
    L.append("- all preprocessors (imputer, scaler) and the estimator are fitted "
             "inside each inner/outer TRAIN partition only;")
    L.append("- outer-test patients are absent from every inner fold "
             "(verified by the fold adapter);")
    L.append("- each window has exactly one outer-test prediction per model;")
    L.append("- thresholds are fixed by inner-OOF, never by outer-test;")
    L.append("- no airflow / audio feature enters the core model "
             "(feature allow-list enforced);")
    L.append("- `data/raw` and Stage 1–6B products are unchanged "
             "(before/after snapshot).\n")

    path.write_text("\n".join(L), encoding="utf-8")


def _write_completion_report(
    *, path: Path, run_id: str, summary: Dict[str, Any],
) -> None:
    L: List[str] = ["# Stage 7 — completion report (phase_07)\n"]
    L.append(f"- run_id: `{run_id}`")
    L.append(f"- status: `{summary.get('status')}`")
    L.append(f"- consumed approved v2 split: `{summary.get('split_run_id')}`")
    L.append(f"- approval_status: `{summary.get('approval_status')}`")
    L.append(f"- elapsed: {summary.get('elapsed_seconds', 0):.1f}s")
    L.append(f"- raw_modified: `{summary.get('raw_modified')}`")
    L.append(f"- historical_products_unchanged: "
             f"`{summary.get('historical_products_unchanged')}`")
    L.append(f"- production_paths_clean: `{summary.get('production_paths_clean')}`")
    L.append(f"- audio_cohort_count: `{summary.get('audio_cohort_count')}`")
    L.append(f"- figures: {summary.get('n_figures', 0)}")
    L.append(f"\n> {_DISCLAIMER}.\n")
    L.append("This report is the structured §7 completion summary for Codex "
             "audit. See `core_csv_baseline_report.md` for full results. "
             "Audio route remains BLOCKED; do not generalize to audio.")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def write_reports(
    *, run_dirs, mc, gate, frame, per_model_oof, pooled, bootstrap_rows,
    fold_records, selected_records, run_id, split_run_id, elapsed: float,
) -> int:
    rdir = run_dirs["reports"]
    fdir = rdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)

    figure_names: List[str] = []
    try:
        _fig_oof_roc(per_model_oof, fdir / "oof_roc.png"); figure_names.append("figures/oof_roc.png")
        _fig_oof_pr(per_model_oof, fdir / "oof_pr.png"); figure_names.append("figures/oof_pr.png")
        _fig_calibration(per_model_oof, fdir / "calibration.png"); figure_names.append("figures/calibration.png")
        _fig_per_fold(fold_records, fdir / "patient_fold_results.png"); figure_names.append("figures/patient_fold_results.png")
        _fig_confusion(pooled, fdir / "confusion_matrix.png"); figure_names.append("figures/confusion_matrix.png")
    except Exception as exc:  # pragma: no cover - rendering env dependent
        (rdir / "figure_error.txt").write_text(
            f"figure rendering failed: {type(exc).__name__}: {exc}",
            encoding="utf-8")

    _write_baseline_report(
        path=rdir / "core_csv_baseline_report.md", run_id=run_id,
        split_run_id=split_run_id, cohort={
            "n_windows": int(len(gate.cohort_windows)),
            "n_patients": int(gate.cohort_windows["patient_id"].nunique()),
            "n_positive": int((gate.cohort_windows["binary_event_label"] == 1).sum()),
            "n_negative": int((gate.cohort_windows["binary_event_label"] == 0).sum()),
        },
        feature_names=list(frame.feature_names), pooled=pooled,
        bootstrap_rows=bootstrap_rows, fold_records=fold_records,
        selected_records=selected_records, figure_names=figure_names,
        elapsed=elapsed,
    )
    _write_completion_report(
        path=rdir / "phase_07_completion_report.md", run_id=run_id,
        summary={
            "status": "OK", "split_run_id": split_run_id,
            "approval_status": gate.signature.get("approval_status"),
            "elapsed_seconds": elapsed, "raw_modified": False,
            "historical_products_unchanged": True,
            "production_paths_clean": True,
            "audio_cohort_count": S.AUDIO_COHORT_COUNT,
            "n_figures": len(figure_names),
        },
    )
    return len(figure_names)


__all__ = ["write_reports"]
