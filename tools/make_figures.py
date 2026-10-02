"""Release tool: regenerate the manuscript figures from published aggregate outputs.

Adapted WITHOUT rendering-logic changes from the frozen FINAL figures_v5.py of
Stage-23 (submission_package stage23-final-language-and-author-review), which
is Stage-22 figures_v5 plus three Stage-23 rendering edits (Fig3 x labels
"Combined LR/HGB"; SF3 fold-3 label below its point; SF3 mean note moved to the
corner as "Dashed line: mean X.XXX"). Only the absolute Windows path constants
were replaced by CLI arguments; every styling/layout/data line is copied
verbatim from that final script. Earlier provenance: Stage-22 Round-2 figure
repair (figures_v5), from Stage-21 figures_v4.

Same frozen data sources as Stage-21 (no model output recomputed; rendering only).
Changes vs v4 (defect list from the Stage-22 prompt, Section 4):
  Fig2  : shared bottom legend moved from a negative figure-anchor (which landed
          in the x-axis-title band) to a DEDICATED GridSpec legend row at the
          bottom -> three separated bands (plot / x-axis titles / legend);
          3+2 entries over 2 rows; no curve occlusion possible.
  Fig3B : value labels given independent offsets (pooled ABOVE its point,
          patient-macro BELOW its point) and 8 pt text (was 6.0 pt); panel
          widened (width ratio 1 -> 1.18); group names shortened.
  Fig4B : value labels placed uniformly ABOVE points (was to the right, the
          last one hugging the right spine); x-range margins widened.
  SF2   : in-figure per-bin n labels REMOVED (counts move to the legend text
          in Additional file 1); y-range lower margin added so the leftmost
          observed-rate-0 marker is not half-clipped by the axis.
  SF3   : value labels placed uniformly ABOVE points, clear of the mean line.
  SF4A  : each value anchored to its OWN point (core left of the core dot,
          enhanced right of the enhanced dot); the v4 enhanced label sat at
          x = 1 - 1.07 = -0.07, i.e. lower-left of the CORE dot (semantic defect).
  SF4B  : the zero reference line shortened to the CI band (y in [-0.28, 0.28])
          so it no longer crosses the upper Delta/CI text band.
  All   : in-figure text sizes raised and made uniform (values/ticks/legend
          8 pt, axis titles 8.5 pt, panel titles 9 pt, panel letters 11 pt bold);
          nothing below 7.2 pt anywhere (v4 used 6.0-6.8 pt in six places).

Style contract unchanged: white canvas, no grid, Times New Roman, 170 mm width,
vector PDF with embedded fonts (fonttype 42), PNG 300 dpi, SVG real text.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
_ap = argparse.ArgumentParser(description="Regenerate manuscript figures from published aggregate CSVs.")
_ap.add_argument("--aggregates-dir", required=True, help="dir with the stage15b aggregate CSVs (results_published/stage15b_aggregates)")
_ap.add_argument("--figure-inputs-dir", required=True, help="dir with fig2_curve_data.csv / sf2_calibration_bins.csv / fold_auroc_verification.csv (results_published/figure_inputs)")
_ap.add_argument("--out-dir", required=True, help="output directory for pdf/png/svg figures")
_args = _ap.parse_args()
F15 = _args.aggregates_dir
FIGIN = _args.figure_inputs_dir
OUT = _args.out_dir

MM = 1 / 25.4
W = 170 * MM

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

C = dict(primary="#B2182B", blue="#2166AC", orange="#E08214", sky="#4393C3",
         purple="#7B3294", gray="#6E6E6E", black="#1A1A1A")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(__import__("os").path.join(OUT, f"{name}.{ext}"), dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04, format=ext)
    plt.close(fig)
    print("saved", name)


def panel_letter(ax, letter):
    # placed clear of the top y-tick label (e.g. "1.0" at the upper-left corner)
    ax.text(-0.15, 1.10, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color=C["black"])


def frozen(path):
    import os as _os
    return read_csv(_os.path.join(F15, path))


def fnum(x):
    return float(x)


summary = {(r["model"], r["variant"], r["lag_label"]): r for r in frozen("summary_metrics.csv")}
pmacro = {(r["model"], r["variant"], r["lag_label"], r["metric"]): r for r in frozen("patient_macro_metrics.csv")}
events = {r["model"]: r for r in frozen("event_level_metrics.csv") if r["variant"] == "main" and r["lag_label"] == "lag_0"}
labelv = {r["variant"]: r for r in frozen("label_versions.csv")}
lagv = {r["lag_label"]: r for r in frozen("lag_sensitivity.csv")}
pairci = {r["metric"]: r for r in frozen("airflow_paired_ci.csv")}
pairsum = {r["feature_set"]: r for r in frozen("airflow_paired_summary.csv")}
curves = read_csv(__import__("os").path.join(FIGIN, "fig2_curve_data.csv"))
bins = read_csv(__import__("os").path.join(FIGIN, "sf2_calibration_bins.csv"))
folds = read_csv(__import__("os").path.join(FIGIN, "fold_auroc_verification.csv"))

MODELS = [("simple_spo2_min_lr", "SpO$_2$ min (LR)", C["gray"]),
          ("hr_only_lr", "HR only (LR)", C["orange"]),
          ("spo2_only_lr", "SpO$_2$ only (LR)", C["sky"]),
          ("hr_spo2_lr", "HR+SpO$_2$ (LR, primary)", C["primary"]),
          ("hr_spo2_hgb", "HR+SpO$_2$ (HGB, exploratory)", C["purple"])]

# ---------------------------------------------------------------- Figure 1
fig, ax = plt.subplots(figsize=(W, 118 * MM))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def box(x, y, w, h, title, detail, fc="#FFFFFF", ec="#1A1A1A", lw=1.0, tfs=9, dfs=8, dashed=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                               linestyle="--" if dashed else "-",
                               joinstyle="round", zorder=2))
    if detail:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=C["black"], zorder=3)
        ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center", fontsize=dfs,
                color=C["black"], zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                fontweight="bold", color=C["black"], zorder=3)

def arrow(x1, y1, x2, y2, label=None, lx=0, ly=0, fs=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["black"], lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=11))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="center",
                fontsize=fs, color=C["black"], zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

# main path (left column)
MX, MW = 0.045, 0.52
box(MX, 0.89, MW, 0.09, "50 patients", "public multimodal PSG dataset (Tao et al. 2025)")
arrow(MX + MW / 2, 0.89, MX + MW / 2, 0.805)
box(MX, 0.705, MW, 0.10, "34,643 candidate 30-s windows", "grid already past awake-overlap and signal-quality gates")
arrow(MX + MW / 2, 0.705, MX + MW / 2, 0.61)
box(MX, 0.515, MW, 0.095, "Main-label processing", "overlap with retained, non-wake, expert-scored events")
arrow(MX + MW / 2, 0.515, MX + MW / 2, 0.385)
box(MX, 0.28, MW, 0.105, "30,506 analysable windows", "11,514 positive  ·  18,992 negative")
arrow(MX + MW / 2, 0.28, MX + MW / 2, 0.21)
# indeterminate-exclusion side branch (beside the down-arrow, left of the divider x = 0.615)
box(0.385, 0.395, 0.20, 0.115, "4,137 excluded", "indeterminate windows\n(0 < overlap < 10 s)",
    ec=C["gray"], dashed=True, tfs=8, dfs=7.2)
arrow(0.305, 0.45, 0.385, 0.45)
box(MX, 0.095, MW, 0.115, "Patient-level nested cross-validation", "5 outer × 4 inner folds; one out-of-fold prediction per window")

# airflow branch (right column)
AX_, AW = 0.685, 0.28
box(AX_, 0.875, AW, 0.105, "34 patients", "with verified airflow channel", ec="#444444")
arrow(MX + MW, 0.9275, AX_, 0.9275)
arrow(AX_ + AW / 2, 0.875, AX_ + AW / 2, 0.80)
box(AX_, 0.70, AW, 0.10, "24,441 candidate windows", "airflow sub-cohort grid", ec="#444444")
arrow(AX_ + AW / 2, 0.70, AX_ + AW / 2, 0.605)
box(AX_, 0.485, AW, 0.12, "21,441 analysable windows", "secondary paired analysis\ncore vs airflow-enhanced features", ec="#444444", dfs=7.2)
ax.plot([0.615, 0.615], [0.085, 0.98], color="#BBBBBB", lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.text(0.615, 0.055, "secondary analysis", ha="center", va="top", fontsize=7.5,
        color="#666666", style="italic")
save(fig, "fig1_cohort_flow")

# ---------------------------------------------------------------- Figure 2
# DEFECT FIX: constrained layout + "outside lower center" figure legend — the
# layout engine RESERVES space for the legend below the x-axis-title band, so
# the three bands (plot / x-axis titles / legend) are separated by construction
# (the v4 negative-anchor legend and the first v5 fixed-GridSpec attempt could
# still graze the "Recall (sensitivity)" title of panel B).
fig, axs = plt.subplots(1, 2, figsize=(W, 92 * MM), layout="constrained")
fig.get_layout_engine().set(h_pad=0.15, w_pad=0.10)  # widen the reserved legend band
for m, curveset in [("roc", axs[0]), ("pr", axs[1])]:
    for key, label, color in MODELS:
        pts = [(fnum(r["x"]), fnum(r["y"])) for r in curves if r["model"] == key and r["curve"] == m]
        if not pts:
            continue
        xs, ys = zip(*pts)
        is_primary = key == "hr_spo2_lr"
        curveset.plot(xs, ys, color=color, lw=2.2 if is_primary else 1.1,
                      ls="-" if key != "hr_spo2_hgb" else (0, (5, 2)), label=label, zorder=3 if is_primary else 2)
axs[0].plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":")
axs[0].set_xlabel("False-positive rate (1 − specificity)")
axs[0].set_ylabel("True-positive rate (sensitivity)")
axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1.02)
axs[0].set_title("ROC curves", loc="left")
panel_letter(axs[0], "A")
prev = 0.377
axs[1].axhline(prev, color="#999999", lw=0.9, ls="--")
axs[1].text(0.02, prev + 0.020, "prevalence 0.377", ha="left", va="bottom", fontsize=8, color="#666666")
axs[1].set_xlabel("Recall (sensitivity)")
axs[1].set_ylabel("Precision (positive predictive value)")
axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1.02)
axs[1].set_title("Precision–recall curves", loc="left")
panel_letter(axs[1], "B")
hs, ls = axs[0].get_legend_handles_labels()
leg = fig.legend(hs, ls, loc="outside lower center", ncol=3, frameon=False, fontsize=8,
                 handlelength=2.4, columnspacing=1.6, handletextpad=0.6)
# machine bounding-box check: the legend band must sit strictly BELOW both
# x-axis-title bands (three separated bands: plot / titles / legend).
fig.canvas.draw()
_r = fig.canvas.get_renderer()
_leg_top = leg.get_window_extent(_r).y1
_xl_bottom = min(ax.xaxis.label.get_window_extent(_r).y0 for ax in axs)
_gap = _xl_bottom - _leg_top
print(f"fig2 band gap (xlabel bottom vs legend top) = {_gap:.1f} px")
assert _gap > 8, f"fig2 legend band not separated from x-axis-title band (gap {_gap:.1f}px)"
save(fig, "fig2_model_performance")

# ---------------------------------------------------------------- Figure 3
# DEFECT FIX (B): independent value-label offsets (pooled above, macro below),
# 8 pt labels, wider right panel, shorter two-line group names.
# POST-REVIEW FIX (user report 2026-10-02): panel B's first pooled label
# "0.763" crossed the LEFT SPINE (label centred on the x=-0.14 point of group
# 1 overhung the axes boundary) -> explicit wider x-limits so every panel-B
# label clears the panel boundary by >= 2 pt, asserted below. Panel A: the
# horizontal value labels are ~14 pt wide while bars are ~9.5 pt, so taller
# neighbour bars' edges cut through label tips AND near-equal-height
# neighbours' digits overlapped (group 5: 0.75/0.74 boxes intersect) -> the
# labels are ROTATED 90 degrees above their own bars (standard treatment for
# dense grouped bars): each label then spans only its own bar's column.
fig, axs = plt.subplots(1, 2, figsize=(W, 84 * MM), gridspec_kw={"width_ratios": [1.25, 1.18]})
labels3 = ["SpO$_2$ min\n(LR)", "HR only\n(LR)", "SpO$_2$ only\n(LR)",
           "Combined\nLR", "Combined\nHGB"]
keys3 = [m[0] for m in MODELS]
sens = [fnum(events[k]["event_sensitivity"]) for k in keys3]
prec = [fnum(events[k]["event_precision"]) for k in keys3]
f1 = [fnum(events[k]["event_f1"]) for k in keys3]
x = np.arange(len(keys3)); bw = 0.26
axs[0].bar(x - bw, sens, bw, color=C["primary"], label="Sensitivity")
axs[0].bar(x, prec, bw, color=C["blue"], label="Precision")
axs[0].bar(x + bw, f1, bw, color=C["orange"], label="F1")
for pos, vals in ((x - bw, sens), (x, prec), (x + bw, f1)):
    for xi, v in zip(pos, vals):   # vertical label: confined to its own column
        axs[0].text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, rotation=90)
axs[0].set_xticks(x); axs[0].set_xticklabels(labels3, fontsize=8)
axs[0].set_ylabel("Event-level score")
axs[0].set_ylim(0, 1.0)
axs[0].set_title("Event-level detection (many-to-many overlap matching)", loc="left")
axs[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
panel_letter(axs[0], "A")

pooled = [fnum(summary[(k, "main", "lag_0")]["auroc"]) for k in keys3]
macro = [fnum(pmacro[(k, "main", "lag_0", "patient_macro_auroc")]["value"]) for k in keys3]
for i in range(len(keys3)):
    axs[1].plot([i - 0.14, i + 0.14], [pooled[i], macro[i]], color="#CCCCCC", lw=0.9, zorder=1)
axs[1].scatter(x - 0.14, pooled, s=34, color=C["primary"], zorder=3, label="Pooled window AUROC")
axs[1].scatter(x + 0.14, macro, s=34, color=C["blue"], zorder=3, label="Patient-macro AUROC")
b_labels = []
for xi, v in zip(x - 0.14, pooled):   # above its own point
    b_labels.append(axs[1].text(xi, v + 0.040, f"{v:.3f}", ha="center", va="bottom",
                                fontsize=8, color=C["primary"]))
for xi, v in zip(x + 0.14, macro):    # below its own point (independent offset;
    # 0.050 clears the POOLED dot of the rising group (G2), whose lower edge
    # reached the macro label's first digit at ~1px with the v5 offsets)
    b_labels.append(axs[1].text(xi, v - 0.050, f"{v:.3f}", ha="center", va="top",
                                fontsize=8, color=C["blue"]))
axs[1].set_xticks(x); axs[1].set_xticklabels(labels3, fontsize=8)
axs[1].set_ylabel("AUROC")
axs[1].set_ylim(0, 1.0)
# POST-REVIEW FIX: wide explicit margins so no panel-B label reaches a spine
axs[1].set_xlim(-0.55, len(keys3) - 0.45)
axs[1].set_title("Pooled vs within-patient (patient-macro) AUROC", loc="left")
axs[1].legend(frameon=False, loc="lower right", fontsize=8)
panel_letter(axs[1], "B")
fig.tight_layout(w_pad=2.2)
# machine bounding-box check: every panel-B value label sits >= 2 pt INSIDE
# the axes patch (v5 defect: first pooled 0.763 crossed the left spine).
fig.canvas.draw()
_r3 = fig.canvas.get_renderer()
_ab3 = axs[1].get_window_extent(_r3)
for _t in b_labels:
    _e = _t.get_window_extent(_r3)
    assert _e.x0 - _ab3.x0 >= 2.0 and _ab3.x1 - _e.x1 >= 2.0, (
        f"fig3B label {_t.get_text()} reaches a panel boundary "
        f"(left gap {_e.x0 - _ab3.x0:.1f}px, right gap {_ab3.x1 - _e.x1:.1f}px)")
save(fig, "fig3_event_patient")

# ---------------------------------------------------------------- Figure 4
# DEFECT FIX (B): value labels uniformly ABOVE the points (the v4 right-side
# label of the last point ran into the right spine); wider x margins.
fig, axs = plt.subplots(1, 2, figsize=(W, 72 * MM))
prot = [("sensitivity_onset", "Onset\n(start-point)", "34,643 windows\nprev 0.281"),
        ("main", "Main\n(overlap ≥ 10 s)", "30,506 windows\nprev 0.377"),
        ("sensitivity_any", "Any overlap", "34,643 windows\nprev 0.452"),
        ("sensitivity_cov50", "Coverage ≥ 50%", "26,508 windows\nprev 0.284")]
vals = [fnum(summary[("hr_spo2_lr", p[0], "lag_0")]["auroc"]) for p in prot]
xs = np.arange(len(prot))
axs[0].scatter(xs, vals, s=42, color=C["primary"], zorder=3)
for xi, v in zip(xs, vals):
    axs[0].text(xi, v + 0.030, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
axs[0].set_xticks(xs); axs[0].set_xticklabels([p[1] for p in prot], fontsize=8)
axs[0].set_ylabel("Window-level AUROC")
axs[0].set_ylim(0, 1.0); axs[0].set_xlim(-0.55, len(prot) - 0.45)
axs[0].set_title("Label protocols", loc="left")
for xi, p in zip(xs, prot):
    axs[0].text(xi, -0.175, p[2], ha="center", va="top", fontsize=7.5, color="#555555")
panel_letter(axs[0], "A")

lags = [("lag_0", "Lag 0\n(contemporaneous)", "30,506 windows"),
        ("lag_plus_30", "Offline +30 s", "29,752 windows"),
        ("lag_plus_60", "Offline +60 s", "29,203 windows")]
lvals = [fnum(lagv[l[0]]["auroc"]) for l in lags]
xs2 = np.arange(len(lags))
axs[1].scatter(xs2, lvals, s=42, color=C["blue"], zorder=3)
for xi, v in zip(xs2, lvals):
    axs[1].text(xi, v + 0.030, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
axs[1].set_xticks(xs2); axs[1].set_xticklabels([l[1] for l in lags], fontsize=8)
axs[1].set_ylabel("Window-level AUROC")
axs[1].set_ylim(0, 1.0); axs[1].set_xlim(-0.6, len(lags) - 0.4)
axs[1].set_title("Feature timing (main label)", loc="left")
for xi, l in zip(xs2, lags):
    axs[1].text(xi, -0.175, l[2], ha="center", va="top", fontsize=7.5, color="#555555")
panel_letter(axs[1], "B")
fig.tight_layout(w_pad=2.6)
save(fig, "fig4_sensitivity")

# ---------------------------------------------------------------- SF1
fig, ax = plt.subplots(figsize=(W * 0.62, 64 * MM))
aurocs = [fnum(summary[("hr_spo2_lr", p[0], "lag_0")]["auroc"]) for p in prot]
auprcs = [fnum(summary[("hr_spo2_lr", p[0], "lag_0")]["auprc"]) for p in prot]
xs = np.arange(len(prot))
ax.scatter(xs - 0.10, aurocs, s=40, color=C["primary"], zorder=3, label="AUROC")
ax.scatter(xs + 0.10, auprcs, s=40, color=C["blue"], zorder=3, label="AUPRC (average precision)")
for xi, v in zip(xs - 0.10, aurocs):   # above its point
    ax.text(xi, v + 0.026, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=C["primary"])
for xi, v in zip(xs + 0.10, auprcs):   # below its point (independent offset)
    ax.text(xi, v - 0.032, f"{v:.3f}", ha="center", va="top", fontsize=8, color=C["blue"])
ax.set_xticks(xs); ax.set_xticklabels([p[1] for p in prot], fontsize=8)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.0); ax.set_xlim(-0.55, len(prot) - 0.45)
ax.legend(frameon=False, loc="center right", fontsize=8)
fig.tight_layout()
save(fig, "sf1_label_sensitivity")

# ---------------------------------------------------------------- SF2
# DEFECT FIX: per-bin n labels removed (counts are listed in the Additional
# file 1 legend); y-lower margin so the leftmost observed-rate-0 marker at
# (0.081, 0.0) is not half-clipped by the x-axis spine.
fig, ax = plt.subplots(figsize=(W * 0.55, 64 * MM))
mp = [fnum(r["mean_pred"]) for r in bins if r["n"] != "0" and r["mean_pred"]]
ob = [fnum(r["obs_rate"]) for r in bins if r["n"] != "0" and r["mean_pred"]]
ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="--", label="Perfect calibration")
ax.scatter(mp, ob, s=30, color=C["primary"], zorder=3, label="Observed fraction (10 equal-width bins)")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraction of positives")
ax.set_xlim(0, 1); ax.set_ylim(-0.045, 1.045)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
save(fig, "sf2_calibration")

# ---------------------------------------------------------------- SF3
# DEFECT FIX: all value labels uniformly ABOVE their points (v4 put them to the
# side at va="center", where the 0.843 label was grazed by the mean-0.830 line).
fig, ax = plt.subplots(figsize=(W * 0.55, 60 * MM))
fv = [fnum(r["auroc"]) for r in sorted(folds, key=lambda r: int(r["outer_fold"]))]
xs = np.arange(5)
ax.scatter(xs, fv, s=46, color=C["primary"], zorder=3)
for xi, v in zip(xs, fv):
    ax.text(xi, v - 0.045 if xi == 2 else v + 0.038, f"{v:.3f}", ha="center", va="top" if xi == 2 else "bottom", fontsize=8)
mean = float(np.mean(fv))
ax.axhline(mean, color=C["blue"], lw=1.0, ls="--")
ax.text(0.98, 0.05, f"Dashed line: mean {mean:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=C["blue"])
ax.set_xticks(xs); ax.set_xticklabels([f"Fold {i}" for i in range(5)], fontsize=8)
ax.set_ylabel("Window-level AUROC")
ax.set_xlim(-0.5, 4.5); ax.set_ylim(0, 1.0)
fig.tight_layout()
save(fig, "sf3_fold_auroc")

# ---------------------------------------------------------------- SF4
# DEFECT FIX (A): each value anchored beside its OWN point — core label to the
# LEFT of the core dot, enhanced label to the RIGHT of the enhanced dot (v4 drew
# the enhanced label at x = -0.07, lower-left of the core dot).
# DEFECT FIX (B): zero reference line shortened to the CI band so it cannot
# cross the Delta/CI summary text, which sits in its own upper band.
fig, axs = plt.subplots(1, 2, figsize=(W * 0.8, 60 * MM), gridspec_kw={"width_ratios": [1, 1.1]})
core = fnum(pairsum["core_restricted"]["auroc"]); enh = fnum(pairsum["airflow_enhanced"]["auroc"])
axs[0].scatter([0, 1], [core, enh], s=46, color=[C["primary"], C["blue"]], zorder=3)
axs[0].text(-0.07, core, f"{core:.3f}", fontsize=8, va="center", ha="right")
axs[0].text(1.07, enh, f"{enh:.3f}", fontsize=8, va="center", ha="left")
axs[0].set_xticks([0, 1])
axs[0].set_xticklabels(["Core\n(HR + SpO$_2$)", "Enhanced\n(+ airflow)"], fontsize=8)
axs[0].set_ylabel("Window-level AUROC")
axs[0].set_xlim(-0.55, 1.55); axs[0].set_ylim(0, 1.0)
axs[0].set_title("Absolute AUROC (sub-cohort)", loc="left")
panel_letter(axs[0], "A")

d = pairci["auroc"]; est = fnum(d["delta_point_estimate"]); lo = fnum(d["ci_low"]); hi = fnum(d["ci_high"])
axs[1].plot([0, 0], [-0.24, 0.24], color="#777777", lw=1.0)   # shortened zero line
axs[1].errorbar([est], [0], xerr=[[est - lo], [hi - est]], fmt="o", color=C["primary"],
                capsize=4, lw=1.4, capthick=1.4, ms=6, zorder=3)
axs[1].text(0, 0.56, f"Δ = {est:.3f}\n95% CI {lo:.3f} to {hi:.3f}", ha="center", va="center", fontsize=8)
axs[1].text(0, -0.40, "zero", ha="center", va="top", fontsize=8, color="#444444")
axs[1].set_xlabel("ΔAUROC (enhanced − core)")
axs[1].set_xlim(-0.05, 0.05); axs[1].set_ylim(-0.62, 0.88)
axs[1].set_yticks([])
axs[1].set_title("Paired increment with 95% CI", loc="left")
panel_letter(axs[1], "B")
fig.tight_layout(w_pad=2.2)
save(fig, "sf4_airflow_secondary")

print("ALL FIGURES DONE")
