# BSPC revision analysis protocol (Stage 15b) — frozen before any re-run

> **Registration kind:** this is a **pre-submission revision registration** for the BSPC
> candidate. It is **NOT** a claim that the Stage 2–14 analysis was prospectively
> pre-registered. It is frozen (UTC `2026-08-08T15:42:29Z`, seed `20250714`) **before any
> Stage-15b run**, and the main choices below are not edited to chase a better number. The
> machine-readable twin is `config/bspc_revision_analysis.yaml`.

## 1. Task and scope

- **Main task:** window-level **PSG-scored respiratory-event classification**.
- **Explicitly NOT:** patient-level OSA diagnosis, PSG replacement, screening, real-time or
  deployment accuracy, external validation, or any audio result. Single public dataset,
  patient-level **internal** cross-validation only.
- **Audio:** BLOCKED (0 eligible windows; not processed). **Airflow:** shared-source /
  incorporation-risk **secondary** analysis only (supplement; never a main result).

## 2. Inputs (all frozen; nothing overwritten)

| input | source |
|---|---|
| events | `annotations/parsed_events.parquet` (Stage 3) |
| canonical awake | parsed_events `overlaps_awake_interval` (Stage 3) |
| windows / time contract | `splits/core_cohort_window_membership.parquet` + Stage-4 contract (30 s, `[start,end)`, relative to `record_start`) |
| HR / SpO2 / airflow features | `features/physiology/{hr,spo2,airflow}_window_features.parquet` (Stage 5) |
| outer split | `splits/outer_patient_folds_core_v2.csv` (approved v2) |
| inner split | `splits/inner_patient_folds_core_v2.parquet` (approved v2) |

Old Stage 4 labels are **re-derived**, never overwritten. The `event_start_point_half_open`
sensitivity variant must reproduce the historical Stage-4 label exactly (sanity check).

## 3. Label protocol (the core revision)

Retained events = scored respiratory events with `overlaps_awake_interval == False`.
`overlap(window,event) = max(0, min(w_end,e_end) − max(w_start,e_start))` seconds.

| variant | rule | role |
|---|---|---|
| **main `overlap_ge_10s`** | positive if overlap ≥ 10 s; **indeterminate** if 0 < overlap < 10 s; negative if overlap = 0 | **PRIMARY** (fixed before results) |
| `event_start_point_half_open` | positive iff retained event start ∈ `[w_start, w_end)` | sensitivity (continuity with Stage 4) |
| `any_overlap` | positive if overlap > 0 | sensitivity |
| `coverage_ge_50pct` | positive if overlap ≥ 15 s (≥50% window); indeterminate if 0 < overlap < 15 s | sensitivity |

**Indeterminate** windows (slight overlap below the main threshold) are **excluded from main
training and evaluation** — never forced positive or negative, never decided by performance.
Rationale for ≥10 s: it is at/above the AASM minimum respiratory-event duration and matches the
ChatGPT-recommended threshold; it was chosen **before** any result.

## 4. Feature families (pre-registered)

Statistics per modality: `mean, median, std_ddof1, min, max, range, iqr, slope_per_second`
(airflow additionally `rms, zero_crossing_count, zero_crossing_rate`). Families: `hr_only`,
`spo2_only`, `hr_spo2` (**primary**, 14 columns), and `simple_spo2_min` (univariate). IDs,
time, quality/coverage, modal-availability, label, annotation, awake, sleep, audio and any
full-cohort-fitted variable are **forbidden** as features. All preprocessing is
fold-internal.

## 5. Models (fixed roles; no outer-test champion)

`dummy_prior` (lower bound), `simple_spo2_min_lr` (simple baseline), `hr_only_lr`,
`spo2_only_lr`, **`hr_spo2_lr` (PRIMARY)**, `hr_spo2_hgb` (exploratory). LR pipeline =
`SimpleImputer(median)→StandardScaler→LogisticRegression`; grid `C∈{0.1,1,10}` ×
`class_weight∈{None,balanced}`. HGB = `SimpleImputer(median)→HistGradientBoostingClassifier`;
grid `learning_rate∈{0.05,0.10}` × `max_leaf_nodes∈{15,31}`. Fixed solver/seed. HGB never
replaces LR by outer-test score.

## 6. Nested CV (inherited v2; never rebuilt)

5 outer × 4 inner patient-level folds. Inner selection = max mean `average_precision`
(tol `1e-9`); tie → higher mean AUROC → ascending grid index. Refit on full outer-train; predict
outer-test **once**. Threshold = inner-OOF max-Youden-J (lowest-probability tie-break),
**never** fit on outer-test; dummy fixed 0.5. Outer-test patients never in that outer run's
inner folds (asserted).

## 7. Physiological latency (SpO2 lags the event)

`lag = +L` shifts the **feature** context +L s relative to the label window (uses signal
`L..L+30 s` after window start). Main = `lag_0` (**current-window / no-future-window**
setting: features use only the window's own 30 s, complete at the window's end — a
retrospective labelling convention, NOT a causal/online/real-time deployment claim; no
real-time deployment was evaluated in this study).
`lag_plus_30`, `lag_plus_60` use future signal → **offline/context sensitivity**, never
real-time, never promoted by performance. End-of-recording tails with no shifted window are
**dropped and reported**.

## 8. Metrics and uncertainty

- **Window:** AUROC, AUPRC, Brier, log loss + threshold metrics at the fixed threshold.
- **Patient-macro:** per-patient AUROC/AUPRC over patients with **both** classes in their OOF,
  then macro-averaged; report n_included and NA reason; **never** conflated with window macro.
- **Event-level** (primary LR + simple baseline): candidate event = maximal run of adjacent
  positive windows; GT = retained non-awake events; match = interval overlap > 0; report event
  sensitivity, precision, F1, **false positives/hour**, counts. **Not** patient diagnosis.
- **Calibration:** Brier, intercept, slope, 10-bin reliability (internal sensitivity; not
  clinical calibration).
- **Uncertainty:** patient-cluster percentile bootstrap (1000, seed 20250714, 95% CI);
  single-class resamples skipped and counted; window-IID forbidden. Paired patient bootstrap
  for `hr_spo2_lr` vs `hr_only_lr` / `spo2_only_lr` / `simple_spo2_min_lr`, and `hgb` vs `lr`
  (exploratory). Main result = pooled AUROC of `hr_spo2_lr`, main label, `lag_0`.

## 9. Airflow secondary (shared-source / incorporation risk)

On the 34-patient selective sub-cohort (HR+SpO2+airflow all available, same windows, inherited
v2 folds): paired `airflow_enhanced_lr − core_restricted_lr` with patient-cluster paired
bootstrap. Framed as a **shared-source construct / upper-bound** check (airflow participates in
PSG event scoring), **supplementary only**, never in the abstract/main result.

## 10. Not-interpretable branches

Any label/event branch that cannot be reliably defined from the data structure is **stopped**
and reported `not_interpretable`; an event-onset result is never substituted under its name.

## 11. Output isolation

All Stage-15b products are isolated under `results/runs/<run_id>/`, `models/runs/<run_id>/`,
`reports/modeling/runs/<run_id>/`. Stage 2–14 products stay byte-identical. No raw, no audio,
no absolute paths, no `pytest`/`Temp`/`AppData`; OOF tables carry only pseudo-anonymous keys.
