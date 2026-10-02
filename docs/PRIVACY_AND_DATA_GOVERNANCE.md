# Privacy and data-governance assessment of the public release

## 1. Items evaluated and their disposition

| Candidate item | Disposition in this release | Rationale |
|---|---|---|
| Raw PSG/CSV/audio files (53 GB) | **EXCLUDED** | Patient data; never redistributable regardless of CC BY 4.0 (project policy is stricter: no raw redistribution) |
| `oof_predictions.parquet` (per-window OOF) | **EXCLUDED** | Per-window derived data: contains pseudonymous patient codes + per-window labels/probabilities + window ordering; re-linkable to official dataset records by anyone holding the dataset → instruction requires separate linkability approval BEFORE inclusion |
| Per-window feature parquets (HR/SpO2/airflow) | **EXCLUDED** | Per-window physiological values per patient — patient-derived, linkable |
| Window membership / split parquets | **EXCLUDED** | Per-patient, per-window mapping incl. exact window times |
| Parsed event tables | **EXCLUDED** | Per-patient event timing (exact times) |
| Per-patient AUROC lists (`per_patient_auroc_primary.csv`) | **EXCLUDED** (kept out of `results_published/`) | Keyed by patient code; already summarized safely by `patient_macro_metrics.csv` (macro + n_included/n_excluded) |
| `results_published/stage15b_aggregates/*.csv` | **INCLUDED** | Pooled/fold/model-level aggregates only; no patient identifiers, no window-level rows, no times (verified by column audit) |
| `results_published/figure_inputs/` (curve points, calibration bins, per-fold AUROC) | **INCLUDED** | Model-level aggregates; `fold_auroc_verification.csv` has per-outer-fold counts only (fold membership of patients is not revealed) |
| `figures_published/`, `tables_published/` | **INCLUDED** | Same numbers as the manuscript under review |
| Download URL list used by the authors | **EXCLUDED** | Machine/session-specific; users must obtain links from the official data page themselves |
| Ethics documents, reviewer correspondence, editorial machinery | **EXCLUDED** | Private project-internal material |

## 2. Re-linkability analysis of the borderline items

- The dataset's patient identifiers are pseudonymous codes (`01`…`50` as used in
  the official tree). Aggregates that do not key on those codes cannot be linked
  back to an individual record without re-running the analysis on the dataset.
- The frozen OOF parquet and per-patient files DO key on those codes and on
  per-window positions; combined with the public dataset they reproduce
  per-patient, per-30-s physiological-model outputs. Even though model outputs
  are not raw biosignals, this granularity was judged above the release bar set
  for this candidate. **A future release may include them only after an explicit
  author approval of a documented linkability assessment** (e.g. considering
  dropping patient codes / per-window granularity).
- `fold_auroc_verification.csv` per-outer-fold rows: the outer-fold → patient
  mapping is NOT published, so fold rows cannot be linked to individuals.

## 3. Sensitive-content scan performed on the candidate

See the stage-24 audit (`reports/release_audit.md` in the run directory, outside
this public bundle) for the full scan: patient-code patterns, absolute personal
paths, credentials/tokens, emails, and dataset file names — zero hits expected;
any hit blocks publication.

## 4. Ethics statement position

The manuscript documents an ethics-exemption determination for this secondary
analysis of a public open-access dataset (no new human-subject data collected by
the authors). No ethics documents are included in the code release; the
manuscript (journal copy) is the authoritative record.

## 5. If the authors later approve releasing derived per-window data

1. Re-run the stage-24 audit with the item added; re-do §2 analysis explicitly.
2. Update `release_manifest.json`, README §1 and this document.
3. Keep raw audio/raw signals excluded regardless (project policy).
