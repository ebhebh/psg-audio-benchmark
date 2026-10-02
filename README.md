# psg-audio-benchmark (author-approved public release)

> Author-approved public code release, version **1.0.0rc2**.
> Repository: https://github.com/ebhebh/psg-audio-benchmark
> Original code: MIT; aggregate outputs: CC BY 4.0 (see LICENSE_DATA.md).
> Zenodo concept DOI: https://doi.org/10.5281/zenodo.23101023. Version 1.0.0rc1 is archived as https://doi.org/10.5281/zenodo.23101024. The associated manuscript is not submitted or published.


Code for the study *"Benchmarking heart-rate and oxygen-saturation features for
respiratory-event window detection in a public polysomnography dataset: a
patient-level cross-validated analysis"* — a secondary analysis of the public
**Tao et al. 2025** multimodal sleep apnea dataset. (The manuscript is
unpublished at the time of this release candidate; nothing here may present it
as a published article — see §10.)

**Scientific boundary (non-negotiable):** this is a public-dataset benchmark with
patient-level internal cross-validation only. It is **not** a clinically
deployable diagnostic system, not a PSG replacement, not an external clinical
validation, and not a screening tool. Window-level PSG-scored respiratory-event
classification — never patient diagnosis (AHI).

---

## 1. What this release contains

| Path | Content |
|---|---|
| `src/psg_audio_benchmark/` | Minimal self-contained analysis package (see §6 dependency map) |
| `scripts/` | Pipeline CLIs, in execution order (`00`…`06b`) plus the Stage-15b analysis (`15_bspc_reanalysis.py`) |
| `config/` | Frozen configuration / analysis registration (incl. `bspc_revision_analysis.yaml`) |
| `tests/` | Synthetic test suite incl. a synthetic end-to-end run of the full analysis |
| `tools/` | `make_figures.py`, `export_figure_data.py`, `recompute_primary_metrics.py`, `bind_run_ids.py` |
| `results_published/` | Published **aggregate** outputs (CSV) of the frozen analysis run + figure inputs |
| `figures_published/` | Final manuscript figures (PDF/PNG/SVG) for reference comparison |
| `tables_published/` | Final manuscript table sources (markdown) |
| `data/` | **Empty** — official acquisition instructions only |
| `docs/` | Reproduction guide, publication steps, third-party licenses, privacy assessment |
| `docs/license_evidence/` | UNFILLED license-confirmation **template** + official-source verification steps (the Stage-2 gate stays `blocked` until a human fills it in) |

**Not included (by design):** raw patient data / audio; per-window OOF
predictions or any per-window derived data; patient-to-recording mappings or
exact per-patient timelines; reference full texts; the authors' internal
editorial/manuscript machinery; historical superseded analysis stages (7–14,
16–19 internal manuscript tooling).

## 2. Installation

**Verified environment: Windows 10, CPython 3.13.9.** The exact package
versions listed in `requirements.txt` (e.g. NumPy 2.3.5) are the ones actually
installed and tested for this release candidate; NumPy 2.3.x itself requires
Python ≥ 3.11, so **Python 3.13 is the recommended and only verified
interpreter**. Other versions (3.11/3.12) are expected to satisfy the
dependency ranges in `pyproject.toml` but were **not** tested for this release
— do not assume they reproduce byte-level figure output. Python 3.10 is not
supported by the pinned NumPy.

```bash
# conda (convenience environment; NOT a build-locked reproduction — see note)
conda env create -f environment.yml
conda activate psg_audio_benchmark
pip install -e .                     # REQUIRED: installs the package + CLI
pip install -r requirements-dev.txt  # exact verified runtime pins + pytest

# or pip only
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate on POSIX
pip install -e .                     # REQUIRED (editable install from repo root)
pip install -r requirements-dev.txt
```

The editable install (`pip install -e .`) is required: it provides the
`psg_audio_benchmark` package import and the `psgb-setup-check` console script
that the verification steps below use. Without it, `python -m
psg_audio_benchmark` and the console scripts are unavailable.

Version pinning layers (they are different things):

* `requirements.txt` / `requirements-dev.txt` — **exact pip pins** of the
  verified environment (authoritative for numeric reproduction).
* `environment.yml` — a **convenience** conda env with *range* constraints,
  not a build lock: conda builds of the "same" version can differ from pip
  wheels (notably matplotlib's bundled FreeType, which affects figure text
  metrics; see §7 and the reproduction guide).
* Figure fidelity additionally depends on **matplotlib's bundled FreeType**
  and on Times New Roman being available to matplotlib; with a different
  FreeType build the regenerated figures are visually near-identical, not
  byte-identical (details in `docs/REPRODUCTION_GUIDE.md`).

Optional: `py7zr` (only if your staged download is 7z) and `soundfile` (audio
metadata probing; audio analysis is BLOCKED in this release and never runs).

No GPU is needed anywhere in the released pipeline (CPU-only scikit-learn).

## 3. Data (official acquisition only — nothing is redistributed)

- **Data DOI:** `10.57760/sciencedb.19070`, version **V5** (Science Data Bank)
- Dataset paper DOI: `10.1038/s41597-025-05583-8`
- License recorded on the data page: **CC BY 4.0** (the paper text is CC BY-NC-ND
  4.0 — a different license that does not govern the data)
- Download size of the raw tree: ~53 GB (50 patients)

See [`data/README.md`](data/README.md) for step-by-step acquisition and the
project's stricter no-redistribution policy.

## 4. Path setup and stage order

The project root is resolved from the location of
`src/psg_audio_benchmark/` (auto), or from the `PSG_BENCHMARK_ROOT`
environment variable. All configured paths are relative to that root; no
absolute machine path is ever hardcoded.

```bash
# from the repository root:
set PSG_BENCHMARK_ROOT=C:\path\to\this\repo        # Windows (optional; auto also works)
export PSG_BENCHMARK_ROOT=/path/to/this/repo       # POSIX

python scripts/00_setup_check.py                   # environment/config sanity check
python scripts/01_data_download.py                 # staging plan (manual download; see data/README.md)
```

Expected `00_setup_check.py` outcome for a correctly configured fresh clone:
**overall `WARNING`**, never `FAIL`, with only these expected warnings —
missing optional audio/other-model packages (`soundfile`, `librosa`,
`lightgbm`, `xgboost` — none is imported by any code path shipped here; audio
analysis is BLOCKED by design), `ffmpeg` not on PATH (audio decoding only),
GPU present but CUDA-unusable torch (CPU mainline unaffected), and git not yet
initialized (it is initialized at publication time). A `FAIL` means a required
directory (`data/interim`, `features`, `reports`, `logs`, `config`) or a
config file is missing or unreadable — the empty `.gitkeep` placeholders
shipped in those directories exist precisely so this check passes out of the
box.

```bash
python scripts/02_data_audit.py                    # raw-file audit + license gate
python scripts/03_annotation_sync.py               # parse PSG/AASM annotations (frozen products)
python scripts/04_window_index.py                  # non-overlapping 30-s window grid + event links
python scripts/05_physiology_features.py           # HR/SpO2/airflow window features
python scripts/06_patient_splits.py                # patient-level cohort + 5x4 nested split
python scripts/06b_split_balance_optimize.py       # v2 split balance optimization (the FROZEN split)
python scripts/15_bspc_reanalysis.py               # THE analysis of the paper (Stage 15b)
```

Each stage writes only into isolated `runs/<run_id>/` directories; earlier
products are never modified (enforced by tests, e.g. `test_no_overwrite.py`).

**Binding YOUR regenerated run IDs (Mode B — bind BEFORE each next stage).**
The `config/*.yaml` files and the `--input-*-run-id` script defaults record
the **authors' historical run IDs** as frozen provenance pins — a fresh clone
contains none of those runs, so running a stage with bare defaults first and
remediating afterwards is NOT the supported path (the input gates BLOCK with
exit 1 on the absent historical ids). Instead, after each upstream stage
completes, validate it and let the binder print the exact next-stage command
with YOUR run ids bound (`--next-stage` requires only the upstream that stage
genuinely needs — the full chain is never required just to advance one stage):

```bash
python scripts/02_data_audit.py                    # license gate record
python scripts/03_annotation_sync.py               # no upstream run-id input
python tools/bind_run_ids.py --next-stage 4        # → prints 04 with YOUR stage-3 id
python tools/bind_run_ids.py --next-stage 5        # → prints 05 with YOUR 3/4 ids
python tools/bind_run_ids.py --next-stage 6        # → prints 06 with YOUR 4/5 ids
python tools/bind_run_ids.py --next-stage 6b       # → prints 06b --v1-comparator-run-id <YOUR stage-6 id>
python tools/bind_run_ids.py --next-stage 15       # + mirror/signature/approval gates
python tools/bind_run_ids.py --next-stage all      # optional final whole-chain check
```

Each invocation enforces (and refuses, exit 2, without): the CURRENT license
evidence re-check (same gate as `scripts/02_data_audit.py`) plus the passed
record, run-dir + product existence, run-id stage prefixes and single-segment
safety (no `..`/absolute paths/shell metacharacters), resolved-config lineage
consistency (empty lineage fields are refused, not skipped), the Stage-6b v1
comparator's signature lineage, and — for Stage 15 — the `approved_for_modeling`
status, the signed SHA-256 of every v2 split product in BOTH the 6b run dir and
the fixed mirror, run-dir/mirror signature identity, the cohort patient-set
fingerprint, and the fixed mirrors Stage 15b actually reads. Stage 15 note
(verified, see `tests/test_bind_run_ids_staged.py`): `scripts/15_bspc_reanalysis.py`
reads the fixed mirrors directly; the frozen `provenance_gate.split_run` pin in
`config/bspc_revision_analysis.yaml` is declarative provenance, not a runtime
gate — the executable gate is the mirror/signature verification above.

The frozen provenance pins in `config/*.yaml` are never edited by this tool
(they stay as the read-only record of the authors' chain); the executable
difference — your run IDs passed via CLI flags — is exactly what the tool
prints and records under `reports/run_bindings/` when asked
(`--write-bindings`). Do not bypass the approval/signature gates by hand-editing
statuses or signature files.

## 5. Random seeds

All randomness is pinned to **`20250714`** (global config `random_seed`, split
seeds `outer_fixed_seed` / `inner_fixed_seed`, and the Stage-15b registration
seed). Re-running the analysis on identical frozen inputs reproduces identical
numbers (verified in `tests/test_stage6_reproducibility.py` and the synthetic
end-to-end determinism assertion in `tests/test_release_synthetic_e2e.py`).

## 6. Primary analysis definition (do not substitute)

- **Windows:** non-overlapping 30-s grid over each record.
- **Main label:** `overlap_ge_10s` — positive if the window overlaps a retained
  (non-awake, scored respiratory) event by ≥ 10 s; **indeterminate** (excluded)
  if overlap is in (0, 10) s; negative if 0.
- **Main feature timing:** `lag_0`, a **current-window / no-future-window
  setting** — each window's features are computed from that same 30-s window,
  which is complete only at the window's end. This is a retrospective labelling
  convention, **not** a claim of causal/online/real-time operation: no
  real-time or streaming deployment was evaluated anywhere in this study.
  `lag_plus_30`/`lag_plus_60` add future signal and are offline context
  sensitivity analyses only.
- **Main model:** `hr_spo2_lr` — logistic regression on 8 HR + 8 SpO2 window
  statistics, inside patient-level 5-outer × 4-inner nested CV with fold-internal
  imputation/scaling/selection/thresholding.
- Comparators: `hr_only_lr`, `spo2_only_lr`, `simple_spo2_min_lr`,
  `dummy_prior`; exploratory: `hr_spo2_hgb`.
- **Airflow secondary** (34-patient subcohort, shared-source caveat):
  `airflow_enhanced` vs `core_restricted`.
- **Audio: BLOCKED** (0 eligible windows; no audio feature is read anywhere).

## 7. Expected results (frozen Stage-15b run, seed 20250714)

Primary battery, main label, main feature timing `lag_0`, 30,506 analyzable
windows, 50 patients (full tables in `results_published/stage15b_aggregates/`):

| Model | AUROC | AUPRC | Brier |
|---|--:|--:|--:|
| dummy_prior (lower bound) | 0.481 | 0.367 | 0.235 |
| simple SpO2-min LR | 0.763 | 0.639 | 0.195 |
| HR-only LR | 0.668 | 0.550 | 0.223 |
| SpO2-only LR | 0.836 | 0.709 | 0.180 |
| **HR+SpO2 LR (primary)** | **0.823** | **0.703** | **0.177** |
| HR+SpO2 HGB (exploratory) | 0.855 | 0.751 | 0.153 |

- Primary AUROC 95% patient cluster bootstrap CI: **0.823 [0.787, 0.854]**;
  AUPRC 0.703 [0.616, 0.782].
- Patient-macro AUROC 0.763 (49 patients included, 1 single-class excluded).
- Event-level (primary): sensitivity 0.660, precision 0.749, F1 0.701,
  FP/h 4.97 over 254.2 analyzable hours — detection at the window-aggregated
  candidate level, **not** patient diagnosis.
- Calibration (primary): intercept −0.528, slope 0.788, ECE 0.117 —
  miscalibration is acknowledged in the manuscript.
- **Label sensitivity (NOT primary):** `event_start_point_half_open` (onset)
  primary-LR AUROC **0.754**; `any_overlap` 0.814; `coverage_ge_50pct` 0.825.
- **Airflow secondary (main pre-registered result):** AUROC
  `airflow_enhanced` 0.8123 vs `core_restricted` 0.8145 → paired Δ
  **−0.002** (95% CI includes 0; no clear incremental improvement — the
  shared-source caveat in §9 applies).

Exact reproduction expectations:

- **Mode A (from this package, no dataset needed):** the synthetic test suite
  runs the full analysis code path on synthetic data, and
  `tools/make_figures.py` regenerates the manuscript figures from
  `results_published/` (see `docs/REPRODUCTION_GUIDE.md` for what "identical"
  means under which matplotlib/FreeType build). The pooled-metric recompute
  (`tools/recompute_primary_metrics.py`) is an OOF-dependent check: the public
  package deliberately excludes per-window OOF data, so that tool runs only
  against a Stage-15b run directory **you** generate in Mode B.
- **Mode B (official data regeneration):** after §3–§4, Stage 15b reproduces
  the table above. Tolerances: logistic-regression/baseline cells are expected
  to match to float round-off given identical library versions; bootstrap CIs
  use 1,000 seeded patient-cluster resamples. Cross-version differences in
  scikit-learn may shift later digits — compare at ≥ 4 decimals.

## 8. Compute

CPU-only. Reference wall-clock on the development machine (Windows 10, desktop
CPU, anaconda CPython 3.13): Stage 3 ≈ 10 min; Stage 4 ≈ 1–2 min; Stage 5
≈ 20 min; Stage 6/6b ≈ minutes; **Stage 15b ≈ 5 min**; figure generation ≈ 1 min.
Dominant cost is the ~53 GB download (hours, network-bound) and ~55 GB of disk.
RAM ≤ 4 GB.

## 9. Known limitations (as stated in the manuscript)

Single public dataset (50 patients); patient-level **internal** CV only — no
external validation; HR/SpO2/airflow channels derive from the same PSG
acquisition that provided the reference annotations (shared-source /
incorporation risk, especially for airflow); event-level matching is lenient
interval overlap, not onset/offset alignment; probability calibration is
imperfect (slope 0.79); no audio result exists (audio blocked); HGB results
are exploratory; no clinical deployability claim of any kind.

## 10. License and citation

- **Code license: MIT (author-approved 2026-10-02)** — see [`LICENSE`](LICENSE).
  Aggregate outputs have a separate CC BY 4.0 notice in [`LICENSE_DATA.md`](LICENSE_DATA.md).
- Third-party components: [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).
- Software citation: [`CITATION.cff`](CITATION.cff). Cite the version DOI for
  the archived snapshot and the concept DOI for the latest version. The
  associated manuscript remains unpublished; do not cite it as a published
  article or imply journal acceptance.
- Derived aggregate outputs in `results_published/` are dataset-derived
  (CC BY 4.0 data source): cite the data DOI `10.57760/sciencedb.19070` (V5)
  in any reuse.
