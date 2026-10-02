# Reproduction guide — two clearly separated modes

Install first (README §2): `pip install -e .` **and**
`pip install -r requirements-dev.txt` in your environment of choice. The
editable install provides the `psg_audio_benchmark` package and the
`psgb-setup-check` console script used below.

Environment honesty note: `requirements*.txt` are exact pip pins of the
verified environment (Windows 10, CPython 3.13.9). `environment.yml` is a
convenience conda env with **range** constraints — it is not a conda
build lock; conda and pip builds of the same version can differ in bundled
libraries (notably matplotlib's FreeType), which matters only for
byte-level figure comparison (A3).

## Mode A — direct reproduction from THIS package (no dataset needed)

What it proves: the released aggregates are internally consistent, the released
code regenerates the published figures/tables, and the full analysis code path
executes end to end (on synthetic data).

What it does NOT prove: any real-data result. Synthetic tests are smoke/
integration evidence only.

```bash
# A1. install + run the synthetic test suite (incl. synthetic end-to-end)
pip install -e .
pip install -r requirements-dev.txt
python -m pytest tests -q

# A2. regenerate the manuscript figures from the published aggregates
python tools/make_figures.py \
    --aggregates-dir results_published/stage15b_aggregates \
    --figure-inputs-dir results_published/figure_inputs \
    --out-dir figures_out
#     compare figures_out/ against figures_published/. Verified behaviour with
#     the conda matplotlib 3.10.6 build that produced the frozen figures:
#     PNG byte-identical; SVG identical after removing the embedded <dc:date>
#     and matplotlib's per-run random element IDs (clipPath/marker ids); PDF
#     pixel-identical at 200 dpi (only the embedded Creation/Mod dates differ).
#     A pip-wheel matplotlib of the same version may carry a different bundled
#     FreeType and Times New Roman availability differs by OS: text metrics
#     can shift by ~0.01 pt, so outputs are then VISUALLY NEAR-IDENTICAL but
#     NOT byte-identical and NOT guaranteed pixel-identical — compare
#     structurally (identical file set, sizes within a few %, side-by-side
#     inspection) rather than asserting byte/pixel equality. This is an
#     environment artifact, not a numerical difference: the underlying data
#     are the released CSVs either way.

# A3. manuscript tables: tables_published/*.md are the final sources; every
#     number traces to results_published/stage15b_aggregates/*.csv.
```

**About `tools/recompute_primary_metrics.py` (NOT runnable in Mode A).** That
tool recomputes the pooled primary metrics from a Stage-15b run directory's
`oof_predictions.parquet`. Per-window OOF data are deliberately **excluded**
from this package (privacy/governance, see `docs/PRIVACY_AND_DATA_GOVERNANCE.md`),
and the public aggregates contain only CSV summaries — so on a fresh clone the
recompute tool has nothing to read and is **not part of Mode A**. Run it in
Mode B (step B4) against a Stage-15b run directory **you** generated. The
authors' frozen OOF store is not redistributed, and nothing in this package
may be used to reconstruct it.

## Mode B — full regeneration from the OFFICIAL dataset

What it proves: the complete pipeline (download → audit → annotations →
windows → features → split → analysis) regenerates the published numbers from
the official public data.

Time estimate (desktop CPU, see README §8): ~53 GB download (hours,
network-bound) + roughly 40–60 minutes of compute for stages 3–6b + ~5 minutes
for stage 15b + ~1 minute for figures.

```bash
# B0. license evidence gate (blocks until a human fills the template)
#     docs/license_evidence/primary_dataset_license_confirmation.md ships as
#     an UNFILLED template; complete it from the official pages following
#     docs/license_evidence/README.md. 02_data_audit writes
#     license_gate_blocked.md and stops while it is unfilled — that is the
#     intended behaviour, not an error.

# B1. obtain the dataset  (data/README.md; Data DOI 10.57760/sciencedb.19070, V5)
#     stage it under data/external/incoming/ preserving V5/Data/<patient>/<file>

# B2. run the pipeline in order, BINDING BEFORE EACH NEXT STAGE (README §4)
#     The script defaults and config/*.yaml pins point at the AUTHORS'
#     historical run ids, which do not exist in your fresh checkout. Do NOT
#     run stages 4/5/6/6b with bare defaults first and remediate afterwards —
#     their input gates BLOCK (exit 1) on the absent historical ids. After
#     each stage completes it writes a LATEST_RUN.txt pointer; validate it and
#     run the PRINTED command for the next stage (--next-stage requires only
#     the upstream that stage genuinely needs, never the whole chain):
python scripts/00_setup_check.py
python scripts/01_data_download.py
python scripts/02_data_audit.py            # writes the license gate record
python scripts/03_annotation_sync.py
python tools/bind_run_ids.py --next-stage 4
#   → python scripts/04_window_index.py --input-run-id "<your-stage3-run-id>"
python tools/bind_run_ids.py --next-stage 5
#   → python scripts/05_physiology_features.py \
#        --input-stage4-run-id "<your-stage4-run-id>" \
#        --input-stage3-run-id "<your-stage3-run-id>"
python tools/bind_run_ids.py --next-stage 6
#   → python scripts/06_patient_splits.py \
#        --input-stage4-run-id "<your-stage4-run-id>" \
#        --input-stage5-run-id "<your-stage5-run-id>"
python tools/bind_run_ids.py --next-stage 6b
#   → python scripts/06b_split_balance_optimize.py \
#        --v1-comparator-run-id "<your-stage6-run-id>"
#     (06b's --v1-comparator-run-id binds YOUR regenerated Stage-6 v1 run as
#      the comparator; the frozen config pin is never edited and the override
#      is recorded in the run's resolved config + signature lineage)

# B2b. gates the binder enforces (exit 2 = refused, nothing was run)
#     license: CURRENT evidence re-check (same gate as 02_data_audit.py) AND
#     the passed record — a stale record alone is refused;
#     every bound id: single safe path segment (no '..'/absolute paths/
#     shell metacharacters; printed commands are safely quoted);
#     stage4→3 / stage5→4,3 resolved-config lineage (empty = refused);
#     stage6 v1 signature lineage 4/5/3;
#     stage15: approved_for_modeling status, signed SHA-256 of every v2
#     product in BOTH the 6b run dir and the fixed mirror, run-dir/mirror
#     signature identity, cohort patient-set fingerprint (queue drift), and
#     the fixed mirrors Stage 15b reads. It never edits the frozen configs or
#     any approval record. With --write-bindings it also drops an audit
#     record under reports/run_bindings/ (git-ignored).

# B3. the analysis of the paper
python tools/bind_run_ids.py --next-stage 15   # mirror/signature/approval gates
python scripts/15_bspc_reanalysis.py
#     → results/runs/stage15b-bspc-reanalysis-<UTC>/summary_metrics.csv etc.
#     (reads the frozen mirror products under splits/, annotations/,
#      features/physiology/ that YOUR 06b run published + approved;
#      the frozen config's provenance_gate.split_run is DECLARATIVE
#      provenance, not a runtime gate — the executable gate is the
#      --next-stage 15 mirror/signature verification)

# B4. verify YOUR run against the published aggregates
python tools/recompute_primary_metrics.py --run-dir results/runs/<your-stage15b-run-id>
python tools/export_figure_data.py \
    --project-root . \
    --run-dir results/runs/<your-stage15b-run-id> \
    --out-dir my_figure_inputs
python tools/make_figures.py \
    --aggregates-dir results/runs/<your-stage15b-run-id> \
    --figure-inputs-dir my_figure_inputs \
    --out-dir my_figures
#     then diff my_figure_inputs/fold_auroc_verification.csv etc. against
#     results_published/figure_inputs/ and compare summary_metrics.csv against
#     results_published/stage15b_aggregates/summary_metrics.csv (>= 4 decimals;
#     see README §7 for version-sensitivity caveats).
```

### Expected divergence sources in Mode B

- Library versions (scikit-learn LBFGS/HGB solver details) — later digits only.
- The frozen v2 split is regenerated deterministically by 06b from identical
  inputs; if your Stage-2/3/4 audit differs (dataset version drift on the
  official side), cohort membership may differ — record and report it rather
  than forcing a match. A regenerated split that does not reach
  `approved_for_modeling` must NOT be forced into Stage 15b.
- Bootstrap CIs: 1,000 resamples under seed 20250714 — identical given
  identical inputs and numpy version.
- Figure byte-fidelity depends on the matplotlib build's bundled FreeType and
  font availability (see the A2 note): numeric outputs are unaffected.
