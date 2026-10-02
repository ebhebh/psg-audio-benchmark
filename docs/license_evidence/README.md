# License evidence — how to fill in the confirmation template

`primary_dataset_license_confirmation.md` in this directory is an **unfilled
template**. The Stage-2 audit (`python scripts/02_data_audit.py`) refuses to
scan `data/raw` until a human has completed it from the OFFICIAL sources. The
gate has exactly two outcomes and both are by design:

* **missing / template / placeholder values or DOI mismatch → `blocked`**:
  the runner writes `reports/data_audit/license_gate_blocked.md` and stops;
* **all required fields genuinely filled + DOI match + affirmative
  non-commercial basis → `passed`**: the run records the evidence path, DOI,
  version and access date in `license_gate_passed.md` and proceeds.

The template is never auto-confirmed: no code path writes `passed` for you,
and the shipped template itself parses to `blocked` (asserted by
`tests/test_license_evidence_template.py`).

## Official-source verification steps

Perform these yourself, on the pages, before filling the template:

1. **Open the official dataset page** on Science Data Bank
   (https://www.scidb.cn) and locate the dataset with
   **Data DOI `10.57760/sciencedb.19070`**. Confirm the version shown there
   (the release was verified against **V5**) and note the access date.
2. **Read the license displayed on the dataset page itself.** Record the
   license name exactly as displayed (during release verification the data
   page recorded **CC BY 4.0**; the dataset *paper* text carries a different
   license, CC BY-NC-ND 4.0, which does not govern the data files). Save a
   copy of the page/notice under `docs/license_evidence/saved_pages/` and
   reference that copy in `evidence_screenshot_relpath`. Do not upload
   screenshots that expose personal account information.
3. **Check the dataset paper** (DOI `10.1038/s41597-025-05583-8`, Tao et al.
   2025) for any usage statements that supplement the page license; cite what
   you actually read in `license_terms` / the policy fields.
4. **Answer the two policy fields from the terms you read**, not from this
   repository's documentation: what is permitted for (a) derived
   features/aggregates and (b) redistribution of raw audio/identifiable
   recordings.
5. **Fill every `<…>` value** in the template, save, and re-run
   `python scripts/02_data_audit.py`. A `blocked` outcome lists the missing
   items; fix and re-run. Do not edit the gate code to pass.

## Scope note

This gate governs YOUR access to the raw dataset for Mode-B reproduction. It
is separate from (and does not decide) the code license of this repository
(`LICENSE_DECISION_REQUIRED.md`) or the dataset-derived aggregate outputs in
`results_published/` (attribute the Data DOI in any reuse).
