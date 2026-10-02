# Primary dataset license confirmation — TEMPLATE (NOT FILLED IN)

> **STATUS: BLOCKED (template).** This file is an unfilled template shipped
> with the code release so that the Stage-2 license gate has a complete,
> field-by-field form to fill in. Every value below is a placeholder. The gate
> (`psg_audio_benchmark.data_audit.license_gate`) parses exactly these
> `- label: value` lines and **blocks** (`license_gate_blocked.md`, pipeline
> stops before reading `data/raw`) while any required field is missing or
> still a placeholder. Nothing here is pre-confirmed, and no `passed` state is
> hardcoded anywhere.
>
> A human reproducer MUST replace every `<…>` value below with what they
> themselves verified on the official pages (see `README.md` in this
> directory), then re-run `python scripts/02_data_audit.py`. Do NOT copy the
> authors' values without repeating the verification yourself; do NOT commit
> screenshots containing personal account information.

- data_doi: <fill in — must equal the primary dataset DOI 10.57760/sciencedb.19070, or the gate blocks with data_doi_mismatch>
- dataset_version: <fill in — e.g. V5 as displayed on the official dataset page>
- access_date: <fill in — YYYY-MM-DD you personally accessed the official page>
- license_terms: <fill in — the license name exactly as stated on the OFFICIAL dataset page, plus where you read it>
- noncommercial_research_basis: <fill in — one affirmative sentence, e.g. "permitted for non-commercial research per <official page/terms section>">
- deriv_features_policy: <fill in — what the official terms say about derived features/aggregates>
- redistribute_audio_policy: <fill in — what the official terms say about redistributing raw audio/identifiable recordings>
- evidence_screenshot_relpath: <fill in — repo-relative path of your saved copy of the official page/notice, e.g. docs/license_evidence/saved_pages/index.html>
- confirmer_name: <fill in — the human who performed the verification>
- confirmation_date: <fill in — YYYY-MM-DD>
- open_items: <fill in — unresolved uncertainties, or "none">

<!-- Required fields (gate blocks if missing/placeholder): data_doi,
     access_date, license_terms, noncommercial_research_basis,
     deriv_features_policy, redistribute_audio_policy, confirmer_name,
     confirmation_date. The DOI must match 10.57760/sciencedb.19070 and the
     non-commercial basis must be an explicit affirmative. -->
