# Manual publication steps — GitHub + Zenodo (to be executed by the AUTHORS)

**LOCAL_CANDIDATE_NOT_PUBLISHED — nothing below has been executed. No URLs,
DOIs or account states are claimed to exist yet.**

## Preconditions (all currently OPEN)

1. Author approval of this release candidate's content (file list = the current
   stage-25 `release_manifest.json` shipped beside `release/`).
2. License decision recorded (`../LICENSE_DECISION_REQUIRED.md`) and a `LICENSE`
   file added to `release/`.
3. Author confirmation of the two manuscript code/data-availability statements.
4. Manually verify (do not assume) current guide requirements of:
   - GitHub: https://docs.github.com and the repository-creation flow
   - Zenodo: https://zenodo.org — GitHub-Zenodo integration (enable the
     repository in Zenodo's GitHub settings, then create a release to trigger a
     DOI). Zenodo's exact menu names may change; follow the live instructions.

## Suggested versioning

- Tag **v1.0.0** for the version corresponding to the submitted manuscript.
- The candidate's internal package version is `1.0.0rc1`; bump to `1.0.0` in
  `pyproject.toml` + `CITATION.cff` at tagging time.
- Future post-review revisions → v1.x with a CHANGELOG entry.

## GitHub steps (manual)

1. Create a new repository (suggested name `psg-audio-benchmark`; choose
   visibility = public only after the checklist above is complete).
2. Initialize from the `release/` directory content EXACTLY as hashed in the
   current release manifest — do not hand-edit files during upload; if an edit is
   needed, redo the audit first.
3. Commit history may be squashed to a single initial commit or kept; the
   manifest hashes matter, not history.
4. Push tag `v1.0.0`.
5. In the repository "About" panel: description, the Data DOI
   `10.57760/sciencedb.19070`, and topics (sleep-apnea, psg, benchmark).
6. Enable Zenodo integration for the repository (Zenodo → GitHub settings).

## Zenodo steps (manual)

1. Create a GitHub release `v1.0.0` on GitHub; Zenodo auto-archives the tag and
   mints a versioned DOI.
2. On Zenodo, complete the record metadata from `CITATION.cff` (fill every
   `<PLACEHOLDER>` — real DOI, authors, license, dates).
3. Reserve the DOI, then backfill (see "Manuscript backfill" below), then
   publish the Zenodo record.
4. Verify the archived file list against the current release manifest (names + sizes;
   Zenodo re-computes checksums).

## Manuscript positions needing backfill AFTER publication (do not pre-fill)

1. **Code availability / Availability of data and materials** statement:
   replace the planned-release wording with the actual GitHub URL and the
   Zenodo DOI (concept DOI + version DOI for v1.0.0).
2. **Declarations → Availability** section: same two identifiers.
3. **Cover letter / submission metadata**: add the repository URL if the journal
   asks at submission time.
4. **CITATION.cff** in future tags: replace `<ZENODO_DOI_PLACEHOLDER>` etc.
5. If the journal requires software citation in the reference list, add the
   software DOI as a new numbered reference (renumbering must then be re-audited
   exactly as previous citation edits were).

## After publication (post-checks)

- Re-run `tools/recompute_primary_metrics.py` and the synthetic suite on a clean
  clone from the public repository and record the result.
- Diff the public clone's `SHA256SUMS.txt` against the current release manifest.
- Remove the LOCAL_CANDIDATE_NOT_PUBLISHED banners in the public repository
  (README top, this docs file, `data/README.md`) as part of the v1.0.0 tag commit
  AFTER the audit of that exact edit.
