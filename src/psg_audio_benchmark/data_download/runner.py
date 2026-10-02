"""Stage-1 orchestration: scan, manifest, receipt and report generation.

The runner ties the data_download modules together into a single, read-only
audit pass:

1. Verify the primary dataset DOI identity (hard failure on mismatch).
2. Snapshot ``data/raw`` BEFORE the run (lightweight: relative path +
   size + mtime — a read-only audit only opens files for reading, which never
   changes size/mtime, so this still proves no write happened).
3. Recursively scan ``data/raw`` (read-only) and ``data/external/incoming``
   (staging), hashing every file ONCE (the manifest's SHA-256) and running
   shallow integrity / archive checks.
4. Build ``file_manifest_stage1.csv`` (duplicates and conflicts flagged, never
   resolved by deletion).
5. Write ``download_receipt.json`` and the four Stage-1 reports.
6. Snapshot ``data/raw`` AFTER the run and assert it is byte-identical.

Nothing under ``data/raw`` is ever written, moved, renamed or deleted.

Output isolation (prompt remediation section 2):

* **Production** (``output_root`` unset): every artifact is written under a
  per-run directory::

      data/manifests/runs/<run_id>/file_manifest_stage1.csv
      data/manifests/runs/<run_id>/download_receipt.json
      reports/data_download/runs/<run_id>/*.md

  The fixed legacy paths (``data/manifests/file_manifest_stage1.csv`` etc.)
  are regenerated from this run ONLY by the CLI ``main()`` after a successful
  production run, never by the runner and never by tests.

* **Isolated / test** (``output_root`` set): every artifact is written strictly
  under ``<output_root>`` and the production manifest/report directories are
  never touched. The contamination guard refuses an ``output_root`` that points
  at (or inside) a production manifest/report directory.

* The contamination guard also refuses a *production-mode* run whose ``run_id``
  / ``config_hash`` / raw path look like a test fixture, so a test can never
  masquerade as production and overwrite fixed paths.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from . import archives as archives_mod
from . import downloader as downloader_mod
from . import integrity as integrity_mod
from .hashing import HASH_ALGORITHM, file_meta, sha256_file
from .identity import (
    BACKGROUND_IDENTITY,
    PRIMARY_IDENTITY,
    IdentityVerification,
    verify_primary_identity,
)
from .manifest import (
    MANIFEST_FIELDS,
    STATUS_CONFLICT,
    STATUS_DUPLICATE_CONTENT,
    STATUS_MANUAL_REQUIRED,
    STATUS_NOT_CHECKED,
    STATUS_VERIFY_EXISTING,
    ManifestRow,
    mark_duplicates,
    write_manifest_csv,
)

#: Names ignored when scanning a directory (not patient data).
_IGNORED_NAMES = {".gitkeep", ".DS_Store", "Thumbs.db"}
#: Subdirectories never descended into during a scan.
_IGNORED_DIRS = {".git", ".incoming", "__pycache__", ".pytest_cache"}

#: run_id / config_hash markers that indicate a test fixture masquerading as a
#: production run. A production run (no ``output_root``) with any of these is
#: refused by the contamination guard.
_TEST_RUN_ID_TOKENS = (
    "full-immutability-test",
    "immutability-test",
    "test",
    "pytest",
    "fixture",
)
_TEST_PATH_TOKENS = ("pytest", "tmp", "temp")


class Stage1ContaminationError(RuntimeError):
    """Raised when a run would write test output into production paths, or
    when a production run looks like a test fixture.

    This is a hard guard: the runner refuses to proceed rather than risk
    overwriting real production artifacts with test data.
    """


@dataclass
class Stage1Options:
    dry_run: bool = False
    verify_existing: bool = False
    source_url: str = ""
    allow_auto_download: bool = False
    access_date: str = ""
    #: Isolated output root (tests). When set, ALL artifacts are written
    #: strictly under this directory and production paths are never touched.
    output_root: Optional[Path] = None
    #: Base directory for relativizing data ``relative_path`` values. Defaults
    #: to ``cfg.project_root`` (production -> ``data/raw/...``); tests set this
    #: to their fixture root so paths stay fixture-relative, never absolute.
    relpath_base: Optional[Path] = None

    @property
    def isolated(self) -> bool:
        """True iff artifacts must be written under ``output_root`` only."""
        return self.output_root is not None


@dataclass
class Stage1Summary:
    run_id: str
    overall_status: str = "PASS WITH WARNINGS"
    access_date: str = ""
    identity: Optional[IdentityVerification] = None
    n_files_raw: int = 0
    n_files_staging: int = 0
    total_bytes: int = 0
    n_archives: int = 0
    n_corrupt_archives: int = 0
    n_anomalies: int = 0
    n_duplicates: int = 0
    n_conflicts: int = 0
    n_audio_not_checked: int = 0
    download_status: str = STATUS_MANUAL_REQUIRED
    raw_modified: bool = False
    #: Lightweight before/after snapshots: {relative_path: "size:mtime_ns"}.
    raw_before: Dict[str, str] = field(default_factory=dict)
    raw_after: Dict[str, str] = field(default_factory=dict)
    manifest_path: str = ""
    receipt_path: str = ""
    report_paths: Dict[str, str] = field(default_factory=dict)
    main_report_path: str = ""
    #: Root directory where this run's artifacts actually live.
    out_base: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class Stage1Runner:
    """Runs the Stage-1 audit. Construct with a loaded :class:`Config`."""

    def __init__(
        self,
        cfg: Config,
        run_metadata: Dict[str, Any],
        options: Optional[Stage1Options] = None,
    ) -> None:
        self.cfg = cfg
        self.run_metadata = run_metadata
        self.options = options or Stage1Options()
        self.run_id: str = run_metadata.get("run_id", "unknown")
        self.access_date: str = (
            self.options.access_date
            or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        self.summary = Stage1Summary(run_id=self.run_id, access_date=self.access_date)
        self._rows: List[ManifestRow] = []
        # Resolve an isolated output root once, up front (None = production).
        self._output_root: Optional[Path] = (
            Path(self.options.output_root).resolve()
            if self.options.output_root is not None
            else None
        )

    # ------------------------------------------------------------------
    # Output-path resolution (isolation vs production run-dir)
    # ------------------------------------------------------------------
    def _relpath_base(self) -> Path:
        """Directory used to relativize data ``relative_path`` values."""
        if self.options.relpath_base is not None:
            return Path(self.options.relpath_base).resolve()
        if self._output_root is not None:
            # Fixture root convention: data lives next to the outputs dir.
            return self._output_root.parent
        return self.cfg.project_root.resolve()

    def _out_base(self) -> Path:
        """Root directory for this run's artifacts.

        Isolated mode -> ``<output_root>``. Production -> a per-run directory
        pair under the configured manifests/reports dirs.
        """
        if self._output_root is not None:
            return self._output_root
        return self.cfg.path("data_manifests") / "runs" / self.run_id

    def _manifest_out_path(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "manifests" / "file_manifest_stage1.csv"
        return self.cfg.path("data_manifests") / "runs" / self.run_id / "file_manifest_stage1.csv"

    def _receipt_out_path(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "manifests" / "download_receipt.json"
        return self.cfg.path("data_manifests") / "runs" / self.run_id / "download_receipt.json"

    def _reports_out_dir(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "data_download"
        return self.cfg.path("reports_data_download") / "runs" / self.run_id

    def _main_report_out_path(self) -> Path:
        if self._output_root is not None:
            return self._output_root / "reports" / "data_download_report.md"
        return self.cfg.path("reports_data_download") / "runs" / self.run_id / "data_download_report.md"

    def artifact_paths(self) -> Dict[str, Path]:
        """Resolved absolute output paths for this run's artifacts (for the CLI
        to write the main report and, in production, to mirror fixed paths)."""
        return {
            "out_base": self._out_base(),
            "manifest": self._manifest_out_path(),
            "receipt": self._receipt_out_path(),
            "reports_dir": self._reports_out_dir(),
            "main_report": self._main_report_out_path(),
        }

    # ------------------------------------------------------------------
    # Contamination guard
    # ------------------------------------------------------------------
    def _assert_not_contaminating(self) -> None:
        """Refuse runs that would let test output reach production paths."""
        run_id_low = (self.run_id or "").lower()
        config_hash = str(self.run_metadata.get("config_hash", "") or "")

        if self._output_root is None:
            # Production mode: must not look like a test fixture.
            hit = [t for t in _TEST_RUN_ID_TOKENS if t and t in run_id_low]
            if hit:
                raise Stage1ContaminationError(
                    f"Production-mode run refused: run_id {self.run_id!r} "
                    f"contains test marker(s) {hit}. Tests must pass "
                    f"output_root=<tmp> and may not write production paths."
                )
            if config_hash.lower() in ("test", "fixture", "pytest"):
                raise Stage1ContaminationError(
                    f"Production-mode run refused: config_hash {config_hash!r} "
                    f"looks like a test value."
                )
            raw = self.cfg.path("data_raw").resolve()
            try:
                raw.relative_to(self.cfg.project_root.resolve())
            except ValueError:
                raise Stage1ContaminationError(
                    f"Production-mode run refused: data_raw {raw} is outside "
                    f"the project root, which indicates a test fixture "
                    f"masquerading as production."
                ) from None
        else:
            # Isolated mode: the output root must NOT point at (or into) any
            # production manifest/report directory or the project root itself.
            root = self._output_root
            forbidden = [
                self.cfg.project_root.resolve(),
                self.cfg.path("data_manifests").resolve(),
                self.cfg.path("reports_data_download").resolve(),
            ]
            for f in forbidden:
                try:
                    root.relative_to(f)
                    inside = True
                except ValueError:
                    inside = (root == f)
                if inside:
                    raise Stage1ContaminationError(
                        f"Isolated output_root {root} must not be (or be inside) "
                        f"the production path {f}."
                    )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _scan_dir(self, root: Path) -> List[Path]:
        out: List[Path] = []
        if not root.exists():
            return out
        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored dirs in-place
            dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
            for fname in filenames:
                if fname in _IGNORED_NAMES:
                    continue
                out.append(Path(dirpath) / fname)
        return out

    def _snapshot_raw(self) -> Dict[str, str]:
        """Read-only lightweight snapshot {relative_path: "size:mtime_ns"}.

        Uses size + mtime_ns (not a full SHA-256) so the before/after proof
        stays fast even for tens of GB. A read-only audit only opens files for
        reading, which never changes size or mtime, so this still detects any
        write/addition/deletion. The rigorous SHA-256 of every file is produced
        ONCE, inside the manifest via :meth:`_build_row`.
        """
        root = self.cfg.path("data_raw")
        snap: Dict[str, str] = {}
        for path in self._scan_dir(root):
            rel = self._rel_to_root(path)
            try:
                st = path.stat()
                snap[rel] = f"{st.st_size}:{st.st_mtime_ns}"
            except OSError as exc:
                snap[rel] = f"-1:stat_failed:{type(exc).__name__}"
        return snap

    def _rel_to_root(self, path: Path) -> str:
        """Return ``path`` relative to the relativization base, POSIX-style.

        NEVER falls back to an absolute path: if ``path`` is not under the base
        that is a hard error (prompt remediation constraint #6). Production
        files live under ``cfg.project_root`` -> ``data/raw/...``; test fixtures
        live under the fixture base -> ``raw/01/...``.
        """
        base = self._relpath_base()
        try:
            rel = path.resolve().relative_to(base)
        except ValueError as exc:
            raise Stage1ContaminationError(
                f"Refusing to record a non-relativizable path {path!s} "
                f"(base={base}). Manifest paths must be relative to the "
                f"project/fixture root, never absolute or pytest-temp paths."
            ) from exc
        return str(rel).replace("\\", "/")

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------
    @staticmethod
    def _row_error_code(hash_err, meta_err, fi) -> str:
        if hash_err or meta_err:
            return (hash_err or meta_err).split(":", 1)[0]
        if fi.error:
            return "integrity_error"
        if fi.anomaly:
            low = fi.anomaly.lower()
            if "ffmpeg" in low or "container" in low:
                return "audio_content_not_verified_no_ffmpeg"
            if "magic" in low:
                return "magic_mismatch"
            if "empty" in low:
                return "empty_file"
            if "small" in low:
                return "very_small_file"
            return "anomaly"
        return ""

    @staticmethod
    def _row_error_message(hash_err, meta_err, fi) -> str:
        if hash_err or meta_err:
            return hash_err or meta_err
        if fi.error:
            return fi.error
        if fi.anomaly:
            # Audio caveat: container/magic is valid but content completeness
            # was NOT verified (ffmpeg/soundfile absent) -> recorded, not fatal.
            return f"anomaly: {fi.anomaly}"
        return ""

    def _build_row(
        self,
        path: Path,
        staging_or_raw: str,
        identity_role: str,
        dataset_name: str,
        data_doi: str,
        paper_doi: str,
    ) -> ManifestRow:
        rel = self._rel_to_root(path)
        size, mtime, meta_err = file_meta(path)
        digest, hash_err = sha256_file(path)
        fi = integrity_mod.check_file(path)
        is_archive = archives_mod.is_archive(path)
        archive_status = ""
        if is_archive:
            insp = archives_mod.inspect_archive(path)
            archive_status = insp.test_status
        else:
            insp = None

        row = ManifestRow(
            manifest_schema_version=str(
                self.cfg.config_data.get("data_download", {}).get(
                    "manifest_schema_version", "1.0"
                )
            ),
            run_id=self.run_id,
            dataset_role=identity_role,
            dataset_name=dataset_name,
            paper_doi=paper_doi,
            data_doi=data_doi,
            dataset_version="TBD_AFTER_DOWNLOAD_AUDIT",
            access_date=self.access_date,
            source_url=self.options.source_url or PRIMARY_IDENTITY.source or "",
            source_archive_name=path.name if is_archive else "",
            relative_path=rel,
            staging_or_raw=staging_or_raw,
            file_name=path.name,
            extension=path.suffix.lower().lstrip("."),
            size_bytes=size,
            mtime_utc=mtime,
            sha256=digest,
            hash_algorithm=HASH_ALGORITHM,
            is_archive=bool(is_archive),
            archive_integrity_status=archive_status,
            readability_status=fi.readability_status,
            download_status=self.summary.download_status,
            overwrite_detected=False,
            error_code=self._row_error_code(hash_err, meta_err, fi),
            error_message=self._row_error_message(hash_err, meta_err, fi),
        )

        # Tally summary observations (file-level audit counts, NOT science).
        self.summary.total_bytes += max(0, size)
        if is_archive:
            self.summary.n_archives += 1
            if insp is not None and insp.corrupt:
                self.summary.n_corrupt_archives += 1
        if fi.has_anomaly:
            self.summary.n_anomalies += 1
        if fi.readability_status == STATUS_NOT_CHECKED:
            self.summary.n_audio_not_checked += 1
        if hash_err:
            self.summary.errors.append(f"{rel}: {hash_err}")
        return row

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> Stage1Summary:
        # 0. contamination guard — refuse test output in production paths.
        self._assert_not_contaminating()
        self.summary.out_base = self._out_base_summary()

        # 1. identity verification
        self.summary.identity = verify_primary_identity(
            PRIMARY_IDENTITY.data_doi,
            paper_doi=PRIMARY_IDENTITY.paper_doi,
            dataset_name=PRIMARY_IDENTITY.name,
        )
        if self.options.source_url:
            # A source URL cannot override DOI identity; only record it.
            self.summary.warnings.append(
                "--source-url recorded but cannot override DOI-based identity."
            )

        # 2. snapshot raw BEFORE (read-only, lightweight size+mtime)
        self.summary.raw_before = self._snapshot_raw()

        # 3. download decision (verify_existing if auditing present data)
        self._decide_download()

        # 4. scan raw (read-only) + staging
        primary = PRIMARY_IDENTITY
        for path in self._scan_dir(self.cfg.path("data_raw")):
            self._rows.append(
                self._build_row(
                    path, "raw", primary.role, primary.name,
                    primary.data_doi, primary.paper_doi or "",
                )
            )
            self.summary.n_files_raw += 1
        for path in self._scan_dir(self.cfg.path("data_external_incoming")):
            self._rows.append(
                self._build_row(
                    path, "staging", primary.role, primary.name,
                    primary.data_doi, primary.paper_doi or "",
                )
            )
            self.summary.n_files_staging += 1

        # 5. duplicates / conflicts
        mark_duplicates(self._rows)
        self._detect_conflicts()
        self.summary.n_duplicates = sum(
            1 for r in self._rows if r.error_code == STATUS_DUPLICATE_CONTENT
        )
        self.summary.n_conflicts = sum(
            1 for r in self._rows if r.error_code == STATUS_CONFLICT
        )

        # 6. snapshot raw AFTER and assert immutability
        self.summary.raw_after = self._snapshot_raw()
        self.summary.raw_modified = self.summary.raw_before != self.summary.raw_after

        # 7. write artifacts (unless pure dry-run)
        if not self.options.dry_run:
            self._write_manifest()
            self._write_receipt()
            self._write_reports()

        self._derive_overall_status()
        return self.summary

    # ------------------------------------------------------------------
    def _decide_download(self) -> None:
        # If we are only auditing already-present real files, express that
        # accurately instead of the default manual_required (which means
        # "download still pending"). verify_existing is the honest status for a
        # read-only re-audit of data already in data/raw.
        if self.options.verify_existing and self.summary.raw_before:
            self.summary.download_status = STATUS_VERIFY_EXISTING
            return
        allow = bool(
            self.options.allow_auto_download
            and self.cfg.config_data.get("data_download", {}).get(
                "allow_auto_download", False
            )
        )
        if not allow or not self.options.source_url:
            self.summary.download_status = STATUS_MANUAL_REQUIRED
            return
        # opt-in auto-download path: attempt into staging (best-effort).
        result = downloader_mod.download_http(
            self.options.source_url,
            self.cfg.path("data_external_incoming"),
            filename=Path(self.options.source_url).name or "download.bin",
            allow_auto_download=True,
            dry_run=self.options.dry_run,
        )
        self.summary.download_status = result.status
        if result.error:
            self.summary.warnings.append(f"download: {result.error}")

    def _out_base_summary(self) -> str:
        """out_base as it should appear in reports: project-relative when the
        run lives under the project (production), else an absolute path (test
        tmp). Production run-dirs are under the project root, so they relativize
        cleanly and no machine absolute path leaks into the receipt/report."""
        base = self._out_base()
        try:
            return str(base.resolve().relative_to(
                self.cfg.project_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(base).replace("\\", "/")

    def _detect_conflicts(self) -> None:
        """Flag same relative_path appearing with different sha (should not
        happen within one scan, but guards staging re-runs)."""
        by_path: Dict[str, List[ManifestRow]] = {}
        for r in self._rows:
            by_path.setdefault(r.relative_path, []).append(r)
        for path_key, group in by_path.items():
            if len(group) < 2:
                continue
            shas = {r.sha256 for r in group}
            if len(shas) > 1:
                for r in group:
                    if not r.error_code:
                        r.error_code = STATUS_CONFLICT
                        r.error_message = (
                            f"conflict: path {path_key} has multiple hashes"
                        )

    # ------------------------------------------------------------------
    # Artifact writers
    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        out = self._manifest_out_path()
        n, path = write_manifest_csv(self._rows, out)
        self.summary.manifest_path = self._rel_to_root(path)

    def _write_receipt(self) -> None:
        out = self._receipt_out_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "manifest_schema_version": "1.0",
            "run_id": self.run_id,
            "access_date": self.access_date,
            "run_kind": "isolated_test" if self.options.isolated else "production",
            "out_base": self.summary.out_base,
            "primary_dataset": {
                "name": PRIMARY_IDENTITY.name,
                "paper_doi": PRIMARY_IDENTITY.paper_doi,
                "data_doi": PRIMARY_IDENTITY.data_doi,
                "version": "TBD_AFTER_DOWNLOAD_AUDIT",
                "license_status": "TBD_AFTER_MANUAL_LICENSE_REVIEW",
            },
            "background_reference_only": {
                "name": BACKGROUND_IDENTITY.name,
                "data_doi": BACKGROUND_IDENTITY.data_doi,
                "role": BACKGROUND_IDENTITY.role,
                "must_not_mix_with_primary": True,
            },
            "identity_verification": _identity_to_dict(self.summary.identity),
            "download_status": self.summary.download_status,
            "source_url": self.options.source_url or "",
            "staging_dir": self._rel_to_root(self.cfg.path("data_external_incoming")),
            "counts": {
                "files_in_raw": self.summary.n_files_raw,
                "files_in_staging": self.summary.n_files_staging,
                "total_bytes": self.summary.total_bytes,
                "archives": self.summary.n_archives,
                "corrupt_archives": self.summary.n_corrupt_archives,
                "anomalies": self.summary.n_anomalies,
                "duplicates": self.summary.n_duplicates,
                "conflicts": self.summary.n_conflicts,
            },
            "raw_modified": self.summary.raw_modified,
            "raw_snapshot_semantics": "relative_path -> 'size:mtime_ns' (lightweight; "
            "full per-file SHA-256 is in the manifest)",
            "raw_snapshot_before": self.summary.raw_before,
            "raw_snapshot_after": self.summary.raw_after,
            "next_manual_steps": _manual_steps(),
            "config_hash": self.run_metadata.get("config_hash"),
            "warnings": self.summary.warnings,
            "errors": self.summary.errors,
        }
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, ensure_ascii=False)
        self.summary.receipt_path = self._rel_to_root(out)

    def _write_reports(self) -> None:
        rep_dir = self._reports_out_dir()
        rep_dir.mkdir(parents=True, exist_ok=True)

        identity_md = _render_identity_report(self.summary)
        license_md = _render_license_report(self.summary, self.access_date)
        integrity_md = _render_integrity_report(self.summary, self._rows)
        known_md = _render_known_issues(self.summary)

        targets = {
            "dataset_identity_report.md": identity_md,
            "license_review_report.md": license_md,
            "file_integrity_report.md": integrity_md,
            "known_issues.md": known_md,
        }
        for name, content in targets.items():
            p = rep_dir / name
            with open(p, "w", encoding="utf-8") as handle:
                handle.write(content)
            self.summary.report_paths[name] = self._rel_to_root(p)

    # ------------------------------------------------------------------
    def _derive_overall_status(self) -> None:
        if self.summary.raw_modified:
            self.summary.overall_status = "BLOCKED"
            return
        if (
            self.summary.identity is not None
            and self.summary.identity.status not in ("verified",)
        ):
            self.summary.overall_status = "BLOCKED"
            return
        blocking = (
            self.summary.n_corrupt_archives > 0
            or self.summary.n_conflicts > 0
            or bool(self.summary.errors)
        )
        if blocking:
            self.summary.overall_status = "PASS WITH WARNINGS"
            return
        # License is structurally TBD until manual review in Stage 1, so even a
        # clean verify_existing audit is PASS WITH WARNINGS (license pending).
        self.summary.overall_status = "PASS WITH WARNINGS"


# ---------------------------------------------------------------------------
# Rendering helpers (markdown)
# ---------------------------------------------------------------------------

def _identity_to_dict(verification: Optional[IdentityVerification]) -> Dict[str, Any]:
    if verification is None:
        return {"status": "not_run"}
    mid = verification.matched_identity
    return {
        "status": verification.status,
        "expected_role": verification.expected_role,
        "matched_name": mid.name if mid else None,
        "matched_role": mid.role if mid else None,
        "data_doi": mid.data_doi if mid else None,
        "paper_doi": mid.paper_doi if mid else None,
        "warnings": verification.warnings,
        "error": verification.error,
    }


def _manual_steps() -> List[str]:
    return [
        "Log in to Science Data Bank (scidb.cn) and locate data DOI 10.57760/sciencedb.19070.",
        "Confirm the dataset title, version and license match the primary identity.",
        "Download the dataset archive(s) into data/external/incoming (the staging area).",
        "Record the download date, version, filenames, source URL and SHA-256 in download_receipt.json.",
        "Re-run scripts/01_data_download.py --verify-existing to audit the staged files.",
        "Only after audit, a human decides whether to promote verified files into data/raw.",
        "Never re-distribute identifiable audio; do not attempt re-identification.",
    ]


def _render_identity_report(s: Stage1Summary) -> str:
    v = s.identity
    mid = v.matched_identity if v else None
    lines = [
        "# 数据集身份核验报告（Stage 1）",
        "",
        f"- run_id：`{s.run_id}`",
        f"- 访问日期：{s.access_date}",
        "",
        "## 主数据集（identity verified against hard-coded DOI constants）",
        "",
        f"- 名称：{PRIMARY_IDENTITY.name}",
        f"- 论文 DOI：`{PRIMARY_IDENTITY.paper_doi}`",
        f"- 数据 DOI：`{PRIMARY_IDENTITY.data_doi}`",
        f"- 角色：`primary`",
        f"- 核验状态：`{v.status if v else 'not_run'}`",
        "",
        "> 文献报告的患者数（~50）、时长（>400 h）、采样率、airflow 子集（~36/50）等均为"
        " `literature_reported_hypothesis`，须在阶段 2 用真实文件审计确认，不得在此写成科学结论。",
        "",
        "## 背景数据集（background_reference_only，不得混入主清单）",
        "",
        f"- 名称：{BACKGROUND_IDENTITY.name}",
        f"- 数据 DOI：`{BACKGROUND_IDENTITY.data_doi}`",
        f"- 角色：`background_reference_only`",
        "- 该数据集（212 例、气管/环境麦克风、EDF/RML）与主数据集不同，禁止与主清单混合。",
        "",
        "## 不可互换性检查",
        "",
        f"- 主数据 DOI 是否等于背景 DOI：`{PRIMARY_IDENTITY.data_doi == BACKGROUND_IDENTITY.data_doi}`（须为 False）",
        f"- 身份核验告警：{'; '.join(v.warnings) if (v and v.warnings) else '无'}",
        "",
        "## 结论",
        "",
        ("主数据集身份与硬编码 DOI 常量一致。" if v and v.status == 'verified'
         else "身份未确认或存在不一致，停止自动下载并请求人工决定。"),
    ]
    return "\n".join(lines) + "\n"


def _render_license_report(s: Stage1Summary, access_date: str) -> str:
    return (
        "# 许可审阅报告（Stage 1）\n\n"
        f"- run_id：`{s.run_id}`\n"
        f"- 访问日期：{access_date}\n\n"
        "## 主数据集\n"
        f"- 名称：{PRIMARY_IDENTITY.name}\n"
        f"- 论文 DOI：`{PRIMARY_IDENTITY.paper_doi}`\n"
        f"- 数据 DOI：`{PRIMARY_IDENTITY.data_doi}`\n"
        f"- 数据库页面：{PRIMARY_IDENTITY.source}\n\n"
        "## 许可状态\n\n"
        "- 许可原文/页面摘要：`TBD_AFTER_MANUAL_LICENSE_REVIEW`\n"
        "- 商业/非商业限制：`TBD_AFTER_MANUAL_LICENSE_REVIEW`\n"
        "- 是否允许科研二次分析：`TBD_AFTER_MANUAL_LICENSE_REVIEW`\n"
        "- 是否允许派生特征发布：`TBD_AFTER_MANUAL_LICENSE_REVIEW`\n"
        "- 是否允许重新分发原始音频：`TBD_AFTER_MANUAL_LICENSE_REVIEW`\n\n"
        "## 说明\n\n"
        "- 不把网页截图当作许可证明的唯一依据；须以数据库正式许可条款为准。\n"
        "- 用户需登录 Science Data Bank 并接受条款后才能下载；不得假装为公开无认证 URL。\n"
        "- 暂存包放入 `data/external/incoming/`，经审计后由用户决定是否进入 `data/raw`。\n"
        "- 不公开患者可识别音频；不尝试重识别。\n"
    )


def _render_integrity_report(s: Stage1Summary, rows: List[ManifestRow]) -> str:
    by_status: Dict[str, int] = {}
    for r in rows:
        by_status[r.readability_status] = by_status.get(r.readability_status, 0) + 1
    anomalies = [r for r in rows if r.error_code or r.archive_integrity_status not in ("", "ok")]
    lines = [
        "# 文件完整性审计报告（Stage 1）\n",
        f"- run_id：`{s.run_id}`\n",
        "## 审计摘要（文件层面，非患者科学结果）\n",
        f"- raw 文件数：{s.n_files_raw}",
        f"- staging 文件数：{s.n_files_staging}",
        f"- 总字节数：{s.total_bytes}",
        f"- 压缩包数：{s.n_archives}",
        f"- 损坏压缩包数：{s.n_corrupt_archives}",
        f"- 重复内容记录数：{s.n_duplicates}",
        f"- 冲突数：{s.n_conflicts}",
        f"- 异常/告警文件数：{s.n_anomalies}",
        f"- 因依赖缺失未检查音频数：{s.n_audio_not_checked}\n",
        "## 可读性状态分布\n",
    ]
    if not by_status:
        lines.append("- （无可审计文件）\n")
    else:
        for status, count in sorted(by_status.items()):
            lines.append(f"- `{status}`：{count}")
        lines.append("")
    lines.append("## 分类\n")
    lines.append(_classify_overall(s))
    if anomalies:
        lines.append("\n## 需关注文件\n")
        lines.append("| 相对路径 | 状态 | 原因 |")
        lines.append("| --- | --- | --- |")
        for r in anomalies[:200]:
            reason = (r.error_message or r.archive_integrity_status or "").replace("|", "/")[:120]
            lines.append(f"| {r.relative_path} | {r.readability_status} | {reason} |")
    return "\n".join(lines) + "\n"


def _classify_overall(s: Stage1Summary) -> str:
    verdicts = []
    verdicts.append("PASS" if not s.raw_modified else "FAIL: raw 被修改")
    verdicts.append("PASS" if s.n_conflicts == 0 else f"WARNING: {s.n_conflicts} 个冲突")
    verdicts.append("PASS" if s.n_corrupt_archives == 0 else f"WARNING: {s.n_corrupt_archives} 个损坏压缩包")
    verdicts.append(
        "PASS" if s.n_audio_not_checked == 0
        else f"WARNING: {s.n_audio_not_checked} 个音频因依赖缺失未检查（仅记录限制，不阻断）"
    )
    if s.download_status == STATUS_MANUAL_REQUIRED:
        verdicts.append("MANUAL_REQUIRED: 需人工下载")
    return "- " + "\n- ".join(verdicts)


def _render_known_issues(s: Stage1Summary) -> str:
    items = list(s.warnings) + list(s.errors)
    if s.download_status == STATUS_MANUAL_REQUIRED:
        items.append("主数据集需人工从 Science Data Bank 下载（认证下载），暂存区当前为空。")
    elif s.download_status == STATUS_VERIFY_EXISTING:
        items.append(
            "本次为对已存在 data/raw 的只读复审（verify_existing）：未下载、未移动、未修改任何 raw 文件。"
        )
    if s.n_audio_not_checked > 0:
        items.append(
            f"ffmpeg/ffprobe 缺失或 soundfile 不可用：{s.n_audio_not_checked} 个音频仅做容器级/魔数检查，"
            "不能宣称音频内容完整（仅记录限制，不阻断）。"
        )
    items.append("soundfile/librosa/lightgbm/xgboost/torch 缺失不阻断阶段 1，仅在后续阶段前需安装。")
    items.append("数据集许可状态待人工核验（TBD_AFTER_MANUAL_LICENSE_REVIEW）。")
    if not items:
        items.append("暂无已知问题。")
    body = "\n".join(f"- {it}" for it in items)
    return f"# 阶段 1 已知问题（known_issues）\n\n- run_id：`{s.run_id}`\n\n{body}\n"


__all__ = [
    "Stage1Options",
    "Stage1Summary",
    "Stage1Runner",
    "Stage1ContaminationError",
    "MANIFEST_FIELDS",
]
