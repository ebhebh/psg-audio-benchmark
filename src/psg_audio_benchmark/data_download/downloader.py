"""Non-destructive staging downloader (Stage 1, prompt sections 3 & 4).

Default behaviour is ``manual_required``: the primary dataset lives behind
Science Data Bank authentication, so the runner refuses to pretend it can fetch
it from a public, no-auth URL. Automatic download is only attempted when an
explicit, verified no-auth ``url`` is supplied **and** ``allow_auto_download``
is enabled — and even then it follows the safe contract below.

Safe-download contract (prompt section 3):
  * Download to a temporary ``.part`` file; only after hash/size checks pass is
    it atomically renamed to its final staging name.
  * If the final target already exists, it is NEVER overwritten: identical hash
    -> ``already_present``; different hash -> ``conflict``.
  * Resume is used only when the server advertises ``Accept-Ranges`` and the
    caller opts in; the HTTP response is recorded.
  * A network failure never deletes an existing file (the ``.part`` is kept so
    a later run may resume; the final target is untouched).
  * No web scripting, no credential submission, no access-control bypass.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .manifest import STATUS_ALREADY_PRESENT, STATUS_CONFLICT

#: Default outcome when no verified public no-auth URL is available.
STATUS_MANUAL_REQUIRED = "manual_required"
STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED_DRY_RUN = "skipped_dry_run"
STATUS_FAILED = "failed"

#: Subdirectory used for in-flight downloads (kept out of the manifest scan).
_INCOMING_PARTS = ".incoming"

#: Stream chunk size for HTTP downloads.
_NET_CHUNK = 1024 * 1024


@dataclass
class DownloadResult:
    status: str
    target_path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    bytes_downloaded: int = 0
    http_status: Optional[int] = None
    accept_ranges: bool = False
    resumed: bool = False
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (
            STATUS_DOWNLOADED, STATUS_ALREADY_PRESENT, STATUS_SKIPPED_DRY_RUN
        )


# ---------------------------------------------------------------------------
# Non-destructive finalize (shared by bytes + http paths)
# ---------------------------------------------------------------------------

def _existing_sha(target: Path) -> str:
    if not target.exists() or not target.is_file():
        return ""
    h = hashlib.sha256()
    with open(target, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_staging(
    staging_dir: Path,
    filename: str,
    temp_path: Path,
    expected_sha256: Optional[str],
    dry_run: bool,
) -> DownloadResult:
    """Atomically promote ``temp_path`` to ``staging_dir/filename``.

    Implements the no-overwrite rule: an existing target is never replaced.
    Returns a :class:`DownloadResult` describing the outcome. On conflict or
    dry-run the ``temp_path`` is removed (it is transient), but an *existing*
    target is left byte-for-byte intact.
    """
    staging_dir = Path(staging_dir)
    target = staging_dir / filename
    new_sha = _sha_of_temp(temp_path)
    size = temp_path.stat().st_size if temp_path.exists() else 0

    if target.exists():
        existing_sha = _existing_sha(target)
        if existing_sha and new_sha and existing_sha == new_sha:
            _safe_remove_temp(temp_path)
            return DownloadResult(
                status=STATUS_ALREADY_PRESENT,
                target_path=str(target),
                sha256=new_sha,
                size_bytes=size,
            )
        # Conflict: NEVER overwrite. Keep the original, drop the temp.
        _safe_remove_temp(temp_path)
        return DownloadResult(
            status=STATUS_CONFLICT,
            target_path=str(target),
            sha256=new_sha,
            size_bytes=size,
            error=(
                f"existing target differs (existing={existing_sha[:12]}..., "
                f"new={new_sha[:12]}...); original left unchanged"
            ),
        )

    # Verify against an expected hash if one was provided.
    if expected_sha256 and new_sha and expected_sha256.lower() != new_sha.lower():
        _safe_remove_temp(temp_path)
        return DownloadResult(
            status=STATUS_FAILED,
            target_path=str(target),
            sha256=new_sha,
            size_bytes=size,
            error=(
                f"hash mismatch: expected {expected_sha256[:12]}..., "
                f"got {new_sha[:12]}..."
            ),
        )

    if dry_run:
        _safe_remove_temp(temp_path)
        return DownloadResult(
            status=STATUS_SKIPPED_DRY_RUN,
            target_path=str(target),
            sha256=new_sha,
            size_bytes=size,
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, target)
    return DownloadResult(
        status=STATUS_DOWNLOADED,
        target_path=str(target),
        sha256=new_sha,
        size_bytes=size,
    )


def _sha_of_temp(temp_path: Path) -> str:
    if not temp_path.exists():
        return ""
    h = hashlib.sha256()
    with open(temp_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_remove_temp(temp_path: Path) -> None:
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass


def _write_temp(temp_path: Path, data: bytes) -> None:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as handle:
        handle.write(data)


# ---------------------------------------------------------------------------
# Public staging helpers
# ---------------------------------------------------------------------------

def stage_bytes(
    staging_dir: Path,
    filename: str,
    data: bytes,
    expected_sha256: Optional[str] = None,
    dry_run: bool = False,
    incoming_subdir: str = _INCOMING_PARTS,
) -> DownloadResult:
    """Stage an in-memory byte blob non-destructively (used by tests & small files)."""
    staging_dir = Path(staging_dir)
    temp_path = staging_dir / incoming_subdir / (filename + ".part")
    _write_temp(temp_path, data)
    return finalize_staging(staging_dir, filename, temp_path, expected_sha256, dry_run)


def stage_local_file(
    staging_dir: Path,
    filename: str,
    source_path: Path,
    expected_sha256: Optional[str] = None,
    dry_run: bool = False,
    incoming_subdir: str = _INCOMING_PARTS,
) -> DownloadResult:
    """Stage a local source file non-destructively (copy via temp + atomic rename)."""
    staging_dir = Path(staging_dir)
    source_path = Path(source_path)
    temp_path = staging_dir / incoming_subdir / (filename + ".part")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, temp_path)
    return finalize_staging(staging_dir, filename, temp_path, expected_sha256, dry_run)


# ---------------------------------------------------------------------------
# HTTP downloader (opt-in)
# ---------------------------------------------------------------------------

def download_http(
    url: str,
    staging_dir: Path,
    filename: str,
    allow_auto_download: bool = False,
    expected_sha256: Optional[str] = None,
    dry_run: bool = False,
    resume: bool = False,
    timeout: float = 60.0,
    incoming_subdir: str = _INCOMING_PARTS,
) -> DownloadResult:
    """Download ``url`` into the staging area following the safe contract.

    Returns ``manual_required`` unless ``allow_auto_download`` is True and a
    real ``url`` is given. Never submits credentials and never bypasses access
    control (no custom auth headers, no cookie jar, no script execution).
    """
    staging_dir = Path(staging_dir)
    if not allow_auto_download or not url:
        return DownloadResult(
            status=STATUS_MANUAL_REQUIRED,
            target_path=str(staging_dir / filename) if filename else "",
            error="no verified public no-auth URL supplied; manual download required",
            extra={"url": url or None},
        )

    target = staging_dir / filename
    temp_path = staging_dir / incoming_subdir / (filename + ".part")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    # If the final target already exists, resolve without any network call.
    if target.exists():
        existing_sha = _existing_sha(target)
        if expected_sha256:
            # An authoritative expected hash was given -> verify against it.
            if existing_sha.lower() == expected_sha256.lower():
                return DownloadResult(
                    status=STATUS_ALREADY_PRESENT,
                    target_path=str(target),
                    sha256=existing_sha,
                    size_bytes=target.stat().st_size,
                )
            return DownloadResult(
                status=STATUS_CONFLICT,
                target_path=str(target),
                sha256=existing_sha,
                size_bytes=target.stat().st_size,
                error="existing target differs from expected hash; not overwritten",
            )
        # No expected hash: the file is already staged. Do NOT re-download or
        # overwrite (idempotent re-run). Its SHA-256 is still recorded by the
        # Stage-1 audit, so integrity is checked at manifest time.
        return DownloadResult(
            status=STATUS_ALREADY_PRESENT,
            target_path=str(target),
            sha256=existing_sha,
            size_bytes=target.stat().st_size,
        )

    headers = {"User-Agent": "PSG_Audio_Benchmark/1.0 (stage1-audit)"}
    offset = 0
    resumed = False
    accept_ranges = False
    http_status: Optional[int] = None

    if resume and temp_path.exists():
        offset = temp_path.stat().st_size
        headers["Range"] = f"bytes={offset}-"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_status = getattr(resp, "status", None) or resp.getcode()
            accept_ranges = (
                resp.headers.get("Accept-Ranges", "").lower() == "bytes"
            )
            mode = "ab" if (resume and offset > 0 and http_status == 206) else "wb"
            if mode == "wb":
                offset = 0
            else:
                resumed = True
            h = hashlib.sha256()
            downloaded = 0
            with open(temp_path, mode) as out:
                while True:
                    chunk = resp.read(_NET_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    downloaded += len(chunk)
    except Exception as exc:
        # Network failure: keep the .part (for resume), never touch the target.
        return DownloadResult(
            status=STATUS_FAILED,
            target_path=str(target),
            http_status=http_status,
            accept_ranges=accept_ranges,
            resumed=resumed,
            error=f"network_error: {type(exc).__name__}: {exc}",
            extra={"url": url, "temp_part_kept": temp_path.exists()},
        )

    # Hash over the whole file after a (possibly resumed) download.
    return finalize_staging(
        staging_dir, filename, temp_path, expected_sha256, dry_run
    )


# ---------------------------------------------------------------------------
# URL-list batch driver (uses download_http per entry)
# ---------------------------------------------------------------------------

@dataclass
class UrlEntry:
    """One parsed line of a scidb.cn download URL list."""

    url: str
    relpath: str          # reconstructed from the `path=` query param, e.g. V5/Data/01/01_phone.wav
    file_name: str
    is_audio: bool = False
    is_archive: bool = False


def parse_url_list(text: str) -> List[UrlEntry]:
    """Parse a newline-separated list of scidb.cn download URLs.

    Each line looks like::

        https://download.scidb.cn/download?fileId=...&path=/V5/Data/01/01_phone.wav&fileName=01_phone.wav

    The relative path is reconstructed from the ``path=`` query parameter so the
    on-disk layout mirrors the source tree (``V5/Data/<patient>/<file>``).
    Blank lines and ``#`` comments are skipped.
    """
    entries: List[UrlEntry] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = urllib.parse.urlparse(line)
        qs = urllib.parse.parse_qs(parsed.query)
        path_vals = qs.get("path") or qs.get("Path") or []
        name_vals = qs.get("fileName") or qs.get("filename") or []
        if not path_vals:
            continue
        raw_path = path_vals[0]
        # normalize to forward slashes and strip a leading slash
        rel = raw_path.replace("\\", "/").lstrip("/")
        fname = name_vals[0] if name_vals else rel.rsplit("/", 1)[-1]
        low = fname.lower()
        entries.append(
            UrlEntry(
                url=line,
                relpath=rel,
                file_name=fname,
                is_audio=low.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")),
                is_archive=low.endswith((".zip", ".tar", ".gz", ".tgz", ".7z")),
            )
        )
    return entries


def filter_entries(
    entries: Iterable[UrlEntry],
    patients: Optional[List[str]] = None,
    include_script: bool = True,
    audio_only: Optional[bool] = None,
) -> List[UrlEntry]:
    """Select entries by patient folder and/or file kind.

    ``patients`` is a list like ``['01', '02']``; an entry is kept if its path
    contains ``/Data/<patient>/`` for any listed patient. Entries outside any
    patient folder (e.g. ``osa_data_eng.py``) are kept only when
    ``include_script`` is True.
    """
    out: List[UrlEntry] = []
    pat_set = {p.strip().lower().lstrip("0") or "0" for p in patients} if patients else None
    # also keep zero-padded forms
    pat_padded = {p.zfill(2) for p in pat_set} if pat_set else None
    for e in entries:
        if audio_only is True and not e.is_audio:
            continue
        if audio_only is False and e.is_audio:
            continue
        parts = [p for p in e.relpath.split("/") if p]
        # locate a "Data" segment; the next segment is the patient id
        try:
            idx = parts.index("Data")
        except ValueError:
            idx = -1
        if idx >= 0 and idx + 1 < len(parts):
            pid = parts[idx + 1].lower()
            if pat_set is not None and pid not in pat_padded and pid.lstrip("0") not in pat_set:
                continue
        else:
            # not under a patient folder (e.g. osa_data_eng.py)
            if not include_script:
                continue
            if pat_set is not None:
                # a patient filter was requested and this is not patient data
                continue
        out.append(e)
    return out


def download_url_list(
    entries: List[UrlEntry],
    staging_dir: Path,
    allow_auto_download: bool = True,
    dry_run: bool = False,
    workers: int = 4,
    retry: int = 3,
    resume: bool = True,
    timeout: float = 120.0,
    backoff_base: float = 5.0,
    progress: Optional[callable] = None,
) -> Tuple[List[Tuple[UrlEntry, "DownloadResult"]], Dict[str, int]]:
    """Download every entry into ``staging_dir`` preserving relative paths.

    Concurrency is bounded by ``workers``. Each file uses the safe contract
    (temp ``.part`` + atomic rename + SHA-256 + no-overwrite + resume). Network
    failures are retried up to ``retry`` times with exponential backoff
    (``backoff_base * 2**(attempt-1)``) so a transient server throttle can clear;
    an existing final file is never deleted. Returns ``(results, tally)``.
    """
    import concurrent.futures
    import time as _time

    staging_dir = Path(staging_dir)
    results: List[Tuple[UrlEntry, DownloadResult]] = []

    def _one(entry: UrlEntry) -> Tuple[UrlEntry, DownloadResult]:
        last: Optional[DownloadResult] = None
        for attempt in range(1, retry + 1):
            res = download_http(
                entry.url,
                staging_dir,
                filename=entry.relpath,
                allow_auto_download=allow_auto_download,
                dry_run=dry_run,
                resume=resume,
                timeout=timeout,
            )
            last = res
            if res.status in (STATUS_DOWNLOADED, STATUS_ALREADY_PRESENT, STATUS_SKIPPED_DRY_RUN):
                if progress is not None:
                    progress(entry, res, attempt)
                return entry, res
            # failed / manual_required -> back off then retry
            if attempt < retry and backoff_base > 0:
                _time.sleep(backoff_base * (2 ** (attempt - 1)))
        if progress is not None:
            progress(entry, last, retry)
        return entry, last  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_one, e) for e in entries]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    tally: Dict[str, int] = {}
    for _e, r in results:
        tally[r.status] = tally.get(r.status, 0) + 1
    return results, tally


__all__ = [
    "STATUS_MANUAL_REQUIRED",
    "STATUS_DOWNLOADED",
    "STATUS_SKIPPED_DRY_RUN",
    "STATUS_FAILED",
    "DownloadResult",
    "UrlEntry",
    "finalize_staging",
    "stage_bytes",
    "stage_local_file",
    "download_http",
    "parse_url_list",
    "filter_entries",
    "download_url_list",
]
