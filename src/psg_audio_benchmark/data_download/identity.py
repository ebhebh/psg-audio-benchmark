"""Primary-dataset identity verification (Stage 1, prompt section 1).

The whole benchmark hinges on downloading the *correct* dataset. Identity is
verified against hard-coded DOI constants, **never** inferred from a filename.
Two datasets must never be confused:

* **Primary** — Tao et al. 2025 (Scientific Data):
  paper DOI ``10.1038/s41597-025-05583-8``, data DOI ``10.57760/sciencedb.19070``.

* **Background-only** — Korompili et al. 2021 (PSG-Audio):
  data DOI ``10.11922/sciencedb.00345``. This is a DIFFERENT dataset (212
  records, tracheal/environment microphones, EDF/RML) and MUST stay
  ``background_reference_only`` — it must never enter the primary manifest.

If the page, file or version DOI does not match the constants below, this
module raises :class:`DatasetIdentityError`; the caller stops auto-download and
emits ``dataset_identity_warning.md`` for a human decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: The only two dataset roles permitted in a Stage-1 manifest.
ROLE_PRIMARY = "primary"
ROLE_BACKGROUND = "background_reference_only"
ALLOWED_ROLES = (ROLE_PRIMARY, ROLE_BACKGROUND)

#: DOI URL prefixes that are stripped during normalization.
_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
)


def normalize_doi(doi: Optional[str]) -> str:
    """Normalize a DOI string for comparison.

    Strips whitespace and any ``doi.org`` URL prefix and lowercases. Returns an
    empty string for ``None``/empty input so comparisons are well defined.
    """
    if not doi or not isinstance(doi, str):
        return ""
    d = doi.strip()
    low = d.lower()
    for prefix in _DOI_URL_PREFIXES:
        if low.startswith(prefix):
            d = d[len(prefix):]
            low = d.lower()
            break
    return low.strip()


@dataclass(frozen=True)
class DatasetIdentity:
    """A known dataset identity (DOI constants live in code, not config)."""

    name: str
    paper_doi: Optional[str]
    data_doi: str
    role: str
    source: Optional[str] = None          # canonical page URL, if known
    note: str = ""

    def __post_init__(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise DatasetIdentityError(
                f"role must be one of {ALLOWED_ROLES}, got {self.role!r}"
            )


# ---------------------------------------------------------------------------
# Hard-coded identity constants (source of truth).
# ---------------------------------------------------------------------------

PRIMARY_IDENTITY = DatasetIdentity(
    name=(
        "Tao et al. 2025 - A multimodal dataset for training deep learning "
        "models aimed at detecting and analyzing sleep apnea"
    ),
    paper_doi="10.1038/s41597-025-05583-8",
    data_doi="10.57760/sciencedb.19070",
    role=ROLE_PRIMARY,
    source="https://www.scidb.cn/  (Science Data Bank; resolved via DOI)",
    note=(
        "Literature reports ~50 patients, >400 h, smartphone + recorder audio, "
        "SpO2/HR/airflow, sleep architecture, PSG/AASM annotations. These are "
        "HYPOTHESES pending the Stage-2 audit of real files."
    ),
)

BACKGROUND_IDENTITY = DatasetIdentity(
    name="PSG-Audio (Korompili et al. 2021)",
    paper_doi="10.1038/s41597-021-00987-0",
    data_doi="10.11922/sciencedb.00345",
    role=ROLE_BACKGROUND,
    source="https://www.scidb.cn/",
    note=(
        "A DIFFERENT dataset (212 records, tracheal/environment microphones, "
        "EDF/RML). Background reference ONLY; never mixed with the primary "
        "dataset."
    ),
)

#: Lookup by normalized data DOI.
KNOWN_BY_DATA_DOI: Dict[str, DatasetIdentity] = {
    normalize_doi(PRIMARY_IDENTITY.data_doi): PRIMARY_IDENTITY,
    normalize_doi(BACKGROUND_IDENTITY.data_doi): BACKGROUND_IDENTITY,
}


class DatasetIdentityError(Exception):
    """Raised when a DOI/name does not match the expected dataset identity."""


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def classify_doi(data_doi: Optional[str]) -> str:
    """Return the known role for ``data_doi`` or ``'unknown'``."""
    norm = normalize_doi(data_doi)
    identity = KNOWN_BY_DATA_DOI.get(norm)
    if identity is None:
        return "unknown"
    return identity.role


@dataclass
class IdentityVerification:
    """Outcome of verifying a presented dataset identity."""

    status: str                       # 'verified' | 'mismatch' | 'unknown' | 'manual_required'
    matched_identity: Optional[DatasetIdentity] = None
    expected_role: str = ROLE_PRIMARY
    checks: Dict[str, Any] = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def verified(self) -> bool:
        return self.status == "verified" and self.matched_identity is not None


def verify_primary_identity(
    data_doi: Optional[str],
    paper_doi: Optional[str] = None,
    dataset_name: Optional[str] = None,
    expected_role: str = ROLE_PRIMARY,
) -> IdentityVerification:
    """Verify that the presented identity is the expected dataset.

    A DOI is matched by its *normalized* value against the known constants.
    The wrong dataset (e.g. the background DOI presented as primary) is a hard
    ``mismatch`` and MUST stop auto-download. An unrecognized DOI is
    ``unknown`` (also non-passing) so it can never slip through silently.
    """
    norm = normalize_doi(data_doi)
    checks: Dict[str, Any] = {
        "presented_data_doi": data_doi,
        "normalized_data_doi": norm,
        "expected_role": expected_role,
    }
    warnings: list = []

    matched = KNOWN_BY_DATA_DOI.get(norm)

    if matched is None:
        return IdentityVerification(
            status="unknown",
            expected_role=expected_role,
            checks=checks,
            warnings=[
                f"DOI {data_doi!r} matches no known dataset identity; refusing "
                "to treat an unrecognized source as the primary dataset."
            ],
            error="unknown_dataset_identity",
        )

    checks["matched_name"] = matched.name
    checks["matched_role"] = matched.role

    # Cross-dataset confusion: background DOI presented as primary (or vice versa)
    if matched.role != expected_role:
        return IdentityVerification(
            status="mismatch",
            matched_identity=matched,
            expected_role=expected_role,
            checks=checks,
            warnings=[
                f"DOI {data_doi!r} is the {matched.role} dataset "
                f"({matched.name}), not the expected {expected_role}. Datasets "
                "must not be interchanged."
            ],
            error="dataset_role_mismatch",
        )

    # Optional extra corroboration: if a paper DOI is given, it must agree with
    # the matched identity's paper DOI (when that is known).
    if paper_doi and matched.paper_doi:
        checks["presented_paper_doi"] = paper_doi
        checks["expected_paper_doi"] = matched.paper_doi
        if normalize_doi(paper_doi) != normalize_doi(matched.paper_doi):
            warnings.append(
                f"paper DOI {paper_doi!r} disagrees with the known paper DOI "
                f"{matched.paper_doi!r} for {matched.name}; verify manually."
            )
            return IdentityVerification(
                status="mismatch",
                matched_identity=matched,
                expected_role=expected_role,
                checks=checks,
                warnings=warnings,
                error="paper_doi_mismatch",
            )

    # Optional name sanity: a presented name must not contradict the matched one
    # by naming the background dataset.
    if dataset_name:
        checks["presented_name"] = dataset_name
        bg_tokens = ("psg-audio", "psgaudio", "korompili", "212")
        low = dataset_name.lower()
        if matched.role == ROLE_PRIMARY and any(t in low for t in bg_tokens):
            warnings.append(
                f"presented name {dataset_name!r} mentions the background "
                "PSG-Audio dataset; verify this is really the primary dataset."
            )
            return IdentityVerification(
                status="manual_required",
                matched_identity=matched,
                expected_role=expected_role,
                checks=checks,
                warnings=warnings,
                error="name_looks_like_background",
            )

    return IdentityVerification(
        status="verified",
        matched_identity=matched,
        expected_role=expected_role,
        checks=checks,
        warnings=warnings,
    )


def assert_primary(data_doi: Optional[str]) -> DatasetIdentity:
    """Convenience: verify and return the primary identity, else raise."""
    result = verify_primary_identity(data_doi, expected_role=ROLE_PRIMARY)
    if not result.verified or result.matched_identity is None:
        raise DatasetIdentityError(
            f"Primary dataset identity not verified (status={result.status}): "
            f"{'; '.join(result.warnings) or result.error}"
        )
    return result.matched_identity


__all__ = [
    "ROLE_PRIMARY",
    "ROLE_BACKGROUND",
    "ALLOWED_ROLES",
    "DatasetIdentity",
    "PRIMARY_IDENTITY",
    "BACKGROUND_IDENTITY",
    "KNOWN_BY_DATA_DOI",
    "DatasetIdentityError",
    "IdentityVerification",
    "normalize_doi",
    "classify_doi",
    "verify_primary_identity",
    "assert_primary",
]
