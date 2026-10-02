"""License confirmation gate (Stage 2, prompt section 1).

Before the Stage-2 audit reads ANY real patient content, a human must have
filled in auditable license evidence::

    docs/license_evidence/primary_dataset_license_confirmation.md

That file MUST be produced by a human from the official Science Data Bank
page/terms. It is never fabricated by code, never bypassed by hard-coding
``true``, and never substituted by a passing test. This module only *reads*
that evidence (it does not touch ``data/raw``) and decides whether the gate is
``passed`` or ``blocked``.

Gate behaviour (prompt section 1 "许可门的行为"):

* Evidence missing / DOI mismatch / terms do not support the current
  non-commercial research use / license still TBD -> ``blocked``. The runner
  then writes ``reports/data_audit/license_gate_blocked.md`` and STOPS, without
  scanning ``data/raw``.
* Evidence complete and DOI matches the primary identity -> ``passed``. The
  runner records the evidence path, DOI, version and access date in
  ``license_gate_passed.md`` and may proceed to the read-only audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import Config
from ..data_download.identity import (
    PRIMARY_IDENTITY,
    normalize_doi,
)

#: Relative path (from project root) of the required evidence file.
EVIDENCE_REL_SUBPATH = "license_evidence/primary_dataset_license_confirmation.md"

#: Markdown-list field labels parsed out of the evidence file. Each label maps
#: to the (lowercased) key used in :class:`LicenseEvidence`. A human edits the
#: markdown; this parser only reads labelled ``- key: value`` lines.
_FIELD_LABELS: Dict[str, str] = {
    "data_doi": "data_doi",
    "dataset_version": "dataset_version",
    "dataset version": "dataset_version",
    "access_date": "access_date",
    "access date": "access_date",
    "license_terms": "license_terms",
    "license terms": "license_terms",
    "noncommercial_research_basis": "noncommercial_research_basis",
    "non-commercial research basis": "noncommercial_research_basis",
    "deriv_features_policy": "deriv_features_policy",
    "derivative features policy": "deriv_features_policy",
    "redistribute_audio_policy": "redistribute_audio_policy",
    "audio redistribution policy": "redistribute_audio_policy",
    "evidence_screenshot_relpath": "evidence_screenshot_relpath",
    "evidence screenshot relpath": "evidence_screenshot_relpath",
    "confirmer_name": "confirmer_name",
    "confirmer name": "confirmer_name",
    "confirmation_date": "confirmation_date",
    "confirmation date": "confirmation_date",
    "open_items": "open_items",
    "open items / uncertainties": "open_items",
}

#: Values that count as "not actually filled in" (a template placeholder).
_PLACEHOLDER_TOKENS = (
    "tbd",
    "待填",
    "待人工",
    "todo",
    "placeholder",
    "xxxx",
    "<",
    "请填写",
)


def _is_placeholder(value: str) -> bool:
    """True if ``value`` looks like an unfilled template placeholder."""
    if not value:
        return True
    low = value.strip().lower()
    if not low:
        return True
    for tok in _PLACEHOLDER_TOKENS:
        if tok in low:
            return True
    return False


@dataclass
class LicenseEvidence:
    """Fields parsed from the human-authored evidence markdown."""

    data_doi: str = ""
    dataset_version: str = ""
    access_date: str = ""
    license_terms: str = ""
    noncommercial_research_basis: str = ""
    deriv_features_policy: str = ""
    redistribute_audio_policy: str = ""
    evidence_screenshot_relpath: str = ""
    confirmer_name: str = ""
    confirmation_date: str = ""
    open_items: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "data_doi": self.data_doi,
            "dataset_version": self.dataset_version,
            "access_date": self.access_date,
            "license_terms": self.license_terms,
            "noncommercial_research_basis": self.noncommercial_research_basis,
            "deriv_features_policy": self.deriv_features_policy,
            "redistribute_audio_policy": self.redistribute_audio_policy,
            "evidence_screenshot_relpath": self.evidence_screenshot_relpath,
            "confirmer_name": self.confirmer_name,
            "confirmation_date": self.confirmation_date,
            "open_items": self.open_items,
        }


@dataclass
class LicenseGateResult:
    """Outcome of the license gate check."""

    status: str  # 'passed' | 'blocked'
    evidence_relpath: str = ""
    evidence: Optional[LicenseEvidence] = None
    missing_items: List[str] = field(default_factory=list)
    data_doi: str = ""
    dataset_version: str = ""
    access_date: str = ""
    license_status: str = "TBD_AFTER_MANUAL_LICENSE_REVIEW"
    notes: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "passed"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

#: A markdown list line:  "- label: value"  (label may contain spaces / hyphens).
_LIST_LINE = re.compile(r"^\s*[-*]\s*([^:：]+?)\s*[:：]\s*(.*)$")


def parse_evidence_md(text: str) -> LicenseEvidence:
    """Parse labelled ``- key: value`` lines from the evidence markdown.

    Tolerant by design: unknown labels are ignored, missing labels stay empty.
    This never raises on a malformed file; the gate's *completeness* check
    decides pass/block.
    """
    fields: Dict[str, str] = {}
    for raw_line in text.splitlines():
        m = _LIST_LINE.match(raw_line)
        if not m:
            continue
        label, value = m.group(1).strip(), m.group(2).strip()
        key = _FIELD_LABELS.get(label.lower())
        if not key:
            # try a normalised lookup (spaces -> underscore already covered)
            continue
        # Only keep the first occurrence per key.
        fields.setdefault(key, value)
    return LicenseEvidence(**{k: fields.get(k, "") for k in LicenseEvidence().as_dict()})


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _evidence_path(cfg: Config) -> Path:
    return cfg.path("docs") / EVIDENCE_REL_SUBPATH


def _required_nonplaceholder(ev: LicenseEvidence) -> List[str]:
    """Return the list of required fields that are missing or placeholder."""
    required = [
        ("data_doi", ev.data_doi),
        ("access_date", ev.access_date),
        ("license_terms", ev.license_terms),
        ("noncommercial_research_basis", ev.noncommercial_research_basis),
        ("deriv_features_policy", ev.deriv_features_policy),
        ("redistribute_audio_policy", ev.redistribute_audio_policy),
        ("confirmer_name", ev.confirmer_name),
        ("confirmation_date", ev.confirmation_date),
    ]
    missing: List[str] = []
    for name, value in required:
        if _is_placeholder(value):
            missing.append(name)
    return missing


def check_license_gate(
    cfg: Config, primary_identity: Any = PRIMARY_IDENTITY
) -> LicenseGateResult:
    """Decide whether the Stage-2 license gate passes.

    Returns a :class:`LicenseGateResult`. This function does NOT read
    ``data/raw`` and never raises on a missing/incomplete evidence file: a
    missing file is simply a ``blocked`` result with a clear ``missing_items``
    list, so the caller can write the ``license_gate_blocked.md`` report.
    """
    ev_path = _evidence_path(cfg)
    try:
        ev_rel = str(ev_path.resolve().relative_to(cfg.project_root.resolve())).replace(
            "\\", "/"
        )
    except ValueError:
        ev_rel = EVIDENCE_REL_SUBPATH

    if not ev_path.is_file():
        return LicenseGateResult(
            status="blocked",
            evidence_relpath=ev_rel,
            missing_items=["evidence_file_absent"],
            notes=[
                f"Required evidence file not found at {ev_rel}. "
                "A human must author it from the official Science Data Bank "
                "page/terms before any patient-data audit."
            ],
        )

    try:
        text = ev_path.read_text(encoding="utf-8")
    except OSError as exc:
        return LicenseGateResult(
            status="blocked",
            evidence_relpath=ev_rel,
            missing_items=["evidence_file_unreadable"],
            notes=[f"read_failed: {type(exc).__name__}: {exc}"],
        )

    ev = parse_evidence_md(text)

    # DOI must match the primary identity (prompt: "DOI 不匹配" -> block).
    doi_ok = bool(ev.data_doi) and normalize_doi(ev.data_doi) == normalize_doi(
        primary_identity.data_doi
    )

    missing = _required_nonplaceholder(ev)
    notes: List[str] = []
    if not doi_ok:
        missing.append("data_doi_mismatch")
        notes.append(
            f"data_doi {ev.data_doi!r} does not match the primary dataset DOI "
            f"{primary_identity.data_doi!r}."
        )

    # The non-commercial research basis must be a genuine affirmative, not TBD.
    # (A bare 'yes'/'permitted' is the minimum human assertion we require.)
    basis_low = ev.noncommercial_research_basis.lower()
    affirmative = any(
        tok in basis_low
        for tok in ("permit", "allow", "yes", "允许", "可", "许可", "支持")
    )

    if missing or not affirmative:
        return LicenseGateResult(
            status="blocked",
            evidence_relpath=ev_rel,
            evidence=ev,
            missing_items=missing if missing else ["noncommercial_basis_not_affirmative"],
            data_doi=ev.data_doi,
            dataset_version=ev.dataset_version,
            access_date=ev.access_date,
            license_status="TBD_AFTER_MANUAL_LICENSE_REVIEW",
            notes=notes,
        )

    # Passed: derive an honest license_status string (not the TBD sentinel).
    version = ev.dataset_version or "version_not_displayed_on_source"
    return LicenseGateResult(
        status="passed",
        evidence_relpath=ev_rel,
        evidence=ev,
        missing_items=[],
        data_doi=ev.data_doi,
        dataset_version=version,
        access_date=ev.access_date,
        license_status="confirmed_for_noncommercial_research",
        notes=notes,
    )


__all__ = [
    "EVIDENCE_REL_SUBPATH",
    "LicenseEvidence",
    "LicenseGateResult",
    "parse_evidence_md",
    "check_license_gate",
]
