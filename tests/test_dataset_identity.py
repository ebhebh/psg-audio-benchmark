"""Tests for primary-dataset identity verification (prompt section 1, test #4).

The primary DOI and the background-only DOI must NEVER be interchangeable, and
an unrecognized DOI must not slip through silently. Identity is verified against
hard-coded constants, never inferred from a filename.
"""

from __future__ import annotations

import pytest

from psg_audio_benchmark.data_download.identity import (
    ALLOWED_ROLES,
    BACKGROUND_IDENTITY,
    DatasetIdentityError,
    PRIMARY_IDENTITY,
    ROLE_BACKGROUND,
    ROLE_PRIMARY,
    assert_primary,
    classify_doi,
    normalize_doi,
    verify_primary_identity,
)


def test_primary_and_background_dois_differ() -> None:
    assert PRIMARY_IDENTITY.data_doi != BACKGROUND_IDENTITY.data_doi
    assert normalize_doi(PRIMARY_IDENTITY.data_doi) != normalize_doi(
        BACKGROUND_IDENTITY.data_doi
    )


def test_classify_known_dois() -> None:
    assert classify_doi(PRIMARY_IDENTITY.data_doi) == ROLE_PRIMARY
    assert classify_doi(BACKGROUND_IDENTITY.data_doi) == ROLE_BACKGROUND


def test_classify_unknown_doi() -> None:
    assert classify_doi("10.9999/does.not.exist") == "unknown"
    assert classify_doi(None) == "unknown"
    assert classify_doi("") == "unknown"


def test_verify_primary_ok() -> None:
    res = verify_primary_identity(PRIMARY_IDENTITY.data_doi)
    assert res.status == "verified"
    assert res.matched_identity is PRIMARY_IDENTITY
    assert res.matched_identity.role == ROLE_PRIMARY


def test_background_doi_presented_as_primary_is_mismatch() -> None:
    res = verify_primary_identity(BACKGROUND_IDENTITY.data_doi)
    assert res.status == "mismatch"
    assert res.error == "dataset_role_mismatch"
    # it matched the background identity, which is NOT primary
    assert res.matched_identity is BACKGROUND_IDENTITY


def test_primary_doi_presented_as_background_is_mismatch() -> None:
    res = verify_primary_identity(
        PRIMARY_IDENTITY.data_doi, expected_role=ROLE_BACKGROUND
    )
    assert res.status == "mismatch"


def test_wrong_doi_does_not_pass_silently() -> None:
    res = verify_primary_identity("10.9999/bogus.doi")
    assert res.status == "unknown"
    assert not res.verified
    assert res.error == "unknown_dataset_identity"


def test_paper_doi_mismatch_is_detected() -> None:
    res = verify_primary_identity(
        PRIMARY_IDENTITY.data_doi, paper_doi="10.9999/wrong.paper"
    )
    assert res.status == "mismatch"
    assert res.error == "paper_doi_mismatch"


def test_assert_primary_raises_on_background() -> None:
    with pytest.raises(DatasetIdentityError):
        assert_primary(BACKGROUND_IDENTITY.data_doi)


def test_assert_primary_raises_on_unknown() -> None:
    with pytest.raises(DatasetIdentityError):
        assert_primary("10.9999/nope")


def test_normalize_strips_url_prefix() -> None:
    assert normalize_doi("https://doi.org/10.57760/sciencedb.19070") == (
        normalize_doi("10.57760/sciencedb.19070")
    )


def test_roles_are_exhaustive() -> None:
    assert set(ALLOWED_ROLES) == {ROLE_PRIMARY, ROLE_BACKGROUND}
