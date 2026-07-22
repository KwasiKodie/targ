"""
test_corpus_builder_page_number.py

Unit tests for RepresentativeCorpusBuilder._resolve_page_number().
"""

from rag.corpus_builder import RepresentativeCorpusBuilder


# ---------------------------------------------------------------------
# Valid metadata
# ---------------------------------------------------------------------


def test_resolve_page_number_from_metadata():
    """
    Metadata page numbers should be converted from zero-based to one-based.
    """

    metadata = {
        "page": 4,
    }

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            metadata,
            fallback_page_index=99,
        )
    )

    assert page == 5


# ---------------------------------------------------------------------
# Missing metadata
# ---------------------------------------------------------------------


def test_resolve_page_number_uses_fallback_when_page_missing():
    """
    Missing page metadata should use the supplied fallback page index.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {},
            fallback_page_index=2,
        )
    )

    assert page == 3


# ---------------------------------------------------------------------
# None page
# ---------------------------------------------------------------------


def test_resolve_page_number_uses_fallback_when_page_is_none():
    """
    None page values should fall back to the supplied page index.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": None},
            fallback_page_index=7,
        )
    )

    assert page == 8


# ---------------------------------------------------------------------
# String integer
# ---------------------------------------------------------------------


def test_resolve_page_number_accepts_numeric_string():
    """
    Numeric strings should be converted to integers.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": "5"},
            fallback_page_index=0,
        )
    )

    assert page == 6


# ---------------------------------------------------------------------
# Invalid string
# ---------------------------------------------------------------------


def test_resolve_page_number_uses_fallback_for_invalid_string():
    """
    Invalid page strings should use the fallback page.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": "page five"},
            fallback_page_index=10,
        )
    )

    assert page == 11


# ---------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------


def test_resolve_page_number_rejects_boolean_page():
    """
    Booleans are subclasses of int and must not be treated as page numbers.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": True},
            fallback_page_index=3,
        )
    )

    assert page == 4


# ---------------------------------------------------------------------
# Float
# ---------------------------------------------------------------------


def test_resolve_page_number_accepts_float():
    """
    Float page numbers are truncated by int().
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": 2.9},
            fallback_page_index=0,
        )
    )

    assert page == 3


# ---------------------------------------------------------------------
# Negative page
# ---------------------------------------------------------------------


def test_resolve_page_number_preserves_negative_page():
    """
    Negative page numbers are currently accepted by the implementation.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": -1},
            fallback_page_index=5,
        )
    )

    assert page == 0


# ---------------------------------------------------------------------
# Fallback zero
# ---------------------------------------------------------------------


def test_resolve_page_number_with_zero_fallback():
    """
    A zero fallback should become page one.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {},
            fallback_page_index=0,
        )
    )

    assert page == 1


# ---------------------------------------------------------------------
# Large page number
# ---------------------------------------------------------------------


def test_resolve_page_number_handles_large_page_numbers():
    """
    Large page numbers should be converted correctly.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {"page": 9999},
            fallback_page_index=0,
        )
    )

    assert page == 10000


# ---------------------------------------------------------------------
# Empty metadata with non-zero fallback
# ---------------------------------------------------------------------


def test_resolve_page_number_empty_metadata():
    """
    Empty metadata should always use the fallback page index.
    """

    page = (
        RepresentativeCorpusBuilder
        ._resolve_page_number(
            {},
            fallback_page_index=25,
        )
    )

    assert page == 26