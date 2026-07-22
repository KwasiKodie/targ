"""
Unit tests for RepresentativeCorpusBuilder._normalize_text().
"""

from rag.corpus_builder import RepresentativeCorpusBuilder


# ---------------------------------------------------------------------
# remove soft hyphens
# ---------------------------------------------------------------------

def test_normalize_text_removes_soft_hyphens():

    text = "Cyber\u00adsecurity"

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert result == "Cybersecurity"


# ---------------------------------------------------------------------
# repair hyphenated line breaks
# ---------------------------------------------------------------------

def test_normalize_text_repairs_hyphenated_line_breaks():

    text = (
        "Retriev-\n"
        "al augmented generation"
    )

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert (
        result
        == "Retrieval augmented generation"
    )


# ---------------------------------------------------------------------
# collapse whitespace
# ---------------------------------------------------------------------

def test_normalize_text_collapses_horizontal_whitespace():

    text = (
        "Adaptive\t\tRAG      improves"
        "      retrieval."
    )

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert (
        result
        == "Adaptive RAG improves retrieval."
    )


# ---------------------------------------------------------------------
# preserve paragraphs
# ---------------------------------------------------------------------

def test_normalize_text_preserves_paragraphs():

    text = (
        "Paragraph one.\n"
        "\n"
        "Paragraph two."
    )

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert (
        result
        == (
            "Paragraph one."
            "\n\n"
            "Paragraph two."
        )
    )


# ---------------------------------------------------------------------
# remove line breaks within paragraphs
# ---------------------------------------------------------------------

def test_normalize_text_replaces_single_line_breaks_with_spaces():

    text = (
        "Language\n"
        "agents\n"
        "reason."
    )

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert (
        result
        == "Language agents reason."
    )


# ---------------------------------------------------------------------
# empty string
# ---------------------------------------------------------------------

def test_normalize_text_accepts_empty_string():

    result = (
        RepresentativeCorpusBuilder._normalize_text("")
    )

    assert result == ""


# ---------------------------------------------------------------------
# whitespace-only page
# ---------------------------------------------------------------------

def test_normalize_text_returns_empty_for_whitespace_only_page():

    text = " \n\t\n   "

    result = (
        RepresentativeCorpusBuilder._normalize_text(text)
    )

    assert result == ""


# ---------------------------------------------------------------------
# invalid input type
# ---------------------------------------------------------------------

import pytest


@pytest.mark.parametrize(
    "value",
    [
        None,
        42,
        3.14,
        [],
        {},
    ],
)
def test_normalize_text_rejects_non_string(value):

    with pytest.raises(TypeError):

        RepresentativeCorpusBuilder._normalize_text(value)