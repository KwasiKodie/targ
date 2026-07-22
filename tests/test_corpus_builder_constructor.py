"""
test_corpus_builder_constructor.py

Unit tests for RepresentativeCorpusBuilder constructor validation.

These tests verify that the builder rejects invalid configuration
before any document loading or chunk generation occurs.
"""

import sys
import os

# Adds the parent directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path

import pytest

from rag.corpus_builder import (
    CorpusChunkingConfig,
    CorpusDocumentSpec,
    RepresentativeCorpusBuilder,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def valid_specifications() -> tuple[CorpusDocumentSpec, ...]:

    return (
        CorpusDocumentSpec(
            corpus_id="targ",
            filename="targ.pdf",
            title="TARG",
            category="rag_research",
        ),
    )


# ---------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------


def test_constructor_rejects_missing_corpus_directory(
    tmp_path: Path,
    valid_specifications,
):
    """
    A missing corpus directory should raise FileNotFoundError.
    """

    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        RepresentativeCorpusBuilder(
            corpus_directory=missing_directory,
            specifications=valid_specifications,
        )


def test_constructor_rejects_non_directory_path(
    tmp_path: Path,
    valid_specifications,
):
    """
    A regular file is not a valid corpus directory.
    """

    file_path = tmp_path / "notes.txt"
    file_path.write_text("test")

    with pytest.raises(NotADirectoryError):
        RepresentativeCorpusBuilder(
            corpus_directory=file_path,
            specifications=valid_specifications,
        )


def test_constructor_rejects_duplicate_filenames(
    tmp_path: Path,
):
    """
    Every source filename must be unique.
    """

    specifications = (

        CorpusDocumentSpec(
            corpus_id="paper_one",
            filename="paper.pdf",
            title="Paper One",
            category="rag",
        ),

        CorpusDocumentSpec(
            corpus_id="paper_two",
            filename="paper.pdf",
            title="Paper Two",
            category="rag",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate filename",
    ):
        RepresentativeCorpusBuilder(
            corpus_directory=tmp_path,
            specifications=specifications,
        )


def test_constructor_rejects_duplicate_corpus_ids(
    tmp_path: Path,
):
    """
    Every corpus identifier must be unique.
    """

    specifications = (

        CorpusDocumentSpec(
            corpus_id="targ",
            filename="paper_one.pdf",
            title="Paper One",
            category="rag",
        ),

        CorpusDocumentSpec(
            corpus_id="targ",
            filename="paper_two.pdf",
            title="Paper Two",
            category="rag",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate corpus_id",
    ):
        RepresentativeCorpusBuilder(
            corpus_directory=tmp_path,
            specifications=specifications,
        )


# ---------------------------------------------------------------------
# Chunk configuration validation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_size",
    [
        0,
        -1,
    ],
)
def test_constructor_rejects_invalid_chunk_size(
    tmp_path: Path,
    valid_specifications,
    chunk_size: int,
):
    """
    Chunk size must be positive.
    """

    with pytest.raises(ValueError):
        RepresentativeCorpusBuilder(
            corpus_directory=tmp_path,
            specifications=valid_specifications,
            config=CorpusChunkingConfig(
                chunk_size=chunk_size,
            ),
        )


@pytest.mark.parametrize(
    "chunk_overlap",
    [
        -1,
        1000,
        1500,
    ],
)
def test_constructor_rejects_invalid_chunk_overlap(
    tmp_path: Path,
    valid_specifications,
    chunk_overlap: int,
):
    """
    Chunk overlap must satisfy:

        0 <= overlap < chunk_size
    """

    with pytest.raises(ValueError):
        RepresentativeCorpusBuilder(
            corpus_directory=tmp_path,
            specifications=valid_specifications,
            config=CorpusChunkingConfig(
                chunk_size=1000,
                chunk_overlap=chunk_overlap,
            ),
        )


@pytest.mark.parametrize(
    "minimum_chunk_size",
    [
        0,
        -5,
        1001,
    ],
)
def test_constructor_rejects_invalid_minimum_chunk_size(
    tmp_path: Path,
    valid_specifications,
    minimum_chunk_size: int,
):
    """
    Minimum chunk size must satisfy:

        0 < minimum_chunk_size <= chunk_size
    """

    with pytest.raises(ValueError):
        RepresentativeCorpusBuilder(
            corpus_directory=tmp_path,
            specifications=valid_specifications,
            config=CorpusChunkingConfig(
                chunk_size=1000,
                minimum_chunk_size=minimum_chunk_size,
            ),
        )