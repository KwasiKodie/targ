"""
test_corpus_builder_build.py

Unit tests for RepresentativeCorpusBuilder.build().
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from rag.corpus_builder import (
    CorpusBuildResult,
    CorpusChunkingConfig,
    CorpusDocumentSpec,
    RepresentativeCorpusBuilder,
    CorpusSourceSummary
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


@pytest.fixture
def specification():

    return CorpusDocumentSpec(
        corpus_id="targ",
        filename="targ.pdf",
        title="Retrieval as a Decision",
        category="rag",
    )


@pytest.fixture
def config():

    return CorpusChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        minimum_chunk_size=10,
    )


def make_builder(
    tmp_path,
    specification,
    config,
):

    pdf = tmp_path / specification.filename
    pdf.write_text("placeholder")

    return RepresentativeCorpusBuilder(
        corpus_directory=tmp_path,
        specifications=(specification,),
        config=config,
    )


# ---------------------------------------------------------------------
# Missing source file
# ---------------------------------------------------------------------


def test_build_raises_when_source_file_missing(
    tmp_path,
    specification,
    config,
):
    """
    Every declared source file must exist.
    """

    builder = RepresentativeCorpusBuilder(
        corpus_directory=tmp_path,
        specifications=(specification,),
        config=config,
    )

    with pytest.raises(FileNotFoundError):
        builder.build()


# ---------------------------------------------------------------------
# Empty PDF
# ---------------------------------------------------------------------


@patch.object(
    RepresentativeCorpusBuilder,
    "_process_source",
)
def test_build_raises_for_empty_pdf(
    process_source,
    tmp_path,
    specification,
    config,
):
    """
    build() should propagate failures from empty PDFs.
    """

    builder = make_builder(
        tmp_path,
        specification,
        config,
    )

    process_source.side_effect = ValueError(
        "No pages were extracted."
    )

    with pytest.raises(
        ValueError,
        match="No pages were extracted",
    ):
        builder.build()


# ---------------------------------------------------------------------
# Valid build
# ---------------------------------------------------------------------


@patch.object(
    RepresentativeCorpusBuilder,
    "_process_source",
)
def test_build_returns_build_result(
    process_source,
    tmp_path,
    specification,
    config,
):
    """
    A successful build should return CorpusBuildResult.
    """

    builder = make_builder(
        tmp_path,
        specification,
        config,
    )

    document = Document(
        page_content="Example chunk.",
        metadata={
            "id": "chunk-1",
        },
    )

    summary = CorpusSourceSummary(
        corpus_id=specification.corpus_id,
        source=specification.filename,
        title=specification.title,
        page_count=1,
        extracted_character_count=13,
        chunk_count=1,
    )

    process_source.return_value = (
        (document,),
        summary,
    )

    result = builder.build()

    assert isinstance(
        result,
        CorpusBuildResult,
    )

    assert len(result.documents) == 1
    assert result.documents[0] == document


# ---------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------


@patch.object(
    RepresentativeCorpusBuilder,
    "_process_source",
)
def test_build_populates_summary_statistics(
    process_source,
    tmp_path,
    specification,
    config,
):
    """
    Build summaries should reflect processed sources.
    """

    builder = make_builder(
        tmp_path,
        specification,
        config,
    )

    document = Document(
        page_content="Example chunk.",
        metadata={
            "id": "chunk-1",
        },
    )

    summary = CorpusSourceSummary(
        corpus_id=specification.corpus_id,
        source=specification.filename,
        title=specification.title,
        page_count=1,
        extracted_character_count=13,
        chunk_count=1,
)

    process_source.return_value = (
        (document,),
        summary,
    )

    result = builder.build()

    assert len(result.source_summaries) == 1

    assert result.total_source_count == 1
    assert result.total_page_count == 1
    assert result.total_chunk_count == 1
    assert result.total_character_count == len(document.page_content)

    summary = result.source_summaries[0]

    assert summary.page_count == 1
    assert summary.chunk_count == 1
    assert summary.extracted_character_count == 13


# ---------------------------------------------------------------------
# Fingerprint determinism
# ---------------------------------------------------------------------


@patch.object(
    RepresentativeCorpusBuilder,
    "_process_source",
)
def test_build_is_deterministic(
    process_source,
    tmp_path,
    specification,
    config,
):
    """
    Identical builds should produce identical fingerprints.
    """

    builder = make_builder(
        tmp_path,
        specification,
        config,
    )

    document = Document(
        page_content="Example chunk.",
        metadata={
            "id": "chunk-1",
        },
    )

    summary = CorpusSourceSummary(
    corpus_id=specification.corpus_id,
    source=specification.filename,
    title=specification.title,
    page_count=1,
    extracted_character_count=13,
    chunk_count=1,
)

    process_source.return_value = (
        (document,),
        summary,
    )

    first = builder.build()
    second = builder.build()

    assert (
        first.corpus_fingerprint
        ==
        second.corpus_fingerprint
    )


# ---------------------------------------------------------------------
# Fingerprint changes
# ---------------------------------------------------------------------


@patch.object(
    RepresentativeCorpusBuilder,
    "_process_source",
)
def test_build_fingerprint_changes_when_corpus_changes(
    process_source,
    tmp_path,
    specification,
    config,
):
    """
    Any corpus content change should alter the fingerprint.
    """

    builder = make_builder(
        tmp_path,
        specification,
        config,
    )

    summary = CorpusSourceSummary(
    corpus_id=specification.corpus_id,
    source=specification.filename,
    title=specification.title,
    page_count=1,
    extracted_character_count=13,
    chunk_count=1,
)

    first_document = Document(
        page_content="Original chunk.",
        metadata={
            "id": "chunk-1",
        },
    )

    second_document = Document(
        page_content="Modified chunk.",
        metadata={
            "id": "chunk-1",
        },
    )

    process_source.return_value = (
        (first_document,),
        summary,
    )

    first = builder.build()

    process_source.return_value = (
        (second_document,),
        summary,
    )

    second = builder.build()

    assert (
        first.corpus_fingerprint
        !=
        second.corpus_fingerprint
    )