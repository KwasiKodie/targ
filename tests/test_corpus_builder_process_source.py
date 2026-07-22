"""
Unit tests for RepresentativeCorpusBuilder._process_source().
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.corpus_builder import (
    CorpusChunkingConfig,
    CorpusDocumentSpec,
    RepresentativeCorpusBuilder,
)

def make_builder(tmp_path):

    pdf = tmp_path / "targ.pdf"
    pdf.write_text("placeholder")

    builder = RepresentativeCorpusBuilder(
        corpus_directory=tmp_path,
        specifications=(
            CorpusDocumentSpec(
                corpus_id="targ",
                filename="targ.pdf",
                title="TARG",
                category="rag",
            ),
        ),
        config=CorpusChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            minimum_chunk_size=10,
        ),
    )

    return builder

@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_filters_small_chunks(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example page",
        metadata={"page": 0},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "small",
        "this chunk is definitely long enough",
    ]

    specification = builder._specifications[0]

    chunks, summary = builder._process_source(
        specification
    )

    assert len(chunks) == 1
    assert (
        chunks[0].page_content
        == "this chunk is definitely long enough"
    )

@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_records_chunk_size_metadata(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example",
        metadata={"page": 0},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "A" * 50,
    ]

    specification = builder._specifications[0]

    chunks, _ = builder._process_source(
        specification
    )

    metadata = chunks[0].metadata

    assert metadata["chunk_size"] == 100

def test_builder_configures_text_splitter(tmp_path):

    config = CorpusChunkingConfig(
        chunk_size=1000,
        chunk_overlap=100,
        minimum_chunk_size=200,
    )

    specification = CorpusDocumentSpec(
        corpus_id="targ",
        filename="targ.pdf",
        title="TARG",
        category="rag",
    )

    (tmp_path / "targ.pdf").write_text("placeholder")

    builder = RepresentativeCorpusBuilder(
        corpus_directory=tmp_path,
        specifications=(specification,),
        config=config,
    )

    assert builder._splitter._chunk_size == 1000
    assert builder._splitter._chunk_overlap == 100


@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_populates_metadata(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example page",
        metadata={"page": 4},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "This is a sufficiently long chunk."
    ]

    specification = builder._specifications[0]

    chunks, _ = builder._process_source(
        specification
    )

    metadata = chunks[0].metadata

    assert metadata["corpus_id"] == "targ"
    assert metadata["title"] == "TARG"
    assert metadata["category"] == "rag"
    assert metadata["page"] == 5
    assert metadata["page_chunk_index"] == 0
    assert metadata["chunk"] == 0
    assert metadata["character_count"] == len(
        chunks[0].page_content
    )

@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_generates_stable_ids(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example",
        metadata={"page": 0},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "This is a sufficiently long chunk."
    ]

    specification = builder._specifications[0]

    first, _ = builder._process_source(
        specification
    )

    second, _ = builder._process_source(
        specification
    )

    assert (
        first[0].metadata["id"]
        ==
        second[0].metadata["id"]
    )

@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_generates_unique_ids(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example",
        metadata={"page": 0},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "Chunk number one is sufficiently long.",
        "Chunk number two is also sufficiently long.",
    ]

    specification = builder._specifications[0]

    chunks, _ = builder._process_source(
        specification
    )

    ids = [
        chunk.metadata["id"]
        for chunk in chunks
    ]

    assert len(ids) == len(set(ids))

@patch("rag.corpus_builder.PyPDFLoader")
def test_process_source_generates_deterministic_fingerprint(
    loader_cls,
    tmp_path,
):

    builder = make_builder(tmp_path)

    page = Document(
        page_content="Example",
        metadata={"page": 0},
    )

    loader = MagicMock()
    loader.load.return_value = [page]
    loader_cls.return_value = loader

    builder._splitter = MagicMock()
    builder._splitter.split_text.return_value = [
        "A deterministic chunk."
    ]

    specification = builder._specifications[0]

    first, _ = builder._process_source(
        specification
    )

    second, _ = builder._process_source(
        specification
    )

    fp1 = builder._create_corpus_fingerprint(
        first
    )

    fp2 = builder._create_corpus_fingerprint(
        second
    )

    assert fp1 == fp2