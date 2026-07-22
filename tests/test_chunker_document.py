"""
Tests for Chunker.chunk_document()
"""

from corpus.chunker import Chunker 
from corpus.corpus_models import CorpusDocument 

def test_chunk_document_single_page():

    chunker = Chunker()

    document = CorpusDocument(
        source="TARG",
        page=1,
        text="A" * 500,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1

    assert chunks[0].source == "TARG"

    assert chunks[0].page == 1

    assert chunks[0].text == document.text 

def test_chunk_document_multiple_chunks():

    chunker = Chunker()

    document = CorpusDocument(
        source="PTES",
        page=5,
        text="A" * 2500,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 3

    assert all(
        chunk.page == 5
        for chunk in chunks 
    )

def test_chunk_document_ids_are_deterministic():

    chunker = Chunker()

    document = CorpusDocument(
        source="TARG",
        page=6,
        text="A" * 2500,
    )

    first = chunker.chunk_document(document)

    second = chunker.chunk_document(document)

    assert [
        c.id
        for c in first
    ] == [
        c.id
        for c in second 
    ]

def test_chunk_document_metadata():

    chunker = Chunker()

    document = CorpusDocument(
        source="PTES",
        page=4,
        text="A" * 1200,
    )

    chunks = chunker.chunk_document(document)

    for index, chunk in enumerate(chunks):

        assert chunk.metadata["source"] == "PTES"

        assert chunk.metadata["page"] == 4

        assert chunk.metadata["chunk"] == index 

        assert chunk.metadata["id"] == chunk.id 


def test_chunk_document_chunk_indexes_are_sequential():

    chunker = Chunker()

    document = CorpusDocument(
        source="NIST",
        page=8,
        text="A" * 3200,
    )

    chunks = chunker.chunk_document(document)

    assert [
        chunk.chunk_index
        for chunk in chunks
    ] == list(range(len(chunks)))


def test_chunk_document_source_propagated():

    chunker = Chunker()

    document = CorpusDocument(
        source="SELF-RAG",
        page=3,
        text="A" * 800,
    )

    chunks = chunker.chunk_document(document)

    assert all(
        chunk.source == "SELF-RAG"
        for chunk in chunks 
    )

def test_chunk_document_page_propagated():

    chunker = Chunker()

    document = CorpusDocument(
        source="Adaptive-RAG",
        page=12,
        text="A" * 800,
    )

    chunks = chunker.chunk_document(document)

    assert all(
        chunk.page == 12
        for chunk in chunks
    )