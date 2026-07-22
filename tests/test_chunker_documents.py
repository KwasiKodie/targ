"""
Tests for chunker.chunk_documents()
"""

import pytest 

from corpus.chunker import Chunker 
from corpus.corpus_models import CorpusDocument 

def test_chunk_documents_single_document():

    chunker = Chunker()

    document = CorpusDocument(
        source="TARG",
        page=1,
        text="A" * 500,
    )

    chunks = chunker.chunk_documents(
        [document]
    )

    assert len(chunks) == 1

    assert chunks[0].source == "TARG"

    assert chunks[0].page == 1

def test_chunk_documents_multiple_documents():

    chunker = Chunker()

    documents = [

        CorpusDocument(
            source="TARG",
            page=1,
            text="A" * 500,
        ),

        CorpusDocument(
            source="PTES",
            page=2,
            text="B" * 500,
        ),
    ]

    chunks = chunker.chunk_documents(
        documents
    )

    assert len(chunks) == 2

    assert chunks[0].source == "TARG"

    assert chunks[1].source == "PTES"

def test_chunk_documents_empty_iterable():

    chunker = Chunker()

    with pytest.raises(
        ValueError,
        match="At least one document is required.",
    ): 
        
        chunker.chunk_documents([])

def test_chunk_documents_accepts_generator():

    chunker = Chunker()

    documents = (

        CorpusDocument(
            source=f"Doc{i}",
            page=1,
            text="A" * 500,
        )
        
        for i in range(3)
    )

    chunks = chunker.chunk_documents(
        documents 
    )

    assert len(chunks) == 3

    assert chunks[0].source == "Doc0"

    assert chunks[2].source == "Doc2"


def test_chunk_documents_preserves_order():

    chunker = Chunker()

    documents = [

        CorpusDocument(
            source="First",
            page=1,
            text="A" * 500,
        ),

        CorpusDocument(
            source="Second",
            page=1,
            text="B" * 500,
        ),

        CorpusDocument(
            source="Third",
            page=1,
            text="C" * 500,
        ),
    ]

    chunks = chunker.chunk_documents(
        documents
    )

    assert [

        chunk.source

        for chunk in chunks
    ] == [

        "First",

        "Second",

        "Third",
    ]