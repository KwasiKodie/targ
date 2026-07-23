"""
Unit tests for retrieval_inspector.py

These tests verify the deterministic retrieval inspection component used
by the TARG reproduction.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from benchmark.benchmark_models import BenchmarkQuery
from retrieval.retrieval_inspector import RetrievalInspector
from retrieval.retrieval_models import (
    RetrievedChunk,
    RetrievalInspection,
)
from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------
# Fake Vector Store
# ---------------------------------------------------------------------


class FakeVectorStore(VectorStore):
    """
    Deterministic fake vector store for unit testing.
    """

    def __init__(self, results=None):

        self.results = results or []

        self.last_query = None

        self.last_k = None

    def similarity_search_with_score(
        self,
        query,
        k=4,
        **kwargs,
    ):

        self.last_query = query
        self.last_k = k

        return self.results

    # -----------------------------------------------------------------
    # Abstract methods required by VectorStore
    # -----------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding,
        metadatas=None,
        **kwargs,
    ):
        return cls()

    def add_texts(
        self,
        texts,
        metadatas=None,
        **kwargs,
    ):
        return []

    def similarity_search(
        self,
        query,
        k=4,
        **kwargs,
    ):
        return []


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_query():

    return BenchmarkQuery(
        question="What is retrieval augmented generation?",
        document_id="rag_survey",
        section_title="Introduction",
        supporting_pages=(2,),
        expected_answer_span="Retrieval augments generation.",
        difficulty="Easy",
        topic="RAG",
    )


def make_document():

    return Document(
        page_content="Example chunk text.",
        metadata={
            "id": "chunk_1",
            "source": "rag_survey",
            "page": 2,
            "chunk": 0,
        },
    )


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------


def test_constructor():

    store = FakeVectorStore()

    inspector = RetrievalInspector(
        vector_store=store,
    )

    assert inspector.vector_store is store

    assert inspector.top_k == 5


def test_constructor_custom_top_k():

    inspector = RetrievalInspector(
        vector_store=FakeVectorStore(),
        top_k=10,
    )

    assert inspector.top_k == 10


def test_constructor_requires_vector_store():

    with pytest.raises(TypeError):

        RetrievalInspector(
            vector_store=None,
        )


def test_constructor_top_k_positive():

    with pytest.raises(ValueError):

        RetrievalInspector(
            vector_store=FakeVectorStore(),
            top_k=0,
        )


# ---------------------------------------------------------------------
# _build_chunk
# ---------------------------------------------------------------------


def test_build_chunk():

    inspector = RetrievalInspector(
        vector_store=FakeVectorStore(),
    )

    chunk = inspector._build_chunk(
        rank=1,
        document=(
            make_document(),
            0.15,
        ),
    )

    assert isinstance(
        chunk,
        RetrievedChunk,
    )

    assert chunk.rank == 1
    assert chunk.score == 0.15
    assert chunk.source == "rag_survey"
    assert chunk.page == 2
    assert chunk.chunk == 0
    assert chunk.chunk_id == "chunk_1"
    assert chunk.text == "Example chunk text."


# ---------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------


def test_inspect():

    document = make_document()

    store = FakeVectorStore(
        results=[
            (
                document,
                0.21,
            ),
        ]
    )

    inspector = RetrievalInspector(
        vector_store=store,
    )

    inspection = inspector.inspect(
        make_query(),
    )

    assert isinstance(
        inspection,
        RetrievalInspection,
    )

    assert inspection.query == (
        "What is retrieval augmented generation?"
    )

    assert inspection.expected_document == "rag_survey"

    assert inspection.expected_pages == (2,)

    assert len(
        inspection.retrieved_chunks
    ) == 1


def test_query_passed_to_vector_store():

    store = FakeVectorStore(
        results=[
            (
                make_document(),
                0.1,
            )
        ]
    )

    inspector = RetrievalInspector(
        vector_store=store,
        top_k=7,
    )

    query = make_query()

    inspector.inspect(query)

    assert (
        store.last_query
        == query.question
    )

    assert store.last_k == 7


def test_empty_retrieval_results():

    store = FakeVectorStore(
        results=[],
    )

    inspector = RetrievalInspector(
        vector_store=store,
    )

    with pytest.raises(ValueError):

        inspector.inspect(
            make_query(),
        )


def test_multiple_results_ranked_correctly():

    document = make_document()

    store = FakeVectorStore(
        results=[
            (
                document,
                0.10,
            ),
            (
                document,
                0.20,
            ),
            (
                document,
                0.30,
            ),
        ]
    )

    inspector = RetrievalInspector(
        vector_store=store,
    )

    inspection = inspector.inspect(
        make_query(),
    )

    assert [
        c.rank
        for c in inspection.retrieved_chunks
    ] == [1, 2, 3]