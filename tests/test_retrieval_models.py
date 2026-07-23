"""
Unit tests for retrieval_models.py

These tests verify the immutable retrieval inspection models used by the
TARG reproduction.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

import pytest

from retrieval.retrieval_models import (
    RetrievedChunk,
    RetrievalInspection,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_chunk(
    *,
    rank: int = 1,
):
    return RetrievedChunk(
        rank=rank,
        score=0.123,
        source="targ",
        page=3,
        chunk=0,
        chunk_id=f"chunk_{rank}",
        text="Example retrieved text.",
    )


# ---------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------


def test_retrieved_chunk_constructs():

    chunk = make_chunk()

    assert chunk.rank == 1
    assert chunk.score == 0.123
    assert chunk.source == "targ"
    assert chunk.page == 3
    assert chunk.chunk == 0
    assert chunk.chunk_id == "chunk_1"
    assert chunk.text == "Example retrieved text."


def test_retrieved_chunk_rank_must_be_positive():

    with pytest.raises(ValueError):
        make_chunk(rank=0)


def test_retrieved_chunk_score_must_be_numeric():

    with pytest.raises(TypeError):

        RetrievedChunk(
            rank=1,
            score="bad",
            source="targ",
            page=1,
            chunk=0,
            chunk_id="id",
            text="text",
        )


def test_retrieved_chunk_source_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievedChunk(
            rank=1,
            score=0.5,
            source="",
            page=1,
            chunk=0,
            chunk_id="id",
            text="text",
        )


def test_retrieved_chunk_page_must_be_positive():

    with pytest.raises(ValueError):

        RetrievedChunk(
            rank=1,
            score=0.5,
            source="targ",
            page=0,
            chunk=0,
            chunk_id="id",
            text="text",
        )


def test_retrieved_chunk_chunk_must_be_non_negative():

    with pytest.raises(ValueError):

        RetrievedChunk(
            rank=1,
            score=0.5,
            source="targ",
            page=1,
            chunk=-1,
            chunk_id="id",
            text="text",
        )


def test_retrieved_chunk_chunk_id_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievedChunk(
            rank=1,
            score=0.5,
            source="targ",
            page=1,
            chunk=0,
            chunk_id="",
            text="text",
        )


def test_retrieved_chunk_text_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievedChunk(
            rank=1,
            score=0.5,
            source="targ",
            page=1,
            chunk=0,
            chunk_id="id",
            text="",
        )


def test_retrieved_chunk_to_dict():

    chunk = make_chunk()

    assert chunk.to_dict() == {
        "rank": 1,
        "score": 0.123,
        "source": "targ",
        "page": 3,
        "chunk": 0,
        "chunk_id": "chunk_1",
        "text": "Example retrieved text.",
    }


# ---------------------------------------------------------------------
# RetrievalInspection
# ---------------------------------------------------------------------


def test_retrieval_inspection_constructs():

    inspection = RetrievalInspection(
        query="What is TARG?",
        expected_document="targ",
        expected_pages=(2,),
        retrieved_chunks=(
            make_chunk(rank=1),
            make_chunk(rank=2),
        ),
    )

    assert inspection.query == "What is TARG?"
    assert inspection.expected_document == "targ"
    assert inspection.expected_pages == (2,)
    assert len(inspection.retrieved_chunks) == 2


def test_query_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="",
            expected_document="targ",
            expected_pages=(1,),
            retrieved_chunks=(make_chunk(),),
        )


def test_expected_document_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="Q",
            expected_document="",
            expected_pages=(1,),
            retrieved_chunks=(make_chunk(),),
        )


def test_expected_pages_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="Q",
            expected_document="targ",
            expected_pages=(),
            retrieved_chunks=(make_chunk(),),
        )


def test_expected_pages_must_be_positive():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="Q",
            expected_document="targ",
            expected_pages=(0,),
            retrieved_chunks=(make_chunk(),),
        )


def test_retrieved_chunks_cannot_be_empty():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="Q",
            expected_document="targ",
            expected_pages=(1,),
            retrieved_chunks=(),
        )


def test_retrieved_chunks_must_contain_retrieved_chunk():

    with pytest.raises(TypeError):

        RetrievalInspection(
            query="Q",
            expected_document="targ",
            expected_pages=(1,),
            retrieved_chunks=("bad",),
        )


def test_chunk_ranks_must_be_strictly_increasing():

    with pytest.raises(ValueError):

        RetrievalInspection(
            query="Q",
            expected_document="targ",
            expected_pages=(1,),
            retrieved_chunks=(
                make_chunk(rank=2),
                make_chunk(rank=1),
            ),
        )


def test_to_dict():

    inspection = RetrievalInspection(
        query="What is TARG?",
        expected_document="targ",
        expected_pages=(2, 3),
        retrieved_chunks=(
            make_chunk(rank=1),
            make_chunk(rank=2),
        ),
    )

    data = inspection.to_dict()

    assert data["query"] == "What is TARG?"
    assert data["expected_document"] == "targ"
    assert data["expected_pages"] == [2, 3]
    assert len(data["retrieved_chunks"]) == 2
    assert data["retrieved_chunks"][0]["rank"] == 1
    assert data["retrieved_chunks"][1]["rank"] == 2