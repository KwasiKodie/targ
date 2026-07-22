"""
Tests for chunk_metadata.py.
"""

from types import MappingProxyType 

import pytest 

from corpus.chunk_metadata import ChunkMetadata 

def test_chunk_id_is_deterministic():

    first = ChunkMetadata.chunk_id(
        source="TARG",
        page=6,
        chunk_index=3,
    )

    second = ChunkMetadata.chunk_id(
        source="TARG",
        page=6,
        chunk_index=3,
    )

    assert first == second 

    assert first == "TARG_p006_c003"

def test_chunk_id_changes_with_inputs():

    ids = {

        ChunkMetadata.chunk_id(
            source="TARG",
            page=1,
            chunk_index=0,
        ),

        ChunkMetadata.chunk_id(
            source="TARG",
            page=2,
            chunk_index=0,
        ),

        ChunkMetadata.chunk_id(
            source="PTES",
            page=1,
            chunk_index=0,
        ),

        ChunkMetadata.chunk_id(
            source="TARG",
            page=1,
            chunk_index=1,
        ),
    }

def test_build_metadata():

    metadata = ChunkMetadata.build(

        source="PTES",

        page=8,

        chunk_index=5,
    )

    assert metadata["id"] == "PTES_p008_c005"

    assert metadata["source"] == "PTES"

    assert metadata["page"] == 8

    assert metadata["chunk"] == 5


def test_metadata_is_immutable():

    metadata = ChunkMetadata.build(

        source="PTES",

        page=1,

        chunk_index=0,
    )

    assert isinstance(
        metadata,
        MappingProxyType,
    )

    with pytest.raises(
        TypeError,
    ):
        
        metadata["page"] = 10


def test_extra_metadata_is_merged():

    metadata = ChunkMetadata.build(

        source="TARG",

        page=4, 

        chunk_index=2, 

        extra={
            "section": "Evaluation",
            "paper": "TARG",
        },
    )

    assert metadata["section"] == "Evaluation"

    assert metadata["paper"] == "TARG"


def test_extra_metadata_cannot_override_standard_fields():

    metadata = ChunkMetadata.build(

        source="PTES",

        page=3,

        chunk_index=1, 

        extra={
            "source": "WRONG",
            "page": 90,
            "chunk": 999,
            "id": "incorrect",
        },
    )

    assert metadata["source"] == "PTES"

    assert metadata["page"] == 3 

    assert metadata["chunk"] == 1

    assert metadata["id"] == "PTES_p003_c001"

def test_empty_extra_metadata():

    metadata = ChunkMetadata.build(

        source="SELF-RAG",

        page=2,

        chunk_index=0,

        extra={},
    )

    assert len(metadata) == 4


def test_none_extra_metadata():

    metadata = ChunkMetadata.build(

        source="SELF-RAG",

        page=2, 

        chunk_index=0
    )

    assert metadata["id"] == "SELF-RAG_p002_c000"