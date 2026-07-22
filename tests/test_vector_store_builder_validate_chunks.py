"""
Tests for VectorStoreBuilder._validate_chunks().
"""

from types import MappingProxyType 

import pytest 
from langchain_core.embeddings import Embeddings 
from langchain_core.vectorstores import VectorStore 

from corpus.corpus_models import Chunk 
from vector_store_builder import VectorStoreBuilder 

# -----------------------------------------------------
# Test doubles
# -----------------------------------------------------

class FakeEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]
    
    def embed_query(self, text):
        return [0.0] * 4
    
class FakeVectorStore(VectorStore):

    @classmethod 
    def from_documents(
        cls,
        documents,
        embedding,
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
    

# -----------------------------------------------
# Fixtures
# -----------------------------------------------

@pytest.fixture
def builder():

    return VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    )

@pytest.fixture 
def chunk():

    return Chunk(
        id="TARG_p001_c000",
        source="TARG",
        page=1,
        chunk_index=0,
        text="This is a chunk.",
        metadata=MappingProxyType(
            {
                "id": "TARG_p001_c000",
                "source": "TARG",
                "page": 1,
                "chunk": 0,
            }
        ),
    )

# --------------------------------------------------
# Valid input
# --------------------------------------------------

def test_validate_chunks(builder, chunk):

    builder._validate_chunks(
        [chunk],
    )

def test_validate_multiple_chunks(builder, chunk):

    chunk2 = Chunk(
        id="TARG_p001_c001",
        source="TARG",
        page=1,
        chunk_index=1,
        text="Another chunk.",
        metadata=MappingProxyType(
            {
                "id": "TARG_p001_c001",
                "source": "TARG",
                "page": 1,
                "chunk": 1,
            }
        ),
    )

    builder._validate_chunks(
        [
            chunk,
            chunk2,
        ],
    )

# ----------------------------------------------------------
# Invalid container
# ----------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        {"a": 1},
        {1,2},
        {c for c in []},
    ],
)
def test_validate_chunks_invalid_container(
    builder, 
    value, 
): 
    
    with pytest.raises(
        TypeError, 
        match="chunks must be a sequence of Chunk objects.",
    ):
        
        builder._validate_chunks(
            value,
        )

# ------------------------------------------------------
# Empty sequence
# ------------------------------------------------------

def test_validate_chunks_empty(builder):

    with pytest.raises(
        ValueError,
        match="At least one chunk is required.",
    ): 
        
        builder._validate_chunks(
            [],
        )

# ------------------------------------------------------
# Invalid element 
# ------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        {},
        [],
        "chunk",
    ],
)
def test_validate_chunks_invalid_element(
    builder, 
    value,
): 
    
    with pytest.raises(
        TypeError, 
        match="All items must be Chunk objects.",
    ):
        
        builder._validate_chunks(
            [value],
        )

def test_validate_chunks_mixed_types(
    builder, 
    chunk,
): 
    
    with pytest.raises(
        TypeError, 
        match="All items must be Chunk objects.",
    ):
        
        builder._validate_chunks(
            [
                chunk,
                "invalid",
            ],
        )

# -----------------------------------------------------
# Empty text
# -----------------------------------------------------

def test_validate_chunks_duplicate_ids(
    builder,
):
    
    chunk1 = Chunk(
        id="duplicate",
        source="PTES",
        page=1,
        chunk_index=0,
        text="Chunk one.",
        metadata=MappingProxyType(
            {
                "id": "duplicate",
                "source": "PTES",
                "page": 1,
                "chunk": 0,
            }
        ),
    )

    chunk2 = Chunk(
        id="duplicate",
        source="PTES",
        page=2,
        chunk_index=0,
        text="Chunk two.",
        metadata=MappingProxyType(
            {
                "id": "duplicate",
                "source": "PTES",
                "page": 2, 
                "chunk": 0,
            }
        ),
    )

    with pytest.raises(
        ValueError,
        match="Duplicate chunk IDs are not permitted.",
    ): 
        
        builder._validate_chunks(
            [
                chunk1,
                chunk2,
            ],
        )

# --------------------------------------------------------------
# Unique IDs
# --------------------------------------------------------------

def test_validate_chunks_unique_ids(builder):

    chunks = []

    for i in range(5):

        chunks.append(
            Chunk(
                id=f"TARG_{i}",
                source="TARG",
                page=i + 1,
                chunk_index=i,
                text=f"Chunk {i}",
                metadata=MappingProxyType(
                    {
                        "id":f"TARG_{i}",
                        "source": "TARG",
                        "page": i + 1,
                        "chunk": i,
                    }
                ),
            )
        )

    builder._validate_chunks(
        chunks, 
    )