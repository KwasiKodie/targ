"""
Tests for VectorStoreBuilder._build_documents().
"""

from types import MappingProxyType 

import pytest 
from langchain_core.documents import Document 
from langchain_core.embeddings import Embeddings 
from langchain_core.vectorstores import VectorStore 

from corpus.corpus_models import Chunk 
from corpus.vector_store_builder import VectorStoreBuilder 

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
    

# --------------------------------------------------
# Fixtures
# --------------------------------------------------

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
        text="Chunk one.",
        metadata=MappingProxyType(
            {
                "id": "TARG_p001_c000",
                "source": "TARG",
                "page": 1,
                "chunk": 0,
            }
        ),
    )

def test_build_documents_single_chunk(
    builder,
    chunk,
):
    
    documents = builder._build_documents(
        [chunk],
    )

    assert len(documents) == 1

    assert isinstance(
        documents[0],
        Document,
    )

    assert documents[0].page_content == chunk.text 


def test_build_documents_metadata(
    builder,
    chunk,
):
    
    document = builder._build_documents(
        [chunk],
    )[0]

    assert document.metadata == dict(
        chunk.metadata,
    )


def test_build_documents_multiple_chunks(
    builder,
):
    
    chunks = []

    for i in range(5):

        chunks.append(
            Chunk(
                id=f"id_{i}",
                source="PTES",
                page=i + 1,
                chunk_index=i,
                text=f"text {i}",
                metadata=MappingProxyType(
                    {
                        "id": f"id_{i}",
                        "source": "PTES",
                        "page": i + 1,
                        "chunk": i,
                    }
                ),
            )
        )

    documents = builder._build_documents(
        chunks,
    )

    assert len(documents) == 5

def test_build_documents_preserves_order(
    builder,
):
    
    chunks = []

    for i in range(5):

        chunks.append(
            Chunk(
                id=f"id_{i}",
                source="TARG",
                page=1,
                chunk_index=i,
                text=f"text {i}",
                metadata=MappingProxyType(
                    {
                        "id": f"id_{i}",
                        "source": "TARG",
                        "page": 1,
                        "chunk": i,
                    }
                ),
            )
        )

    documents = builder._build_documents(
        chunks,
    )

    for i in range(5):

        assert documents[i].page_content == f"text {i}"


def test_build_documents_metadata_is_copied(
        builder,
        chunk,
):
    
    document = builder._build_documents(
        [chunk],
    )[0]

    assert document.metadata is not chunk.metadata 


def test_build_documents_returns_tuple(
    builder,
    chunk,
):
    
    documents = builder._build_documents(
        [chunk],
    )

    assert isinstance(
        documents,
        tuple,
    )