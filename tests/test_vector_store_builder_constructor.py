import pytest 

from langchain_core.embeddings import Embeddings 
from langchain_core.vectorstores import VectorStore 

from corpus.vector_store_builder import VectorStoreBuilder 

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
    
def test_constructor():

    builder = VectorStoreBuilder(
        embedding_model=FakeEmbeddings(),
        vector_store_class=FakeVectorStore,
    )

    assert builder.embedding_model.__class__ is FakeEmbeddings

    assert builder.vector_store_class is FakeVectorStore 


def test_constructor_none_embedding():

    with pytest.raises(
        TypeError, 
        match="embedding_model must not be None.",
    ):
        
        VectorStoreBuilder(
            embedding_model=None,
            vector_store_class=FakeVectorStore,
        )

def test_constructor_none_vector_store():

    with pytest.raises(
        TypeError,
        match="vector_store_class must not be None.",
    ):
        
        VectorStoreBuilder(
            embedding_model=FakeEmbeddings(),
            vector_store_class=None,
        )

@pytest.mark.parametrize(
    "value",
    [
        1,
        [],
        {},
        object(),
    ],
)
def test_constructor_invalid_embedding(value):

    with pytest.raises(
        TypeError,
        match="embedding_model must implement Embeddings.",
    ):
        
        VectorStoreBuilder(
            embedding_model=value, 
            vector_store_class=FakeVectorStore,
        )

@pytest.mark.parametrize(
    "value",
    [
        1,
        [],
        {},
        object(),
    ],
)
def test_constructor_invalid_vector_store(value):

    with pytest.raises(
        TypeError,
        match="vector_store_class must be a VectorStore subclass.",
    ):
        
        VectorStoreBuilder(
            embedding_model=FakeEmbeddings(),
            vector_store_class=value, 
        )

def _validate_constructor(self) -> None: 

    if self.embedding_model is None:
        raise TypeError(
            "embedding_model must not be None."
        )
    
    if not isinstance(
        self.embedding_model,
        Embeddings,
    ):
        
        raise TypeError(
            "embedding_model must implement Embeddings."
        )
    
    if self.vector_store_class is None: 
        raise TypeError(
            "vector_store_class must not be None."
        )
    
    if (
        not isinstance(self.vector_store_class, type)
        or not issubclass(
            self.vector_store_class,
            VectorStore,
        )
    ): 
        raise TypeError(
            "vector_store_class must be a VectorStore subclass."
        )