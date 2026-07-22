"""
vector_store_builder.py

Deterministic construction of the vector store used by the TARG
reproduction.

Responsibilities
----------------
* Validate builder configuration
* Convert immutable Chunk objects into LangChain Documents.
* Construct the configured vector store.
* Return the initialized vector store. 

The VectorStoreBuilder NEVER:

* Reads PDFs.
* Chunks documents.
* Performs retrieval.
* Generates answers.
* Reranks retrieved documents.

Author:
Retrieval as a Decision (TARG) Reproduction
"""

from __future__ import annotations 

from collections.abc import Iterable 
from typing import Sequence

from langchain_core.documents import Document 
from langchain_core.embeddings import Embeddings 
from langchain_core.vectorstores import VectorStore 

from corpus.corpus_models import Chunk 

class VectorStoreBuilder:
    """
    Deterministic builder for LangChain vector stores.
    """

    def __init__(
        self,
        *,
        embedding_model: Embeddings, 
        vector_store_class: type[VectorStore],
    ) -> None: 
        
        self.embedding_model = embedding_model
        self.vector_store_class = vector_store_class

        self._validate_constructor()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def build(
        self,
        chunks: Sequence[Chunk],
    ) -> VectorStore:
        """
        Build a vector store from validated corpus chunks.
        """

        self._validate_chunks(chunks)

        documents = self._build_documents(chunks)

        return self._build_vector_store(documents)
    
    def _build_documents(
        self,
        chunks: Sequence[Chunk],
    ) -> tuple[Document, ...]:
        """
        Convert validated Chunk objects into LangChain Documents. 

        Parameters
        ----------
        chunks 
            Validated Chunk objects.

        Returns
        -------
        tuple[Document, ...]
            Documents preserving the original odering and metadata.
        """

        return tuple(
            Document(
                page_content=chunk.text,
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        )
    
    def _build_vector_store(
        self,
        documents: tuple[Document, ...],
    ) -> VectorStore:
        """
        Construct the configured vector store. 

        Parameters
        ----------
        documents
            Validated LangChain documents to embed and index.

        Returns
        -------
        VectorStore
            Initialised vector store containing the documents.
        """

        return self.vector_store_class.from_documents(
            documents=list(documents),
            embedding=self.embedding_model,
        )
    
    # ---------------------------------------------------
    # Validation
    # ---------------------------------------------------

    def _validate_constructor(
        self,
    ) -> None:
        """
        Validate the injected embedding model and vector-store class.

        Raises
        ------
        TypeError
           If the embedding model is missing or does not implement the 
           LangChain Embeddings interface.

           If the vector-store class is missing, is not a class, or is not 
           a subclass of LangChain's VectorStore abstraction.
        """

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
        
        if not isinstance(
            self.vector_store_class,
            type,
        ):
            raise TypeError(
                "vector_store_class must be a VectorStore subclass."
            )
        
        if not issubclass(
            self.vector_store_class,
            VectorStore,
        ):
            raise TypeError(
                "vector_store_class must be a VectorStore subclass."
            )
    
    def _validate_chunks(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """
        Validate chunks before vectorisation.

        Raises
        ------
        TypeError
            If chunks is not a sequence of Chunk objects.

        ValueError
            If no chunks are supplied, a chunk contains empty text,
            or duplicate chunk IDs are present.
        """

        if not isinstance(chunks, Sequence):
            raise TypeError(
                "chunks must be a sequence of Chunk objects."
            )
        
        if len(chunks) == 0:
            raise ValueError(
                "At least one chunk is required."
            )
        
        seen_ids: set[str] = set()

        for chunk in chunks: 

            if not isinstance(chunk, Chunk):
                raise TypeError(
                    "All items must be Chunk objects."
                )
            
            # if not chunk.text.strip():
            #     raise ValueError(
            #         "Chunk text must not be empty."
            #     )
            
            if chunk.id in seen_ids:
                raise ValueError(
                    "Duplicate chunk IDs are not permitted."
                )
            
            seen_ids.add(chunk.id)