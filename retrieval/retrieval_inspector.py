"""
retrieval_inspector.py

Retrieval inspection for TARG reproduction.

The RetrievalInspector executes a benchmark query against a vector store
and records the retrieved evidence without performing any evaluation.

Responsibilities
----------------
- Execute similarity search.
- Preserve retrieval rank.
- Preserve retrieval metadata.
- Build immutable RetrievalInspection objects.

The RetrievalInspection does NOT:
- compute retrieval metrics;
- assess relevance;
- compare against ground truth;
- load benchmark datasets.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations 

from langchain_core.vectorstores import VectorStore 

from benchmark.benchmark_models import BenchmarkQuery

from retrieval.retrieval_models import (
    RetrievedChunk,
    RetrievalInspection,
)

class RetrievalInspector:
    """
    Executes benchmark queries against a vector store.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore, 
        top_k: int = 5,
    ) -> None: 
        
        self.vector_store = vector_store 
        self.top_k = top_k 

        self._validate_constructor()

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------

    def inspect(
        self,
        benchmark_query: BenchmarkQuery,
    ) -> RetrievalInspection:
        """
        Retrieve the top-k chunks for a benchmark query.
        """

        if not isinstance(benchmark_query, BenchmarkQuery):
            raise TypeError(
                "benchmark_query must be a BenchmarkQuery."
            )
        
        documents = self._retrieve(
            benchmark_query.question,
        )

        retrieved_chunks = tuple(
            self._build_chunk(
                rank=i,
                document=document,
            )
            for i, document in enumerate(
                documents,
                start=1,
            )
        )

        return RetrievalInspection(
            query=benchmark_query.question,
            expected_document=benchmark_query.document_id,
            expected_pages=tuple(
                benchmark_query.supporting_pages
            ),
            retrieved_chunks=retrieved_chunks,
        )
    
    # -------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------

    def _retrieve(
        self,
        query: str,
    ):
        """
        Execute similarity search.
        """

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k,
        )
    
    # -------------------------------------------------------
    # Conversion
    # -------------------------------------------------------

    def _build_chunk(
        self,
        *,
        rank: int, 
        document,
    ) -> RetrievedChunk:
        """
        Convert a LangChain retrieval result into RetrievedChunk.
        """

        doc, score = document 

        metadata = doc.metadata 

        return RetrievedChunk(
            rank=rank,
            score=float(score),
            source=metadata["source"],
            page=metadata["page"],
            chunk=metadata["chunk"],
            chunk_id=metadata["id"],
            text=doc.page_content,
        )
    
    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def _validate_constructor(
        self,
    ) -> None:
        
        if self.vector_store is None:
            raise TypeError(
                "vector_store cannot be None."
            )
        
        if not isinstance(
            self.vector_store,
            VectorStore,
        ):
            raise TypeError(
                "vector_store must inherit from VectorStore."
            )
        
        if not isinstance(
            self.top_k,
            int,
        ):
            raise TypeError(
                "top_k must be an integer."
            )
        
        if self.top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )