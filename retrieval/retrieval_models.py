"""
retrieval_models.py

Immutable data models for retrieval inspection.

These models capture the output of the retrieval stage before any
manual relevance assessment or metric computation. They provide a 
deterministic representation of retrieved evidence for each benchmark 
query. 

Responsibilities
----------------
- Represent retrieved chunks.
- Represent retrieval inspection results.
- Provide deterministic serialization.

These models do NOT:
- perform retrieval;
- compute evaluation metrics;
- assess relevance.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass 
from typing import Any 

# ----------------------------------------------------------
# Retrieved Chunk
# ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """
    One retrieved chunk returned by the vector store.
    """

    rank: int 

    score: float 

    source: str

    page: int 

    chunk: int 

    chunk_id: str 

    text: str 

    def __post_init__(self) -> None: 

        if self.rank < 1:
            raise ValueError("rank must be >= 1.")
        
        if not isinstance(self.score, (int, float)):
            raise TypeError("score must be numeric.")
        
        if not self.source.strip():
            raise ValueError("source cannot be empty.")
        
        if self.page < 1:
            raise ValueError("page must be >= 1.")
        
        if self.chunk < 0:
            raise ValueError("chunk must be >= 0.")
        
        if not self.chunk_id.strip():
            raise ValueError("chunk_id cannot be empty.")
        
        if not self.text.strip():
            raise ValueError("text cannot be empty.")
        
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serialization dictionary.
        """
        return asdict(self)
    
# -----------------------------------------------------------------------
# Retrieval Inspection
# -----------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievalInspection:
    """
    Retrieval results for one benchmark query.
    """

    query: str 

    expected_document: str 

    expected_pages: tuple[int, ...]

    retrieved_chunks: tuple[RetrievedChunk, ...]

    def __post_init__(self) -> None: 

        if not self.query.strip():
            raise ValueError("query cannot be empty.")
        
        if not self.expected_document.strip():
            raise ValueError("expected_document cannot be empty.")
        
        if len(self.expected_pages) == 0:
            raise ValueError(
                "expected_pages must contain at least one page."
            )
        
        for page in self.expected_pages:

            if page < 1:
                raise ValueError(
                    "expected_pages must contain positive integers."
                )
            
        if len(self.retrieved_chunks) == 0:
            raise ValueError(
                "retrieved_chunks cannot be empty."
            )
        
        previous_rank = 0

        for chunk in self.retrieved_chunks:

            if not isinstance(chunk, RetrievedChunk):
                raise TypeError(
                    "retrieved_chunks must contain RetrievedChunk objects."
                )
            
            if chunk.rank <= previous_rank:
                raise ValueError(
                    "Retrieved chunk ranks must be strictly increasing."
                )
            
            previous_rank = chunk.rank

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.
        """

        return {
            "query": self.query,
            "expected_document": self.expected_document,
            "expected_pages": list(self.expected_pages),
            "retrieved_chunks": [
                chunk.to_dict()
                for chunk in self.retrieved_chunks
            ]
        }