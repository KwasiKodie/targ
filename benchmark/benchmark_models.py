"""
benchmark_models.py

Immutable benchmark models for retrieval evaluation.

The benchmark defines a deterministic set of questions together with 
their expected evidence in the authoritative corpus.

Responsibilities
----------------
- Represent benchmark queries.
- Validate benchmark integrity
- Provide deterministic serialization.

These models do NOT:
- load benchmark files;
- perform retrieval;
- compute evaluation metrics.

Author: 
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations 

from dataclasses import asdict 
from dataclasses import dataclass 
from typing import Any 


# ---------------------------------------------------------
# Benchmark Query
# ---------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    """
    One benchmark question together with its expected evidence.
    """

    question: str 

    document_id: str 

    section_title: str 

    supporting_pages: tuple[int, ...]

    expected_answer_span: str 

    difficulty: str 

    topic: str 

    def __post_init__(self) -> None: 

        if not self.question.strip():
            raise ValueError(
                "question cannot be empty."
            )
        
        if not self.document_id.strip():
            raise ValueError(
                "document_id cannot be empty."
            )
        
        if not self.section_title.strip():
            raise ValueError(
                "section_title cannot be empty."
            )
        
        if len(self.supporting_pages) == 0:
            raise ValueError(
                "supporting_pages cannot be empty."
            )
        
        for page in self.supporting_pages: 

            if page < 1:
                raise ValueError(
                    "supporting_pages must contain positive integers."
                )
            
        if not self.expected_answer_span.strip():
            raise ValueError(
                "expected_answer_span cannot be empty."
            )
        
        allowed = {
            "Easy",
            "Medium",
            "Hard",
        }

        if self.difficulty not in allowed:
            raise ValueError(
                f"difficulty must be one of {sorted(allowed)}."
            )
        
        if not self.topic.strip():
            raise ValueError(
                "topic cannot be empty."
            )
        
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a JSON-serializable dictionary.
        """

        data = asdict(self)

        data["supporting_pages"] = list(
            self.supporting_pages
        )

        return data 
    

# --------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Benchmark:
    """
    Complete benchmark collection.
    """

    queries: tuple[BenchmarkQuery, ...]

    def __post_init__(self) -> None: 

        if len(self.queries) == 0:
            raise ValueError(
                "Benchmark must contain at least one query."
            )
        
        seen = set()

        for query in self.queries: 

            if not isinstance(
                query, 
                BenchmarkQuery,
            ):
                raise TypeError(
                    "queries must contain BenchmarkQuery objects."
                )
            
            key = (
                query.question,
                query.document_id,
            )

            if key in seen:
                raise ValueError(
                    "Duplicate benchmark query detected."
                )
            
            seen.add(key)

        
    def __len__(self) -> int:
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)
    
    def __getitem__(self, index):
        return self.queries[index]
    
    def to_dict(self) -> list[dict[str, Any]]:
        """
        Convert to a JSON-serializable list.
        """

        return [
            query.to_dict()
            for query in self.queries
        ]