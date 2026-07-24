"""
retrieval_verification.py

Research-grade retrieval verification for TARG.

This experiment validates that the retrieval subsystem
returns the expected evidence for benchmark questions
before evaluating adaptive retrieval gating.

Responsibilities
----------------
1. Execute benchmark retrieval queries.
2. Preserve complete retrieval evidence.
3. Compute retrieval-quality metrics.
4. Produce immutable experiment summaries.

The runner NEVER 

- generates answers
- computes uncertainty
- calibrates thresholds
- applies retrieval gating
- evaluates generated answers 

Author:
TARG Reproduction 
"""

from __future__ import annotations

from dataclasses import dataclass 
from math import isfinite 
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable 

from benchmark.benchmark_models import Benchmark, BenchmarkQuery

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)

BaseRetriever = retrieval.BaseRetriever
RetrievalResult = retrieval.RetrievalResult
RetrievedDocument = retrieval.RetrievedDocument 


# ----------------------------------------------------------
# Benchmark
# ----------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RetrievalBenchmark: 

    """
    One benchmark retrieval query
    """

    benchmark_id: str

    query: str 

    expected_sources: tuple[str, ...]

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "benchmark_id",
            self._validate_text(
                self.benchmark_id,
                "benchmark_id",
            ),
        )

        object.__setattr__(
            self,
            "query",
            self._validate_text(
                self.query,
                "query",
            ),
        )

        if not self.expected_sources:
            raise ValueError(
                "expected_sources must not be empty."
            )
        
        normalized = []

        for source in self.expected_sources:

            normalized.append(
                self._validate_text(
                    source,
                    "expected_source",
                )
            )

        object.__setattr__(
            self,
            "expected_sources",
            tuple(normalized),
        )

    @staticmethod 
    def _validate_text(
        value: str,
        field: str,
    ) -> str: 
        
        if not isinstance(value, str):
            raise TypeError(
                f"{field} must be a string."
            )
        
        value = value.strip()

        if not value: 
            raise ValueError(
                f"{field} must not be empty."
            )
        
        return value 
    

# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalVerification: 
    """
    Retrieval evidence for one benchmark query.
    """

    benchmark: RetrievalBenchmark

    retrieval: RetrievalResult 

    retrieved_sources: tuple[str, ...]

    top1_correct: bool 

    first_correct_rank: int | None 

    retrieved_document_count: int 
    
    hit_at_k: bool 


# ---------------------------------------------------
# Aggregate result 
# ---------------------------------------------------

@dataclass(frozen=True)
class RetrievalVerificationResult:

    verifications: tuple[RetrievalVerification, ...]

    top1_accuracy: float 

    hit_at_k: float 

    mean_first_correct_rank: float | None 

    @property 
    def benchmark_count(self) -> int: 

        return len(self.verifications)
    
    def __post_init__(self) -> None: 

        return len(self.verifications)
    
    def __post_init__(self) -> None: 

        if not self.verifications:
            raise ValueError(
                "At least one verification is required."
            )
        
        for field in (
            self.top1_accuracy,
            self.hit_at_k,
        ):
            
            if not isfinite(field):
                raise ValueError(
                    "Metric must be finite."
                )
            
            if field < 0 or field > 1:
                raise ValueError(
                    "Metric must lie in [0,1]."
                )
            
        if self.mean_first_correct_rank is not None:

            if not isfinite(
                self.mean_first_correct_rank
            ):
                raise ValueError(
                    "Invalid mean_first_correct_rank."
                )
            
            if self.mean_first_correct_rank <= 0:
                raise ValueError(
                    "Invalid mean_first_correct_rank."
                )
    

    # --------------------------------------------------------
    # Retriever protocol
    # --------------------------------------------------------

    @runtime_checkable 
    class RetrieverProtocol(Protocol):

        def retrieve(
            self,
            query: str, 
            top_k: int = 5,
        ) -> RetrievalResult: 
            ...


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

class RetrievalVerificationRunner:
    """
    Execute retrieval verification experiments.
    """

    def __init__(
        self,
        *,
        retriever: BaseRetriever, 
        top_k: int = 5,
    ): 
        
        if retriever is None:
            raise TypeError(
                "retriever must not be None."
            )
        
        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )
        
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )
        
        self.retriever = retriever 
        self.top_k = top_k 

    def run(
        self, 
        benchmarks: Iterable[RetrievalBenchmark],
    ) -> RetrievalVerification: 
        
        benchmarks = self._materialize(
            benchmarks 
        )

        verifications = tuple(
            self._verify_benchmark(
                benchmark 
            )
            for benchmark in benchmarks 
        )

        return self._summarise(
            verifications 
        )
    
    # ---------------------------------------------

    def _verify_benchmark(
        self,
        benchmark: RetrievalBenchmark,
    ) -> RetrievalVerification: 
        
        retrieval = self.retriever.retrieve(
            query=benchmark.question,
            top_k=self.top_k,
        )

        if retrieval is None:
            raise RuntimeError(
                "Retriever returned None."
            )
        
        retrieved_sources = []

        for document in retrieval.documents:

            source = (
                document.metadata
                .get("source").strip()
            )

            if not source.endswith(".pdf"):
                source += ".pdf"

            if (
                not isinstance(source, str)
                or
                not source.strip()
            ):
                raise ValueError(
                    "Every retrieved document "
                    "must contain metadata['source']"
                )
            
            retrieved_sources.append(
                source.strip()
            )

        retrieved_sources = tuple(
            retrieved_sources 
        )

        top1_correct = (
            len(retrieved_sources) > 0
            and 
            retrieved_sources[0]
            in benchmark.expected_source 
        )

        hit_at_k = any(
            source
            in benchmark.expected_source 
            for source in retrieved_sources 
        )

        first_rank = None 

        for index, source in enumerate(
            retrieved_sources, 
            start=1,
        ): 
            
            if (
                source
                in benchmark.expected_source
            ): 
                first_rank = index 
                break 

        return RetrievalVerification(
            benchmark=benchmark,
            retrieval=retrieval,
            retrieved_sources=retrieved_sources,
            top1_correct=top1_correct,
            first_correct_rank=first_rank,
            retrieved_document_count=len(retrieval.documents),
            hit_at_k=hit_at_k,
        )
    
    # -----------------------------------------------------------

    @staticmethod 
    def _summarise(
        verifications:
        Sequence[RetrievalVerification],
    ) -> RetrievalVerificationResult:
        
        n = len(verifications)

        top1 = sum(
            item.top1_correct
            for item in verifications 
        ) / n 

        hit_at_k = sum(
            item.hit_at_k
            for item in verifications 
        ) / n 

        ranks = [
            item.first_correct_rank
            for item in verifications 
            if item.first_correct_rank 
            is not None
        ]

        mean_rank = (
            None 
            if not ranks 
            else sum(ranks) / len(ranks)
        )

        return RetrievalVerificationResult(
            verifications=tuple(
                verifications
            ),
            top1_accuracy=top1,
            hit_at_k=hit_at_k,
            mean_first_correct_rank=mean_rank,
        )
    
    # ---------------------------------------------------

    def _materialize(
        self,
        benchmarks: Benchmark,
    ) -> tuple[BenchmarkQuery, ...]:
        """
        Validate and materialize a Benchmark into an immutable
        sequence of BenchmarkQuery objects.
        """

        if not isinstance(benchmarks, Benchmark):
            raise TypeError(
                "Expected Benchmark."
            )

        queries = tuple(benchmarks.queries)

        if not queries:
            raise ValueError(
                "At least one benchmark is required."
            )

        seen: set[str] = set()

        for query in queries:

            if not isinstance(
                query,
                BenchmarkQuery,
            ):
                raise TypeError(
                    "Expected BenchmarkQuery."
                )

            if query.benchmark_id in seen:
                raise ValueError(
                    f"Duplicate benchmark_id: "
                    f"{query.benchmark_id}"
                )

            seen.add(
                query.benchmark_id
            )

        return queries 