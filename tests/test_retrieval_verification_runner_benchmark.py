from retrieval_verification import (
    RetrievalBenchmark,
    RetrievalVerificationRunner,
)

from retrieval import (
    RetrievedDocument,
    RetrievalResult,
)

import pytest

class DummyRetriever:

    def __init__(self, retrieval):
        self._retrieval = retrieval

    def retrieve(
        self,
        query,
        top_k=5,
    ) -> RetrievalResult:
        return self._retrieval

class NoneRetriever:

    def retrieve(
        self,
        *,
        query,
        top_k,
    ):
        return None
    

@pytest.fixture
def benchmark():

    return RetrievalBenchmark(
        benchmark_id="rb-001",
        query="What is TARG?",
        expected_sources=(
            "TARG",
            "SELF-RAG",
        ),
    )


def test_retrieval_returns_none(benchmark):

    runner = RetrievalVerificationRunner(
        retriever=NoneRetriever(),
        )

    with pytest.raises(RuntimeError):
        runner._verify_benchmark(benchmark)

def test_missing_source_metadata(benchmark):
    retrieval = RetrievalResult(
        query="What is TARG?",
        retrieved=True,
        retrieval_backend="Dummy",
        documents=[
            RetrievedDocument(
                id="doc1",
                text="text",
                score=0.9,
                metadata={},
        ),
        RetrievedDocument(
            id="doc2",
            text="...",
            score=0.82,
            metadata={},
        ),
    ],
)
    runner = RetrievalVerificationRunner(
            retriever=DummyRetriever(retrieval),
        )

    with pytest.raises(ValueError):
        runner._verify_benchmark(benchmark)

def test_empty_source_metadata(benchmark):

    retrieval = RetrievalResult(
    query="What is TARG?",
    retrieved=True,
    retrieval_backend="Dummy",
    documents=[
        RetrievedDocument(
            id="doc1",
            text="...",
            score=0.95,
            metadata={
                "source": "",
            },
        ),
        RetrievedDocument(
            id="doc2",
            text="...",
            score=0.82,
            metadata={
                "source": "",
            },
        ),
    ],
)

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
        )

    with pytest.raises(ValueError):
        runner._verify_benchmark(benchmark)

def test_top1_success(benchmark):

    retrieval = make_retrieval(
        "TARG",
        "PTES",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.top1_correct is True

    
def test_top1_failure(benchmark):

    retrieval = make_retrieval(
        "PTES",
        "TARG",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.top1_correct is False

    
def test_recall_success(benchmark):

    retrieval = make_retrieval(
        "SELF-RAG",
        "PTES",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.hit_at_k is True

def test_recall_failure(benchmark):

    retrieval = make_retrieval(
        "PTES",
        "NIST",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.hit_at_k is False

def test_first_rank_one(benchmark):
    retrieval = make_retrieval(
        "TARG",
        "PTES",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.first_correct_rank == 1

    
def test_first_rank_three(benchmark):
    retrieval = make_retrieval(
        "PTES",
        "NIST",
        "TARG",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.first_correct_rank == 3

def test_no_correct_document(benchmark):
    retrieval = make_retrieval(
        "PTES",
        "NIST",
    )

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(retrieval),
    )

    result = runner._verify_benchmark(
        benchmark,
    )

    assert result.first_correct_rank is None

def make_retrieval(*sources: str) -> RetrievalResult:

    return RetrievalResult(
        query="What is TARG?",
        retrieved=True,
        retrieval_backend="Dummy",
        documents=[
            RetrievedDocument(
                id=f"doc_{i}",
                text=f"Document {i}",
                score=1.0 - (i * 0.05),
                metadata={
                    "source": source,
                },
            )
            for i, source in enumerate(sources)
        ],
    )