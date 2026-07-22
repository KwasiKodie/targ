import pytest

from retrieval import (
    RetrievedDocument,
    RetrievalResult,
)

from retrieval_verification import (
    RetrievalBenchmark,
    RetrievalVerificationRunner,
)

class DummyRetriever:

    def retrieve(
        self,
        query,
        top_k=5,
    ) -> RetrievalResult:

        return RetrievalResult(
            query=query,
            retrieved=True,
            retrieval_backend="Dummy",
            documents=[
                RetrievedDocument(
                    id="doc1",
                    text="...",
                    score=0.95,
                    metadata={
                        "source": "TARG",
                    },
                ),
            ],
        )
    
@pytest.fixture
def benchmark():

    return RetrievalBenchmark(
        benchmark_id="rb-001",
        query="What is TARG?",
        expected_sources=(
            "TARG",
        ),
    )

def test_run_accepts_generator(
    benchmark,
):

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(),
    )

    benchmarks = (
        b
        for b in [benchmark]
    )

    result = runner.run(
        benchmarks,
    )

    assert result.benchmark_count == 1

    assert result.top1_accuracy == 1.0

    assert result.hit_at_k == 1.0


def test_run_empty_benchmark_collection():

    runner = RetrievalVerificationRunner(
        retriever=DummyRetriever(),
    )

    with pytest.raises(
        ValueError,
    ):
        runner.run([])

@pytest.fixture
def benchmark_two():

    return RetrievalBenchmark(
        benchmark_id="rb-002",
        query="Explain SELF-RAG",
        expected_sources=(
            "SELF-RAG",
        ),
    )

class MultiRetriever:

    def retrieve(
        self,
        query,
        top_k=5,
    ):

        if query == "What is TARG?":

            source = "TARG"

        else:

            source = "PTES"

        return RetrievalResult(
            query=query,
            retrieved=True,
            retrieval_backend="Dummy",
            documents=[
                RetrievedDocument(
                    id="doc",
                    text="...",
                    score=0.9,
                    metadata={
                        "source": source,
                    },
                ),
            ],
        )
    
def test_run_aggregate_metrics(
    benchmark,
    benchmark_two,
):

    runner = RetrievalVerificationRunner(
        retriever=MultiRetriever(),
    )

    result = runner.run(
        (
            benchmark,
            benchmark_two,
        ),
    )

    assert result.benchmark_count == 2

    assert result.top1_accuracy == pytest.approx(
        0.5,
    )

    assert result.hit_at_k == pytest.approx(
        0.5,
    )

    assert result.mean_first_correct_rank == 1.0

