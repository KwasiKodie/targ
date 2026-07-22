import pytest

from retrieval import (
    RetrievalResult,
)

from retrieval_verification import (
    RetrievalBenchmark,
    RetrievalVerification,
    RetrievalVerificationRunner,
)

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

def make_verification(
    benchmark,
    *,
    top1: bool,
    hit: bool,
    rank: int | None,
    count: int = 2,
) -> RetrievalVerification:

    retrieval = RetrievalResult(
        query=benchmark.query,
        retrieved=True,
        retrieval_backend="Dummy",
        documents=[],
    )

    return RetrievalVerification(
        benchmark=benchmark,
        retrieval=retrieval,
        retrieved_sources=(),
        top1_correct=top1,
        first_correct_rank=rank,
        retrieved_document_count=count,
        hit_at_k=hit,
    )

def test_summarise_single_verification(
    benchmark,
):

    verification = make_verification(
        benchmark,
        top1=True,
        hit=True,
        rank=1,
    )

    result = RetrievalVerificationRunner._summarise(
        (verification,),
    )

    assert result.benchmark_count == 1

    assert result.top1_accuracy == 1.0

    assert result.hit_at_k == 1.0

    assert result.mean_first_correct_rank == 1.0


def test_summarise_multiple_verifications(
    benchmark,
):

    result = RetrievalVerificationRunner._summarise(

        (

            make_verification(
                benchmark,
                top1=True,
                hit=True,
                rank=1,
            ),

            make_verification(
                benchmark,
                top1=False,
                hit=True,
                rank=2,
            ),

            make_verification(
                benchmark,
                top1=False,
                hit=False,
                rank=None,
            ),

        )

    )

    assert result.benchmark_count == 3

    assert result.top1_accuracy == pytest.approx(
        1 / 3,
    )

    assert result.hit_at_k == pytest.approx(
        2 / 3,
    )

    assert result.mean_first_correct_rank == pytest.approx(
        1.5,
    )

def test_summarise_no_successful_retrievals(
    benchmark,
):

    result = RetrievalVerificationRunner._summarise(

        (

            make_verification(
                benchmark,
                top1=False,
                hit=False,
                rank=None,
            ),

            make_verification(
                benchmark,
                top1=False,
                hit=False,
                rank=None,
            ),

        )

    )

    assert result.top1_accuracy == 0.0

    assert result.hit_at_k == 0.0

    assert result.mean_first_correct_rank is None

def test_summarise_all_successful_retrievals(
    benchmark,
):

    result = RetrievalVerificationRunner._summarise(

        (

            make_verification(
                benchmark,
                top1=True,
                hit=True,
                rank=1,
            ),

            make_verification(
                benchmark,
                top1=True,
                hit=True,
                rank=1,
            ),

            make_verification(
                benchmark,
                top1=True,
                hit=True,
                rank=1,
            ),

        )

    )

    assert result.top1_accuracy == 1.0

    assert result.hit_at_k == 1.0

    assert result.mean_first_correct_rank == 1.0


def test_summarise_mean_rank(
    benchmark,
):

    result = RetrievalVerificationRunner._summarise(

        (

            make_verification(
                benchmark,
                top1=False,
                hit=True,
                rank=1,
            ),

            make_verification(
                benchmark,
                top1=False,
                hit=True,
                rank=2,
            ),

            make_verification(
                benchmark,
                top1=False,
                hit=True,
                rank=5,
            ),

        )

    )

    assert result.mean_first_correct_rank == pytest.approx(
        8 / 3,
    )

    