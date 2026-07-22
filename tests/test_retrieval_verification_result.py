"""
Unit tests for RetrievalVerificationResult.
"""

import pytest

from retrieval_verification import (
    RetrievalBenchmark,
    RetrievalVerificationResult,
)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------


@pytest.fixture
def benchmark():

    return RetrievalBenchmark(
        benchmark_id="rb-001",
        query="What is Retrieval-Augmented Generation?",
        expected_sources=(
            "TARG",
        ),
    )


# ---------------------------------------------------------------------
# Valid metrics
# ---------------------------------------------------------------------


def test_valid_metrics(benchmark):

    result = RetrievalVerificationResult(
        verifications=(benchmark,),
        top1_accuracy=0.80,
        recall_at_k=0.90,
        mean_first_correct_rank=1.75,
    )

    assert result.verifications == (benchmark,)
    assert result.top1_accuracy == 0.80
    assert result.recall_at_k == 0.90
    assert result.mean_first_correct_rank == 1.75


# ---------------------------------------------------------------------
# No verifications
# ---------------------------------------------------------------------


def test_no_verifications():

    with pytest.raises(ValueError):

        RetrievalVerificationResult(
            verifications=(),
            top1_accuracy=0.80,
            recall_at_k=0.90,
            mean_first_correct_rank=1.75,
        )


# ---------------------------------------------------------------------
# Invalid accuracy
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "accuracy",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_accuracy(
    benchmark,
    accuracy,
):

    with pytest.raises(ValueError):

        RetrievalVerificationResult(
            verifications=(benchmark,),
            top1_accuracy=accuracy,
            recall_at_k=0.90,
            mean_first_correct_rank=1.75,
        )


# ---------------------------------------------------------------------
# Invalid recall
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "recall",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_recall(
    benchmark,
    recall,
):

    with pytest.raises(ValueError):

        RetrievalVerificationResult(
            verifications=(benchmark,),
            top1_accuracy=0.80,
            recall_at_k=recall,
            mean_first_correct_rank=1.75,
        )


# ---------------------------------------------------------------------
# Invalid mean rank
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "mean_rank",
    [
        0,
        -1,
        -0.5,
    ],
)
def test_invalid_mean_rank(
    benchmark,
    mean_rank,
):

    with pytest.raises(ValueError):

        RetrievalVerificationResult(
            verifications=(benchmark,),
            top1_accuracy=0.80,
            recall_at_k=0.90,
            mean_first_correct_rank=mean_rank,
        )