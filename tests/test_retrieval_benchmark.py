"""
Unit tests for RetrievalBenchmark.
"""

import pytest

from retrieval_verification import RetrievalBenchmark


# ---------------------------------------------------------------------
# Valid benchmark
# ---------------------------------------------------------------------


def test_valid_benchmark():

    benchmark = RetrievalBenchmark(
        benchmark_id="rb-001",
        query="What is Retrieval-Augmented Generation?",
        expected_sources=(
            "TARG",
            "SELF-RAG",
        ),
    )

    assert benchmark.benchmark_id == "rb-001"
    assert benchmark.query == (
        "What is Retrieval-Augmented Generation?"
    )
    assert benchmark.expected_sources == (
        "TARG",
        "SELF-RAG",
    )


# ---------------------------------------------------------------------
# Empty benchmark_id
# ---------------------------------------------------------------------


def test_empty_benchmark_id():

    with pytest.raises(ValueError):

        RetrievalBenchmark(
            benchmark_id="",
            query="Example",
            expected_sources=("TARG",),
        )


# ---------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------


def test_empty_query():

    with pytest.raises(ValueError):

        RetrievalBenchmark(
            benchmark_id="rb-001",
            query="",
            expected_sources=("TARG",),
        )


# ---------------------------------------------------------------------
# Empty expected_sources
# ---------------------------------------------------------------------


def test_empty_expected_sources():

    with pytest.raises(ValueError):

        RetrievalBenchmark(
            benchmark_id="rb-001",
            query="Example",
            expected_sources=(),
        )
