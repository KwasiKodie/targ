"""
Unit tests for benchmark_loader.py

Tests loading benchmark queries from JSON.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.benchmark_loader import BenchmarkLoader
from benchmark.benchmark_models import BenchmarkQuery,Benchmark


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def benchmark_record(
    *,
    question="What is RAG?",
    document_id="rag_survey",
    section_title="Introduction",
    supporting_pages=(2, 3),
    expected_answer_span="Retrieval augments generation.",
    difficulty="Easy",
    topic="RAG",
):

    return {
        "question": question,
        "document_id": document_id,
        "section_title": section_title,
        "supporting_pages": list(supporting_pages),
        "expected_answer_span": expected_answer_span,
        "difficulty": difficulty,
        "topic": topic,
    }


def write_json(path: Path, records):

    path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------


def test_constructor_accepts_path(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(path, [])

    loader = BenchmarkLoader(
        benchmark_path=path,
    )

    assert loader.benchmark_path == path


def test_constructor_accepts_string(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(path, [])

    loader = BenchmarkLoader(
        benchmark_path=str(path),
        )

    assert loader.benchmark_path == path


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def test_empty_file_rejected(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(path, [])

    loader = BenchmarkLoader(
        benchmark_path=path,
    )

    with pytest.raises(ValueError):
        loader.load()


def test_load_single_query(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(
        path,
        [
            benchmark_record(),
        ],
    )

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    benchmark = loader.load()

    assert len(benchmark) == 1

    assert isinstance(
        benchmark[0],
        BenchmarkQuery,
    )

    assert benchmark[0].question == "What is RAG?"


def test_load_multiple_queries(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(
        path,
        [
            benchmark_record(question="Q1"),
            benchmark_record(question="Q2"),
            benchmark_record(question="Q3"),
        ],
    )

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    benchmark = loader.load()

    assert len(benchmark) == 3

    assert benchmark[0].question == "Q1"

    assert benchmark[1].question == "Q2"

    assert benchmark[2].question == "Q3"


# ---------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------


def test_missing_file():

    with pytest.raises(FileNotFoundError):

        BenchmarkLoader(
            benchmark_path="does_not_exist.json",
        )


def test_invalid_json(tmp_path):

    path = tmp_path / "broken.json"

    path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    with pytest.raises(json.JSONDecodeError):

        loader.load()


def test_missing_required_field(tmp_path):

    path = tmp_path / "benchmark.json"

    record = benchmark_record()

    del record["question"]

    write_json(path, [record])

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    with pytest.raises(
        (
            TypeError,
            KeyError,
        )
    ):

        loader.load()


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------


def test_supporting_pages_preserved(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(
        path,
        [
            benchmark_record(
                supporting_pages=(1, 5, 7),
            )
        ],
    )

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    benchmark = loader.load()

    assert benchmark[0].supporting_pages == (
        1, 
        5, 
        7
    )


def test_returns_tuple(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(
        path,
        [
            benchmark_record(),
        ],
    )

    loader = BenchmarkLoader(
        benchmark_path=path,
        )

    benchmark = loader.load()

    assert isinstance(
        benchmark,
        Benchmark,
    )


def test_all_items_are_benchmark_queries(tmp_path):

    path = tmp_path / "benchmark.json"

    write_json(
        path,
        [
            benchmark_record(question="Q1"),
            benchmark_record(question="Q2"),
        ],
    )

    loader = BenchmarkLoader(
        benchmark_path=str(path),
    )

    benchmark = loader.load()

    assert all(
        isinstance(
            q,
            BenchmarkQuery,
        )
        for q in benchmark
    )