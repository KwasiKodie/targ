"""
Unit tests for benchmark_models.py

Tests immutable benchmark data models.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from benchmark.benchmark_models import BenchmarkQuery


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_query():

    return BenchmarkQuery(
        question="What is Retrieval-Augmented Generation?",
        document_id="rag_survey",
        section_title="Introduction",
        supporting_pages=(2, 3),
        expected_answer_span="Retrieval augments generation.",
        difficulty="Easy",
        topic="RAG",
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_constructor():

    query = make_query()

    assert query.question == "What is Retrieval-Augmented Generation?"
    assert query.document_id == "rag_survey"
    assert query.section_title == "Introduction"
    assert query.supporting_pages == (2, 3)
    assert query.expected_answer_span == "Retrieval augments generation."
    assert query.difficulty == "Easy"
    assert query.topic == "RAG"


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_model_is_frozen():

    query = make_query()

    with pytest.raises(FrozenInstanceError):

        query.question = "Modified"


# ---------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------


def test_equal_objects():

    q1 = make_query()

    q2 = make_query()

    assert q1 == q2


def test_not_equal_when_field_changes():

    q1 = make_query()

    q2 = BenchmarkQuery(
        question="Different question",
        document_id="rag_survey",
        section_title="Introduction",
        supporting_pages=(2, 3),
        expected_answer_span="Retrieval augments generation.",
        difficulty="Easy",
        topic="RAG",
    )

    assert q1 != q2


# ---------------------------------------------------------------------
# Supporting pages
# ---------------------------------------------------------------------


def test_supporting_pages_is_tuple():

    query = make_query()

    assert isinstance(
        query.supporting_pages,
        tuple,
    )


def test_multiple_supporting_pages():

    query = BenchmarkQuery(
        question="Question",
        document_id="doc",
        section_title="Section",
        supporting_pages=(1, 2, 5, 8),
        expected_answer_span="answer",
        difficulty="Medium",
        topic="Topic",
    )

    assert len(query.supporting_pages) == 4


# ---------------------------------------------------------------------
# Hashability
# ---------------------------------------------------------------------


def test_hashable():

    query = make_query()

    d = {
        query: "stored",
    }

    assert d[query] == "stored"


# ---------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------


def test_repr_contains_class_name():

    query = make_query()

    assert "BenchmarkQuery" in repr(query)


# ---------------------------------------------------------------------
# Dataclass conversion
# ---------------------------------------------------------------------


def test_asdict():

    from dataclasses import asdict

    query = make_query()

    data = asdict(query)

    assert data["question"] == query.question
    assert data["document_id"] == query.document_id
    assert data["section_title"] == query.section_title
    assert data["supporting_pages"] == (2, 3)
    assert data["expected_answer_span"] == query.expected_answer_span
    assert data["difficulty"] == "Easy"
    assert data["topic"] == "RAG"


# ---------------------------------------------------------------------
# Type validation (if implemented)
# ---------------------------------------------------------------------


def test_supporting_pages_accepts_list():

    query = BenchmarkQuery(
        question="Question",
        document_id="doc",
        section_title="Section",
        supporting_pages=[1, 2],
        expected_answer_span="answer",
        difficulty="Easy",
        topic="Topic",
    )

    assert query.supporting_pages == [1, 2]


def test_empty_question_rejected():

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):

        BenchmarkQuery(
            question="",
            document_id="doc",
            section_title="Section",
            supporting_pages=(1,),
            expected_answer_span="answer",
            difficulty="Easy",
            topic="Topic",
        )