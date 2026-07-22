"""
Unit tests for RetrievalVerificationRunner constructor.
"""

import pytest

from retrieval_verification import RetrievalVerificationRunner


# ---------------------------------------------------------------------
# Dummy retriever
# ---------------------------------------------------------------------


class DummyRetriever:
    pass


# ---------------------------------------------------------------------
# Valid constructor
# ---------------------------------------------------------------------


def test_valid_constructor():

    retriever = DummyRetriever()

    runner = RetrievalVerificationRunner(
        retriever=retriever,
        top_k=5,
    )

    assert runner.retriever is retriever
    assert runner.top_k == 5


# ---------------------------------------------------------------------
# retriever None
# ---------------------------------------------------------------------


def test_retriever_none():

    with pytest.raises(TypeError):

        RetrievalVerificationRunner(
            retriever=None,
            top_k=5,
        )


# ---------------------------------------------------------------------
# invalid top_k type
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "top_k",
    [
        "5",
        3.5,
        [],
        {},
        None,
    ],
)
def test_invalid_top_k_type(top_k):

    with pytest.raises(TypeError):

        RetrievalVerificationRunner(
            retriever=DummyRetriever(),
            top_k=top_k,
        )


# ---------------------------------------------------------------------
# top_k <= 0
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        -5,
    ],
)
def test_invalid_top_k_value(top_k):

    with pytest.raises(ValueError):

        RetrievalVerificationRunner(
            retriever=DummyRetriever(),
            top_k=top_k,
        )