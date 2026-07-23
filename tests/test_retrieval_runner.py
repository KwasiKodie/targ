"""
Unit tests for retrieval_runner.py

Tests the orchestration logic of RetrievalRunner.

Author:
Training-Free Adaptive Retrieval Gating (TARG) Reproduction
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.benchmark_models import BenchmarkQuery
from retrieval.retrieval_models import (
    RetrievedChunk,
    RetrievalInspection,
)
from retrieval.retrieval_runner import RetrievalRunner


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def make_query(index: int = 1):

    return BenchmarkQuery(
        question=f"Question {index}",
        document_id=f"doc_{index}",
        section_title="Section",
        supporting_pages=(1,),
        expected_answer_span="answer",
        difficulty="Easy",
        topic="Topic",
    )


def make_inspection(index: int = 1):

    return RetrievalInspection(
        query=f"Question {index}",
        expected_document=f"doc_{index}",
        expected_pages=(1,),
        retrieved_chunks=(
            RetrievedChunk(
                rank=1,
                score=0.10,
                source=f"doc_{index}",
                page=1,
                chunk=0,
                chunk_id=f"chunk_{index}",
                text="Example text.",
            ),
        ),
    )


# ---------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------


class FakeBenchmarkLoader:

    def __init__(self, queries):

        self.queries = queries

        self.load_called = False

    def load(self):

        self.load_called = True

        return self.queries


class FakeRetrievalInspector:

    def __init__(self, inspection):

        self.inspection = inspection

        self.calls = []

    def inspect(self, query):

        self.calls.append(query)

        return self.inspection


# ---------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------


def test_constructor():

    loader = FakeBenchmarkLoader([])

    inspector = FakeRetrievalInspector(
        make_inspection(),
    )

    runner = RetrievalRunner(
        benchmark_loader=loader,
        retrieval_inspector=inspector,
    )

    assert runner.benchmark_loader is loader
    assert runner.retrieval_inspector is inspector


def test_constructor_requires_loader():

    with pytest.raises(TypeError):

        RetrievalRunner(
            benchmark_loader=None,
            retrieval_inspector=FakeRetrievalInspector(
                make_inspection(),
            ),
        )


def test_constructor_requires_inspector():

    with pytest.raises(TypeError):

        RetrievalRunner(
            benchmark_loader=FakeBenchmarkLoader([]),
            retrieval_inspector=None,
        )


# ---------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------


def test_run_empty_benchmark():

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    results = runner.run()

    assert results == ()


def test_run_single_query():

    query = make_query()

    inspection = make_inspection()

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([query]),
        retrieval_inspector=FakeRetrievalInspector(
            inspection,
        ),
    )

    results = runner.run()

    assert len(results) == 1

    assert results[0] == inspection


def test_run_multiple_queries():

    queries = [
        make_query(1),
        make_query(2),
        make_query(3),
    ]

    inspection = make_inspection()

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader(
            queries,
        ),
        retrieval_inspector=FakeRetrievalInspector(
            inspection,
        ),
    )

    results = runner.run()

    assert len(results) == 3


def test_inspector_called_for_every_query():

    queries = [
        make_query(1),
        make_query(2),
        make_query(3),
    ]

    inspector = FakeRetrievalInspector(
        make_inspection(),
    )

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader(
            queries,
        ),
        retrieval_inspector=inspector,
    )

    runner.run()

    assert inspector.calls == queries


# ---------------------------------------------------------------------
# run_and_save()
# ---------------------------------------------------------------------


def test_run_and_save_returns_results(
    monkeypatch,
    tmp_path,
):

    inspection = make_inspection()

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader(
            [make_query()],
        ),
        retrieval_inspector=FakeRetrievalInspector(
            inspection,
        ),
    )

    called = {}

    def fake_save(
        *,
        inspections,
        output_path,
    ):
        called["inspections"] = inspections
        called["output_path"] = output_path

    monkeypatch.setattr(
        runner,
        "save",
        fake_save,
    )

    output = tmp_path / "results.json"

    results = runner.run_and_save(
        output,
    )

    assert called["inspections"] == (inspection,)
    assert called["output_path"] == output


# ---------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------


def test_save_creates_directory(
    tmp_path,
):

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    output = (
        tmp_path
        / "nested"
        / "results.json"
    )

    runner.save(
        inspections=(make_inspection(),),
        output_path=output,
    )

    assert output.exists()


def test_save_writes_valid_json(
    tmp_path,
):

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    output = tmp_path / "results.json"

    inspection = make_inspection()

    runner.save(
        inspections=(inspection,),
        output_path=output,
    )

    with output.open() as f:

        data = json.load(f)

    assert isinstance(
        data,
        list,
    )

    assert data[0] == inspection.to_dict()


def test_save_overwrites_existing_file(
    tmp_path,
):

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    output = tmp_path / "results.json"

    output.write_text("old")

    runner.save(
        inspections=(make_inspection(),),
        output_path=output,
    )

    with output.open() as f:

        data = json.load(f)

    assert isinstance(
        data,
        list,
    )


def test_save_accepts_string_path(
    tmp_path,
):

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    output = str(
        tmp_path / "results.json"
    )

    runner.save(
        inspections=(make_inspection(),),
        output_path=output,
    )

    assert Path(output).exists()


def test_save_accepts_path_object(
    tmp_path,
):

    runner = RetrievalRunner(
        benchmark_loader=FakeBenchmarkLoader([]),
        retrieval_inspector=FakeRetrievalInspector(
            make_inspection(),
        ),
    )

    output = tmp_path / "results.json"

    runner.save(
        inspections=(make_inspection(),),
        output_path=output,
    )

    assert output.exists()