"""Notebook- and script-friendly benchmark orchestration for TARG.

This module contains no argparse dependency.  It can therefore be imported and
used directly from Jupyter, Python scripts, tests, or the CLI wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmark.benchmark_loader import BenchmarkLoader
from parallel_benchmark_executor import (
    BenchmarkDatasetWriter,
    BenchmarkInferenceResult,
    BenchmarkRecord,
    ParallelBenchmarkConfig,
    ParallelBenchmarkExecutor,
)


@dataclass(frozen=True, slots=True)
class BenchmarkRunConfig:
    model_name: str
    embedding_model_name: str
    vector_store_directory: Path
    calibration_path: Path
    benchmark_path: Path
    device: str = "auto"
    worker_count: int | None = None
    gpu_ids: tuple[int, ...] | None = None
    prefix_length: int = 20
    beta: float = 3.0
    max_new_tokens: int = 256
    retrieval_top_k: int = 5
    embedding_device: str = "cpu"
    add_special_tokens: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        for name in ("vector_store_directory", "calibration_path", "benchmark_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if not self.benchmark_path.is_file():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmark_path}")
        if not self.vector_store_directory.exists():
            raise FileNotFoundError(f"Vector store not found: {self.vector_store_directory}")
        if not self.calibration_path.is_file():
            raise FileNotFoundError(f"Calibration artifact not found: {self.calibration_path}")


@dataclass(frozen=True, slots=True)
class BenchmarkRunOutput:
    results: tuple[BenchmarkInferenceResult, ...]
    csv_path: Path | None = None
    jsonl_path: Path | None = None


class BenchmarkRunner:
    """High-level benchmark API independent of command-line parsing."""

    def __init__(self, *, config: BenchmarkRunConfig) -> None:
        if not isinstance(config, BenchmarkRunConfig):
            raise TypeError("config must be BenchmarkRunConfig.")
        self.config = config

    def load_records(self) -> tuple[BenchmarkRecord, ...]:
        benchmark = BenchmarkLoader(benchmark_path=self.config.benchmark_path).load()
        records = tuple(self._to_record(query) for query in benchmark)
        if not records:
            raise RuntimeError(f"No benchmark examples loaded from {self.config.benchmark_path}")
        return records

    def run(
        self,
        *,
        parallel: bool = True,
        records: Iterable[BenchmarkRecord] | None = None,
    ) -> tuple[BenchmarkInferenceResult, ...]:
        materialized = tuple(records) if records is not None else self.load_records()
        executor = ParallelBenchmarkExecutor(
            config=self._executor_config(parallel=parallel)
        )
        return executor.run(records=materialized)

    def run_and_save(
        self,
        *,
        csv_path: str | Path | None = None,
        jsonl_path: str | Path | None = None,
        append: bool = False,
        parallel: bool = True,
        records: Iterable[BenchmarkRecord] | None = None,
    ) -> BenchmarkRunOutput:
        results = self.run(parallel=parallel, records=records)
        saved_csv = (
            BenchmarkDatasetWriter.write_csv(results, csv_path, append=append)
            if csv_path is not None else None
        )
        saved_jsonl = (
            BenchmarkDatasetWriter.write_jsonl(results, jsonl_path, append=append)
            if jsonl_path is not None else None
        )
        return BenchmarkRunOutput(results=results, csv_path=saved_csv, jsonl_path=saved_jsonl)

    def _executor_config(self, *, parallel: bool) -> ParallelBenchmarkConfig:
        cfg = self.config
        return ParallelBenchmarkConfig(
            model_name=cfg.model_name,
            embedding_model_name=cfg.embedding_model_name,
            vector_store_directory=cfg.vector_store_directory,
            calibration_path=cfg.calibration_path,
            device=cfg.device,
            worker_count=cfg.worker_count if parallel else 1,
            gpu_ids=cfg.gpu_ids,
            prefix_length=cfg.prefix_length,
            beta=cfg.beta,
            max_new_tokens=cfg.max_new_tokens,
            retrieval_top_k=cfg.retrieval_top_k,
            embedding_device=cfg.embedding_device,
            add_special_tokens=cfg.add_special_tokens,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
        )

    @staticmethod
    def _to_record(query) -> BenchmarkRecord:
        expected_sources = getattr(query, "expected_source", ())
        return BenchmarkRecord(
            benchmark_id=str(query.benchmark_id),
            question=str(query.question),
            expected_answer=str(query.expected_answer_span),
            expected_sources=tuple(str(item) for item in expected_sources),
            section_title=(
                str(query.section_title)
                if getattr(query, "section_title", None) is not None else None
            ),
            supporting_pages=tuple(int(item) for item in getattr(query, "supporting_pages", ())),
            difficulty=(str(query.difficulty) if getattr(query, "difficulty", None) is not None else None),
            topic=(str(query.topic) if getattr(query, "topic", None) is not None else None),
        )
