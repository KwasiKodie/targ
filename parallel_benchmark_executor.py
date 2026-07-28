"""Parallel, analysis-free benchmark data collection for TARG inference."""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence
from uuid import uuid4

import torch

ExecutionDevice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    benchmark_id: str
    question: str
    expected_answer: str
    expected_sources: tuple[str, ...] = ()
    section_title: str | None = None
    supporting_pages: tuple[int, ...] = ()
    difficulty: str | None = None
    topic: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkInferenceResult:
    run_id: str
    recorded_at_utc: str
    model_name: str
    embedding_model_name: str
    benchmark_id: str
    question: str
    expected_answer: str
    expected_sources: tuple[str, ...]
    section_title: str | None
    supporting_pages: tuple[int, ...]
    difficulty: str | None
    topic: str | None
    generated_answer: str
    draft_text: str
    uncertainty_score: float
    threshold: float
    retrieval_triggered: bool
    retrieved_document_count: int
    retrieved_documents: tuple[dict[str, Any], ...]
    evaluation: dict[str, Any] | None
    timing: dict[str, float]
    worker_device: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelBenchmarkConfig:
    model_name: str
    embedding_model_name: str
    vector_store_directory: Path
    calibration_path: Path
    device: ExecutionDevice = "auto"
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
        object.__setattr__(self, "vector_store_directory", Path(self.vector_store_directory).expanduser().resolve())
        object.__setattr__(self, "calibration_path", Path(self.calibration_path).expanduser().resolve())
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.worker_count is not None and self.worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if self.gpu_ids is not None:
            if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
                raise ValueError("gpu_ids must be non-empty and unique.")
            if any(not isinstance(item, int) or item < 0 for item in self.gpu_ids):
                raise ValueError("gpu_ids must contain non-negative integers.")
        if min(self.prefix_length, self.max_new_tokens, self.retrieval_top_k) <= 0:
            raise ValueError("Token limits and retrieval_top_k must be positive.")
        if self.beta <= 0:
            raise ValueError("beta must be positive.")


@dataclass(frozen=True)
class _WorkerTask:
    index: int
    device: str
    records: tuple[BenchmarkRecord, ...]
    config: ParallelBenchmarkConfig
    run_id: str
    recorded_at_utc: str


def _plain(value: Any) -> Any:
    """Convert dataclasses and common model objects to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if hasattr(value, "dict"):
        return _plain(value.dict())
    if hasattr(value, "__dict__"):
        return {key: _plain(item) for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


def _document_payload(document: Any) -> dict[str, Any]:
    if hasattr(document, "page_content"):
        return {
            "page_content": str(getattr(document, "page_content", "")),
            "metadata": _plain(getattr(document, "metadata", {})),
        }
    payload = _plain(document)
    return payload if isinstance(payload, dict) else {"value": payload}


class _TopKRetriever:
    """Bind retrieval_top_k without changing TARGPipeline."""

    def __init__(self, retriever: Any, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def retrieve(self, query: str):
        return self._retriever.retrieve(query=query, top_k=self._top_k)


def _build_pipeline(task: _WorkerTask):
    """Construct all non-picklable runtime components inside a worker."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from answer_generator import AnswerGenerator
    from corpus.threshold_calibrator import ThresholdCalibrator
    from draft_generator import DraftGenerator
    from margin_uncertainty_scorer import MarginUncertaintyScorer
    from module_loader import load
    from retrieval_gate import RetrievalGate
    from targ_pipeline import TARGPipeline

    cfg = task.config
    device = torch.device(task.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32 if device.type == "cpu" else "auto",
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
    )
    model.to(device)
    model.eval()

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.embedding_model_name,
        model_kwargs={"device": cfg.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.load_local(
        folder_path=str(cfg.vector_store_directory),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    retrieval_module = load("retrieval_runtime", "retrieval.py")
    retriever = _TopKRetriever(
        retrieval_module.VectorRetriever(vector_store=vector_store),
        cfg.retrieval_top_k,
    )
    calibration = ThresholdCalibrator.load(cfg.calibration_path)

    # TARGPipeline accepts exactly five runtime components. Evaluation is
    # intentionally deferred to Jupyter and is not passed into the pipeline.
    pipeline = TARGPipeline(
        draft_generator=DraftGenerator(model=model, tokenizer=tokenizer, prefix_length=cfg.prefix_length),
        scorer=MarginUncertaintyScorer(beta=cfg.beta),
        gate=RetrievalGate(calibration=calibration),
        retriever=retriever,
        answer_generator=AnswerGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            add_special_tokens=cfg.add_special_tokens,
        ),
    )
    return pipeline, calibration


def _run_partition(task: _WorkerTask) -> tuple[BenchmarkInferenceResult, ...]:
    if task.device == "cpu":
        workers = max(1, task.config.worker_count or 1)
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))

    pipeline, calibration = _build_pipeline(task)
    output: list[BenchmarkInferenceResult] = []
    for record in task.records:
        result = pipeline.run(record.question)
        documents = tuple(_document_payload(item) for item in result.retrieval.documents)
        evaluation = None
        output.append(
            BenchmarkInferenceResult(
                run_id=task.run_id,
                recorded_at_utc=task.recorded_at_utc,
                model_name=task.config.model_name,
                embedding_model_name=task.config.embedding_model_name,
                benchmark_id=record.benchmark_id,
                question=record.question,
                expected_answer=record.expected_answer,
                expected_sources=record.expected_sources,
                section_title=record.section_title,
                supporting_pages=record.supporting_pages,
                difficulty=record.difficulty,
                topic=record.topic,
                generated_answer=str(
                    getattr(result.answer, "generated_text", result.answer)
                ),
                draft_text=str(getattr(result.draft, "generated_text", getattr(result.draft, "text", ""))),
                uncertainty_score=float(result.margin.score),
                threshold=float(calibration.threshold),
                retrieval_triggered=bool(result.gate.retrieve),
                retrieved_document_count=len(documents),
                retrieved_documents=documents,
                evaluation=evaluation if isinstance(evaluation, dict) else None,
                timing={str(key): float(value) for key, value in result.timing.items()},
                worker_device=task.device,
            )
        )
    return tuple(output)


class ParallelBenchmarkExecutor:
    """Collect benchmark inference data in parallel without analysing it."""

    def __init__(self, *, config: ParallelBenchmarkConfig) -> None:
        if not isinstance(config, ParallelBenchmarkConfig):
            raise TypeError("config must be ParallelBenchmarkConfig.")
        self.config = config

    def run(self, *, records: Iterable[BenchmarkRecord]) -> tuple[BenchmarkInferenceResult, ...]:
        materialized = tuple(records)
        if not materialized:
            raise ValueError("records must not be empty.")
        devices = self._resolve_devices(len(materialized))
        partitions = self._partition(materialized, len(devices))
        run_id = uuid4().hex
        recorded_at = datetime.now(timezone.utc).isoformat()
        tasks = tuple(
            _WorkerTask(index, device, partition, self.config, run_id, recorded_at)
            for index, (device, partition) in enumerate(zip(devices, partitions, strict=True))
        )
        if len(tasks) == 1:
            return _run_partition(tasks[0])

        completed: dict[int, tuple[BenchmarkInferenceResult, ...]] = {}
        with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp.get_context("spawn")) as pool:
            future_indexes = {pool.submit(_run_partition, task): task.index for task in tasks}
            try:
                for future in as_completed(future_indexes):
                    completed[future_indexes[future]] = future.result()
            except BaseException:
                for future in future_indexes:
                    future.cancel()
                raise
        return tuple(item for index in range(len(tasks)) for item in completed[index])

    def _resolve_devices(self, record_count: int) -> tuple[str, ...]:
        cfg = self.config
        cuda_available = torch.cuda.is_available()
        if cfg.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but unavailable.")
        use_cuda = cfg.device == "cuda" or (cfg.device == "auto" and cuda_available)
        if use_cuda:
            count = torch.cuda.device_count()
            ids = cfg.gpu_ids or tuple(range(count))
            invalid = tuple(item for item in ids if item >= count)
            if invalid:
                raise ValueError(f"Invalid GPU IDs {invalid}; detected {count} GPU(s).")
            limit = cfg.worker_count or len(ids)
            return tuple(f"cuda:{item}" for item in ids[: min(limit, record_count)])

        workers = min(cfg.worker_count or min(4, os.cpu_count() or 1), record_count)
        return tuple("cpu" for _ in range(workers))

    @staticmethod
    def _partition(records: Sequence[BenchmarkRecord], count: int) -> tuple[tuple[BenchmarkRecord, ...], ...]:
        size = ceil(len(records) / count)
        return tuple(tuple(records[start:start + size]) for start in range(0, len(records), size))


class BenchmarkDatasetWriter:
    """Persist raw benchmark observations for later notebook analysis."""

    CSV_FIELDS = (
        "run_id", "recorded_at_utc", "model_name", "embedding_model_name",
        "benchmark_id", "question", "expected_answer", "expected_sources",
        "section_title", "supporting_pages", "difficulty", "topic",
        "generated_answer", "draft_text", "uncertainty_score", "threshold",
        "retrieval_triggered", "retrieved_document_count", "retrieved_documents",
        "evaluation", "timing", "worker_device",
    )

    @classmethod
    def write_jsonl(cls, results: Sequence[BenchmarkInferenceResult], path: str | Path, *, append: bool = False) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a" if append else "w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result.as_dict(), ensure_ascii=False) + "\n")
        return destination

    @classmethod
    def write_csv(cls, results: Sequence[BenchmarkInferenceResult], path: str | Path, *, append: bool = False) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        exists = destination.exists() and destination.stat().st_size > 0
        with destination.open("a" if append else "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=cls.CSV_FIELDS)
            if not append or not exists:
                writer.writeheader()
            for result in results:
                row = result.as_dict()
                for key in ("expected_sources", "supporting_pages", "retrieved_documents", "evaluation", "timing"):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
                writer.writerow({key: row[key] for key in cls.CSV_FIELDS})
        return destination
