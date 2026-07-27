"""Parallel CPU/GPU executor for Stage 2.5 experiments."""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch

from answer_evaluator import AnswerEvaluator
from answer_generator import AnswerGenerator
from corpus.development_record import DevelopmentRecord
from corpus.stage2_5_experiment import ExperimentResult, Stage2_5ExperimentRunner
from draft_generator import DraftGenerator
from margin_uncertainty_scorer import MarginUncertaintyScorer
from retrieval_comparison import RetrievalComparison

ExecutionDevice = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class ParallelExperimentConfig:
    model_name: str
    embedding_model_name: str
    vector_store_directory: Path
    device: ExecutionDevice = "auto"
    worker_count: int | None = None
    gpu_ids: tuple[int, ...] | None = None
    prefix_length: int = 20
    beta: float = 3.0
    max_new_tokens: int = 128
    retrieval_top_k: int = 5
    embedding_device: str = "cpu"
    add_special_tokens: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vector_store_directory",
            Path(self.vector_store_directory).expanduser().resolve(),
        )
        if not self.model_name.strip() or not self.embedding_model_name.strip():
            raise ValueError("Model names must not be empty.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda.")
        if self.worker_count is not None and self.worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        if self.gpu_ids is not None:
            if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
                raise ValueError("gpu_ids must be non-empty and unique.")
            if any(not isinstance(i, int) or i < 0 for i in self.gpu_ids):
                raise ValueError("gpu_ids must be non-negative integers.")
        if self.prefix_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("Token limits must be positive.")
        if self.beta <= 0 or self.retrieval_top_k <= 0:
            raise ValueError("beta and retrieval_top_k must be positive.")


@dataclass(frozen=True)
class _WorkerTask:
    index: int
    device: str
    records: tuple[DevelopmentRecord, ...]
    config: ParallelExperimentConfig


class _TopKRetriever:
    def __init__(self, retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def retrieve(self, query: str):
        return self._retriever.retrieve(query=query, top_k=self._top_k)


def _build_runner(task: _WorkerTask) -> Stage2_5ExperimentRunner:
    """Build non-picklable model and retrieval objects inside the worker."""
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from module_loader import load

    cfg = task.config
    device = torch.device(task.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        local_files_only=cfg.local_files_only,
    )
    dtype = torch.float32 if device.type == "cpu" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
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
    if not cfg.vector_store_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found: {cfg.vector_store_directory}"
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

    return Stage2_5ExperimentRunner(
        draft_generator=DraftGenerator(
            model=model,
            tokenizer=tokenizer,
            prefix_length=cfg.prefix_length,
        ),
        uncertainty_scorer=MarginUncertaintyScorer(beta=cfg.beta),
        answer_generator=AnswerGenerator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            add_special_tokens=cfg.add_special_tokens,
        ),
        retriever=retriever,
        answer_evaluator=AnswerEvaluator(),
    )


def _run_partition(task: _WorkerTask) -> tuple[RetrievalComparison, ...]:
    if task.device == "cpu":
        workers = max(1, task.config.worker_count or 1)
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // workers))
    return _build_runner(task).run(
        development_dataset=task.records
    ).comparisons


class ParallelExperimentExecutor:
    """Run Stage 2.5 on CPU workers or one spawned worker per GPU."""

    def __init__(self, *, config: ParallelExperimentConfig) -> None:
        if not isinstance(config, ParallelExperimentConfig):
            raise TypeError("config must be ParallelExperimentConfig.")
        self.config = config

    def run(
        self,
        *,
        development_dataset: Iterable[DevelopmentRecord],
    ) -> ExperimentResult:
        records = Stage2_5ExperimentRunner._materialize_dataset(
            development_dataset
        )
        devices = self._resolve_devices(len(records))
        partitions = self._partition(records, len(devices))
        tasks = tuple(
            _WorkerTask(i, device, partition, self.config)
            for i, (device, partition) in enumerate(
                zip(devices, partitions, strict=True)
            )
        )

        if len(tasks) == 1:
            return self._summarise(_run_partition(tasks[0]))

        results: dict[int, tuple[RetrievalComparison, ...]] = {}
        with ProcessPoolExecutor(
            max_workers=len(tasks),
            mp_context=mp.get_context("spawn"),
        ) as pool:
            future_indexes = {
                pool.submit(_run_partition, task): task.index
                for task in tasks
            }
            try:
                for future in as_completed(future_indexes):
                    results[future_indexes[future]] = future.result()
            except BaseException:
                for future in future_indexes:
                    future.cancel()
                raise

        comparisons = tuple(
            comparison
            for index in range(len(tasks))
            for comparison in results[index]
        )
        return self._summarise(comparisons)

    def _resolve_devices(self, record_count: int) -> tuple[str, ...]:
        cfg = self.config
        cuda_available = torch.cuda.is_available()
        if cfg.device == "cuda" and not cuda_available:
            raise RuntimeError("CUDA requested but unavailable.")
        use_cuda = cfg.device == "cuda" or (
            cfg.device == "auto" and cuda_available
        )
        if use_cuda:
            count = torch.cuda.device_count()
            ids = cfg.gpu_ids or tuple(range(count))
            invalid = tuple(i for i in ids if i >= count)
            if invalid:
                raise ValueError(
                    f"Invalid GPU IDs {invalid}; detected {count} GPU(s)."
                )
            limit = cfg.worker_count or len(ids)
            return tuple(
                f"cuda:{i}" for i in ids[: min(limit, record_count)]
            )

        workers = min(
            cfg.worker_count or min(4, os.cpu_count() or 1),
            record_count,
        )
        return tuple("cpu" for _ in range(workers))

    @staticmethod
    def _partition(
        records: Sequence[DevelopmentRecord],
        count: int,
    ) -> tuple[tuple[DevelopmentRecord, ...], ...]:
        size = ceil(len(records) / count)
        return tuple(
            tuple(records[start : start + size])
            for start in range(0, len(records), size)
        )

    @staticmethod
    def _summarise(
        comparisons: Sequence[RetrievalComparison],
    ) -> ExperimentResult:
        comparisons = tuple(comparisons)
        if not comparisons:
            raise ValueError("Cannot summarise empty comparisons.")
        count = len(comparisons)
        return ExperimentResult(
            comparisons=comparisons,
            development_examples=tuple(
                item.calibration_example for item in comparisons
            ),
            mean_no_retrieval_score=(
                sum(float(item.no_retrieval_score) for item in comparisons)
                / count
            ),
            mean_retrieval_score=(
                sum(float(item.retrieval_score) for item in comparisons)
                / count
            ),
            mean_improvement=(
                sum(float(item.improvement) for item in comparisons)
                / count
            ),
            retrieval_helped_count=sum(
                1 for item in comparisons if item.retrieval_helped
            ),
            retrieval_hurt_count=sum(
                1 for item in comparisons if item.retrieval_hurt
            ),
            retrieval_no_effect_count=sum(
                1 for item in comparisons if item.retrieval_no_effect
            ),
        )
