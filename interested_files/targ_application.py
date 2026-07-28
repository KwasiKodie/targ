"""
targ_application.py

Application-level orchestration for the complete Training-Free Adaptive
Retrieval Gating (TARG) workflow.

This module coordinates three existing workflows without reimplementing
their internal algorithms:

1. Corpus preparation:
   PDF loading -> deterministic chunking -> embedding -> vector store.

2. Threshold calibration:
   benchmark loading -> Stage 2.5 retrieval comparison -> accuracy- or
   budget-based threshold calibration -> atomic calibration persistence.

3. Runtime inference:
   calibrated retrieval gate -> TARGPipeline -> optional answer evaluation.

The application deliberately keeps corpus construction, generation,
uncertainty scoring, retrieval, calibration, and evaluation logic inside
their existing dedicated components.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from langchain_core.vectorstores import VectorStore

from benchmark.benchmark_loader import BenchmarkLoader
from corpus.chunker import Chunker
from corpus.corpus_models import Chunk, CorpusDocument

from corpus.development_record import DevelopmentRecord
from corpus.representative_corpus import RepresentativeCorpusBuilder
from corpus.stage2_5_experiment import (
    ExperimentResult,
    Stage2_5ExperimentRunner,
)
from corpus.threshold_calibrator import (
    CalibrationResult,
    ThresholdCalibrator,
)
from corpus.vector_store_builder import VectorStoreBuilder

from answer_evaluator import AnswerEvaluator, EvaluationResult
from answer_generator import AnswerGenerator
from draft_generator import DraftGenerator
from margin_uncertainty_scorer import MarginUncertaintyScorer

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)

VectorRetriever = retrieval.VectorRetriever

from retrieval_gate import RetrievalGate
from targ_pipeline import PipelineResult, TARGPipeline


# ---------------------------------------------------------------------
# Application models
# ---------------------------------------------------------------------


class CalibrationStrategy(str, Enum):
    """Supported application-level threshold-calibration strategies."""

    ACCURACY = "accuracy"
    BUDGET = "budget"


@dataclass(frozen=True, slots=True)
class CorpusPreparationResult:
    """Evidence produced by the corpus-preparation workflow."""

    pdf_directory: Path
    documents: tuple[CorpusDocument, ...]
    chunks: tuple[Chunk, ...]
    vector_store: VectorStore

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True, slots=True)
class CalibrationWorkflowResult:
    """Evidence produced by Stage 2.5 and threshold calibration."""

    benchmark_path: Path
    strategy: CalibrationStrategy
    experiment: ExperimentResult
    calibration: CalibrationResult
    calibration_path: Path


@dataclass(frozen=True, slots=True)
class InferenceWorkflowResult:
    """Runtime pipeline result plus optional reference-based evaluation."""

    pipeline: PipelineResult
    evaluation: EvaluationResult | None

    @property
    def answer_text(self) -> str:
        answer = self.pipeline.answer
        generated_text = getattr(answer, "generated_text", None)

        if isinstance(generated_text, str):
            return generated_text

        if isinstance(answer, str):
            return answer

        raise TypeError(
            "PipelineResult.answer must be a string or expose "
            "generated_text."
        )


# ---------------------------------------------------------------------
# TARG application
# ---------------------------------------------------------------------


class TARGApplication:
    """
    Coordinate corpus preparation, calibration, and online inference.

    The application owns workflow state, but delegates all research logic
    to injected components.
    """

    def __init__(
        self,
        *,
        chunker: Chunker,
        vector_store_builder: VectorStoreBuilder,
        draft_generator: DraftGenerator,
        uncertainty_scorer: MarginUncertaintyScorer,
        answer_generator: AnswerGenerator,
        answer_evaluator: AnswerEvaluator,
        threshold_calibrator: ThresholdCalibrator,
        calibration_path: str | Path,
        retrieval_top_k: int = 5,
        vector_store_persist: Callable[[VectorStore], object] | None = None,
    ) -> None:
        """
        Initialize the application with already-configured components.

        Parameters
        ----------
        chunker:
            Existing deterministic corpus chunker.

        vector_store_builder:
            Existing embedding and vector-store builder.

        draft_generator:
            Existing retrieval-free draft generator.

        uncertainty_scorer:
            Existing margin uncertainty scorer.

        answer_generator:
            Existing final answer generator.

        answer_evaluator:
            Existing answer evaluator used by Stage 2.5 and optional
            runtime evaluation.

        threshold_calibrator:
            Existing calibrator supporting accuracy and budget modes.

        calibration_path:
            JSON destination for the selected calibration artifact.

        retrieval_top_k:
            Number of documents requested by the runtime retriever.

        vector_store_persist:
            Optional application-level persistence callback. It receives the
            completed vector store after construction. This is necessary
            because VectorStoreBuilder itself exposes construction only and
            vector-store persistence APIs vary by backend.
        """

        self._require_method("chunker", chunker, "chunk_documents")
        self._require_method(
            "vector_store_builder",
            vector_store_builder,
            "build",
        )
        self._require_method(
            "draft_generator",
            draft_generator,
            "generate",
        )
        self._require_method(
            "uncertainty_scorer",
            uncertainty_scorer,
            "score",
        )
        self._require_method(
            "answer_generator",
            answer_generator,
            "generate",
        )
        self._require_method(
            "answer_evaluator",
            answer_evaluator,
            "evaluate",
        )
        self._require_method(
            "threshold_calibrator",
            threshold_calibrator,
            "calibrate_for_accuracy",
        )
        self._require_method(
            "threshold_calibrator",
            threshold_calibrator,
            "calibrate_for_budget",
        )
        self._require_method(
            "threshold_calibrator",
            threshold_calibrator,
            "save",
        )

        if isinstance(retrieval_top_k, bool) or not isinstance(
            retrieval_top_k,
            int,
        ):
            raise TypeError("retrieval_top_k must be an integer.")

        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")

        if (
            vector_store_persist is not None
            and not callable(vector_store_persist)
        ):
            raise TypeError(
                "vector_store_persist must be callable or None."
            )

        self.chunker = chunker
        self.vector_store_builder = vector_store_builder
        self.draft_generator = draft_generator
        self.uncertainty_scorer = uncertainty_scorer
        self.answer_generator = answer_generator
        self.answer_evaluator = answer_evaluator
        self.threshold_calibrator = threshold_calibrator
        self.calibration_path = (
            Path(calibration_path).expanduser().resolve()
        )
        self.retrieval_top_k = retrieval_top_k
        self.vector_store_persist = vector_store_persist

        self._corpus_result: CorpusPreparationResult | None = None
        self._calibration_result: CalibrationResult | None = None
        self._retriever: VectorRetriever | None = None
        self._pipeline: TARGPipeline | None = None

    # -----------------------------------------------------------------
    # Workflow 1: corpus preparation
    # -----------------------------------------------------------------

    def prepare_corpus(
        self,
        *,
        pdf_directory: str | Path,
    ) -> CorpusPreparationResult:
        """
        Load PDFs, construct corpus documents, chunk them, and build the
        vector store.
        """

        directory = Path(pdf_directory).expanduser().resolve()

        corpus_builder = RepresentativeCorpusBuilder(
            pdf_directory=directory,
        )
        documents = corpus_builder.build()

        if not documents:
            raise ValueError(
                f"No extractable PDF content was found in {directory}."
            )

        chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            raise ValueError(
                "Corpus chunking produced no chunks. Check document text "
                "and chunk-size configuration."
            )

        vector_store = self.vector_store_builder.build(chunks)

        if self.vector_store_persist is not None:
            self.vector_store_persist(vector_store)

        result = CorpusPreparationResult(
            pdf_directory=directory,
            documents=documents,
            chunks=chunks,
            vector_store=vector_store,
        )

        self._corpus_result = result
        self._retriever = VectorRetriever(vector_store)
        self._rebuild_pipeline_when_ready()

        return result

    # -----------------------------------------------------------------
    # Workflow 2: Stage 2.5 and threshold calibration
    # -----------------------------------------------------------------

    def calibrate_threshold(
        self,
        *,
        benchmark_path: str | Path,
        strategy: CalibrationStrategy | str = CalibrationStrategy.ACCURACY,
        target_retrieval_rate: float | None = None,
    ) -> CalibrationWorkflowResult:
        """
        Run Stage 2.5, calibrate a threshold, and save it atomically.

        `target_retrieval_rate` is required only for budget calibration.
        """

        retriever = self._require_retriever()
        normalized_strategy = self._normalize_strategy(strategy)
        source = Path(benchmark_path).expanduser().resolve()

        benchmark = BenchmarkLoader(
            benchmark_path=source,
        ).load()

        development_dataset = tuple(
            DevelopmentRecord(
                example_id=query.benchmark_id,
                question=query.question,
                reference_answer=query.expected_answer_span,
            )
            for query in benchmark
        )

        runner = Stage2_5ExperimentRunner(
            draft_generator=self.draft_generator,
            uncertainty_scorer=self.uncertainty_scorer,
            answer_generator=self.answer_generator,
            retriever=retriever,
            answer_evaluator=self.answer_evaluator,
        )

        experiment = runner.run(
            development_dataset=development_dataset,
        )

        if normalized_strategy is CalibrationStrategy.ACCURACY:
            if target_retrieval_rate is not None:
                raise ValueError(
                    "target_retrieval_rate must be omitted for "
                    "accuracy calibration."
                )

            calibration = (
                self.threshold_calibrator.calibrate_for_accuracy(
                    experiment.development_examples
                )
            )

        else:
            if target_retrieval_rate is None:
                raise ValueError(
                    "target_retrieval_rate is required for budget "
                    "calibration."
                )

            calibration = (
                self.threshold_calibrator.calibrate_for_budget(
                    experiment.development_examples,
                    target_retrieval_rate=target_retrieval_rate,
                )
            )

        saved_path = self.threshold_calibrator.save(
            calibration,
            self.calibration_path,
        )

        self._calibration_result = calibration
        self._rebuild_pipeline_when_ready()

        return CalibrationWorkflowResult(
            benchmark_path=source,
            strategy=normalized_strategy,
            experiment=experiment,
            calibration=calibration,
            calibration_path=saved_path,
        )

    def load_calibration(
        self,
        path: str | Path | None = None,
    ) -> CalibrationResult:
        """Load a saved calibration artifact and prepare the runtime gate."""

        source = (
            self.calibration_path
            if path is None
            else Path(path).expanduser().resolve()
        )

        calibration = ThresholdCalibrator.load(source)
        self._calibration_result = calibration
        self._rebuild_pipeline_when_ready()

        return calibration

    # -----------------------------------------------------------------
    # Workflow 3: online inference
    # -----------------------------------------------------------------

    def answer_question(
        self,
        *,
        question: str,
        reference_answer: str | None = None,
    ) -> InferenceWorkflowResult:
        """
        Answer one question using the calibrated TARGPipeline.

        When `reference_answer` is supplied, evaluation is performed after
        the existing pipeline completes. No evaluation logic is added to or
        substituted for TARGPipeline.
        """

        pipeline = self._require_pipeline()
        normalized_question = self._validate_text(
            question,
            "question",
        )

        pipeline_result = pipeline.run(
            normalized_question,
        )

        evaluation: EvaluationResult | None = None

        if reference_answer is not None:
            normalized_reference = self._validate_text(
                reference_answer,
                "reference_answer",
            )

            candidate = self._extract_answer_text(
                pipeline_result
            )

            evaluation = self.answer_evaluator.evaluate(
                reference=normalized_reference,
                candidate=candidate,
            )

        return InferenceWorkflowResult(
            pipeline=pipeline_result,
            evaluation=evaluation,
        )

    # -----------------------------------------------------------------
    # Complete sequential workflow
    # -----------------------------------------------------------------

    def run(
        self,
        *,
        pdf_directory: str | Path,
        benchmark_path: str | Path,
        questions: list[str] | tuple[str, ...],
        calibration_strategy: CalibrationStrategy | str = (
            CalibrationStrategy.ACCURACY
        ),
        target_retrieval_rate: float | None = None,
        reference_answers: Mapping[str, str] | None = None,
    ) -> tuple[
        CorpusPreparationResult,
        CalibrationWorkflowResult,
        tuple[InferenceWorkflowResult, ...],
    ]:
        """Execute all three application workflows sequentially."""

        if not isinstance(questions, (list, tuple)):
            raise TypeError("questions must be a list or tuple.")

        if not questions:
            raise ValueError("questions must not be empty.")

        corpus_result = self.prepare_corpus(
            pdf_directory=pdf_directory,
        )

        calibration_result = self.calibrate_threshold(
            benchmark_path=benchmark_path,
            strategy=calibration_strategy,
            target_retrieval_rate=target_retrieval_rate,
        )

        inference_results = tuple(
            self.answer_question(
                question=question,
                reference_answer=(
                    None
                    if reference_answers is None
                    else reference_answers.get(question)
                ),
            )
            for question in questions
        )

        return (
            corpus_result,
            calibration_result,
            inference_results,
        )

    # -----------------------------------------------------------------
    # Runtime assembly
    # -----------------------------------------------------------------

    def _rebuild_pipeline_when_ready(self) -> None:
        """
        Rebuild the runtime pipeline only when both corpus and calibration
        state are available.
        """

        if self._retriever is None or self._calibration_result is None:
            self._pipeline = None
            return

        gate = RetrievalGate(
            calibration=self._calibration_result,
        )

        self._pipeline = TARGPipeline(
            draft_generator=self.draft_generator,
            scorer=self.uncertainty_scorer,
            gate=gate,
            retriever=self._retriever,
            answer_generator=self.answer_generator,
        )

    # -----------------------------------------------------------------
    # State guards
    # -----------------------------------------------------------------

    def _require_retriever(self) -> VectorRetriever:
        if self._retriever is None:
            raise RuntimeError(
                "The corpus has not been prepared. Call "
                "prepare_corpus() before calibrate_threshold()."
            )

        return self._retriever

    def _require_pipeline(self) -> TARGPipeline:
        if self._pipeline is None:
            missing: list[str] = []

            if self._retriever is None:
                missing.append("prepared corpus/vector store")

            if self._calibration_result is None:
                missing.append("loaded or calibrated threshold")

            raise RuntimeError(
                "Runtime inference is not ready; missing "
                + " and ".join(missing)
                + "."
            )

        return self._pipeline

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _require_method(
        name: str,
        component: object,
        method_name: str,
    ) -> None:
        if component is None:
            raise TypeError(f"{name} must not be None.")

        method = getattr(component, method_name, None)

        if method is None or not callable(method):
            raise TypeError(
                f"{name} must expose a callable {method_name}() method."
            )

    @staticmethod
    def _normalize_strategy(
        strategy: CalibrationStrategy | str,
    ) -> CalibrationStrategy:
        if isinstance(strategy, CalibrationStrategy):
            return strategy

        if not isinstance(strategy, str):
            raise TypeError(
                "strategy must be a CalibrationStrategy or string."
            )

        normalized = strategy.strip().lower()

        try:
            return CalibrationStrategy(normalized)
        except ValueError as exc:
            allowed = ", ".join(
                item.value for item in CalibrationStrategy
            )
            raise ValueError(
                f"strategy must be one of: {allowed}."
            ) from exc

    @staticmethod
    def _validate_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

        normalized = value.strip()

        if not normalized:
            raise ValueError(f"{name} must not be empty.")

        return normalized

    @staticmethod
    def _extract_answer_text(
        pipeline_result: PipelineResult,
    ) -> str:
        answer = pipeline_result.answer
        generated_text = getattr(answer, "generated_text", None)

        if isinstance(generated_text, str) and generated_text.strip():
            return generated_text.strip()

        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        raise TypeError(
            "PipelineResult.answer must be a non-empty string or expose "
            "a non-empty generated_text attribute."
        )

    # -----------------------------------------------------------------
    # Read-only state access
    # -----------------------------------------------------------------

    @property
    def corpus_result(self) -> CorpusPreparationResult | None:
        return self._corpus_result

    @property
    def calibration_result(self) -> CalibrationResult | None:
        return self._calibration_result

    @property
    def pipeline(self) -> TARGPipeline | None:
        return self._pipeline
