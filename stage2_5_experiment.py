"""
stage2_5_experiment.py

Orchestrate the Stage 2.5 retrieval-comparison experiment.

The runner evaluates every development example twice:

1. Generate an answer without retrieval.
2. Generate an answer using retrieved evidence.
3. Evaluate both answer against the reference answer.
4. Preserve the complete evidence chain in RetrievalComparison.
5. Convert each comparison into a DevelopmentExample for threshold calibration.
6. Produce deterministic aggregate experiment statistics.

The runner performs no generation, retrieval, uncertainty estimation,
evaluation, or threshold-calibration algorithms itself.

Author:
TARG Reproduction
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from answer_evaluator import AnswerEvaluator, EvaluationResult
from answer_generator import AnswerGenerator, AnswerOutput
from retrieval_comparison import RetrievalComparison 
from development_record import DevelopmentRecord

# ---------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------

@runtime_checkable
class DevelopmentRecordProtocol(Protocol):
    """
    Minimum inference required from a development-dataset record.
    """

    @property
    def example_id(self) -> str:
        ...

    @property
    def question(self) -> str:
        ...

    @property
    def reference_answer(self) -> str:
        ...

@runtime_checkable
class DraftGeneratorProtocol(Protocol):
    """
    Interface required from the draft generator.
    """

    def generate(
        self,
        query: str,
    ) -> Any:
        ...


@runtime_checkable
class UncertaintyScorerProtocol(Protocol):
    """
    Interface required from the uncertainty scorer.

    The concrete scorer may either return a numeric score directly or
    return an immutable result object exposing one of the supported
    score attributes handled by `_extract_uncertainty_score`.
    """

    def score(
        self,
        draft_output: Any,
    ) -> Any:
        ...

@runtime_checkable
class RetrieverProtocol(Protocol):
    """
    Interface required from the retriever.
    """

    def retrieve(
        self,
        query: str,
    ) -> Any:
        ...


# ------------------------------------------------------------
# Experiment result
# ------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentResult:
    """
    Immutable result of a complete Stage 2.5 experiment.

    Attributes
    ----------
    comparisons
        Per-example retrieval comparisons containing uncertainty,
        retrieval, generation, and evaluation evidence.

    development_examples
        Calibration-ready examples derived from the comparisons.

    mean_no_retrieval_score
        Mean answer score without retrieval.

    mean_retrieval_score
        Mean answer score with retrieval.

    mean_improvement
        Mean retrieval score minus mean no-retrieval score.

    retrieval_helped_count
        Number of examples for which retrieval improved the score.

    retrieval_hurt_count
        Number of examples for which retrieval reduced the score.

    retrieval_no_effect_count
        Number of examples for which retrieval did not change the score.
    """

    comparisons: tuple[RetrievalComparison, ...]

    development_example: tuple[Any, ...]

    mean_no_retrieval_score: float

    mean_retrieval_score: float

    mean_improvement: float

    retrieval_hurt_count: int

    retrieval_no_effect_count: int

    # -------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------

    @property
    def example_count(self) -> int:
        """
        Number of evaluated development examples.
        """

        return len(self.comparisons)
    
    @property
    def retrieval_help_rate(self) -> float:
        """
        Proportion of examples for which retrieval improved performance.
        """

        if self.example_count == 0:
            return 0.0
        
        return self.retrieval_helped_count / self.example_count
    
    @property
    def retrieval_no_effect_rate(self) -> float:
        """
        Proportion of examples for which retrieval had no score effect.
        """

        if self.example_count == 0:
            return 0.0
        
        return self.retrieval_no_effect_count / self.example_count
    

    @property
    def retrieval_improvement(self) -> float:
        """
        Alias for the aggregate mean improvement.
        """

        return self.mean_improvement 
    
    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def __post_init__(self) -> None:
        comparison_count = len(self.comparisons)
        calibration_count = len(self.development_example)

        if comparison_count == 0:
            raise ValueError(
                "ExperimentResult must contain at least one comparison."
            )
        
        if calibration_count != comparison_count:
            raise ValueError(
                "The number of development_examples must equal the "
                "number of camparisons."
            )
        
        category_count = (
            self.retrieval_helped_count
            + self.retrieval_hurt_count
            + self.retrieval_no_effect_count
        )

        if category_count != comparison_count:
            raise ValueError(
                "Retrieval outcome counts must sum to the number of "
                "comparisons."
            )
        
        for field_name, value in (
            (
                "mean_no_retrieval_score",
                self.mean_no_retrieval_score,
            ),
            (
                "mean_retrieval_score",
                self.mean_retrieval_score,
            ),
            (
                "mean_improvement",
                self.mean_improvement,
            )
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{field_name} must be numeric."
                )
            
            if not isfinite(float(value)):
                raise ValueError(
                    f"{field_name} must be finite."
                )
            
        for field_name, value in (
            (
                "retrieval_helped_count",
                self.retrieval_helped_count,
            ),
            (
                "retrieval_no_effect_count",
                self.retrieval_no_effect_count,
            ),
        ):
            if not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be an integer."
                )
            
            if value < 0:
                raise ValueError(
                    f"{field_name} must not be negative."
                )
            
class Stage2_5ExperimentRunner:
    """
    Orchestrate Stage 2.5 retrieval-comparison experiments.

    Responsibilities
    ----------------
    1. Validate development-dataset records.
    2. Generate a draft for uncertainty estimation.
    3. Estimate uncertainty from the draft.
    4. Generate an answer without retrieval.
    5. Evaluate the no-retrieval answer.
    6. Retrieve relevant evidence.
    7. Generate an answer using the retrieved evidence.
    8. Evaluate the retrieval-augmented answer.
    9. Preserve all evidence in RetrievalComparison.
    10. Produce calibration examples and experiment summaries.

    The runner never:
    - implements generation;
    - implements retrieval;
    - computes uncertainty itself;
    - implements answer metrics;
    - calibrates a threshold;
    - applies a retrieval gate.
    """

    def __init__(
        self,
        *,
        draft_generator: DraftGeneratorProtocol,
        uncertainty_scorer: UncertaintyScorerProtocol,
        answer_generator: AnswerGenerator,
        retriever: RetrieverProtocol,
        answer_evaluator: AnswerEvaluator,
    ) -> None:
        """
        Initialize the Stage 2.5 experiment runner.
        """

        self._validate_component(
            name="draft_generator",
            component=draft_generator,
            required_method="generate",
        )

        self._validate_component(
            name="uncertainty_scorer",
            component=uncertainty_scorer,
            required_method="score",
        )

        self._validate_component(
            name="answer_generator",
            component=answer_generator,
            required_method="generate",
        )

        self._validate_component(
            name="retriever",
            component=retriever,
            required_method="retrieve",
        )

        self._validate_component(
            name="answer_evaluator",
            component=answer_evaluator,
            required_method="evaluate",
        )

        self.draft_generator = draft_generator
        self.uncertainty_scorer = uncertainty_scorer
        self.answer_generator = answer_generator
        self.retriever = retriever
        self.answer_evaluator = answer_evaluator


    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(
        self,
        *,
        development_dataset: Iterable[DevelopmentRecord],
    ) -> ExperimentResult:
        """
        Run the complete stage 2.5 experiment.

        Parameters
        ----------
        developement_dataset
            Iterable of development records exposing:

            - example_id
            - question
            - reference_answer

        Returns
        -------
        ExperimentResult
            Immutable aggregate experiment result.
        """

        records = self._materialize_dataset(
            development_dataset
        )

        comparisons = tuple(
            self._run_example(record)
            for record in records
        )

        development_examples = tuple(
            comparison.calibration_example
            for comparison in comparisons
        )

        return self._summarise_results(
            comparisons=comparisons,
            development_examples=development_examples,
        )
    
    # ------------------------------------------------------------------
    # Per-example orchestration
    # ------------------------------------------------------------------

    def _run_example(
        self,
        record: DevelopmentRecord,
    ) -> RetrievalComparison:
        """
        Execute the full experiment flow for one development example.
        """

        base_prompt = (
            "Answer the following question concisely.\n\n"
            f"Question:\n{record.question}\n\n"
            "Answer:"
        )
        example_id = self._validate_example_id(
            record.example_id
        )

        question = self._validate_text(
            name="question",
            value=record.question,
        )

        reference_answer = self._validate_text(
            name="reference_answer",
            value=record.reference_answer,
        )

        draft_output = self.draft_generator.generate(
            base_prompt=base_prompt,
        )

        uncertainty_result = self.uncertainty_scorer.score(
            draft=draft_output
        )

        uncertainty_score = self._extract_uncertainty_score(
            uncertainty_result
        )

        no_retrieval_output = (
            self._generate_without_retrieval(
                question=question,
            )
        )

        no_retrieval_evaluation = self._evaluate(
            reference_answer=reference_answer,
            answer_output=no_retrieval_output
        )

        retrieval_result = self.retriever.retrieve(
            query=question
        )

        if retrieval_result is None:
            raise RuntimeError(
                "Retriever returned None for example "
                f"{example_id!r}."
            )
        
        retrieval_output = self._generate_with_retrieval(
            question=question,
            retrieval_result=retrieval_result,
        )

        retrieval_evaluation = self._evaluate(
            reference_answer=reference_answer,
            answer_output=retrieval_output,
        )

        comparison = RetrievalComparison(
            example_id=example_id,
            question=question,
            reference_answer=reference_answer,
            uncertainty_score=uncertainty_score,
            retrieval_result=retrieval_result,
            no_retrieval_output=no_retrieval_output,
            retrieval_output=retrieval_output,
            no_retrieval_evaluation=no_retrieval_evaluation,
            retrieval_evaluation=retrieval_evaluation,
        )

        self._validate_comparison(
            comparison=comparison,
        )

        return comparison
    
    def _generate_without_retrieval(
        self,
        *,
        question: str,
    ) -> AnswerOutput:
        """
        Generate an answer without retrieved context.
        """

        output = self.answer_generator.generate(
            query=question,
        )

        self._validate_answer_output(
            output=output,
            expected_query=question,
            expected_retrieval=False,
        )

        return output 
    
    def _generate_with_retrieval(
        self,
        *,
        question: str,
        retrieval_result: Any,
    ) -> AnswerOutput:
        """
        Generate an answer using retrieved context.
        """

        output = self.answer_generator.generate(
            query=question,
            retrieval=retrieval_result,
        )

        self._validate_answer_output(
            output=output,
            expected_query=question,
            expected_retrieval=True,
        )

        return output
    
    def _evaluate(
        self,
        *,
        reference_answer: str,
        answer_output: AnswerOutput,
    ) -> EvaluationResult:
        """
        Evaluate one generated answer against its reference answer.
        """

        evaluation = self.answer_evaluator.evaluate(
            reference=reference_answer,
            candidate=answer_output.generated_text,
        )

        if not isinstance(evaluation, EvaluationResult):
            raise TypeError(
                "answer_evaluator.evaluate() must return "
                "EvaluationResult."
            )
        
        if (
            evaluation.reference_answer
            != reference_answer
        ):
            raise ValueError(
                "EvaluationResult did not preserve the supplied "
                "reference answer."
            )
        
        if (
            evaluation.candidate_answer
            != answer_output.generated_text
        ): 
            raise ValueError(
                "EvaluationResult did not preserve the generated "
                "candidate answer"
            )
        
        return evaluation 
    
    # ---------------------------------------------------------------------------
    # Experiment summarisation
    # ---------------------------------------------------------------------------

    def _summarise_results(
        self,
        *,
        comparisons: Sequence[RetrievalComparison],
        development_examples: Sequence[Any],
    ) -> ExperimentResult:
        """
        Build deterministic aggregate experiment statistics.
        """

        if not comparisons:
            raise ValueError(
                "Cannot summarise an empty comparison sequence."
            )
        
        no_retrieval_scores = tuple(
            float(comparison.no_retrieval_score)
            for comparison in comparisons
        )

        retrieval_scores = tuple(
            float(comparison.retrieval_score)
            for comparison in comparisons
        )

        improvements = tuple(
            float(comparison.improvement)
            for comparison in comparisons
        )

        helped_count = sum(
            1
            for comparison in comparisons
            if comparison.retrieval_helped
        )

        hurt_count = sum(
            1
            for comparison in comparisons
            if comparison.retrieval_hurt
        )

        no_effect_count = sum(
            1
            for comparison in comparisons
            if comparison.retrieval_no_effect
        )

        comparison_count = len(comparisons)

        return ExperimentResult(
            comparisons=tuple(comparisons),
            development_example=tuple(
                development_examples
            ),
            mean_no_retrieval_score=(
                sum(no_retrieval_scores)
                / comparison_count
            ),
            mean_retrieval_score = (
                sum(retrieval_scores)
                / comparison_count
            ),
            mean_improvement=(
                sum(improvements)
                / comparison_count
            ),
            retrieval_helped_count=helped_count,
            retrieval_hurt_count=hurt_count,
            retrieval_no_effect_count=no_effect_count,
        )
    
    # -------------------------------------------------------------
    # Uncertainty extraction
    # -------------------------------------------------------------

    @staticmethod
    def _extract_uncertainty_score(
        uncertainty_result: Any,
    ) -> float:
        """
        Extract a numeric uncertainty score from a scorer result.

        Supported scorer outputs
        ------------------------
        - float
        - int
        - object exposing `uncertainty_score`
        - object exposing `score`
        - object exposing `margin_uncertainty`
        - object exposing `uncertainty`

        This adapter keeps orchestration independent of the precise
        immutable evidence-object name used by the uncertainty scorer.
        """

        if isinstance(uncertainty_result, bool):
            raise TypeError(
                "Uncertainty score must not be boolean."
            )
        
        if isinstance(uncertainty_result, (int, float)):
            score = float(uncertainty_result)

        else:
            score = None

            for attribute_name in (
                "uncertainty_score",
                "score",
                "margin_uncertainty",
                "uncertainty",
            ):
                if hasattr(
                    uncertainty_result,
                    attribute_name,
                ):
                    candidate = getattr(
                        uncertainty_result,
                        attribute_name,
                    )

                    if isinstance(candidate, bool):
                        continue

                    if isinstance(candidate, (int, float)):
                        score = float(candidate)
                        break

            if score is None:
                raise TypeError(
                    "Unable to extract an uncertainty score. "
                    "The scorer must return a numeric value or an "
                    "object exposing one of: uncertainty_score, "
                    "score, margin_uncertainty, uncertainty."
                )

                
        if not isfinite(score):
            raise ValueError(
                "Uncertainty score must be finite."
            )
        
        return score 
    
    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    @staticmethod
    def _materialize_dataset(
        development_dataset: Iterable[
            DevelopmentRecord
        ],
    ) -> tuple[DevelopmentRecord, ...]:
        """
        Materialize and validate the development dataset.
        """

        if development_dataset is None:
            raise TypeError(
                "development_dataset must not be None."
            )
        
        if isinstance(
            development_dataset,
            (str, bytes),
        ):
            raise TypeError(
                "development_dataset must be an iterable of "
                "development records, not a string."
            )
        
        try:
            records = tuple(development_dataset)
        
        except TypeError as error:
            raise TypeError(
                "development_dataset must contain at least one "
                "record."
            )
        
        seen_example_ids: str[str] = set()

        for index, record in enumerate(records):
            if record is None:
                raise ValueError(
                    "development_dataset contains None at "
                    f"index {index}."
                )
            
            for attribute_name in (
                "example_id",
                "question",
                "reference_answer",
            ):
                if not hasattr(record, attribute_name):
                    raise TypeError(
                        "Develpment record at index "
                        f"{index} does not expose "
                        f"{attribute_name!r}."
                    )
                
            example_id = (
                Stage2_5ExperimentRunner
                ._validate_example_id(
                    record.example_id
                )
            )

            if example_id in seen_example_ids:
                raise ValueError(
                    "Duplicate example_id detected: "
                    f"{example_id!r}."
                )
            
            seen_example_ids.add(example_id)

            Stage2_5ExperimentRunner._validate_text(
                name="question",
                value=record.question,
            )

            Stage2_5ExperimentRunner._validate_text(
                name="reference_answer",
                value=record.reference_answer,
            )

        return records 
    
    @staticmethod
    def _validate_component(
        *,
        name: str,
        component: Any,
        required_method: str,
    ) -> None:
        """
        Validate one injected experiment component.
        """

        if component is None:
            raise TypeError(
                f"{name} must not be None."
            )
        
        method = getattr(
            component,
            required_method,
            None,
        )

        if method is None or not callable(method):
            raise TypeError(
                f"{name} must expose a callable "
                f"{required_method}() method."
            )
        
    @staticmethod
    def _validate_example_id(
        value: Any,
    ) -> str:
        """
        Validate and normalize a development-example identifier.
        """

        if not isinstance(value, str):
            raise TypeError(
                "example_id must be a string."
            )
        
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "example_id must not be empty."
            )
        
        return normalized 
    
    @staticmethod
    def _validate_text(
        *,
        name: str,
        value: Any,
    ) -> str:
        """
        Validate and normalize a required text field.
        """

        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string."
            )
        
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{name} must not be empty."
            )
        
        return normalized
    
    @staticmethod
    def _validate_answer_output(
        *,
        output: Any,
        expected_query: str,
        expected_retrieval: bool,
    ) -> None:
        """
        Validate evidence returned by AnswerGenerator.
        """

        if not isinstance(output, AnswerOutput):
            raise TypeError(
                "answer_generator.generate() must return "
                "AnswerOutput."
            )
        
        if output.query != expected_query:
            raise ValueError(
                "AnswerOutput did not preserve the expected query."
            )
        
        if not isinstance(
            output.generated_text,
            str,
        ):
            raise TypeError(
                "AnswerOutput.generated_text must be a string."
            )
        
        if not output.generated_text.strip():
            raise ValueError(
                "AnswerOutput.generated_text must not be empty."
            )
        
        if (
            output.used_retrieval
            is not expected_retrieval
        ): 
            raise ValueError(
                "AnswerOutput.used_retrieval is inconsistent "
                "with the requested generation mode."
            )
        
        if expected_retrieval:
            if output.retrieved_document_count < 0:
                raise ValueError(
                    "retrieved_document_count must not be negative."
                )
            
        elif output.retrieved_document_count != 0:
            raise ValueError(
                "No-retrieval generation must report zero "
                "retrieved documents."
            )
        
    @staticmethod
    def _validate_comparison(
        *,
        comparison: RetrievalComparison,
    ) -> None:
        """
        Validate cross-object evidence consistency.
        """

        if(
            comparison.no_retrieval_output.query
            != comparison.question
        ):
            raise ValueError(
                "No-retrieval output query is inconsistent with "
                "the comparison question."
            )
        
        if (
            comparison.retrieval_output.query
            != comparison.question 
        ):
            raise ValueError(
                "Retrieval output query is inconsistent with "
                "the comparison question."
            )
        
        if (
            comparison.no_retrieval_evaluation
            .reference_answer
            != comparison.reference_answer
        ):
            raise ValueError(
                "No-retrieval evaluation reference is "
                "inconsistent with RetrievalComparison."
            )
        
        if (
            comparison.retrieval_evaluation
            .reference_answer
            != comparison.reference_answer
        ):
            raise ValueError(
                "Retrieval evaluation reference is inconsistent "
                "with RetrievalComparison."
            )
        
        if (
            comparison.no_retrieval_evaluation
            .candidate_answer
            != comparison.no_retrieval_output.generated_text
        ):
            raise ValueError(
                "No-retrieval evaluation candidate does not match "
                "the generated answer."
            )
        
        if (
            comparison.retrieval_evaluation
            .candidate_answer
            != comparison.retrieval_output.generated_text
        ):
            raise ValueError(
                "Retrieval evaluation candidate does not match "
                "the generated answer."
            )
        
        outcome_count = sum(
            (
                bool(comparison.retrieval_helped),
                bool(comparison.retrieval_hurt),
                bool(comparison.retrieval_no_effect),
            )
        )

        if outcome_count != 1:
            raise ValueError(
                "Exactly one retrieval outcome must be true: "
                "helped, hurt, or no effect."
            )
        
        