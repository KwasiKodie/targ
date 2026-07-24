"""
retrieval_comparison.py

Immutable experiment models for Training-Free Adaptive Retrieval
Gating (TARG).

These models preserve the complete evidence from one retrieval 
comparison experiment.

A RetrievalComparison represents one development-set example after
both retrieval-free and retrieval-augmented generation have been
evaluated.

Responsibilities
----------------
1. Preserve uncertainty evidence.
2. Preserve retrieval evidence.
3. Preserve answer-generation evidence.
4. Preserve evaluation evidence.
5. Provide convenient derived properties.

The models NEVER 

- generate answers
- perform retrieval
- evaluate answers
- compute uncertainty
- calibrate thresholds
"""

from __future__ import annotations

from dataclasses import dataclass

from answer_generator import AnswerOutput
from answer_evaluator import EvaluationResult

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)

RetrievalResult = retrieval.RetrievalResult 

# -----------------------------------------------------------------
# Immutable experiment record
# -----------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalComparison:
    """
    Complete evidence for one retrieval experiment.
    """

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    example_id: str

    question: str

    reference_answer: str

    # -------------------------------------------------------------
    # Retrieval decision evidence
    # -------------------------------------------------------------

    uncertainty_score: float

    # -------------------------------------------------------------
    # Retrieval evidence
    # -------------------------------------------------------------
    retrieval_result: RetrievalResult 

    # -------------------------------------------------------------
    # Generation evidence
    # -------------------------------------------------------------

    no_retrieval_output: AnswerOutput

    retrieval_output: AnswerOutput

    # -------------------------------------------------------------
    # Evaluation evidence
    # -------------------------------------------------------------

    no_retrieval_evaluation: EvaluationResult

    retrieval_evaluation: EvaluationResult

    # -------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------

    @property
    def no_retrieval_score(self) -> float:
        """
        Evaluation score without retrieval.
        """

        return self.no_retrieval_evaluation.overall_score
    
    @property
    def retrieval_score(self) -> float:
        """
        Evaluation score obtained with retrieval.
        """

        return self.retrieval_evaluation.overall_score 
    
    @property
    def retrieval_no_effect(self) -> bool:
        """
        Whether retrieval produced exactly the same evaluation score.
        """

        return self.improvement == 0.0
    
    @property
    def improvement(self) -> float:
        """
        Absolute improvement obtained by retrieval.
        """

        return (
            self.retrieval_evaluation.overall_score
            - self.no_retrieval_evaluation.overall_score
        )
    
    @property
    def retrieval_helped(self) -> bool:
        """
        Whether retrieval improved the answer.
        """

        return self.improvement > 0.0
    
    @property
    def retrieval_hurt(self) -> bool:
        """
        Whether retrieval reduced answer quality.
        """

        return self.improvement < 0.0
    
    @property
    def calibration_example(self):
        """
        Convert directly into the format expected by
        ThresholdCalibrator.

        Imported lazily to avoid circular imports
        """

        from threshold_calibrator import DevelopmentExample

        return DevelopmentExample(
            example_id=self.example_id,
            uncertainty_score=self.uncertainty_score,
            no_retrieval_score=self.no_retrieval_score,
            retrieval_score=self.retrieval_score,
        )
    
    @property
    def preferred_answer(self) -> AnswerOutput:
        """
        Return the higher-scoring answer output.

        If both answers receive identical scores,
        prefer the retrieval-free answer because it
        incurs no retrieval cost.
        """

        if self.retrieval_score > self.no_retrieval_score:
            return self.retrieval_output
        
        return self.no_retrieval_output
    
    @property
    def retrieved_documents(self):
        """
        Documents returned by the retriever.
        """

        return self.retrieval_result.documents
    
    @property
    def retrieved_document_count(self) -> int:
        """
        Number of retrieved documents.
        """

        return len(self.retrieval_result.documents)
    
    @property
    def retrieval_performed(self) -> bool:
        """
        Whether any documents were retrieved.
        """

        return self.retrieval_result.retrieved