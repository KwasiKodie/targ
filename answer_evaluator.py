"""
answer_evaluator.py

Answer evaluation for Training-Free Adaptive Retrieval Gating (TARG).

This module evaluates generated answer against reference answers.

Responsibilities
----------------
1. Compare a candidate answer with a reference answer.
2. Compute one or more evaluation metrics.
3. Produce immutable evaluation evidence.

The evaluator NEVER 

- generates answers
- performs retrieval
- computes uncertainty
- calibrates thresholds
- makes retrieval decisions

Initially only Exact Match is implemented.
The design intentionally allows future metrics such as:

- Token F1
- ROUGE 
- BLEU
- BERTScore

Paper
-----
Wang, Wei, and Ling,
"Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG"
arXiv:2511.09803
"""

from __future__ import annotations 

import re
import string

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------
# Metric enumeration
# --------------------------------------------------------------

class EvaluationMetric(str, Enum):

    EXACT_MATCH = "exact_match"

# --------------------------------------------------------------
# Immutable outputs
# --------------------------------------------------------------

@dataclass(frozen=True)
class MetricResult:
    """
    Result for one evaluation metric.
    """

    metric: EvaluationMetric

    score: float

@dataclass(frozen=True)
class EvaluationResult:
    """
    Immutable evidence describing one answer evaluation.
    """

    # ----------------------------------------------------------------------
    # Raw evidence
    # ----------------------------------------------------------------------

    reference_answer: str

    candidate_answer: str

    # ----------------------------------------------------------------------
    # Normaized evidence
    # ----------------------------------------------------------------------

    normalized_reference: str

    normalized_candidate: str

    # ----------------------------------------------------------------------
    # Evaluation evidence
    # ----------------------------------------------------------------------

    metric_results: tuple[MetricResult, ...]
    
    overall_score: float

    @property
    def metrics(self) -> dict[str, float]:
        """
        Convenience mapping from metric name to score.
        """

        return {
            result.metric.value: result.score
            for result in self.metric_results
        }
    
    @property
    def matched_after_normalization(self) -> bool:
        """
        Whether the normalized strings are identical.
        """

        return (
            self.normalized_reference
            == self.normalized_candidate
        )
# --------------------------------------------------------------------------
# Abstract evaluator
# --------------------------------------------------------------------------

class BaseAnswerEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        *,
        reference: str,
        candidate: str,
    ) -> EvaluationResult:
        
        reference_normalized = self._normalize(reference)
        
        candidate_normalized = self._normalize(candidate)


# --------------------------------------------------------------------------
# Exact Match evaluator
# --------------------------------------------------------------------------

class AnswerEvaluator(BaseAnswerEvaluator):

    """
    Exact Match evaluator.

    Exact Match is currently the default metric because
    ThresholdCalibrator already defaults to 

        metric_name="exact_match"

    Additional metrics can be added later without changing
    the public interface.
    """

    # ---------------------------------------------------------------------

    def evaluate(
        self,
        *,
        reference: str,
        candidate: str,
    ) -> EvaluationResult:
        
        reference = self._validate_text(
            reference,
            "reference",
        )

        candidate = self._validate_text(
            candidate,
            "candidate",
        )

        normalized_reference = self._normalize(reference)

        normalized_candidate = self._normalize(candidate)

        score = self._exact_match(
            normalized_reference,
            normalized_candidate,
        )

        result = MetricResult(
            metric=EvaluationMetric.EXACT_MATCH,
            score=score,
        )

        return EvaluationResult(
            reference_answer=reference,
            candidate_answer=candidate,
            normalized_reference=normalized_reference,
            normalized_candidate=normalized_candidate,
            metric_results=(result,),
            overall_score=score,
        )
    
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_match(
        normalized_reference: str,
        normalized_candidate: str,
    ) -> float:
        """
        Compute Exact Match from already-normalized text.
        """

        return float(
            normalized_reference
            == normalized_candidate
        )
    
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize text before Exact Match.

        Steps
        -----
        1. lowercase
        2. remove punctuation
        3. collapse whitespace
        """

        text = text.lower()

        text = text.translate(
            str.maketrans(
                "",
                "",
                string.punctuation,
            )
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()
    
    # ------------------------------------------------------

    @staticmethod
    def _validate_text(
        text: str,
        name: str,
    ) -> str:
        
        if not isinstance(text, str):
            raise TypeError(
                f"{name} must be a string."
            )
        
        text = text.strip()

        if not text:
            raise ValueError(
                f"{name} must not be empty."
            )
        
        return text