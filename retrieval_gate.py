"""
retrieval_gate.py

Training-Free Adaptive Retrieval Gating (TARG)

Implements the deterministic retrieval decision described in

    Training-free Retrieval Adaptive Gating (TARG)

Responsibilities
----------------
1. Receive a calibrated threshold τ*
2. Receive an uncertainty score U(q)
3. Decide whether retrieval should be triggered
4. Produce an immutable audit record

The RetrievalGate NEVER

- computes uncertainty
- calibrates thresholds
- performs retrieval
- generates answers

Author:
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from threshold_calibrator import CalibrationResult


# ---------------------------------------------------------
# Decision reason
# ---------------------------------------------------------


class RetrievalDecisionReason(str, Enum):

    RETRIEVAL_REQUIRED = "retrieval_required"

    RETRIEVAL_NOT_REQUIRED = "retrieval_not_required"


# ---------------------------------------------------------
# Immutable decision
# ---------------------------------------------------------


@dataclass(frozen=True)
class RetrievalDecision:

    retrieve: bool

    uncertainty_score: float

    threshold: float

    margin_type: str

    calibration_method: str

    reason: RetrievalDecisionReason

    explanation: str


# ---------------------------------------------------------
# Retrieval Gate
# ---------------------------------------------------------


class RetrievalGate:

    """
    Applies the TARG retrieval rule.

        Retrieve if

            U(q) > τ*

    otherwise answer directly.

    The gate performs no retrieval itself.
    """

    def __init__(

        self,

        calibration: CalibrationResult,

    ):

        self._validate_calibration(calibration)

        self.calibration = calibration

    # -------------------------------------------------

    def decide(

        self,

        uncertainty_score: float,

    ) -> RetrievalDecision:

        """
        Apply

            U(q) > τ*

        Returns
        -------
        RetrievalDecision
        """

        self._validate_score(uncertainty_score)

        retrieve = uncertainty_score > self.calibration.threshold

        if retrieve:

            reason = RetrievalDecisionReason.RETRIEVAL_REQUIRED

            explanation = (
                f"Retrieval triggered because "
                f"uncertainty ({uncertainty_score:.6f}) "
                f"exceeds calibrated threshold "
                f"({self.calibration.threshold:.6f})."
            )

        else:

            reason = RetrievalDecisionReason.RETRIEVAL_NOT_REQUIRED

            explanation = (
                f"No retrieval required because "
                f"uncertainty ({uncertainty_score:.6f}) "
                f"is below or equal to calibrated threshold "
                f"({self.calibration.threshold:.6f})."
            )

        return RetrievalDecision(
    retrieve=retrieve,
    uncertainty_score=uncertainty_score,
    threshold=self.calibration.threshold,
    margin_type=self.calibration.gate_name,
    calibration_method=self.calibration.mode,
    reason=reason,
    explanation=explanation,
)

    # -------------------------------------------------

    @staticmethod
    def _validate_score(score: float) -> None:

        if not isinstance(score, (int, float)):
            raise TypeError("uncertainty_score must be numeric.")

        if score < 0:
            raise ValueError(
                "Uncertainty score cannot be negative."
            )

    # -------------------------------------------------

    @staticmethod
    def _validate_calibration(

        calibration: CalibrationResult,

    ) -> None:

        if not isinstance(calibration, CalibrationResult):

            raise TypeError(
                "Expected CalibrationResult."
            )

        if calibration.threshold < 0:

            raise ValueError(
                "Threshold cannot be negative."
            )
