"""
development_record.py

Immutable labelled question-answer example used during the 
Stage 2.5 retrieval comparison experiment.

A DevelopmentRecord represents one example from the labelled 
development dataset. It contains only the information required to
execute the experiment.

It is intentionally distinct from DevelopmentExample, which is
produced after the experiment and contains only the statistics
required for threshold calibration.

Author:
TARG Reproduction
"""

from __future__ import annotations

from dataclasses import dataclass 

@dataclass(frozen=True, slots=True)
class DevelopmentRecord:
    """
    One labelled development-set question.

    Attributes
    ----------
    example_id
        Stable identifier for the example

    question
        Natural-languate question presented to the model.

    reference-answer
        Ground-truth answer used during evaluation.
    """

    example_id: str

    question: str

    reference_answer: str 

    def __post_init__(self) -> None:

        object.__setattr__(
            self,
            "example_id",
            self._validate(
                self.example_id,
                "example_id",
            ),
        )

        object.__setattr__(
            self,
            "question",
            self._validate(
                self.question,
                "question",
            ),
        )

        object.__setattr__(
            self,
            "reference_answer",
            self._validate(
                self.reference_answer,
                "reference_answer",
            ),
        )

    @staticmethod
    def _validate(
        value: str,
        field_name: str,
    ) -> str:
        
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )
        
        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} must not be empty."
            )
        
        return value 