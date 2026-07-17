"""
margin_uncertainty_scorer.py

Margin-based uncertainty scoring for Training-free Adaptive Retrieval
Gating (TARG).

This module implements the margin uncertainty score defined by:

    g_t = l_(t,1) - l_(t,2)

    u_t = phi(g_t) = exp(-g_t / beta)

    U_mar(k; beta) = (1 / k) * sum_{t=1}^{k} u_t

where l_(t,1) and l_(t,2) are the largest and second-largest logits at
draft position t.

The scorer:
- consumes retrieval-free prefix logits;
- computes margins directly from logits, not probabilities;
- maps each margin to uncertainty;
- averages token uncertainties;
- performs no retrieval decision or threshold calibration.

Reference:
    Wang, Wei, and Ling,
    "Retrieval as a Decision: Training-Free Adaptive Gating for
    Efficient RAG", arXiv:2511.09803.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import TYPE_CHECKING, Iterable, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from draft_generator import DraftOutput


# ---------------------------------------------------------------------
# Immutable result models
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TokenMarginScore:
    """
    Margin evidence for one generated draft token.

    Attributes
    ----------
    position:
        Zero-based position within the retrieval-free draft.

    token_id:
        Token selected by DraftGenerator at this position.

    top1_logit:
        Largest next-token logit at this position.

    top2_logit:
        Second-largest next-token logit at this position.

    margin:
        Top-1/top-2 logit gap:

            g_t = top1_logit - top2_logit

    uncertainty:
        Margin-derived token uncertainty:

            u_t = exp(-g_t / beta)

        Small margins produce values nearer 1. Large margins produce
        values nearer 0.
    """

    position: int
    token_id: int
    top1_logit: float
    top2_logit: float
    margin: float
    uncertainty: float


@dataclass(frozen=True)
class MarginUncertaintyResult:
    """
    Complete output of margin-based TARG uncertainty scoring.

    Attributes
    ----------
    beta:
        Positive scale parameter used by the exponential link.

    score:
        Prefix-level margin uncertainty U_mar.

    token_scores:
        Per-token top-1/top-2 margins and mapped uncertainties.

    token_count:
        Number of draft positions included in the aggregate.

    mean_margin:
        Arithmetic mean of the raw logit margins.

        This is retained for diagnostics. The TARG retrieval score is
        `score`, not `mean_margin`.

    minimum_margin:
        Smallest observed top-1/top-2 margin.

    maximum_margin:
        Largest observed top-1/top-2 margin.

    minimum_uncertainty:
        Smallest mapped token uncertainty.

    maximum_uncertainty:
        Largest mapped token uncertainty.
    """

    beta: float
    score: float
    token_scores: tuple[TokenMarginScore, ...]
    token_count: int
    mean_margin: float
    minimum_margin: float
    maximum_margin: float
    minimum_uncertainty: float
    maximum_uncertainty: float

    @property
    def margins(self) -> tuple[float, ...]:
        """Return all raw top-1/top-2 logit gaps."""

        return tuple(item.margin for item in self.token_scores)

    @property
    def uncertainties(self) -> tuple[float, ...]:
        """Return all mapped token uncertainties."""

        return tuple(item.uncertainty for item in self.token_scores)

    def as_dict(self) -> dict[str, object]:
        """
        Return a serialisable representation suitable for logging,
        calibration datasets, or experiment records.
        """

        return {
            "gate": "margin",
            "beta": self.beta,
            "score": self.score,
            "token_count": self.token_count,
            "mean_margin": self.mean_margin,
            "minimum_margin": self.minimum_margin,
            "maximum_margin": self.maximum_margin,
            "minimum_uncertainty": self.minimum_uncertainty,
            "maximum_uncertainty": self.maximum_uncertainty,
            "token_scores": [
                {
                    "position": item.position,
                    "token_id": item.token_id,
                    "top1_logit": item.top1_logit,
                    "top2_logit": item.top2_logit,
                    "margin": item.margin,
                    "uncertainty": item.uncertainty,
                }
                for item in self.token_scores
            ],
        }


# ---------------------------------------------------------------------
# Margin uncertainty scorer
# ---------------------------------------------------------------------


class MarginUncertaintyScorer:
    """
    Compute the margin-based TARG uncertainty score.

    TARG obtains next-token logits from a short retrieval-free draft.
    At every generated position, this scorer finds the two largest logits,
    calculates their gap, maps the gap through an exponential decay, and
    averages the mapped values.

    The scorer deliberately does not:
    - apply softmax;
    - calculate probability margins;
    - compare the score with a threshold;
    - calibrate a threshold;
    - invoke retrieval;
    - generate a final answer.

    Those operations belong to separate pipeline components.
    """

    DEFAULT_BETA: float = 3.0

    def __init__(
        self,
        *,
        beta: float = DEFAULT_BETA,
    ) -> None:
        """
        Parameters
        ----------
        beta:
            Positive scale parameter in:

                phi(g) = exp(-g / beta)

            The paper uses beta=3 by default and tunes beta on the
            development split.
        """

        self._beta = self._validate_beta(beta)

    @property
    def beta(self) -> float:
        """Return the configured exponential-link scale."""

        return self._beta

    def score(
        self,
        draft: DraftOutput,
    ) -> MarginUncertaintyResult:
        """
        Compute U_mar from a DraftOutput.

        The top-two logits are recomputed directly from every stored
        vocabulary logit vector. This avoids relying on cached probability
        values or precomputed margins.

        Parameters
        ----------
        draft:
            Retrieval-free draft generated by DraftGenerator.

        Returns
        -------
        MarginUncertaintyResult
            Immutable prefix-level and token-level uncertainty evidence.

        Raises
        ------
        TypeError
            If the draft does not expose the expected attributes.

        ValueError
            If no draft tokens exist or the logits are malformed.
        """

        self._validate_draft(draft)

        logits = draft.logits
        generated_token_ids = draft.generated_token_ids

        return self.score_logits(
            logits=logits,
            token_ids=generated_token_ids,
        )

    def score_logits(
        self,
        *,
        logits: Tensor,
        token_ids: Tensor | Sequence[int] | None = None,
    ) -> MarginUncertaintyResult:
        """
        Compute U_mar directly from a matrix of prefix logits.

        Parameters
        ----------
        logits:
            Tensor with shape:

                [prefix_length, vocabulary_size]

            Each row must contain the complete next-token logit vector
            produced at one draft position.

        token_ids:
            Optional generated token IDs corresponding to each logit row.
            If omitted, token IDs are recorded as -1.

        Returns
        -------
        MarginUncertaintyResult
            Prefix uncertainty and per-token evidence.
        """

        normalized_logits = self._validate_logits(logits)

        prefix_length = normalized_logits.shape[0]
        normalized_token_ids = self._normalize_token_ids(
            token_ids=token_ids,
            expected_length=prefix_length,
        )

        # Compute the two largest raw logits for every prefix position.
        #
        # Input:
        #     [k, vocabulary_size]
        #
        # Output:
        #     [k, 2]
        top2_logits = torch.topk(
            normalized_logits,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        ).values

        top1 = top2_logits[:, 0]
        top2 = top2_logits[:, 1]

        # Paper-defined logit margin:
        #
        #     g_t = l_(t,1) - l_(t,2)
        #
        # No softmax operation is applied.
        margins = top1 - top2

        self._validate_margins(margins)

        # Paper-default monotone link:
        #
        #     phi(g_t) = exp(-g_t / beta)
        token_uncertainties = self.margin_to_uncertainty(margins)

        # Prefix-level margin uncertainty:
        #
        #     U_mar = (1 / k) * sum_t phi(g_t)
        uncertainty_score = self.aggregate_uncertainties(
            token_uncertainties,
        )

        token_scores = tuple(
            TokenMarginScore(
                position=position,
                token_id=normalized_token_ids[position],
                top1_logit=float(top1[position].item()),
                top2_logit=float(top2[position].item()),
                margin=float(margins[position].item()),
                uncertainty=float(token_uncertainties[position].item()),
            )
            for position in range(prefix_length)
        )

        margin_values = tuple(
            item.margin for item in token_scores
        )
        uncertainty_values = tuple(
            item.uncertainty for item in token_scores
        )

        return MarginUncertaintyResult(
            beta=self._beta,
            score=uncertainty_score,
            token_scores=token_scores,
            token_count=prefix_length,
            mean_margin=fmean(margin_values),
            minimum_margin=min(margin_values),
            maximum_margin=max(margin_values),
            minimum_uncertainty=min(uncertainty_values),
            maximum_uncertainty=max(uncertainty_values),
        )

    def compute_margin(
        self,
        logits: Tensor,
    ) -> Tensor:
        """
        Compute the top-1/top-2 logit margin for one or more positions.

        Accepted shapes
        ---------------
        [vocabulary_size]
            Returns a scalar tensor.

        [prefix_length, vocabulary_size]
            Returns shape [prefix_length].

        Formula
        -------
            g_t = l_(t,1) - l_(t,2)
        """

        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a torch.Tensor")

        if logits.ndim not in (1, 2):
            raise ValueError(
                "logits must have shape [vocabulary_size] or "
                "[prefix_length, vocabulary_size]"
            )

        if logits.shape[-1] < 2:
            raise ValueError(
                "the vocabulary dimension must contain at least two logits"
            )

        if not torch.is_floating_point(logits):
            logits = logits.float()

        if not torch.isfinite(logits).all():
            raise ValueError("logits contain NaN or infinite values")

        top2 = torch.topk(
            logits,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        ).values

        margins = top2[..., 0] - top2[..., 1]

        self._validate_margins(margins)

        return margins

    def margin_to_uncertainty(
        self,
        margin: Tensor | float,
    ) -> Tensor:
        """
        Apply the paper's exponential margin-to-uncertainty link.

        Formula
        -------
            phi(g) = exp(-g / beta)

        Interpretation
        --------------
        - g near zero: uncertainty approaches 1.
        - large g: uncertainty approaches 0.

        Returns
        -------
        torch.Tensor
            Values in the interval (0, 1], subject to finite-precision
            underflow for extremely large margins.
        """

        if isinstance(margin, Tensor):
            margin_tensor = margin
        elif isinstance(margin, (int, float)):
            margin_tensor = torch.tensor(
                margin,
                dtype=torch.float64,
            )
        else:
            raise TypeError(
                "margin must be a number or torch.Tensor"
            )

        if not torch.is_floating_point(margin_tensor):
            margin_tensor = margin_tensor.float()

        if not torch.isfinite(margin_tensor).all():
            raise ValueError(
                "margin contains NaN or infinite values"
            )

        if torch.any(margin_tensor < 0):
            raise ValueError(
                "top-1/top-2 logit margins must be non-negative"
            )

        return torch.exp(
            -margin_tensor / self._beta
        )

    def aggregate_uncertainties(
        self,
        uncertainties: Tensor | Sequence[float],
    ) -> float:
        """
        Average per-token uncertainties to produce U_mar.

        Formula
        -------
            U_mar(k; beta)
                = (1 / k) * sum_{t=1}^{k} phi(g_t)

        The aggregate is computed over the actual number of available
        draft positions. Under the paper's normal protocol, this is k
        unless generation reaches an applicable stopping condition.
        """

        if isinstance(uncertainties, Tensor):
            values = uncertainties
        else:
            values = torch.tensor(
                list(uncertainties),
                dtype=torch.float64,
            )

        if values.ndim != 1:
            raise ValueError(
                "uncertainties must be one-dimensional"
            )

        if values.numel() == 0:
            raise ValueError(
                "at least one token uncertainty is required"
            )

        if not torch.isfinite(values).all():
            raise ValueError(
                "uncertainties contain NaN or infinite values"
            )

        if torch.any(values < 0) or torch.any(values > 1):
            raise ValueError(
                "margin uncertainties must be within [0, 1]"
            )

        return float(values.mean().item())

    @staticmethod
    def _validate_beta(
        beta: float,
    ) -> float:
        if isinstance(beta, bool) or not isinstance(
            beta,
            (int, float),
        ):
            raise TypeError("beta must be a real number")

        normalized = float(beta)

        if not math.isfinite(normalized):
            raise ValueError("beta must be finite")

        if normalized <= 0:
            raise ValueError("beta must be greater than zero")

        return normalized

    @staticmethod
    def _validate_draft(
        draft: DraftOutput,
    ) -> None:
        if draft is None:
            raise ValueError("draft must not be None")

        required_attributes = (
            "logits",
            "generated_token_ids",
            "actual_prefix_length",
        )

        for attribute in required_attributes:
            if not hasattr(draft, attribute):
                raise TypeError(
                    f"draft does not expose required attribute: "
                    f"{attribute}"
                )

        if draft.actual_prefix_length <= 0:
            raise ValueError(
                "draft must contain at least one generated token"
            )

        if draft.logits.shape[0] != draft.actual_prefix_length:
            raise ValueError(
                "draft logit count does not match "
                "actual_prefix_length"
            )

        if (
            draft.generated_token_ids.numel()
            != draft.actual_prefix_length
        ):
            raise ValueError(
                "generated token count does not match "
                "actual_prefix_length"
            )

    @staticmethod
    def _validate_logits(
        logits: Tensor,
    ) -> Tensor:
        if not isinstance(logits, Tensor):
            raise TypeError("logits must be a torch.Tensor")

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape "
                "[prefix_length, vocabulary_size]"
            )

        if logits.shape[0] <= 0:
            raise ValueError(
                "logits must contain at least one prefix position"
            )

        if logits.shape[1] < 2:
            raise ValueError(
                "vocabulary size must be at least two"
            )

        if not torch.is_floating_point(logits):
            logits = logits.float()

        if not torch.isfinite(logits).all():
            raise ValueError(
                "logits contain NaN or infinite values"
            )

        return logits

    @staticmethod
    def _validate_margins(
        margins: Tensor,
    ) -> None:
        if not torch.isfinite(margins).all():
            raise ValueError(
                "computed margins contain NaN or infinite values"
            )

        # torch.topk guarantees top1 >= top2. The tolerance only protects
        # against unexpected floating-point behaviour.
        if torch.any(margins < -1e-7):
            raise RuntimeError(
                "computed top-1/top-2 margin is negative"
            )

    @staticmethod
    def _normalize_token_ids(
        *,
        token_ids: Tensor | Sequence[int] | None,
        expected_length: int,
    ) -> tuple[int, ...]:
        if token_ids is None:
            return tuple(-1 for _ in range(expected_length))

        if isinstance(token_ids, Tensor):
            if token_ids.ndim != 1:
                raise ValueError(
                    "token_ids must be one-dimensional"
                )

            normalized = tuple(
                int(value)
                for value in token_ids.detach().cpu().tolist()
            )
        else:
            normalized = tuple(
                int(value)
                for value in token_ids
            )

        if len(normalized) != expected_length:
            raise ValueError(
                "token_ids length must match the number of logit rows"
            )

        return normalized
