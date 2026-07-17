"""
threshold_calibrator.py

Offline threshold calibration for Training-free Adaptive Retrieval
Gating (TARG).

This module determines the retrieval threshold tau from a development
set after uncertainty scores and both retrieval conditions have already
been evaluated.

Supported paper-aligned calibration modes
-----------------------------------------
1. Accuracy maximisation:
       tau* = argmax_tau Accuracy_gate(tau)

2. Retrieval-budget calibration:
       tau = F_U^{-1}(1 - rho)

The runtime retrieval decision is:

       retrieve(q) iff U(q) > tau

This module performs no:
- draft generation;
- uncertainty computation;
- retrieval;
- answer generation;
- model training.

Reference
---------
"Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG"
arXiv:2511.09803.
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Iterable, Literal, Mapping, Sequence


# ---------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------


class CalibrationMode(str, Enum):
    """Supported TARG threshold-calibration strategies."""

    MAXIMIZE_ACCURACY = "maximize_accuracy"
    TARGET_RETRIEVAL_BUDGET = "target_retrieval_budget"


# ---------------------------------------------------------------------
# Development-set records
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DevelopmentExample:
    """
    One pre-evaluated development-set query.

    Attributes
    ----------
    example_id:
        Stable identifier for the development example.

    uncertainty_score:
        Scalar gate score U(q), such as U_mar produced by
        MarginUncertaintyScorer.

    no_retrieval_score:
        Evaluation score obtained when generating without retrieval:
            A^(0)(q)

        For exact match this is normally 0.0 or 1.0. It may also be a
        continuous metric such as token-level F1.

    retrieval_score:
        Evaluation score obtained when generating with retrieval:
            A^(1)(q)

    metadata:
        Optional experiment metadata. It is not used for calibration.
    """

    example_id: str
    uncertainty_score: float
    no_retrieval_score: float
    retrieval_score: float
    metadata: Mapping[str, object] | None = None

    @property
    def retrieval_gain(self) -> float:
        """
        Return Delta(q) = A^(1)(q) - A^(0)(q).
        """

        return self.retrieval_score - self.no_retrieval_score


# ---------------------------------------------------------------------
# Per-threshold evaluation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdEvaluation:
    """
    Performance of one candidate threshold on the development set.

    Attributes
    ----------
    threshold:
        Candidate tau.

    gated_score:
        Mean development metric after routing each example with:
            retrieve iff U > tau

    retrieval_rate:
        Fraction of development examples routed to retrieval.

    retrieved_count:
        Number of examples for which U > tau.

    skipped_count:
        Number of examples for which U <= tau.

    mean_selected_gain:
        Mean retrieval gain included by the gate:
            mean(Delta(q) * 1[U(q) > tau])

    no_retrieval_baseline:
        Mean metric if retrieval is never used.

    always_retrieval_baseline:
        Mean metric if retrieval is always used.
    """

    threshold: float
    gated_score: float
    retrieval_rate: float
    retrieved_count: int
    skipped_count: int
    mean_selected_gain: float
    no_retrieval_baseline: float
    always_retrieval_baseline: float


# ---------------------------------------------------------------------
# Saved calibration artifact
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """
    Final calibrated threshold and supporting evidence.

    Attributes
    ----------
    threshold:
        Selected threshold tau.

    mode:
        Calibration strategy used.

    development_size:
        Number of examples used.

    achieved_score:
        Mean gated development metric at the selected threshold.

    achieved_retrieval_rate:
        Actual fraction satisfying U > tau.

    target_retrieval_rate:
        Requested rho for budget calibration; otherwise None.

    no_retrieval_score:
        Development score under Never-RAG.

    always_retrieval_score:
        Development score under Always-RAG.

    beta:
        Margin-link beta associated with the scores.

    prefix_length:
        Prefix length k associated with the scores.

    gate_name:
        Name of uncertainty gate.

    metric_name:
        Metric used for calibration, for example "exact_match" or "f1".

    calibrated_at:
        UTC ISO-8601 timestamp.

    schema_version:
        Persistence schema version.
    """

    threshold: float
    mode: str
    development_size: int
    achieved_score: float
    achieved_retrieval_rate: float
    target_retrieval_rate: float | None
    no_retrieval_score: float
    always_retrieval_score: float
    beta: float
    prefix_length: int
    gate_name: str
    metric_name: str
    calibrated_at: str
    schema_version: int = 1

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""

        return asdict(self)


# ---------------------------------------------------------------------
# Threshold calibrator
# ---------------------------------------------------------------------


class ThresholdCalibrator:
    """
    Calibrate a scalar TARG threshold on a development set.

    The calibrator expects each development query to have already been
    evaluated in both conditions:

    1. Never-RAG:
           A^(0)(q)

    2. With-RAG:
           A^(1)(q)

    It also requires the uncertainty score U(q) computed from the
    retrieval-free draft.

    This separation prevents calibration from rerunning expensive model
    inference during every threshold sweep.
    """

    def __init__(
        self,
        *,
        gate_name: str = "margin",
        metric_name: str = "exact_match",
        beta: float = 3.0,
        prefix_length: int = 20,
        tie_breaker: Literal[
            "lower_retrieval",
            "higher_retrieval",
            "lowest_threshold",
            "highest_threshold",
        ] = "lower_retrieval",
    ) -> None:
        """
        Parameters
        ----------
        gate_name:
            Gate being calibrated. For this pipeline it should be "margin".

        metric_name:
            Development metric used to optimise accuracy.

        beta:
            Margin uncertainty parameter associated with U_mar.

        prefix_length:
            Draft prefix length k associated with the scores.

        tie_breaker:
            Rule applied when several thresholds have identical gated
            development scores.

            "lower_retrieval":
                Prefer the threshold with the smallest retrieval rate.
                This is a cost-conscious default.

            "higher_retrieval":
                Prefer the largest retrieval rate.

            "lowest_threshold":
                Prefer the numerically smallest threshold.

            "highest_threshold":
                Prefer the numerically largest threshold.

        Notes
        -----
        The paper specifies maximising development accuracy or matching
        a target retrieval budget, but it does not prescribe a universal
        tie-breaking rule. Therefore the selected rule is recorded as an
        implementation policy rather than presented as a paper requirement.
        """

        self._gate_name = self._validate_nonempty_text(
            gate_name,
            "gate_name",
        )
        self._metric_name = self._validate_nonempty_text(
            metric_name,
            "metric_name",
        )
        self._beta = self._validate_positive_finite(beta, "beta")
        self._prefix_length = self._validate_positive_integer(
            prefix_length,
            "prefix_length",
        )

        allowed_tie_breakers = {
            "lower_retrieval",
            "higher_retrieval",
            "lowest_threshold",
            "highest_threshold",
        }

        if tie_breaker not in allowed_tie_breakers:
            raise ValueError(
                f"tie_breaker must be one of "
                f"{sorted(allowed_tie_breakers)}"
            )

        self._tie_breaker = tie_breaker

    # -----------------------------------------------------------------
    # Public calibration API
    # -----------------------------------------------------------------

    def calibrate_for_accuracy(
        self,
        examples: Iterable[DevelopmentExample],
    ) -> CalibrationResult:
        """
        Select tau that maximises the development-set gated metric.

        For each candidate threshold:

            selected_score(q, tau) =
                A^(1)(q), if U(q) > tau
                A^(0)(q), otherwise

        The selected threshold is:

            tau* = argmax_tau mean_q selected_score(q, tau)

        Returns
        -------
        CalibrationResult
            Threshold and associated development performance.
        """

        data = self._validate_examples(examples)
        evaluations = self.sweep(data)

        best = self._select_best_accuracy_evaluation(evaluations)

        return self._build_result(
            selected=best,
            mode=CalibrationMode.MAXIMIZE_ACCURACY,
            target_retrieval_rate=None,
            development_size=len(data),
        )

    def calibrate_for_budget(
        self,
        examples: Iterable[DevelopmentExample],
        *,
        target_retrieval_rate: float,
    ) -> CalibrationResult:
        """
        Select tau using the empirical uncertainty-score CDF.

        Paper-defined calibration:

            tau = F_U^{-1}(1 - rho)

        where:
            F_U is the empirical CDF of U on the development set;
            rho is the target retrieval rate.

        The runtime rule remains:

            retrieve iff U > tau

        Because an empirical distribution is discrete and duplicate scores
        can occur, the achieved retrieval rate may differ slightly from the
        requested rate.
        """

        data = self._validate_examples(examples)
        rho = self._validate_probability(
            target_retrieval_rate,
            "target_retrieval_rate",
        )

        scores = sorted(
            item.uncertainty_score
            for item in data
        )

        threshold = self._empirical_quantile(
            scores=scores,
            probability=1.0 - rho,
        )

        evaluation = self.evaluate_threshold(
            data,
            threshold=threshold,
        )

        return self._build_result(
            selected=evaluation,
            mode=CalibrationMode.TARGET_RETRIEVAL_BUDGET,
            target_retrieval_rate=rho,
            development_size=len(data),
        )

    def sweep(
        self,
        examples: Iterable[DevelopmentExample],
        *,
        candidate_thresholds: Sequence[float] | None = None,
    ) -> tuple[ThresholdEvaluation, ...]:
        """
        Evaluate the accuracy-efficiency frontier over thresholds.

        If no threshold grid is supplied, the method constructs every
        threshold needed to represent all distinct routing partitions under:

            retrieve iff U > tau

        Candidate thresholds include:
        - one value immediately below the minimum score, producing
          Always-RAG;
        - each unique observed uncertainty score, including the maximum,
          which produces Never-RAG under the strict `>` rule.

        This exact partition sweep is more reliable than imposing an
        arbitrary fixed grid.
        """

        data = self._validate_examples(examples)

        if candidate_thresholds is None:
            thresholds = self._routing_complete_thresholds(data)
        else:
            thresholds = self._validate_thresholds(
                candidate_thresholds
            )

        return tuple(
            self.evaluate_threshold(
                data,
                threshold=threshold,
            )
            for threshold in thresholds
        )

    def evaluate_threshold(
        self,
        examples: Iterable[DevelopmentExample],
        *,
        threshold: float,
    ) -> ThresholdEvaluation:
        """
        Evaluate one threshold using TARG's strict retrieval condition.
        """

        data = self._validate_examples(examples)
        tau = self._validate_finite_number(
            threshold,
            "threshold",
        )

        selected_scores: list[float] = []
        selected_gains: list[float] = []
        retrieved_count = 0

        for example in data:
            should_retrieve = (
                example.uncertainty_score > tau
            )

            if should_retrieve:
                retrieved_count += 1
                selected_scores.append(
                    example.retrieval_score
                )
                selected_gains.append(
                    example.retrieval_gain
                )
            else:
                selected_scores.append(
                    example.no_retrieval_score
                )
                selected_gains.append(0.0)

        development_size = len(data)
        skipped_count = development_size - retrieved_count

        no_retrieval_baseline = fmean(
            item.no_retrieval_score
            for item in data
        )
        always_retrieval_baseline = fmean(
            item.retrieval_score
            for item in data
        )

        return ThresholdEvaluation(
            threshold=tau,
            gated_score=fmean(selected_scores),
            retrieval_rate=(
                retrieved_count / development_size
            ),
            retrieved_count=retrieved_count,
            skipped_count=skipped_count,
            mean_selected_gain=fmean(selected_gains),
            no_retrieval_baseline=no_retrieval_baseline,
            always_retrieval_baseline=always_retrieval_baseline,
        )

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(
        self,
        result: CalibrationResult,
        path: str | Path,
    ) -> Path:
        """
        Save a calibration artifact atomically as JSON.

        Atomic replacement prevents inference workers from reading a
        partially written threshold file.
        """

        if not isinstance(result, CalibrationResult):
            raise TypeError(
                "result must be a CalibrationResult"
            )

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = json.dumps(
            result.as_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)

            os.replace(temporary_path, destination)

        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

        return destination

    @staticmethod
    def load(
        path: str | Path,
    ) -> CalibrationResult:
        """
        Load and validate a saved calibration artifact.
        """

        source = Path(path).expanduser().resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Calibration file does not exist: {source}"
            )

        with source.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(
                "Calibration artifact must contain a JSON object"
            )

        required_fields = {
            "threshold",
            "mode",
            "development_size",
            "achieved_score",
            "achieved_retrieval_rate",
            "target_retrieval_rate",
            "no_retrieval_score",
            "always_retrieval_score",
            "beta",
            "prefix_length",
            "gate_name",
            "metric_name",
            "calibrated_at",
            "schema_version",
        }

        missing = required_fields - payload.keys()

        if missing:
            raise ValueError(
                "Calibration artifact is missing fields: "
                + ", ".join(sorted(missing))
            )

        result = CalibrationResult(
            threshold=float(payload["threshold"]),
            mode=str(payload["mode"]),
            development_size=int(
                payload["development_size"]
            ),
            achieved_score=float(
                payload["achieved_score"]
            ),
            achieved_retrieval_rate=float(
                payload["achieved_retrieval_rate"]
            ),
            target_retrieval_rate=(
                None
                if payload["target_retrieval_rate"] is None
                else float(
                    payload["target_retrieval_rate"]
                )
            ),
            no_retrieval_score=float(
                payload["no_retrieval_score"]
            ),
            always_retrieval_score=float(
                payload["always_retrieval_score"]
            ),
            beta=float(payload["beta"]),
            prefix_length=int(payload["prefix_length"]),
            gate_name=str(payload["gate_name"]),
            metric_name=str(payload["metric_name"]),
            calibrated_at=str(payload["calibrated_at"]),
            schema_version=int(payload["schema_version"]),
        )

        ThresholdCalibrator._validate_loaded_result(result)

        return result

    # -----------------------------------------------------------------
    # Internal selection
    # -----------------------------------------------------------------

    def _select_best_accuracy_evaluation(
        self,
        evaluations: Sequence[ThresholdEvaluation],
    ) -> ThresholdEvaluation:
        if not evaluations:
            raise ValueError(
                "at least one threshold evaluation is required"
            )

        best_score = max(
            item.gated_score
            for item in evaluations
        )

        tied = [
            item
            for item in evaluations
            if math.isclose(
                item.gated_score,
                best_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        if self._tie_breaker == "lower_retrieval":
            return min(
                tied,
                key=lambda item: (
                    item.retrieval_rate,
                    -item.threshold,
                ),
            )

        if self._tie_breaker == "higher_retrieval":
            return max(
                tied,
                key=lambda item: (
                    item.retrieval_rate,
                    -item.threshold,
                ),
            )

        if self._tie_breaker == "lowest_threshold":
            return min(
                tied,
                key=lambda item: item.threshold,
            )

        return max(
            tied,
            key=lambda item: item.threshold,
        )

    def _build_result(
        self,
        *,
        selected: ThresholdEvaluation,
        mode: CalibrationMode,
        target_retrieval_rate: float | None,
        development_size: int,
    ) -> CalibrationResult:
        return CalibrationResult(
            margin_type = self.callibration.gate_name
            callibration_method = self.callibration.mode
            threshold=selected.threshold,
            mode=mode.value,
            development_size=development_size,
            achieved_score=selected.gated_score,
            achieved_retrieval_rate=(
                selected.retrieval_rate
            ),
            target_retrieval_rate=target_retrieval_rate,
            no_retrieval_score=(
                selected.no_retrieval_baseline
            ),
            always_retrieval_score=(
                selected.always_retrieval_baseline
            ),
            beta=self._beta,
            prefix_length=self._prefix_length,
            gate_name=self._gate_name,
            metric_name=self._metric_name,
            calibrated_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

    # -----------------------------------------------------------------
    # Threshold construction
    # -----------------------------------------------------------------

    @staticmethod
    def _routing_complete_thresholds(
        examples: Sequence[DevelopmentExample],
    ) -> tuple[float, ...]:
        unique_scores = sorted({
            item.uncertainty_score
            for item in examples
        })

        minimum = unique_scores[0]

        # Under U > tau, a threshold just below the minimum routes every
        # development example to retrieval.
        all_retrieve_threshold = math.nextafter(
            minimum,
            -math.inf,
        )

        return tuple(
            [all_retrieve_threshold, *unique_scores]
        )

    @staticmethod
    def _empirical_quantile(
        *,
        scores: Sequence[float],
        probability: float,
    ) -> float:
        """
        Return the inverse empirical CDF:

            F_n^{-1}(p) = inf{x : F_n(x) >= p}

        This is the order-statistic definition rather than an interpolated
        numerical quantile.
        """

        if not scores:
            raise ValueError(
                "scores must not be empty"
            )

        if probability <= 0.0:
            # A threshold immediately below min gives retrieval rate 1.
            return math.nextafter(
                scores[0],
                -math.inf,
            )

        if probability >= 1.0:
            # tau=max(U) gives retrieval rate 0 because routing uses U > tau.
            return scores[-1]

        n = len(scores)

        # Smallest one-based rank r for which r/n >= probability.
        rank = math.ceil(probability * n)

        # Convert to zero-based index.
        index = max(0, min(n - 1, rank - 1))

        return scores[index]

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @classmethod
    def _validate_examples(
        cls,
        examples: Iterable[DevelopmentExample],
    ) -> tuple[DevelopmentExample, ...]:
        if examples is None:
            raise ValueError(
                "examples must not be None"
            )

        data = tuple(examples)

        if not data:
            raise ValueError(
                "development set must contain at least one example"
            )

        seen_ids: set[str] = set()

        for position, example in enumerate(data):
            if not isinstance(
                example,
                DevelopmentExample,
            ):
                raise TypeError(
                    f"development item {position} must be "
                    "a DevelopmentExample"
                )

            example_id = cls._validate_nonempty_text(
                example.example_id,
                f"examples[{position}].example_id",
            )

            if example_id in seen_ids:
                raise ValueError(
                    f"duplicate example_id: {example_id}"
                )

            seen_ids.add(example_id)

            uncertainty = cls._validate_finite_number(
                example.uncertainty_score,
                f"examples[{position}].uncertainty_score",
            )

            # Margin U_mar should lie in (0, 1], although zero can occur
            # numerically through exponential underflow.
            if uncertainty < 0.0 or uncertainty > 1.0:
                raise ValueError(
                    f"examples[{position}].uncertainty_score "
                    "must be within [0, 1]"
                )

            cls._validate_finite_number(
                example.no_retrieval_score,
                f"examples[{position}].no_retrieval_score",
            )
            cls._validate_finite_number(
                example.retrieval_score,
                f"examples[{position}].retrieval_score",
            )

        return data

    @classmethod
    def _validate_thresholds(
        cls,
        thresholds: Sequence[float],
    ) -> tuple[float, ...]:
        if not thresholds:
            raise ValueError(
                "candidate_thresholds must not be empty"
            )

        normalized = sorted({
            cls._validate_finite_number(
                threshold,
                "candidate threshold",
            )
            for threshold in thresholds
        })

        return tuple(normalized)

    @staticmethod
    def _validate_loaded_result(
        result: CalibrationResult,
    ) -> None:
        finite_fields = {
            "threshold": result.threshold,
            "achieved_score": result.achieved_score,
            "achieved_retrieval_rate": (
                result.achieved_retrieval_rate
            ),
            "no_retrieval_score": (
                result.no_retrieval_score
            ),
            "always_retrieval_score": (
                result.always_retrieval_score
            ),
            "beta": result.beta,
        }

        for name, value in finite_fields.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"loaded {name} must be finite"
                )

        if result.development_size <= 0:
            raise ValueError(
                "loaded development_size must be positive"
            )

        if result.prefix_length <= 0:
            raise ValueError(
                "loaded prefix_length must be positive"
            )

        if result.beta <= 0:
            raise ValueError(
                "loaded beta must be positive"
            )

        if not 0.0 <= result.achieved_retrieval_rate <= 1.0:
            raise ValueError(
                "loaded achieved_retrieval_rate must lie in [0, 1]"
            )

        if result.target_retrieval_rate is not None:
            if not 0.0 <= result.target_retrieval_rate <= 1.0:
                raise ValueError(
                    "loaded target_retrieval_rate must lie in [0, 1]"
                )

        valid_modes = {
            item.value
            for item in CalibrationMode
        }

        if result.mode not in valid_modes:
            raise ValueError(
                f"loaded calibration mode is invalid: {result.mode}"
            )

        if result.schema_version != 1:
            raise ValueError(
                "unsupported calibration schema version: "
                f"{result.schema_version}"
            )

    @staticmethod
    def _validate_probability(
        value: float,
        name: str,
    ) -> float:
        normalized = ThresholdCalibrator._validate_finite_number(
            value,
            name,
        )

        if normalized < 0.0 or normalized > 1.0:
            raise ValueError(
                f"{name} must lie in [0, 1]"
            )

        return normalized

    @staticmethod
    def _validate_positive_finite(
        value: float,
        name: str,
    ) -> float:
        normalized = ThresholdCalibrator._validate_finite_number(
            value,
            name,
        )

        if normalized <= 0.0:
            raise ValueError(
                f"{name} must be greater than zero"
            )

        return normalized

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than zero"
            )

        return value

    @staticmethod
    def _validate_finite_number(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be a real number"
            )

        normalized = float(value)

        if not math.isfinite(normalized):
            raise ValueError(
                f"{name} must be finite"
            )

        return normalized

    @staticmethod
    def _validate_nonempty_text(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{name} must not be empty"
            )

        return normalized
