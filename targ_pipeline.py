from __future__ import annotations

from dataclasses import dataclass

from draft_generator import DraftGenerator
from margin_uncertainty import MarginUncertaintyScorer
from retrieval_gate import RetrievalGate
from retrieval import BaseRetriever, RetrievalResult
from answer_generator import AnswerGenerator


@dataclass(frozen=True)
class PipelineResult:

    query: str

    draft

    margin

    gate

    retrieval: RetrievalResult

    answer: str

class TARGPipeline:

    """
    Implements the complete TARG workflow.

    The pipeline contains no decision logic beyond
    orchestrating the independent components.
    """

    def __init__(

        self,

        draft_generator: DraftGenerator,

        scorer: MarginUncertaintyScorer,

        gate: RetrievalGate,

        retriever: BaseRetriever,

        answer_generator: AnswerGenerator,

    ):

        self.draft_generator = draft_generator

        self.scorer = scorer

        self.gate = gate

        self.retriever = retriever

        self.answer_generator = answer_generator

    def run(

        self,

        query: str,

    ) -> PipelineResult:

        draft = self.draft_generator.generate(
            query
        )

        margin = self.scorer.compute(
            draft=draft
        )

        gate = self.gate.decide(
            uncertainty_score=margin.score
        )

        if gate.retrieve:

            retrieval = self.retriever.retrieve(
                query=query
            )

        else:

            retrieval = RetrievalResult(

                query=query,

                retrieved=False,

                retrieval_backend="None",

                documents=[],
            )

        answer = self.answer_generator.generate(

            query=query,

            draft=draft,

            retrieval=retrieval,
        )

        return PipelineResult(

            query=query,

            draft=draft,

            margin=margin,

            gate=gate,

            retrieval=retrieval,

            answer=answer,
        )
