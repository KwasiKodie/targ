from __future__ import annotations
import time 

from dataclasses import dataclass, field

from draft_generator import DraftGenerator
from margin_uncertainty_scorer import MarginUncertaintyScorer
from retrieval_gate import RetrievalGate

from module_loader import load 

retrieval = load(
    "retrieval_runtime",
    "retrieval.py"
)

RetrievalResult = retrieval.RetrievalResult 
BaseRetriever = retrieval.BaseRetriever 


from answer_generator import AnswerGenerator


@dataclass(frozen=True)
class PipelineResult:

    query: str

    draft: DraftGenerator

    margin: MarginUncertaintyScorer

    gate: RetrievalGate

    retrieval: RetrievalResult

    answer: str

    timing: dict[str, float] = field(default_factory=dict)

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
        pipeline_start = time.perf_counter()

        t0 = time.perf_counter()
        draft = self.draft_generator.generate(
            query
        )
        draft_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        margin = self.scorer.score(
            draft=draft
        )
        margin_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        gate = self.gate.decide(
            uncertainty_score=margin.score
        )
        gate_time = time.perf_counter() - t0

        t0 = time.perf_counter()
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
        retrieval_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        answer = self.answer_generator.generate(

            query=query,

            retrieval=retrieval,
        )
        answer_time = time.perf_counter() - t0

        total_time = time.perf_counter() - pipeline_start

        return PipelineResult(

            query=query,

            draft=draft,

            margin=margin,

            gate=gate,

            retrieval=retrieval,

            answer=answer,

            timing={
                "draft_generation":draft_time,
                "margin_scoring":margin_time,
                "gate_decision":gate_time,
                "retrieval":retrieval_time,
                "answer_generation":answer_time,  
                "total":total_time 
            },
        )
