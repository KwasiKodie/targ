"""
answer_generator.py

Final answer generation for Training-Free Adaptive Retrieval Gating (TARG).

This module generates the final answer after the retrieval decision has already
been made.

Responsibilities
----------------
1. Generate an answer without retrieval.
2. Generate an answer using retrieval context.
3. Reuse the same underlying language model used by DraftGenerator.
4. Produce immutable answer evidence.

The AnswerGenerator NEVER

- computes uncertainty
- calibrates thresholds
- performs retrieval
- decides whether retrieval is required 
- evalutates answers

Paper
-----
Wang, Wei, and Ling,
"Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG"
arXiv:2511.09803
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from retrieval import RetrievalResult

# -------------------------------------------------------------------
# Tokenizer protocol
# -------------------------------------------------------------------


class TokenizerProtocol(Protocol):

    eos_token_id: int | None

    def __call__(
            self,
            text: str,
            *,
            return_tensors: str,
            add_special_tokens: bool = True,
            truncation: bool = False,
    ):
        ...

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ...


# --------------------------------------------------------------------
# Immutable outputs
#---------------------------------------------------------------------

@dataclass(frozen=True)
class AnswerOutput:
    """
    Immutable evidence describing one final answer generation.

    Unlike DraftOutput, this object stores the complete generation
    evidence required for evaluation, auditing and reproducibility.
    """

    # ----------------------------------------------------------------
    # Input evidence
    # ----------------------------------------------------------------

    query: str

    prompt: str

    prompt_input_ids: Tensor 

    prompt_attention_mask: Tensor 

    # ----------------------------------------------------------------
    # Generated evidence
    # ----------------------------------------------------------------

    generated_token_ids: Tensor

    generated_text: str

    requested_max_new_tokens: int

    actual_generated_tokens: int

    stopped_on_eos: bool

    # ----------------------------------------------------------------
    # Retrieval evidence
    # ----------------------------------------------------------------

    used_retrieval: bool

    retrieved_document_count: int 


    @property
    def answer_length(self) -> int:
        """
        Number of whitespace-separated words in the decoded answer.
        """

        return len(self.generated_text.split())

# --------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------

class AnswerGenerator:

    """
    Generate the final answer.

    The same LLM is used irrespective of whether retrieval occurs.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: TokenizerProtocol,
        max_new_tokens: int = 256,
        add_special_tokens: bool = True,
    ):
        
        if model is None:
            raise ValueError("model must not be None.")
        
        if tokenizer is None:
            raise ValueError("tokenizer must not be None.")
        
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.add_special_tokens = add_special_tokens

    # -------------------------------------------------------------

    @property
    def device(self):

        return next(self.model.parameters()).device
    
    # ------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        *,
        query: str,
        retrieval: RetrievalResult | None = None,
    ) -> AnswerOutput:
        
        query = self._validate_query(query)

        prompt = self._build_prompt(
            query=query,
            retrieval=retrieval,
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=self.add_special_tokens,
        )

        encoded = {
            k: v.to(self.device)
            for k, v in encoded.items()
        }

        prompt_input_ids = (
            encoded["input_ids"]
            .detach()
            .clone()
            .cpu()
        )

        prompt_attention_mask = (
            encoded["attention_mask"]
            .detach()
            .clone()
            .cpu()
        )

        outputs = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][
            encoded["input_ids"].shape[-1]:
        ]

        generated_token_ids = (
            generated
            .detach()
            .clone()
            .cpu()
        )

        generated_text = self.tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()

        stopped_on_eos = False

        if (
            generated_token_ids.numel() > 0
            and self.tokenizer.eos_token_id is not None
        ):
            
            stopped_on_eos = (
                int(generated_token_ids[-1].item())
                == self.tokenizer.eos_token_id
            )

        document_count = (
            0
            if retrieval is None
            else len(retrieval.documents)
        )

        return AnswerOutput(
            query=query,

            prompt=prompt,

            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            requested_max_new_tokens=self.max_new_tokens,
            actual_generated_tokens=generated_token_ids.numel(),
            stopped_on_eos=stopped_on_eos,
            used_retrieval=document_count > 0,
            retrieved_document_count=document_count,
        )
    
    def _build_prompt(
            self,
            *,
            query: str,
            retrieval: RetrievalResult | None,
    ) -> str:
        
        if (
            retrieval is None
            or not retrieval.documents
        ):
            
            return (
                "Answer the following question concisely.\n\n"
                f"Question:\n{query}\n\n"
                "Answer:\n"
            )
        context = "\n\n".join(
            document.text
            for document in retrieval.documents
        )

        return (
            "Answer the question using ONLY the provided context.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:\n"
        )