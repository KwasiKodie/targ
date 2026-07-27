"""
draft_generator.py

Retrieval-free prefix generation for Training-free Adaptive Retrieval
Gating (TARG).

This module implements only the DecodePrefix stage of TARG. It does not
calculate uncertainty, calibrate thresholds, perform retrieval, or generate
the final answer.

Paper:
    Wang, Wei, and Ling,
    "Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG"
    arXiv:2511.09803
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

import torch
from torch import Tensor


# ---------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------


class TokenizerProtocol(Protocol):
    """
    Minimal tokenizer interface required by DraftGenerator.

    A Hugging Face tokenizer satisfies this interface.
    """

    eos_token_id: Optional[int]
    pad_token_id: Optional[int]

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool = True,
        truncation: bool = False,
    ) -> Mapping[str, Tensor]:
        ...

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        ...


# ---------------------------------------------------------------------
# Immutable outputs
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DraftStep:
    """
    Information recorded for one autoregressive prefix position.

    Attributes
    ----------
    position:
        Zero-based generated-token position within the retrieval-free draft.

    token_id:
        Token selected greedily at this position.

    logits:
        Complete next-token logit vector before token selection.
        Shape: [vocabulary_size].

    top1_logit:
        Largest next-token logit.

    top2_logit:
        Second-largest next-token logit.

    logit_margin:
        Difference between the largest and second-largest logits:
        g_t = l_(1) - l_(2).
    """

    position: int
    token_id: int
    logits: Tensor
    top1_logit: float
    top2_logit: float
    logit_margin: float


@dataclass(frozen=True)
class DraftOutput:
    """
    Immutable result of retrieval-free prefix decoding.

    Attributes
    ----------
    base_prompt:
        The retrieval-free prompt B(q).

    prompt_input_ids:
        Tokenized base prompt. Shape: [1, prompt_length].

    prompt_attention_mask:
        Attention mask for the base prompt. Shape: [1, prompt_length].

    generated_token_ids:
        Token IDs generated after the base prompt. Shape: [generated_length].

    generated_text:
        Decoded retrieval-free draft prefix.

    steps:
        One DraftStep for every generated token.

    requested_prefix_length:
        Configured k.

    actual_prefix_length:
        Number of generated tokens. This may be less than k if EOS or
        another configured stop token is produced.

    stopped_on_eos:
        Whether decoding terminated because the model emitted EOS.

    stopped_on_custom_token:
        Whether decoding terminated because a configured stop token was
        produced.
    """

    base_prompt: str
    prompt_input_ids: Tensor
    prompt_attention_mask: Tensor
    generated_token_ids: Tensor
    generated_text: str
    steps: tuple[DraftStep, ...]
    requested_prefix_length: int
    actual_prefix_length: int
    stopped_on_eos: bool
    stopped_on_custom_token: bool

    @property
    def logits(self) -> Tensor:
        """
        Stack per-step logits into shape [generated_length, vocabulary_size].
        """
        if not self.steps:
            return torch.empty((0, 0), dtype=torch.float32)

        return torch.stack(
            [step.logits for step in self.steps],
            dim=0,
        )

    @property
    def logit_margins(self) -> Tensor:
        """
        Return g_1, ..., g_k as a one-dimensional tensor.
        """
        return torch.tensor(
            [step.logit_margin for step in self.steps],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------
# Draft generator
# ---------------------------------------------------------------------


class DraftGenerator:
    """
    Generate the short retrieval-free prefix required by TARG.

    The paper performs a single up-front gate using a short prefix generated
    from B(q), without retrieved context. Main experiments use greedy
    decoding, batch size 1, and k=20 by default.

    This class intentionally does not:
    - call a retriever;
    - inject retrieved documents;
    - compute entropy, margin uncertainty, or variance;
    - decide whether retrieval is required;
    - reuse the draft as the final answer.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: TokenizerProtocol,
        prefix_length: int = 20,
        stop_token_ids: Optional[Sequence[int]] = None,
        add_special_tokens: bool = True,
        use_cache: bool = True,
        clone_logits_to_cpu: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Autoregressive causal language model exposing:
                outputs.logits
            and optionally:
                outputs.past_key_values

        tokenizer:
            Tokenizer associated with the generator model.

        prefix_length:
            Number of retrieval-free draft tokens k.
            The paper uses k=20 by default and evaluates k in {10, 20, 30}.

        stop_token_ids:
            Optional additional token IDs that should terminate prefix
            decoding. The model's EOS token is handled separately.

        add_special_tokens:
            Whether tokenizer-defined special tokens should be included.

        use_cache:
            Reuse past_key_values during autoregressive decoding.
            This is an implementation optimization; it does not alter the
            TARG decision rule.

        clone_logits_to_cpu:
            Store detached logits on CPU. This avoids retaining GPU memory
            after draft generation. Set False only when the downstream scorer
            must operate directly on-device.
        """
        if model is None:
            raise ValueError("model must not be None")

        if tokenizer is None:
            raise ValueError("tokenizer must not be None")

        if not isinstance(prefix_length, int):
            raise TypeError("prefix_length must be an integer")

        if prefix_length <= 0:
            raise ValueError("prefix_length must be greater than zero")

        normalized_stop_ids: set[int] = set()

        for token_id in stop_token_ids or ():
            if not isinstance(token_id, int):
                raise TypeError("every stop token ID must be an integer")

            if token_id < 0:
                raise ValueError("stop token IDs must be non-negative")

            normalized_stop_ids.add(token_id)

        self._model = model
        self._tokenizer = tokenizer
        self._prefix_length = prefix_length
        self._stop_token_ids = frozenset(normalized_stop_ids)
        self._add_special_tokens = add_special_tokens
        self._use_cache = use_cache
        self._clone_logits_to_cpu = clone_logits_to_cpu

    @property
    def prefix_length(self) -> int:
        return self._prefix_length

    @property
    def device(self) -> torch.device:
        """
        Return the device on which model parameters reside.
        """
        try:
            return next(self._model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError(
                "Unable to determine model device because the model "
                "has no parameters."
            ) from exc

    @torch.inference_mode()
    def generate(
        self,
        base_prompt: str,
    ) -> DraftOutput:
        """
        Decode a short retrieval-free prefix from B(q).

        Processing
        ----------
        1. Tokenize the base prompt.
        2. Run autoregressive greedy generation.
        3. At every generated position:
           - preserve the complete next-token logits;
           - identify the top-1 and top-2 logits;
           - record their margin;
           - select argmax as the next token.
        4. Stop after k tokens or a configured stopping token.
        5. Return the draft and per-step logits.

        Returns
        -------
        DraftOutput
            Immutable draft-generation evidence for later uncertainty
            scoring.
        """
        normalized_prompt = self._validate_prompt(base_prompt)
        encoded = self._tokenize(normalized_prompt)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        original_input_ids = input_ids.detach().clone().cpu()
        original_attention_mask = attention_mask.detach().clone().cpu()

        generated_ids: list[int] = []
        steps: list[DraftStep] = []

        stopped_on_eos = False
        stopped_on_custom_token = False

        model_was_training = bool(getattr(self._model, "training", False))
        self._model.eval()

        try:
            current_input_ids = input_ids
            current_attention_mask = attention_mask
            past_key_values = None

            for position in range(self._prefix_length):
                outputs = self._forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                )

                next_token_logits = self._extract_next_token_logits(outputs)
                step = self._build_step(
                    logits=next_token_logits,
                    position=position,
                )

                generated_ids.append(step.token_id)
                steps.append(step)

                if self._is_eos(step.token_id):
                    stopped_on_eos = True
                    break

                if step.token_id in self._stop_token_ids:
                    stopped_on_custom_token = True
                    break

                next_token_tensor = torch.tensor(
                    [[step.token_id]],
                    dtype=input_ids.dtype,
                    device=self.device,
                )

                if self._use_cache:
                    past_key_values = getattr(
                        outputs,
                        "past_key_values",
                        None,
                    )

                if self._use_cache and past_key_values is not None:
                    current_input_ids = next_token_tensor
                else:
                    current_input_ids = torch.cat(
                        [current_input_ids, next_token_tensor],
                        dim=-1,
                    )

                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=current_attention_mask.dtype,
                            device=current_attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )

        finally:
            if model_was_training:
                self._model.train()

        generated_token_ids = torch.tensor(
            generated_ids,
            dtype=torch.long,
        )

        generated_text = self._tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        )

        return DraftOutput(
            base_prompt=normalized_prompt,
            prompt_input_ids=original_input_ids,
            prompt_attention_mask=original_attention_mask,
            generated_token_ids=generated_token_ids,
            generated_text=generated_text,
            steps=tuple(steps),
            requested_prefix_length=self._prefix_length,
            actual_prefix_length=len(steps),
            stopped_on_eos=stopped_on_eos,
            stopped_on_custom_token=stopped_on_custom_token,
        )

    def _validate_prompt(
        self,
        base_prompt: str,
    ) -> str:
        if not isinstance(base_prompt, str):
            raise TypeError("base_prompt must be a string")

        normalized = base_prompt.strip()

        if not normalized:
            raise ValueError("base_prompt must not be empty")

        return normalized

    def _tokenize(
        self,
        base_prompt: str,
    ) -> dict[str, Tensor]:
        encoded = self._tokenizer(
            base_prompt,
            return_tensors="pt",
            add_special_tokens=self._add_special_tokens,
            truncation=False,
        )

        if "input_ids" not in encoded:
            raise RuntimeError(
                "Tokenizer output does not contain input_ids."
            )

        input_ids = encoded["input_ids"]

        if not isinstance(input_ids, Tensor):
            raise TypeError("tokenizer input_ids must be a tensor")

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch_size, sequence_length]"
            )

        if input_ids.shape[0] != 1:
            raise ValueError(
                "DraftGenerator requires batch size 1."
            )

        attention_mask = encoded.get("attention_mask")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if not isinstance(attention_mask, Tensor):
            raise TypeError("attention_mask must be a tensor")

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids"
            )

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def _forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Any,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": self._use_cache,
            "return_dict": True,
        }

        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values

        outputs = self._model(**kwargs)

        if outputs is None:
            raise RuntimeError("Model returned no output.")

        if not hasattr(outputs, "logits"):
            raise RuntimeError(
                "Model output does not expose logits."
            )

        return outputs

    def _extract_next_token_logits(
        self,
        outputs: Any,
    ) -> Tensor:
        logits = outputs.logits

        if not isinstance(logits, Tensor):
            raise TypeError("model logits must be a torch.Tensor")

        if logits.ndim != 3:
            raise ValueError(
                "model logits must have shape "
                "[batch_size, sequence_length, vocabulary_size]"
            )

        if logits.shape[0] != 1:
            raise ValueError(
                "DraftGenerator requires model batch size 1."
            )

        if logits.shape[-1] < 2:
            raise ValueError(
                "vocabulary must contain at least two tokens "
                "to compute a top-1/top-2 margin"
            )

        return logits[:, -1, :]

    def _build_step(
        self,
        *,
        logits: Tensor,
        position: int,
    ) -> DraftStep:
        """
        Build one greedy draft step directly from logits.

        The selected token is argmax(logits). The top-1/top-2 values are
        retained so the later MarginUncertaintyScorer can compute or verify
        the logit gap.
        """
        top2 = torch.topk(
            logits,
            k=2,
            dim=-1,
            largest=True,
            sorted=True,
        )

        top1_logit_tensor = top2.values[0, 0]
        top2_logit_tensor = top2.values[0, 1]

        token_id_tensor = torch.argmax(
            logits,
            dim=-1,
        )

        token_id = int(token_id_tensor.item())
        top1_logit = float(top1_logit_tensor.item())
        top2_logit = float(top2_logit_tensor.item())
        logit_margin = top1_logit - top2_logit

        stored_logits = logits[0].detach()

        if self._clone_logits_to_cpu:
            stored_logits = stored_logits.float().cpu().clone()
        else:
            stored_logits = stored_logits.clone()

        return DraftStep(
            position=position,
            token_id=token_id,
            logits=stored_logits,
            top1_logit=top1_logit,
            top2_logit=top2_logit,
            logit_margin=logit_margin,
        )

    def _is_eos(
        self,
        token_id: int,
    ) -> bool:
        eos_token_id = self._tokenizer.eos_token_id

        if eos_token_id is None:
            return False

        return token_id == eos_token_id
