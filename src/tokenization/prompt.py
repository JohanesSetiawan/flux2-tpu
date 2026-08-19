"""
Prompt tokenization.

The tokenizer files and the chat template are shipped inside the
checkpoint bundle rather than fetched separately, so the exact template
the model was conditioned against travels with the weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKENIZER_SUBDIRECTORY_NAME = "tokenizer"

# The prompt is wrapped in a chat template before tokenization, since
# the text encoder is an instruction-tuned model and was conditioned on
# templated input. Thinking mode is disabled: it injects a reasoning
# block into the prompt, which would consume context positions and
# change the conditioning.
CHAT_TEMPLATE_ROLE = "user"
ENABLE_THINKING = False

# Padding goes on the right, and the resulting padded positions are fed
# to the diffusion transformer along with the real tokens. See
# src/layers/masking.py for why that makes key-side padding suppression
# necessary rather than redundant.
PADDING_SIDE = "right"


@dataclass(frozen=True)
class TokenizedPrompt:
    """
    A batch of prompts tokenized to a fixed length.

    Both arrays are needed downstream: the identifiers select
    embeddings, and the mask tells the attention layers which positions
    are padding.
    """

    token_ids: np.ndarray
    token_is_real: np.ndarray

    def __post_init__(self) -> None:
        if self.token_ids.shape != self.token_is_real.shape:
            raise ValueError(
                f"Token identifiers have shape {self.token_ids.shape} but the mask "
                f"has shape {self.token_is_real.shape}; they must match"
            )


def load_tokenizer(bundle_path: Path, logger: logging.Logger):
    """
    Load the tokenizer shipped inside the checkpoint bundle.

    The transformers library is used purely for its tokenizer, which is
    a pure Python and Rust component; this does not pull in a PyTorch
    backend and does not make PyTorch a dependency of this package.

    Reimplementing the tokenizer would be a mistake even though it is
    conceptually simple: byte-pair merge order, special token handling
    and the chat template must match the reference exactly, and any
    divergence produces a well-formed but differently-conditioned
    result that no shape or dtype check can detect.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "The transformers library is required for tokenization. Install it "
            "with: pip install transformers"
        ) from error

    tokenizer_path = bundle_path / TOKENIZER_SUBDIRECTORY_NAME
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"No tokenizer directory found at {tokenizer_path}. The checkpoint "
            f"bundle should contain one."
        )

    logger.info("Loading tokenizer from %s", tokenizer_path)
    return AutoTokenizer.from_pretrained(str(tokenizer_path))


def apply_chat_template(tokenizer, prompt: str) -> str:
    """
    Wrap a raw prompt in the chat template the model expects.

    The template is taken from the tokenizer configuration shipped with
    the checkpoint rather than reconstructed here, so it stays correct
    if the upstream template changes.
    """
    return tokenizer.apply_chat_template(
        [{"role": CHAT_TEMPLATE_ROLE, "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def tokenize_prompts(
    tokenizer,
    prompts: list[str],
    sequence_length: int,
    logger: logging.Logger,
) -> TokenizedPrompt:
    """
    Apply the chat template to each prompt and tokenize to a fixed
    length.

    Prompts longer than the fixed length are truncated. That is a real
    loss of conditioning rather than a formatting detail, so it is
    logged as a warning rather than passing silently.

    Parameters
    ----------
    tokenizer:
        A tokenizer as returned by load_tokenizer.
    prompts:
        Raw prompt strings, before templating.
    sequence_length:
        Fixed padded length. Every prompt produces exactly this many
        positions, which is what allows the whole pipeline to run with
        static shapes and therefore compile once.
    logger:
        Receives a warning if any prompt was truncated.
    """
    templated = [apply_chat_template(tokenizer, prompt) for prompt in prompts]

    encoded = tokenizer(
        templated,
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="np",
        add_special_tokens=False,
    )

    token_ids = np.asarray(encoded["input_ids"])
    token_is_real = np.asarray(encoded["attention_mask"])

    for index, single in enumerate(templated):
        untruncated_length = len(tokenizer(single, add_special_tokens=False)["input_ids"])
        if untruncated_length > sequence_length:
            logger.warning(
                "Prompt %d was truncated from %d tokens to %d; conditioning from the "
                "removed text is lost",
                index,
                untruncated_length,
                sequence_length,
            )

    return TokenizedPrompt(token_ids=token_ids, token_is_real=token_is_real)
