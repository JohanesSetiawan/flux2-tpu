"""
Prompt tokenization.

The tokenizer files and the chat template are shipped inside the
checkpoint bundle rather than fetched separately, so the exact template
the model was conditioned against travels with the weights.
"""

from __future__ import annotations

import logging
import os
import threading
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

# transformers selects a deep learning backend at import time. Setting
# this to a falsy value makes it skip PyTorch, which it would otherwise
# import in full for a tokenizer that does not need it.
BACKEND_SELECTION_VARIABLE = "USE_TORCH"
BACKEND_DISABLED_VALUE = "0"


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
    Load the fastest tokenizer the bundle supports.

    Prefers the self-contained path, which needs only the tokenizers and
    Jinja libraries and avoids importing a deep learning framework for a
    job that has nothing to do with one. Falls back to transformers when
    the bundle lacks the full pipeline definition, since a slower load
    is far better than none.

    The fallback is not a formality. Reconstructing a BPE pipeline from
    vocabulary and merge files alone risks tokenization that differs
    subtly from the reference, and a differing token produces a
    different image with nothing to signal it. If the definition is
    absent, defer to the library that knows how to rebuild it.
    """
    from .fast import TokenizerFilesMissingError, load_fast_tokenizer

    tokenizer_directory = bundle_path / TOKENIZER_SUBDIRECTORY_NAME
    if not tokenizer_directory.is_dir():
        raise FileNotFoundError(
            f"No tokenizer directory found at {tokenizer_directory}. The checkpoint "
            f"bundle should contain one."
        )

    try:
        return load_fast_tokenizer(tokenizer_directory, logger)
    except TokenizerFilesMissingError as reason:
        logger.info(
            "Falling back to the transformers tokenizer: %s", reason
        )
        return _load_transformers_tokenizer(bundle_path, logger)


def _load_transformers_tokenizer(bundle_path: Path, logger: logging.Logger):
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
    # The transformers library imports a deep learning backend on first
    # use, and on a machine with PyTorch installed that costs tens of
    # seconds. Measured on Colab: importing it took 47 seconds, of which
    # nearly all was PyTorch and torch_xla being pulled in behind it.
    #
    # Only the tokenizer is wanted here, which is pure Python and Rust
    # and needs no backend at all. Declaring that before the import
    # skips the backend entirely. It must be set before transformers is
    # first imported anywhere in the process, which is why it sits here
    # rather than in a configuration object read later.
    os.environ.setdefault(BACKEND_SELECTION_VARIABLE, BACKEND_DISABLED_VALUE)

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
    # The fast tokenizer renders, encodes and pads in one step, since it
    # owns all three pieces. The transformers path keeps its original
    # sequence.
    if hasattr(tokenizer, "encode_to_fixed_length"):
        token_ids, token_is_real = tokenizer.encode_to_fixed_length(
            prompts, sequence_length, logger
        )
        return TokenizedPrompt(token_ids=token_ids, token_is_real=token_is_real)

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


def preload_tokenizer_library(logger: logging.Logger) -> threading.Thread:
    """
    Begin importing the tokenizer library on a background thread.

    Importing transformers costs tens of seconds, and on a cold machine
    that is comparable to downloading the checkpoint. Both are waiting
    rather than computing, so they can overlap: start the import, let
    the download proceed, and by the time a tokenizer is wanted the
    import has usually finished.

    Returns the thread so a caller can join it before first use. Joining
    is not optional: without it the first tokenizer call would race the
    import, and Python would serialise them anyway but with the wait
    moved somewhere less obvious.
    """
    def run_import() -> None:
        os.environ.setdefault(BACKEND_SELECTION_VARIABLE, BACKEND_DISABLED_VALUE)
        try:
            import transformers  # noqa: F401
        except ImportError:
            # Reported when the tokenizer is actually needed, with a
            # message explaining what to install. Failing here, on a
            # background thread, would surface at a confusing moment.
            pass

    logger.info("Importing the tokenizer library in the background")
    thread = threading.Thread(target=run_import, name="tokenizer-import", daemon=True)
    thread.start()
    return thread
