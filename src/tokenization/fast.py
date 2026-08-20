"""
Tokenization without the transformers library.

Only three things are needed to turn a prompt into model input: a BPE
tokenizer, the chat template, and padding to a fixed length. The
transformers library provides all three, but importing it costs several
seconds because it loads a deep learning backend and thousands of
modules for a job that needs neither.

Measured: importing transformers takes about 6 seconds against 1 for
the tokenizers library, and on the cold, network-backed filesystems of
hosted notebooks the gap widens to tens of seconds. Loading the
tokenizer was 19 seconds on Kaggle and 47 on Colab, in both cases more
than restoring the text encoder it feeds.

This module reproduces those three behaviours exactly, using the Rust
tokenizers library for the BPE and Jinja for the template. "Exactly" is
not a figure of speech: the accompanying tests compare token
identifiers against transformers across a range of prompts, and a
single differing identifier means a different image with no error to
warn anyone.

What makes this safe rather than reckless is `tokenizer.json`. That
file defines the entire tokenization pipeline, normalizer,
pre-tokenizer, post-processor and decoder, so nothing has to be
reconstructed from vocabulary and merge files and guessed at. If a
future checkpoint ships without it, fall back to transformers rather
than rebuilding the pipeline by hand.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOKENIZER_DEFINITION_FILENAME = "tokenizer.json"
TOKENIZER_CONFIG_FILENAME = "tokenizer_config.json"

CHAT_TEMPLATE_KEY = "chat_template"
PAD_TOKEN_KEY = "pad_token"

# The prompt is wrapped as a single user turn with a generation prompt
# appended, which is what the reference does.
CHAT_TEMPLATE_ROLE = "user"

# Thinking mode injects a reasoning block into the prompt. The template
# emits an empty one when this is false, which consumes a few context
# positions; leaving it undefined instead would emit nothing and change
# the conditioning.
ENABLE_THINKING = False

# Padding goes on the right, and the padded positions are fed to the
# diffusion transformer along with the real tokens. See
# src/layers/masking.py for why that makes key-side padding suppression
# necessary rather than redundant.
PAD_ON_RIGHT = True


class TokenizerFilesMissingError(FileNotFoundError):
    """
    Raised when the bundle lacks a file this tokenizer needs.

    A distinct type so a caller can fall back to the transformers path
    rather than failing outright: a slower load is much better than no
    load, and the two situations warrant different responses.
    """


@dataclass(frozen=True)
class FastTokenizer:
    """
    A tokenizer assembled from the bundle's own files.

    Holds the BPE encoder, the rendered chat template, and the padding
    identifier. Frozen because none of these change after construction,
    and sharing one across threads is then safe by construction.
    """

    encoder: object
    chat_template: object
    pad_token_id: int

    def render_prompt(self, prompt: str) -> str:
        """
        Wrap a raw prompt in the chat template.

        The template comes from the bundle rather than being written out
        here, so it stays correct if the upstream template changes.
        """
        return self.chat_template.render(
            messages=[{"role": CHAT_TEMPLATE_ROLE, "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )

    def encode_to_fixed_length(
        self, prompts: list[str], sequence_length: int, logger: logging.Logger
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Render, tokenize, and pad or truncate to a fixed length.

        Returns token identifiers and a mask marking which positions are
        real. Both are needed downstream: the identifiers select
        embeddings, and the mask tells the attention layers which
        positions are padding.

        Truncation is a real loss of conditioning rather than a
        formatting detail, so it is logged as a warning rather than
        passing silently.
        """
        token_ids = np.full(
            (len(prompts), sequence_length), self.pad_token_id, dtype=np.int32
        )
        token_is_real = np.zeros((len(prompts), sequence_length), dtype=np.int32)

        for index, prompt in enumerate(prompts):
            rendered = self.render_prompt(prompt)
            # Special tokens are already present in the rendered
            # template, so adding more would duplicate them.
            encoded = self.encoder.encode(rendered, add_special_tokens=False).ids

            if len(encoded) > sequence_length:
                logger.warning(
                    "Prompt %d was truncated from %d tokens to %d; conditioning from "
                    "the removed text is lost",
                    index,
                    len(encoded),
                    sequence_length,
                )
                encoded = encoded[:sequence_length]

            token_ids[index, : len(encoded)] = encoded
            token_is_real[index, : len(encoded)] = 1

        return token_ids, token_is_real


def load_fast_tokenizer(tokenizer_directory: Path, logger: logging.Logger) -> FastTokenizer:
    """
    Build a tokenizer from the files in a bundle's tokenizer directory.

    Raises TokenizerFilesMissingError if anything required is absent, so
    a caller can fall back rather than fail.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError(
            "The tokenizers library is required. Install it with: pip install tokenizers"
        ) from error

    try:
        from jinja2 import Template
    except ImportError as error:
        raise ImportError(
            "Jinja2 is required to render the chat template. Install it with: "
            "pip install jinja2"
        ) from error

    definition_path = tokenizer_directory / TOKENIZER_DEFINITION_FILENAME
    config_path = tokenizer_directory / TOKENIZER_CONFIG_FILENAME

    for path in (definition_path, config_path):
        if not path.is_file():
            raise TokenizerFilesMissingError(
                f"{path.name} not found in {tokenizer_directory}. This tokenizer needs "
                f"the full pipeline definition rather than vocabulary and merge files "
                f"alone; reconstructing it by hand risks silently different tokenization."
            )

    logger.info("Loading tokenizer from %s", definition_path)
    encoder = Tokenizer.from_file(str(definition_path))

    configuration = json.loads(config_path.read_text())

    template_source = configuration.get(CHAT_TEMPLATE_KEY)
    if not template_source:
        raise TokenizerFilesMissingError(
            f"No chat template found in {config_path.name}. The text encoder is "
            f"instruction tuned and was conditioned on templated input, so an "
            f"untemplated prompt would produce different conditioning."
        )

    # Jinja must be told to strip the whitespace the template's control
    # blocks would otherwise emit. transformers configures its
    # environment the same way, and without it every control block would
    # leave a stray newline in the rendered prompt.
    chat_template = Template(
        template_source, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )

    pad_token = configuration.get(PAD_TOKEN_KEY)
    pad_token_id = encoder.token_to_id(pad_token) if pad_token else None
    if pad_token_id is None:
        raise TokenizerFilesMissingError(
            f"Could not resolve the padding token {pad_token!r} against the vocabulary"
        )

    logger.info(
        "Tokenizer ready: %d tokens in vocabulary, padding with %r (id %d)",
        encoder.get_vocab_size(),
        pad_token,
        pad_token_id,
    )
    return FastTokenizer(
        encoder=encoder, chat_template=chat_template, pad_token_id=pad_token_id
    )
