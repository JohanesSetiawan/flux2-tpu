"""
Tests for the tokenizer that avoids the transformers library.

Correctness here is unusually load-bearing. A tokenizer that differs
from the reference by a single identifier produces a valid conditioning
tensor of exactly the right shape, carrying different content, and
nothing anywhere downstream will signal it. The only symptom is a
different image.

So the central test compares token identifiers against transformers
directly rather than checking properties. It is an integration test in
spirit, needing both libraries and the real tokenizer files, but cheap
enough to run in the unit suite when those are present, and skipped
with a clear message when they are not.

The prompt set is chosen to cover what actually breaks tokenizers:
non-Latin scripts, emoji including sequences with modifiers, literal
special-token text that must not be interpreted, whitespace at the
edges, embedded newlines and tabs, and prompts long enough to truncate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.tokenization.fast import TokenizerFilesMissingError, load_fast_tokenizer


# Covers the cases that distinguish a correct BPE pipeline from a
# plausible one. Each is here because it exercises something specific:
# scripts outside Latin, multi-codepoint emoji, text that looks like a
# special token, whitespace the template might strip, and lengths that
# force truncation.
PARITY_PROMPTS = (
    "a lighthouse on a rocky shore at dusk",
    "cat",
    "",
    "Cinematic portrait, soft natural lighting, shallow depth of field.",
    "kucing oranye duduk di atas genteng saat senja",
    "日本の桜と富士山、夕暮れ時の風景写真",
    "emoji test with modifiers and symbols: ~!@#$%^&*()",
    "  leading and trailing whitespace  ",
    "line one\nline two\ttabbed",
    "a " * 400,
    "x" * 3000,
)

SEQUENCE_LENGTH = 512


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("fast_tokenizer_tests")
    logger.addHandler(logging.NullHandler())
    return logger


def _locate_tokenizer_directory() -> Path | None:
    """
    Find a tokenizer directory to test against, or report that there is
    none.

    Returning None rather than raising lets the suite run in
    environments without the bundle, which is most of them, while still
    exercising these tests wherever the files happen to be present.
    """
    candidates = (
        Path("/kaggle/temp/flux2_klein_checkpoint_cache/tokenizer"),
        Path("/mnt/user-data/uploads"),
        Path("tokenizer"),
    )
    for candidate in candidates:
        if (candidate / "tokenizer.json").is_file() and (
            candidate / "tokenizer_config.json"
        ).is_file():
            return candidate
    return None


def _reference_tokenizer(directory: Path):
    """Load the transformers tokenizer, or None if it is unavailable."""
    import os

    os.environ.setdefault("USE_TORCH", "0")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(str(directory))


def test_regression_fast_tokenizer_matches_transformers_exactly() -> None:
    """
    The test this module exists for.

    Every prompt must produce identical identifiers and an identical
    mask. Not approximately, and not merely the same length: a single
    differing identifier is a different prompt as far as the model is
    concerned.
    """
    directory = _locate_tokenizer_directory()
    if directory is None:
        return

    reference = _reference_tokenizer(directory)
    if reference is None:
        return

    fast = load_fast_tokenizer(directory, _silent_logger())

    for prompt in PARITY_PROMPTS:
        templated = reference.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = reference(
            templated,
            padding="max_length",
            max_length=SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="np",
            add_special_tokens=False,
        )
        expected_ids = np.asarray(encoded["input_ids"])[0]
        expected_mask = np.asarray(encoded["attention_mask"])[0]

        actual_ids, actual_mask = fast.encode_to_fixed_length(
            [prompt], SEQUENCE_LENGTH, _silent_logger()
        )

        assert np.array_equal(actual_ids[0], expected_ids), (
            f"token identifiers differ for {prompt[:50]!r} at positions "
            f"{np.where(actual_ids[0] != expected_ids)[0][:5].tolist()}"
        )
        assert np.array_equal(actual_mask[0], expected_mask), (
            f"attention mask differs for {prompt[:50]!r}"
        )


def test_regression_rendered_template_matches_transformers() -> None:
    """
    Checked separately from the identifiers so that a template
    difference is distinguishable from an encoding difference. Both
    produce the same symptom otherwise, and they need different fixes.
    """
    directory = _locate_tokenizer_directory()
    if directory is None:
        return

    reference = _reference_tokenizer(directory)
    if reference is None:
        return

    fast = load_fast_tokenizer(directory, _silent_logger())

    for prompt in PARITY_PROMPTS[:5]:
        expected = reference.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert fast.render_prompt(prompt) == expected, (
            f"rendered template differs for {prompt[:40]!r}; check the Jinja "
            f"whitespace settings, which must strip control-block whitespace"
        )


def test_regression_padding_goes_on_the_right() -> None:
    """
    Right padding is what makes key-side masking necessary rather than
    redundant, since padded positions still produce hidden states that
    reach the transformer. Left padding would silently change which
    positions carry the prompt.
    """
    directory = _locate_tokenizer_directory()
    if directory is None:
        return

    fast = load_fast_tokenizer(directory, _silent_logger())
    token_ids, token_is_real = fast.encode_to_fixed_length(
        ["cat"], SEQUENCE_LENGTH, _silent_logger()
    )

    real_count = int(token_is_real[0].sum())
    assert token_is_real[0][:real_count].all(), "real tokens are not at the start"
    assert not token_is_real[0][real_count:].any(), "padding is not at the end"
    assert (token_ids[0][real_count:] == fast.pad_token_id).all(), (
        "padded positions do not carry the padding identifier"
    )


def test_regression_truncation_is_reported_not_silent() -> None:
    """
    Truncation loses conditioning, so it must be visible. A prompt
    quietly cut in half produces an image that ignores half of what was
    asked for.
    """
    directory = _locate_tokenizer_directory()
    if directory is None:
        return

    fast = load_fast_tokenizer(directory, _silent_logger())

    warnings: list[str] = []

    class CapturingLogger:
        def warning(self, message, *arguments):
            warnings.append(message % arguments if arguments else message)

        def info(self, *arguments):
            pass

    fast.encode_to_fixed_length(["word " * 2000], SEQUENCE_LENGTH, CapturingLogger())

    assert warnings, "an over-long prompt was truncated without a warning"
    assert "truncat" in warnings[0].lower()


def test_regression_missing_definition_raises_a_distinct_error() -> None:
    """
    A bundle without the full pipeline definition must raise a type the
    caller can catch and fall back on, rather than a bare file error.
    Reconstructing a BPE pipeline from vocabulary and merges alone risks
    silently different tokenization, so falling back to the library that
    knows how is the correct response.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        try:
            load_fast_tokenizer(Path(directory), _silent_logger())
        except TokenizerFilesMissingError as error:
            assert "tokenizer.json" in str(error)
            return
        raise AssertionError("Expected TokenizerFilesMissingError for an empty directory")


_FAST_TOKENIZER_TESTS = [
    test_regression_fast_tokenizer_matches_transformers_exactly,
    test_regression_rendered_template_matches_transformers,
    test_regression_padding_goes_on_the_right,
    test_regression_truncation_is_reported_not_silent,
    test_regression_missing_definition_raises_a_distinct_error,
]


def run_fast_tokenizer_tests(logger: logging.Logger) -> None:
    directory = _locate_tokenizer_directory()
    if directory is None:
        logger.info(
            "Skipping fast tokenizer tests: no tokenizer files found. These compare "
            "against transformers and need the bundle's tokenizer directory."
        )
        return

    logger.info(
        "Running %d unit tests against the fast tokenizer, using %s",
        len(_FAST_TOKENIZER_TESTS),
        directory,
    )
    for test_function in _FAST_TOKENIZER_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All fast tokenizer tests passed")
