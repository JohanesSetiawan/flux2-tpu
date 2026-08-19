"""
Tests for src.layers.masking.

The mask is the most consequential piece of the text encoder and the
one whose errors are hardest to detect downstream, so it is tested
directly and structurally rather than only through its effect on
outputs.

The test that matters most is
test_regression_padded_query_cannot_attend_to_padded_key. Under causal
attention with right-side padding it is tempting to conclude that
padding needs no mask, since real tokens always precede padding. That
holds for real queries but not for padded ones, and padded positions
are part of the conditioning handed to the diffusion transformer, so
their hidden states are real input rather than discarded leftovers.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.layers.masking import MASKED_SCORE_BIAS, causal_padding_mask


SEQUENCE_LENGTHS = (1, 4, 9)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _right_padded_mask(batch: int, sequence_length: int, real_lengths: list[int]) -> np.ndarray:
    mask = np.zeros((batch, sequence_length), dtype=np.int64)
    for index, real_length in enumerate(real_lengths):
        mask[index, :real_length] = 1
    return mask


def test_smoke_mask_has_broadcastable_head_axis() -> None:
    token_is_real = _right_padded_mask(2, 6, [6, 3])

    mask = causal_padding_mask(jnp.asarray(token_is_real), 6)

    assert mask.shape == (2, 1, 6, 6), (
        "mask must carry a singleton head axis so it broadcasts across heads"
    )


def test_regression_mask_rejects_length_disagreement() -> None:
    token_is_real = _right_padded_mask(1, 5, [5])

    try:
        causal_padding_mask(jnp.asarray(token_is_real), sequence_length=8)
    except ValueError as error:
        assert "length" in str(error)
        return
    raise AssertionError("Expected ValueError when the requested length disagrees with the mask")


def test_regression_future_positions_are_always_masked() -> None:
    """Causality: no query may attend to a key at a later position."""
    for sequence_length in SEQUENCE_LENGTHS:
        token_is_real = np.ones((1, sequence_length), dtype=np.int64)
        mask = np.asarray(causal_padding_mask(jnp.asarray(token_is_real), sequence_length))[0, 0]

        for query, key in itertools.product(range(sequence_length), range(sequence_length)):
            if key > query:
                assert mask[query, key] == MASKED_SCORE_BIAS, (
                    f"query {query} was permitted to attend to future key {key}"
                )
            else:
                assert mask[query, key] == 0.0, (
                    f"query {query} was blocked from attending to past key {key}"
                )


def test_regression_padded_query_cannot_attend_to_padded_key() -> None:
    """
    The case that makes key-side padding suppression necessary rather
    than redundant.

    With three real tokens in a sequence of six, query position 5 is
    padding. Causality alone would let it attend to positions 0 through
    5, which includes the padded positions 3 and 4. The mask must block
    those while still permitting the real positions 0 through 2.
    """
    sequence_length, real_length = 6, 3
    token_is_real = _right_padded_mask(1, sequence_length, [real_length])

    mask = np.asarray(causal_padding_mask(jnp.asarray(token_is_real), sequence_length))[0, 0]

    padded_query = 5
    for key in range(real_length):
        assert mask[padded_query, key] == 0.0, (
            f"padded query {padded_query} was blocked from real key {key}"
        )
    for key in range(real_length, padded_query + 1):
        assert mask[padded_query, key] == MASKED_SCORE_BIAS, (
            f"padded query {padded_query} was permitted to attend to padded key {key}; "
            f"causality alone does not prevent this, which is why the key-side "
            f"padding condition exists"
        )


def test_regression_no_row_is_entirely_masked() -> None:
    """
    Every query must retain at least one permitted key, or its softmax
    would be undefined.

    This holds for any non-empty prompt because position zero is always
    real and always at or before every query. The test asserts it across
    padding levels rather than assuming it.
    """
    sequence_length = 8
    for real_length in range(1, sequence_length + 1):
        token_is_real = _right_padded_mask(1, sequence_length, [real_length])
        mask = np.asarray(causal_padding_mask(jnp.asarray(token_is_real), sequence_length))[0, 0]

        permitted_per_query = (mask == 0.0).sum(axis=-1)
        assert np.all(permitted_per_query >= 1), (
            f"some query had no permitted key at real_length={real_length}"
        )


def test_regression_mask_is_independent_per_batch_element() -> None:
    """
    Prompts of different lengths in one batch must receive different
    masks. A mask computed from the batch as a whole, rather than per
    element, would silently apply one prompt's padding to another.
    """
    sequence_length = 6
    token_is_real = _right_padded_mask(2, sequence_length, [6, 2])

    mask = np.asarray(causal_padding_mask(jnp.asarray(token_is_real), sequence_length))

    fully_real, mostly_padded = mask[0, 0], mask[1, 0]
    assert fully_real[5, 4] == 0.0, "fully real prompt should permit attention to key 4"
    assert mostly_padded[5, 4] == MASKED_SCORE_BIAS, (
        "prompt with two real tokens should block attention to padded key 4"
    )


def test_regression_masked_bias_is_finite() -> None:
    """
    The bias must be large but finite. Negative infinity produces NaN
    for any fully masked row, since softmax then divides zero by zero;
    a large finite value degrades to a uniform distribution instead.
    """
    assert np.isfinite(MASKED_SCORE_BIAS), "masked bias must be finite to avoid NaN in softmax"
    assert MASKED_SCORE_BIAS < -1e6, "masked bias must be large enough to zero out attention weight"


_MASKING_TESTS = [
    test_smoke_mask_has_broadcastable_head_axis,
    test_regression_mask_rejects_length_disagreement,
    test_regression_future_positions_are_always_masked,
    test_regression_padded_query_cannot_attend_to_padded_key,
    test_regression_no_row_is_entirely_masked,
    test_regression_mask_is_independent_per_batch_element,
    test_regression_masked_bias_is_finite,
]


def run_masking_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against the attention mask", len(_MASKING_TESTS))
    for test_function in _MASKING_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All masking tests passed")
