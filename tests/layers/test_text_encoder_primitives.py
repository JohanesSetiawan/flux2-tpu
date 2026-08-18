"""
Tests for the text encoder primitives in src.layers.

Both primitives here are places where a plausible-looking alternative
implementation produces output that is wrong in a way no shape check
can detect, so both are compared against oracles written directly from
the reference implementation's own sequence of operations:

RMS normalization is order-sensitive. Casting back to the input dtype
before applying the scale, rather than after, changes the rounding.
The oracle reproduces the reference order explicitly so a reordering
here would fail rather than pass with a slightly different result.

Rotary embedding is convention-sensitive. The half-split pairing used
by Qwen3 pairs element i with element i + head_dim/2; the interleaved
convention pairs element 2i with 2i+1. Both are self-consistent, both
produce correctly shaped output, and only one matches the checkpoint.
test_regression_rotary_uses_half_split_not_interleaved_pairing asserts
the distinction directly, since an oracle sharing the implementation's
assumption could not.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.config import TextEncoderConfig
from src.layers import apply_rotary_embedding, rms_normalization, rotary_frequency_table


NUMERICAL_TOLERANCE = 1e-10

# Swept so that an error visible only at one particular width or length
# is caught at another. Odd and even head counts, and a head dimension
# that is not a power of two, are all included deliberately.
FEATURE_WIDTHS = (4, 6, 16)
SEQUENCE_LENGTHS = (1, 5, 12)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rms_normalization_oracle(
    activations: np.ndarray, scale: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Reference RMS normalization, following the reference
    implementation's operation order exactly: promote, compute the mean
    square, scale by its reciprocal square root, demote, then multiply
    by the learned scale. Test-only.
    """
    input_dtype = activations.dtype
    promoted = activations.astype(np.float64)
    mean_square = np.mean(promoted ** 2, axis=-1, keepdims=True)
    normalized = promoted / np.sqrt(mean_square + epsilon)
    return normalized.astype(input_dtype) * scale


def _rotary_tables_oracle(
    sequence_length: int, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reference rotary tables, built by explicit index arithmetic rather
    than vectorised operations. Test-only.
    """
    cosine = np.zeros((sequence_length, head_dim), dtype=np.float64)
    sine = np.zeros((sequence_length, head_dim), dtype=np.float64)
    half = head_dim // 2

    for position, pair_index in itertools.product(range(sequence_length), range(half)):
        inverse_frequency = 1.0 / (theta ** ((2.0 * pair_index) / head_dim))
        angle = position * inverse_frequency
        # The same angle governs both members of the pair, which sit
        # half a head dimension apart.
        cosine[position, pair_index] = np.cos(angle)
        cosine[position, pair_index + half] = np.cos(angle)
        sine[position, pair_index] = np.sin(angle)
        sine[position, pair_index + half] = np.sin(angle)

    return cosine, sine


def _apply_rotary_oracle(
    vectors: np.ndarray, cosine: np.ndarray, sine: np.ndarray
) -> np.ndarray:
    """
    Reference rotary application, rotating each pair as an explicit two
    dimensional rotation rather than through a concatenate. Test-only.
    """
    output = np.zeros_like(vectors)
    head_dim = vectors.shape[-1]
    half = head_dim // 2

    for pair_index in range(half):
        first = vectors[..., pair_index]
        second = vectors[..., pair_index + half]
        cosine_component = cosine[:, pair_index]
        sine_component = sine[:, pair_index]

        output[..., pair_index] = first * cosine_component - second * sine_component
        output[..., pair_index + half] = second * cosine_component + first * sine_component

    return output


def test_smoke_rms_normalization_preserves_shape_and_dtype() -> None:
    rng = _random_generator(seed=0)
    activations = jnp.asarray(rng.standard_normal((2, 7, 16)), dtype=jnp.float32)
    scale = jnp.asarray(rng.standard_normal((16,)), dtype=jnp.float32)

    output = rms_normalization(activations, scale, epsilon=1e-6)

    assert output.shape == activations.shape
    assert output.dtype == jnp.float32


def test_regression_rms_normalization_matches_oracle_across_widths() -> None:
    for width, sequence_length in itertools.product(FEATURE_WIDTHS, SEQUENCE_LENGTHS):
        rng = _random_generator(seed=width * 10 + sequence_length)
        epsilon = 1e-6

        activations = rng.standard_normal((2, sequence_length, width))
        scale = rng.standard_normal((width,))

        expected = _rms_normalization_oracle(activations, scale, epsilon)
        actual = np.asarray(
            rms_normalization(
                jnp.asarray(activations, dtype=jnp.float64),
                jnp.asarray(scale, dtype=jnp.float64),
                epsilon,
            )
        )

        assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE), (
            f"mismatch at width={width} sequence_length={sequence_length}, "
            f"max abs diff {np.max(np.abs(actual - expected))}"
        )


def test_regression_rms_normalization_does_not_recentre() -> None:
    """
    RMS normalization divides by the root mean square and subtracts no
    mean, which is what distinguishes it from layer normalization.

    The assertion is derived rather than chosen. Dividing every element
    by the same constant divides their mean by that constant too, so for
    a unit scale the output mean must equal the input mean divided by
    the input root mean square. An implementation that subtracted the
    mean would instead produce an output mean of zero.

    Note that the output mean cannot exceed one in magnitude, since the
    output has unit root mean square by construction. An earlier version
    of this test asserted a mean greater than one and failed against a
    correct implementation; the bound was picked arbitrarily rather than
    derived, which is exactly the mistake this version avoids.
    """
    rng = _random_generator(seed=100)
    width = 16
    activations = rng.standard_normal((1, 1, width)) + 10.0

    output = np.asarray(
        rms_normalization(
            jnp.asarray(activations, dtype=jnp.float64),
            jnp.ones((width,), dtype=jnp.float64),
            1e-6,
        )
    )

    input_root_mean_square = float(np.sqrt(np.mean(activations ** 2)))
    expected_output_mean = float(np.mean(activations)) / input_root_mean_square

    assert abs(float(np.mean(output)) - expected_output_mean) < 1e-6, (
        f"output mean is {float(np.mean(output))}, expected {expected_output_mean}; "
        f"a value near zero would indicate mean subtraction was applied"
    )

    root_mean_square = float(np.sqrt(np.mean(output ** 2)))
    assert abs(root_mean_square - 1.0) < 1e-6, (
        f"output root mean square is {root_mean_square}, expected approximately one"
    )


def test_regression_rotary_tables_match_oracle() -> None:
    theta = TextEncoderConfig().rope_theta

    for head_dim, sequence_length in itertools.product(FEATURE_WIDTHS, SEQUENCE_LENGTHS):
        expected_cosine, expected_sine = _rotary_tables_oracle(sequence_length, head_dim, theta)
        actual_cosine, actual_sine = rotary_frequency_table(sequence_length, head_dim, theta)

        assert np.allclose(np.asarray(actual_cosine), expected_cosine, atol=1e-6), (
            f"cosine table mismatch at head_dim={head_dim} length={sequence_length}"
        )
        assert np.allclose(np.asarray(actual_sine), expected_sine, atol=1e-6), (
            f"sine table mismatch at head_dim={head_dim} length={sequence_length}"
        )


def test_regression_rotary_table_rejects_odd_head_dimension() -> None:
    try:
        rotary_frequency_table(sequence_length=4, head_dim=7, theta=10000.0)
    except ValueError as error:
        assert "even" in str(error)
        return
    raise AssertionError("Expected ValueError for an odd head dimension")


def test_regression_apply_rotary_matches_oracle() -> None:
    theta = TextEncoderConfig().rope_theta

    for head_dim, sequence_length in itertools.product(FEATURE_WIDTHS, SEQUENCE_LENGTHS):
        rng = _random_generator(seed=head_dim * 100 + sequence_length)
        vectors = rng.standard_normal((2, 3, sequence_length, head_dim))

        cosine, sine = _rotary_tables_oracle(sequence_length, head_dim, theta)
        expected = _apply_rotary_oracle(vectors, cosine, sine)

        table_cosine, table_sine = rotary_frequency_table(sequence_length, head_dim, theta)
        actual = np.asarray(
            apply_rotary_embedding(
                jnp.asarray(vectors, dtype=jnp.float64),
                table_cosine.astype(jnp.float64),
                table_sine.astype(jnp.float64),
            )
        )

        assert np.allclose(actual, expected, atol=1e-6), (
            f"mismatch at head_dim={head_dim} length={sequence_length}, "
            f"max abs diff {np.max(np.abs(actual - expected))}"
        )


def test_regression_rotary_uses_half_split_not_interleaved_pairing() -> None:
    """
    Distinguish the two rotary pairing conventions directly.

    A unit vector placed in a single feature position is rotated. Under
    the half-split convention its partner is at index i + head_dim/2;
    under the interleaved convention the partner of index 0 would be
    index 1. Checking which position receives the rotated component
    identifies the convention unambiguously, which an oracle sharing the
    implementation's assumption cannot do.
    """
    head_dim = 8
    half = head_dim // 2
    sequence_length = 2
    theta = 10000.0

    vectors = np.zeros((1, 1, sequence_length, head_dim))
    vectors[0, 0, 1, 0] = 1.0  # position 1, feature 0

    cosine, sine = rotary_frequency_table(sequence_length, head_dim, theta)
    rotated = np.asarray(
        apply_rotary_embedding(
            jnp.asarray(vectors, dtype=jnp.float64),
            cosine.astype(jnp.float64),
            sine.astype(jnp.float64),
        )
    )[0, 0, 1]

    assert abs(rotated[half]) > 1e-9, (
        "no rotated component appeared at index head_dim/2, which means the "
        "implementation is not using the half-split pairing Qwen3 requires"
    )
    assert abs(rotated[1]) < 1e-12, (
        "a rotated component appeared at index 1, which indicates the interleaved "
        "pairing convention rather than the half-split one"
    )


def test_regression_rotary_preserves_vector_norm() -> None:
    """
    Rotation is norm-preserving by construction. A violation would mean
    the cosine and sine components are not describing a rotation, for
    instance through a table built with mismatched halves.
    """
    rng = _random_generator(seed=900)
    head_dim, sequence_length = 16, 6
    vectors = rng.standard_normal((2, 3, sequence_length, head_dim))

    cosine, sine = rotary_frequency_table(sequence_length, head_dim, TextEncoderConfig().rope_theta)
    rotated = np.asarray(
        apply_rotary_embedding(
            jnp.asarray(vectors, dtype=jnp.float64),
            cosine.astype(jnp.float64),
            sine.astype(jnp.float64),
        )
    )

    original_norms = np.linalg.norm(vectors, axis=-1)
    rotated_norms = np.linalg.norm(rotated, axis=-1)

    assert np.allclose(original_norms, rotated_norms, atol=1e-9), (
        "rotation changed vector norms, so the transform is not a rotation"
    )


def test_regression_rotary_leaves_position_zero_unchanged() -> None:
    """
    At position zero every angle is zero, so the rotation is the
    identity. This catches an off-by-one in the position index, which
    would otherwise shift every token's encoding by one place.
    """
    rng = _random_generator(seed=901)
    head_dim, sequence_length = 8, 4
    vectors = rng.standard_normal((1, 2, sequence_length, head_dim))

    cosine, sine = rotary_frequency_table(sequence_length, head_dim, TextEncoderConfig().rope_theta)
    rotated = np.asarray(
        apply_rotary_embedding(
            jnp.asarray(vectors, dtype=jnp.float64),
            cosine.astype(jnp.float64),
            sine.astype(jnp.float64),
        )
    )

    assert np.allclose(rotated[:, :, 0, :], vectors[:, :, 0, :], atol=1e-12), (
        "position zero was modified, which indicates an off-by-one in the position index"
    )


_TEXT_ENCODER_PRIMITIVE_TESTS = [
    test_smoke_rms_normalization_preserves_shape_and_dtype,
    test_regression_rms_normalization_matches_oracle_across_widths,
    test_regression_rms_normalization_does_not_recentre,
    test_regression_rotary_tables_match_oracle,
    test_regression_rotary_table_rejects_odd_head_dimension,
    test_regression_apply_rotary_matches_oracle,
    test_regression_rotary_uses_half_split_not_interleaved_pairing,
    test_regression_rotary_preserves_vector_norm,
    test_regression_rotary_leaves_position_zero_unchanged,
]


def run_text_encoder_primitive_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests against the text encoder primitives",
        len(_TEXT_ENCODER_PRIMITIVE_TESTS),
    )
    for test_function in _TEXT_ENCODER_PRIMITIVE_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All text encoder primitive tests passed")
