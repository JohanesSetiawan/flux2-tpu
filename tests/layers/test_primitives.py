"""
Tests for src.layers.

Two tiers, both driven by dynamically generated inputs rather than
stored golden arrays:

SMOKE tests confirm each primitive runs, produces the expected shape,
and preserves dtype. They are cheap and catch gross breakage.

REGRESSION tests confirm each primitive is numerically correct by
comparing it against an independent oracle implemented from the
mathematical definition in plain numpy. The oracles below are written
deliberately naively (explicit loops, no vectorisation tricks) so that
they share no implementation with the JAX code under test. An oracle
that reused the same reshape-and-reduce strategy would agree with a
buggy implementation for the same wrong reason and prove nothing.

Both tiers sweep a matrix of shapes generated at run time from
SHAPE_MATRIX rather than testing one fixed size, so an indexing error
that happens to be invisible at one particular channel count or kernel
size is caught at another.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.config import NumericPrecision, VaeLayerConfig
from src.layers import (
    convolution_2d,
    group_normalization,
    nearest_neighbor_upsample_2d,
    sigmoid_linear_unit,
)


# Every combination below is exercised by the regression sweeps. The
# values are chosen to vary each dimension that could plausibly be
# mis-indexed: non-square spatial extents catch height/width
# transposition, differing in/out channel counts catch kernel axis
# ordering, and both kernel sizes catch padding derivation.
SHAPE_MATRIX = {
    "spatial": ((5, 7), (8, 8), (4, 6)),
    "channels": ((3, 5), (4, 4), (6, 2)),
    "kernel_size": (1, 3),
}

# Regression comparisons run in float64 through both the oracle and the
# implementation, so the only difference between them is the algorithm,
# not the arithmetic precision. A tight tolerance is therefore
# meaningful here; it would not be if one side were computing in
# bfloat16.
NUMERICAL_TOLERANCE = 1e-10

_FLOAT64_TEST_CONFIG = VaeLayerConfig(precision=NumericPrecision.HIGHEST)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _naive_convolution_2d_oracle(
    activations_nhwc: np.ndarray,
    kernel_hwio: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    """
    Reference stride-1, shape-preserving 2D convolution, written as
    explicit nested loops directly from the definition. Test-only.
    """
    batch, height, width, in_channels = activations_nhwc.shape
    kernel_height, kernel_width, kernel_in, out_channels = kernel_hwio.shape
    assert kernel_in == in_channels

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded = np.pad(
        activations_nhwc,
        pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width), (0, 0)),
    )

    output = np.zeros((batch, height, width, out_channels), dtype=np.float64)
    for n, y, x, o in itertools.product(
        range(batch), range(height), range(width), range(out_channels)
    ):
        total = 0.0
        for ky, kx, i in itertools.product(
            range(kernel_height), range(kernel_width), range(in_channels)
        ):
            total += padded[n, y + ky, x + kx, i] * kernel_hwio[ky, kx, i, o]
        output[n, y, x, o] = total

    if bias is not None:
        output = output + bias
    return output


def _naive_group_normalization_oracle(
    activations_nhwc: np.ndarray,
    scale: np.ndarray,
    shift: np.ndarray,
    num_groups: int,
    epsilon: float,
) -> np.ndarray:
    """
    Reference group normalization, computing each group's statistics by
    explicitly gathering that group's elements. Test-only.
    """
    batch, height, width, channels = activations_nhwc.shape
    channels_per_group = channels // num_groups
    output = np.zeros_like(activations_nhwc, dtype=np.float64)

    for n, g in itertools.product(range(batch), range(num_groups)):
        channel_start = g * channels_per_group
        channel_stop = channel_start + channels_per_group
        group_values = activations_nhwc[n, :, :, channel_start:channel_stop]

        mean = group_values.mean()
        variance = ((group_values - mean) ** 2).mean()
        normalized = (group_values - mean) / np.sqrt(variance + epsilon)

        output[n, :, :, channel_start:channel_stop] = normalized

    return output * scale + shift


def _naive_nearest_upsample_oracle(activations_nhwc: np.ndarray, factor: int) -> np.ndarray:
    """
    Reference nearest-neighbour upsample, assigning each output pixel
    from its source pixel by integer division. Test-only.
    """
    batch, height, width, channels = activations_nhwc.shape
    output = np.zeros(
        (batch, height * factor, width * factor, channels), dtype=activations_nhwc.dtype
    )
    for y, x in itertools.product(range(height * factor), range(width * factor)):
        output[:, y, x, :] = activations_nhwc[:, y // factor, x // factor, :]
    return output


def test_smoke_convolution_produces_expected_shape_and_dtype() -> None:
    rng = _random_generator(seed=0)
    activations = jnp.asarray(rng.standard_normal((2, 8, 6, 4)), dtype=jnp.float32)
    kernel = jnp.asarray(rng.standard_normal((3, 3, 4, 7)), dtype=jnp.float32)
    bias = jnp.asarray(rng.standard_normal((7,)), dtype=jnp.float32)

    output = convolution_2d(activations, kernel, bias, _FLOAT64_TEST_CONFIG)

    assert output.shape == (2, 8, 6, 7)
    assert output.dtype == jnp.float32


def test_smoke_convolution_accepts_absent_bias() -> None:
    rng = _random_generator(seed=1)
    activations = jnp.asarray(rng.standard_normal((1, 4, 4, 2)), dtype=jnp.float32)
    kernel = jnp.asarray(rng.standard_normal((3, 3, 2, 3)), dtype=jnp.float32)

    output = convolution_2d(activations, kernel, None, _FLOAT64_TEST_CONFIG)

    assert output.shape == (1, 4, 4, 3)


def test_smoke_group_normalization_preserves_shape_and_dtype() -> None:
    rng = _random_generator(seed=2)
    config = VaeLayerConfig(num_groups=4)
    activations = jnp.asarray(rng.standard_normal((2, 5, 5, 8)), dtype=jnp.float32)
    scale = jnp.asarray(rng.standard_normal((8,)), dtype=jnp.float32)
    shift = jnp.asarray(rng.standard_normal((8,)), dtype=jnp.float32)

    output = group_normalization(activations, scale, shift, config)

    assert output.shape == activations.shape
    assert output.dtype == jnp.float32


def test_smoke_upsample_produces_expected_shape() -> None:
    rng = _random_generator(seed=3)
    config = VaeLayerConfig(upsample_scale_factor=2)
    activations = jnp.asarray(rng.standard_normal((2, 5, 7, 3)), dtype=jnp.float32)

    output = nearest_neighbor_upsample_2d(activations, config)

    assert output.shape == (2, 10, 14, 3)


def test_smoke_sigmoid_linear_unit_preserves_shape() -> None:
    rng = _random_generator(seed=4)
    activations = jnp.asarray(rng.standard_normal((2, 3, 3, 5)), dtype=jnp.float32)

    output = sigmoid_linear_unit(activations)

    assert output.shape == activations.shape


def test_regression_convolution_matches_naive_oracle_across_shape_matrix() -> None:
    seed = 0
    for (height, width), (in_channels, out_channels), kernel_size in itertools.product(
        SHAPE_MATRIX["spatial"], SHAPE_MATRIX["channels"], SHAPE_MATRIX["kernel_size"]
    ):
        seed += 1
        rng = _random_generator(seed)

        activations = rng.standard_normal((2, height, width, in_channels))
        kernel = rng.standard_normal((kernel_size, kernel_size, in_channels, out_channels))
        bias = rng.standard_normal((out_channels,))

        expected = _naive_convolution_2d_oracle(activations, kernel, bias)
        actual = np.asarray(
            convolution_2d(
                jnp.asarray(activations, dtype=jnp.float64),
                jnp.asarray(kernel, dtype=jnp.float64),
                jnp.asarray(bias, dtype=jnp.float64),
                _FLOAT64_TEST_CONFIG,
            )
        )

        assert actual.shape == expected.shape, (
            f"shape mismatch at spatial=({height},{width}) "
            f"channels=({in_channels},{out_channels}) kernel={kernel_size}"
        )
        assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE), (
            f"value mismatch at spatial=({height},{width}) "
            f"channels=({in_channels},{out_channels}) kernel={kernel_size}, "
            f"max abs diff {np.max(np.abs(actual - expected))}"
        )


def test_regression_group_normalization_matches_naive_oracle_across_shape_matrix() -> None:
    seed = 100
    group_counts = (1, 2, 4)
    channel_counts = (4, 8)

    for (height, width), channels, num_groups in itertools.product(
        SHAPE_MATRIX["spatial"], channel_counts, group_counts
    ):
        if channels % num_groups != 0:
            continue
        seed += 1
        rng = _random_generator(seed)
        config = VaeLayerConfig(num_groups=num_groups)

        activations = rng.standard_normal((2, height, width, channels))
        scale = rng.standard_normal((channels,))
        shift = rng.standard_normal((channels,))

        expected = _naive_group_normalization_oracle(
            activations, scale, shift, num_groups, config.normalization_epsilon
        )
        actual = np.asarray(
            group_normalization(
                jnp.asarray(activations, dtype=jnp.float64),
                jnp.asarray(scale, dtype=jnp.float64),
                jnp.asarray(shift, dtype=jnp.float64),
                config,
            )
        )

        assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE), (
            f"value mismatch at spatial=({height},{width}) channels={channels} "
            f"groups={num_groups}, max abs diff {np.max(np.abs(actual - expected))}"
        )


def test_regression_group_normalization_standardizes_each_group_independently() -> None:
    """
    A property check independent of the oracle: with an identity affine
    transform, every group of every batch element must come out with
    approximately zero mean and unit variance. This catches a class of
    bug the oracle comparison cannot, namely both implementations
    grouping channels the same wrong way.
    """
    rng = _random_generator(seed=200)
    num_groups = 4
    channels = 8
    config = VaeLayerConfig(num_groups=num_groups)

    # Give each channel a different scale and offset, so a bug that
    # pooled statistics across groups instead of within them would leave
    # a clearly non-standardized result.
    activations = rng.standard_normal((3, 6, 5, channels))
    activations = activations * np.arange(1, channels + 1) + np.arange(channels) * 10.0

    output = np.asarray(
        group_normalization(
            jnp.asarray(activations, dtype=jnp.float64),
            jnp.ones((channels,), dtype=jnp.float64),
            jnp.zeros((channels,), dtype=jnp.float64),
            config,
        )
    )

    channels_per_group = channels // num_groups
    for n, g in itertools.product(range(output.shape[0]), range(num_groups)):
        group_slice = output[n, :, :, g * channels_per_group : (g + 1) * channels_per_group]
        assert abs(group_slice.mean()) < 1e-8, f"group {g} of batch {n} is not zero-mean"
        assert abs(group_slice.var() - 1.0) < 1e-6, f"group {g} of batch {n} is not unit-variance"


def test_regression_group_normalization_rejects_indivisible_channel_count() -> None:
    config = VaeLayerConfig(num_groups=3)
    activations = jnp.zeros((1, 2, 2, 8), dtype=jnp.float32)
    scale = jnp.ones((8,), dtype=jnp.float32)
    shift = jnp.zeros((8,), dtype=jnp.float32)

    try:
        group_normalization(activations, scale, shift, config)
    except ValueError as error:
        assert "divisible" in str(error)
        return
    raise AssertionError("Expected ValueError when channels are not divisible by group count")


def test_regression_upsample_matches_naive_oracle_across_shape_matrix() -> None:
    seed = 300
    for (height, width), (channels, _unused), factor in itertools.product(
        SHAPE_MATRIX["spatial"], SHAPE_MATRIX["channels"], (2, 3)
    ):
        seed += 1
        rng = _random_generator(seed)
        config = VaeLayerConfig(upsample_scale_factor=factor)

        activations = rng.standard_normal((2, height, width, channels))

        expected = _naive_nearest_upsample_oracle(activations, factor)
        actual = np.asarray(
            nearest_neighbor_upsample_2d(jnp.asarray(activations, dtype=jnp.float64), config)
        )

        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected), (
            f"value mismatch at spatial=({height},{width}) channels={channels} factor={factor}"
        )


def test_regression_sigmoid_linear_unit_matches_definition() -> None:
    rng = _random_generator(seed=400)
    activations = rng.standard_normal((2, 4, 4, 6)) * 5.0

    expected = activations / (1.0 + np.exp(-activations))
    actual = np.asarray(sigmoid_linear_unit(jnp.asarray(activations, dtype=jnp.float64)))

    assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE)


_LAYER_TESTS = [
    test_smoke_convolution_produces_expected_shape_and_dtype,
    test_smoke_convolution_accepts_absent_bias,
    test_smoke_group_normalization_preserves_shape_and_dtype,
    test_smoke_upsample_produces_expected_shape,
    test_smoke_sigmoid_linear_unit_preserves_shape,
    test_regression_convolution_matches_naive_oracle_across_shape_matrix,
    test_regression_group_normalization_matches_naive_oracle_across_shape_matrix,
    test_regression_group_normalization_standardizes_each_group_independently,
    test_regression_group_normalization_rejects_indivisible_channel_count,
    test_regression_upsample_matches_naive_oracle_across_shape_matrix,
    test_regression_sigmoid_linear_unit_matches_definition,
]


def run_layer_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against layers.py", len(_LAYER_TESTS))
    for test_function in _LAYER_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All layer tests passed")
