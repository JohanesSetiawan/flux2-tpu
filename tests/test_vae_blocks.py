"""
Tests for flux2_klein.parameters and flux2_klein.vae_blocks.

The most important test in this module is
test_regression_chunked_attention_equals_unchunked. Query chunking
exists purely to bound memory, and is only legitimate if it changes
nothing about the result. That test asserts the equivalence across a
sweep of sequence lengths and chunk sizes, including sizes that do not
divide the sequence length evenly, which is where a padding or slicing
error would appear.

Parameters throughout are generated at run time from a seeded random
generator with shapes drawn from a sweep, never stored as golden
arrays.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from flux2_klein.config import NumericPrecision, VaeLayerConfig
from flux2_klein.parameters import (
    MissingParameterError,
    has_parameter_group,
    require_parameter,
    select_parameter_group,
)
from flux2_klein.vae_blocks import (
    _chunked_self_attention,
    attention_block,
    residual_block,
)


NUMERICAL_TOLERANCE = 1e-10
_TEST_CONFIG = VaeLayerConfig(num_groups=2, precision=NumericPrecision.HIGHEST)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_residual_block_parameters(
    rng: np.random.Generator,
    in_channels: int,
    out_channels: int,
    include_shortcut: bool,
) -> dict[str, np.ndarray]:
    """
    Build a complete, correctly shaped residual block parameter group.

    Shapes follow the converted checkpoint's conventions: convolution
    kernels are HWIO, normalization parameters are per-channel vectors
    sized to that layer's input.
    """
    parameters = {
        "norm1_weight": rng.standard_normal((in_channels,)),
        "norm1_bias": rng.standard_normal((in_channels,)),
        "conv1_weight": rng.standard_normal((3, 3, in_channels, out_channels)),
        "conv1_bias": rng.standard_normal((out_channels,)),
        "norm2_weight": rng.standard_normal((out_channels,)),
        "norm2_bias": rng.standard_normal((out_channels,)),
        "conv2_weight": rng.standard_normal((3, 3, out_channels, out_channels)),
        "conv2_bias": rng.standard_normal((out_channels,)),
    }
    if include_shortcut:
        parameters["nin_shortcut_weight"] = rng.standard_normal((1, 1, in_channels, out_channels))
        parameters["nin_shortcut_bias"] = rng.standard_normal((out_channels,))
    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in parameters.items()}


def _make_attention_block_parameters(
    rng: np.random.Generator, channels: int
) -> dict[str, np.ndarray]:
    """Build a complete, correctly shaped attention block parameter group."""
    parameters = {
        "norm_weight": rng.standard_normal((channels,)),
        "norm_bias": rng.standard_normal((channels,)),
    }
    for prefix in ("q", "k", "v", "proj_out"):
        parameters[f"{prefix}_weight"] = rng.standard_normal((1, 1, channels, channels))
        parameters[f"{prefix}_bias"] = rng.standard_normal((channels,))
    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in parameters.items()}


def _unchunked_attention_oracle(
    queries: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """
    Reference single-head attention computing the full score matrix at
    once, with a numerically stable softmax. Test-only; this is the
    computation chunking must reproduce exactly.
    """
    channels = queries.shape[-1]
    scores = np.einsum("bqc,bkc->bqk", queries, keys) / np.sqrt(channels)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exponentiated = np.exp(scores)
    weights = exponentiated / exponentiated.sum(axis=-1, keepdims=True)
    return np.einsum("bqk,bkc->bqc", weights, values)


def test_smoke_select_parameter_group_strips_prefix() -> None:
    parameters = {
        "mid_block_1_conv1_weight": np.zeros((1,)),
        "mid_block_1_norm1_bias": np.zeros((1,)),
        "mid_attn_1_q_weight": np.zeros((1,)),
    }

    selected = select_parameter_group(parameters, "mid_block_1")

    assert set(selected.keys()) == {"conv1_weight", "norm1_bias"}


def test_smoke_select_parameter_group_returns_empty_for_no_match() -> None:
    assert select_parameter_group({"a_b": np.zeros((1,))}, "nonexistent") == {}


def test_regression_require_parameter_reports_missing_key_and_available_keys() -> None:
    parameters = {"conv1_weight": np.zeros((1,)), "conv1_bias": np.zeros((1,))}

    try:
        require_parameter(parameters, "norm1_weight", "some_block")
    except MissingParameterError as error:
        message = str(error)
        assert "norm1_weight" in message
        assert "some_block" in message
        assert "conv1_weight" in message, "error should list what was available"
        return
    raise AssertionError("Expected MissingParameterError for an absent key")


def test_regression_has_parameter_group_detects_presence_and_absence() -> None:
    parameters = {"nin_shortcut_weight": np.zeros((1,)), "conv1_weight": np.zeros((1,))}

    assert has_parameter_group(parameters, "nin_shortcut") is True
    assert has_parameter_group(parameters, "norm1") is False


def test_smoke_residual_block_preserves_shape_when_channels_match() -> None:
    rng = _random_generator(seed=10)
    parameters = _make_residual_block_parameters(rng, 4, 4, include_shortcut=False)
    activations = jnp.asarray(rng.standard_normal((2, 6, 5, 4)), dtype=jnp.float64)

    output = residual_block(activations, parameters, _TEST_CONFIG)

    assert output.shape == (2, 6, 5, 4)


def test_smoke_residual_block_changes_channels_with_shortcut() -> None:
    rng = _random_generator(seed=11)
    parameters = _make_residual_block_parameters(rng, 4, 6, include_shortcut=True)
    activations = jnp.asarray(rng.standard_normal((2, 6, 5, 4)), dtype=jnp.float64)

    output = residual_block(activations, parameters, _TEST_CONFIG)

    assert output.shape == (2, 6, 5, 6)


def test_regression_residual_block_uses_shortcut_projection_when_present() -> None:
    """
    With both convolution kernels zeroed, the block's residual path
    contributes only its convolution biases, so the output is dominated
    by the shortcut. Comparing the shortcut-present and identity cases
    proves the projection is actually applied rather than silently
    skipped.
    """
    rng = _random_generator(seed=12)
    channels = 4
    parameters = _make_residual_block_parameters(rng, channels, channels, include_shortcut=True)
    parameters["conv1_weight"] = jnp.zeros_like(parameters["conv1_weight"])
    parameters["conv2_weight"] = jnp.zeros_like(parameters["conv2_weight"])
    parameters["conv1_bias"] = jnp.zeros_like(parameters["conv1_bias"])
    parameters["conv2_bias"] = jnp.zeros_like(parameters["conv2_bias"])

    activations = jnp.asarray(rng.standard_normal((1, 4, 4, channels)), dtype=jnp.float64)

    with_shortcut = np.asarray(residual_block(activations, parameters, _TEST_CONFIG))

    parameters_without_shortcut = {
        key: value for key, value in parameters.items() if not key.startswith("nin_shortcut")
    }
    without_shortcut = np.asarray(
        residual_block(activations, parameters_without_shortcut, _TEST_CONFIG)
    )

    assert not np.allclose(with_shortcut, without_shortcut), (
        "shortcut projection appears not to have been applied"
    )
    assert np.allclose(without_shortcut, np.asarray(activations), atol=NUMERICAL_TOLERANCE), (
        "identity shortcut path should pass the input through unchanged when the "
        "residual branch contributes nothing"
    )


def test_regression_residual_block_reports_missing_parameter() -> None:
    rng = _random_generator(seed=13)
    parameters = _make_residual_block_parameters(rng, 4, 4, include_shortcut=False)
    del parameters["norm2_weight"]
    activations = jnp.asarray(rng.standard_normal((1, 4, 4, 4)), dtype=jnp.float64)

    try:
        residual_block(activations, parameters, _TEST_CONFIG)
    except MissingParameterError as error:
        assert "norm2_weight" in str(error)
        return
    raise AssertionError("Expected MissingParameterError for an incomplete parameter group")


def test_regression_chunked_attention_equals_unchunked() -> None:
    """
    Chunking must be exactly equivalent to computing attention in one
    pass. Sequence lengths and chunk sizes are swept together,
    deliberately including combinations where the chunk size does not
    divide the sequence length, since that is the case requiring
    padding and re-slicing.
    """
    sequence_lengths = (16, 24, 30)
    chunk_sizes = (4, 7, 16, 64)
    channels = 8

    seed = 500
    for sequence_length, chunk_size in itertools.product(sequence_lengths, chunk_sizes):
        seed += 1
        rng = _random_generator(seed)

        queries = rng.standard_normal((2, sequence_length, channels))
        keys = rng.standard_normal((2, sequence_length, channels))
        values = rng.standard_normal((2, sequence_length, channels))

        expected = _unchunked_attention_oracle(queries, keys, values)
        actual = np.asarray(
            _chunked_self_attention(
                jnp.asarray(queries, dtype=jnp.float64),
                jnp.asarray(keys, dtype=jnp.float64),
                jnp.asarray(values, dtype=jnp.float64),
                query_chunk_size=chunk_size,
            )
        )

        assert actual.shape == expected.shape, (
            f"shape mismatch at sequence_length={sequence_length} chunk_size={chunk_size}"
        )
        assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE), (
            f"chunked attention diverged from unchunked at "
            f"sequence_length={sequence_length} chunk_size={chunk_size}, "
            f"max abs diff {np.max(np.abs(actual - expected))}"
        )


def test_regression_attention_output_is_invariant_to_chunk_size() -> None:
    """
    The same assertion viewed from the block level rather than the
    kernel level: changing only the configured chunk size must not
    change what attention_block returns.
    """
    rng = _random_generator(seed=600)
    channels = 4
    height, width = 5, 5
    parameters = _make_attention_block_parameters(rng, channels)
    activations = jnp.asarray(
        rng.standard_normal((2, height, width, channels)), dtype=jnp.float64
    )

    outputs = []
    for chunk_size in (3, 8, height * width, height * width * 2):
        config = VaeLayerConfig(
            num_groups=2,
            precision=NumericPrecision.HIGHEST,
            attention_query_chunk_size=chunk_size,
        )
        outputs.append(np.asarray(attention_block(activations, parameters, config)))

    for index in range(1, len(outputs)):
        assert np.allclose(outputs[0], outputs[index], atol=NUMERICAL_TOLERANCE), (
            f"attention_block output changed with chunk size, "
            f"max abs diff {np.max(np.abs(outputs[0] - outputs[index]))}"
        )


def test_smoke_attention_block_preserves_shape() -> None:
    rng = _random_generator(seed=601)
    channels = 4
    parameters = _make_attention_block_parameters(rng, channels)
    activations = jnp.asarray(rng.standard_normal((2, 6, 4, channels)), dtype=jnp.float64)
    config = VaeLayerConfig(num_groups=2, attention_query_chunk_size=8)

    output = attention_block(activations, parameters, config)

    assert output.shape == activations.shape


def test_regression_attention_block_is_residual() -> None:
    """
    With the output projection zeroed, the block must return its input
    unchanged, confirming the residual addition is present and that the
    attention path feeds the projection rather than bypassing it.
    """
    rng = _random_generator(seed=602)
    channels = 4
    parameters = _make_attention_block_parameters(rng, channels)
    parameters["proj_out_weight"] = jnp.zeros_like(parameters["proj_out_weight"])
    parameters["proj_out_bias"] = jnp.zeros_like(parameters["proj_out_bias"])

    activations = jnp.asarray(rng.standard_normal((1, 4, 4, channels)), dtype=jnp.float64)
    config = VaeLayerConfig(num_groups=2, attention_query_chunk_size=8)

    output = np.asarray(attention_block(activations, parameters, config))

    assert np.allclose(output, np.asarray(activations), atol=NUMERICAL_TOLERANCE)


_VAE_BLOCK_TESTS = [
    test_smoke_select_parameter_group_strips_prefix,
    test_smoke_select_parameter_group_returns_empty_for_no_match,
    test_regression_require_parameter_reports_missing_key_and_available_keys,
    test_regression_has_parameter_group_detects_presence_and_absence,
    test_smoke_residual_block_preserves_shape_when_channels_match,
    test_smoke_residual_block_changes_channels_with_shortcut,
    test_regression_residual_block_uses_shortcut_projection_when_present,
    test_regression_residual_block_reports_missing_parameter,
    test_regression_chunked_attention_equals_unchunked,
    test_regression_attention_output_is_invariant_to_chunk_size,
    test_smoke_attention_block_preserves_shape,
    test_regression_attention_block_is_residual,
]


def run_vae_block_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against parameters.py and vae_blocks.py", len(_VAE_BLOCK_TESTS))
    for test_function in _VAE_BLOCK_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All VAE block tests passed")
