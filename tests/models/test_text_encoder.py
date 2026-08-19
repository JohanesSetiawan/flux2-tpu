"""
Tests for src.models.text_encoder.

The subtlest property tested here is hidden state depth. Conditioning
is built from the states after specific numbers of layers, and an
off-by-one would take the state one layer too early or too late while
producing exactly the right shape. test_regression_hidden_state_depth
pins it by making each layer's contribution individually identifiable.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from src.config import NumericPrecision, TextEncoderConfig
from src.models.text_encoder import (
    LayerCountMismatchError,
    embed_tokens,
    encode_prompt,
)
from tests.blocks.test_text_encoder_blocks import make_layer_parameters


NUMERICAL_TOLERANCE = 1e-10

_TEST_CONFIG = TextEncoderConfig(
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    vocab_size=100,
    hidden_states_output_layers=(1, 2, 4),
    sequence_length=6,
    precision=NumericPrecision.HIGHEST,
)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_parameters(
    rng: np.random.Generator, config: TextEncoderConfig, num_layers: int
) -> dict:
    """Stack per-layer parameters along a leading axis, as the checkpoint does."""
    per_layer = [make_layer_parameters(rng, config) for _ in range(num_layers)]
    stacked = {
        key: jnp.stack([layer[key] for layer in per_layer], axis=0)
        for key in per_layer[0]
    }
    embedding = jnp.asarray(
        rng.standard_normal((config.vocab_size, config.hidden_size)), dtype=jnp.float64
    )
    return {"embed_tokens": {"weight": embedding}, "layers": stacked}


def _token_inputs(rng: np.random.Generator, config: TextEncoderConfig, real_length: int):
    token_ids = jnp.asarray(
        rng.integers(0, config.vocab_size, size=(2, config.sequence_length))
    )
    token_is_real = np.zeros((2, config.sequence_length), dtype=np.int64)
    token_is_real[:, :real_length] = 1
    return token_ids, jnp.asarray(token_is_real)


def test_smoke_encode_prompt_returns_conditioning_of_expected_width() -> None:
    rng = _random_generator(seed=0)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    token_ids, token_is_real = _token_inputs(rng, config, real_length=4)

    conditioning = encode_prompt(token_ids, token_is_real, parameters, config)

    assert conditioning.shape == (2, config.sequence_length, config.conditioning_dimension)
    assert config.conditioning_dimension == config.hidden_size * 3


def test_regression_embed_tokens_gathers_rows_not_columns() -> None:
    """
    The embedding table keeps its (vocabulary, hidden) row layout while
    every projection weight was transposed during conversion. Gathering
    along the wrong axis would still produce a plausible tensor here, so
    the lookup is checked against the table directly.
    """
    rng = _random_generator(seed=1)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    table = np.asarray(parameters["embed_tokens"]["weight"])

    token_ids = jnp.asarray(np.array([[3, 17, 0]]))
    embedded = np.asarray(embed_tokens(token_ids, parameters, config))

    for position, token_id in enumerate([3, 17, 0]):
        assert np.array_equal(embedded[0, position], table[token_id]), (
            f"position {position} did not receive row {token_id} of the table"
        )


def test_regression_embed_tokens_rejects_vocabulary_mismatch() -> None:
    rng = _random_generator(seed=2)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    parameters["embed_tokens"]["weight"] = parameters["embed_tokens"]["weight"][:10]

    try:
        embed_tokens(jnp.asarray(np.array([[1]])), parameters, config)
    except ValueError as error:
        assert "vocabulary" in str(error)
        return
    raise AssertionError("Expected ValueError when the table disagrees with the configuration")


def test_regression_hidden_state_depth() -> None:
    """
    Confirm each selected hidden state is taken after exactly the
    configured number of layers.

    Every layer is neutralised except one, whose output projections are
    left intact. That layer is therefore the only one that changes the
    activations, so the conditioning slices before it must equal the
    embedding output and the slices at or after it must differ. An
    off-by-one in depth selection moves that boundary by one slice.
    """
    rng = _random_generator(seed=3)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)

    active_layer = 1
    for key in ("self_attn_o_proj_weight", "mlp_down_proj_weight"):
        stacked = np.asarray(parameters["layers"][key]).copy()
        for layer_index in range(stacked.shape[0]):
            if layer_index != active_layer:
                stacked[layer_index] = 0.0
        parameters["layers"][key] = jnp.asarray(stacked)

    token_ids, token_is_real = _token_inputs(rng, config, real_length=config.sequence_length)
    conditioning = np.asarray(encode_prompt(token_ids, token_is_real, parameters, config))
    embedded = np.asarray(embed_tokens(token_ids, parameters, config))

    width = config.hidden_size
    for slice_index, depth in enumerate(config.hidden_states_output_layers):
        captured = conditioning[..., slice_index * width : (slice_index + 1) * width]
        matches_embedding = np.allclose(captured, embedded, atol=NUMERICAL_TOLERANCE)

        if depth <= active_layer:
            assert matches_embedding, (
                f"state at depth {depth} differs from the embedding even though only "
                f"layer {active_layer} is active; depth selection may be too late"
            )
        else:
            assert not matches_embedding, (
                f"state at depth {depth} equals the embedding even though layer "
                f"{active_layer} ran before it; depth selection may be too early"
            )


def test_regression_encode_prompt_rejects_insufficient_layers() -> None:
    rng = _random_generator(seed=4)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required - 1)
    token_ids, token_is_real = _token_inputs(rng, config, real_length=3)

    try:
        encode_prompt(token_ids, token_is_real, parameters, config)
    except LayerCountMismatchError as error:
        assert str(config.num_layers_required) in str(error)
        return
    raise AssertionError("Expected LayerCountMismatchError when the checkpoint is too shallow")


def test_regression_encode_prompt_ignores_extra_layers() -> None:
    """
    A checkpoint deeper than required must still work, running only the
    layers needed for the configured hidden states rather than failing
    or silently using the wrong depth.
    """
    rng = _random_generator(seed=5)
    config = _TEST_CONFIG
    exact = _make_parameters(rng, config, config.num_layers_required)

    deeper = {
        "embed_tokens": exact["embed_tokens"],
        "layers": {
            key: jnp.concatenate([value, value[:1]], axis=0)
            for key, value in exact["layers"].items()
        },
    }

    token_ids, token_is_real = _token_inputs(rng, config, real_length=4)
    from_exact = np.asarray(encode_prompt(token_ids, token_is_real, exact, config))
    from_deeper = np.asarray(encode_prompt(token_ids, token_is_real, deeper, config))

    assert np.allclose(from_exact, from_deeper, atol=NUMERICAL_TOLERANCE), (
        "extra layers beyond the required depth changed the result"
    )


def test_regression_padding_does_not_affect_real_positions() -> None:
    """
    End-to-end version of the masking property: changing the token
    identifiers at padded positions must leave the conditioning at real
    positions untouched.
    """
    rng = _random_generator(seed=6)
    config = _TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    real_length = 3

    token_ids = np.asarray(
        rng.integers(0, config.vocab_size, size=(1, config.sequence_length))
    )
    token_is_real = np.zeros((1, config.sequence_length), dtype=np.int64)
    token_is_real[0, :real_length] = 1

    altered = token_ids.copy()
    altered[0, real_length:] = (altered[0, real_length:] + 7) % config.vocab_size

    first = np.asarray(
        encode_prompt(jnp.asarray(token_ids), jnp.asarray(token_is_real), parameters, config)
    )
    second = np.asarray(
        encode_prompt(jnp.asarray(altered), jnp.asarray(token_is_real), parameters, config)
    )

    assert np.allclose(
        first[0, :real_length], second[0, :real_length], atol=NUMERICAL_TOLERANCE
    ), "changing padded tokens altered the conditioning at real positions"


_TEXT_ENCODER_TESTS = [
    test_smoke_encode_prompt_returns_conditioning_of_expected_width,
    test_regression_embed_tokens_gathers_rows_not_columns,
    test_regression_embed_tokens_rejects_vocabulary_mismatch,
    test_regression_hidden_state_depth,
    test_regression_encode_prompt_rejects_insufficient_layers,
    test_regression_encode_prompt_ignores_extra_layers,
    test_regression_padding_does_not_affect_real_positions,
]


def run_text_encoder_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests against the text encoder", len(_TEXT_ENCODER_TESTS)
    )
    for test_function in _TEXT_ENCODER_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All text encoder tests passed")
