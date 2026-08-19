"""
Tests for the text encoder blocks.

Parameters are synthesised at run time at reduced width, preserving the
structural properties that make the real model easy to get wrong:
head_dim is not hidden_size divided by head count, and there are fewer
key/value heads than query heads.

Two tests here assert properties no shape check and no oracle could
establish. test_regression_grouped_query_heads_share_key_value_heads_in_contiguous_groups
pins which query heads share which key/value head, distinguishing the
correct adjacent repeat from an interleaved one that produces
identically shaped output. test_regression_feedforward_gates_only_the_gate_branch
distinguishes applying the activation to the gate branch from applying
it to both branches or to their product.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from src.blocks import gated_feedforward, grouped_query_attention, transformer_layer
from src.blocks.grouped_query_attention import _repeat_key_value_heads
from src.config import NumericPrecision, TextEncoderConfig
from src.layers import causal_padding_mask, rotary_frequency_table


NUMERICAL_TOLERANCE = 1e-10

# Deliberately mirrors the real model's awkward properties at small
# scale: 8 query heads over 2 key/value heads (a ratio of four), and a
# head dimension whose product with the head count (8 times 16 = 128)
# differs from the hidden size (64).
_TEST_CONFIG = TextEncoderConfig(
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=200,
    hidden_states_output_layers=(1, 2),
    sequence_length=10,
    precision=NumericPrecision.HIGHEST,
)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_layer_parameters(
    rng: np.random.Generator, config: TextEncoderConfig
) -> dict[str, jnp.ndarray]:
    """
    Build one layer's parameters in the converted checkpoint's layout:
    projections stored as (in_features, out_features), normalization
    scales as per-feature vectors.
    """
    query_width = config.num_attention_heads * config.head_dim
    key_value_width = config.num_key_value_heads * config.head_dim

    parameters = {
        "input_layernorm_weight": rng.standard_normal((config.hidden_size,)) * 0.1 + 1.0,
        "post_attention_layernorm_weight": rng.standard_normal((config.hidden_size,)) * 0.1 + 1.0,
        "self_attn_q_proj_weight": rng.standard_normal((config.hidden_size, query_width)) * 0.1,
        "self_attn_k_proj_weight": rng.standard_normal((config.hidden_size, key_value_width)) * 0.1,
        "self_attn_v_proj_weight": rng.standard_normal((config.hidden_size, key_value_width)) * 0.1,
        "self_attn_o_proj_weight": rng.standard_normal((query_width, config.hidden_size)) * 0.1,
        "self_attn_q_norm_weight": rng.standard_normal((config.head_dim,)) * 0.1 + 1.0,
        "self_attn_k_norm_weight": rng.standard_normal((config.head_dim,)) * 0.1 + 1.0,
        "mlp_gate_proj_weight": rng.standard_normal((config.hidden_size, config.intermediate_size)) * 0.1,
        "mlp_up_proj_weight": rng.standard_normal((config.hidden_size, config.intermediate_size)) * 0.1,
        "mlp_down_proj_weight": rng.standard_normal((config.intermediate_size, config.hidden_size)) * 0.1,
    }
    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in parameters.items()}


def _attention_inputs(rng: np.random.Generator, config: TextEncoderConfig, batch: int = 2):
    activations = jnp.asarray(
        rng.standard_normal((batch, config.sequence_length, config.hidden_size)),
        dtype=jnp.float64,
    )
    cosine, sine = rotary_frequency_table(
        config.sequence_length, config.head_dim, config.rope_theta, dtype=jnp.float64
    )
    token_is_real = jnp.ones((batch, config.sequence_length), dtype=jnp.int32)
    bias = causal_padding_mask(token_is_real, config.sequence_length)
    return activations, cosine, sine, bias


def test_smoke_grouped_query_attention_preserves_shape() -> None:
    rng = _random_generator(seed=0)
    parameters = make_layer_parameters(rng, _TEST_CONFIG)
    activations, cosine, sine, bias = _attention_inputs(rng, _TEST_CONFIG)

    output = grouped_query_attention(
        activations, parameters, cosine, sine, bias, _TEST_CONFIG
    )

    assert output.shape == activations.shape


def test_regression_grouped_query_heads_share_key_value_heads_in_contiguous_groups() -> None:
    """
    Pin which query heads share which key/value head.

    With four query heads per key/value head, query heads 0 through 3
    must all see key/value head 0. An interleaved repeat would instead
    give query head 1 key/value head 1, producing identically shaped
    output that pairs every head wrongly.
    """
    num_key_value_heads, repeats = 2, 4
    head_dim, sequence_length = 3, 2

    # Give each key/value head a distinct constant so its identity is
    # recoverable from the expanded result.
    key_value = jnp.stack(
        [
            jnp.full((1, sequence_length, head_dim), float(head_index))
            for head_index in range(num_key_value_heads)
        ],
        axis=1,
    )

    expanded = np.asarray(_repeat_key_value_heads(key_value, repeats))

    assert expanded.shape == (1, num_key_value_heads * repeats, sequence_length, head_dim)
    for query_head in range(num_key_value_heads * repeats):
        expected_source = query_head // repeats
        assert np.allclose(expanded[0, query_head], float(expected_source)), (
            f"query head {query_head} received key/value head "
            f"{expanded[0, query_head].flat[0]}, expected {expected_source}; "
            f"this indicates an interleaved rather than contiguous repeat"
        )


def test_regression_attention_respects_padding_mask() -> None:
    """
    Changing a padded position's input must not change any real
    position's output. If it does, the mask is not suppressing padded
    keys.
    """
    rng = _random_generator(seed=1)
    config = _TEST_CONFIG
    parameters = make_layer_parameters(rng, config)
    real_length = 4

    token_is_real = np.zeros((1, config.sequence_length), dtype=np.int64)
    token_is_real[0, :real_length] = 1
    bias = causal_padding_mask(jnp.asarray(token_is_real), config.sequence_length)
    cosine, sine = rotary_frequency_table(
        config.sequence_length, config.head_dim, config.rope_theta, dtype=jnp.float64
    )

    base = rng.standard_normal((1, config.sequence_length, config.hidden_size))
    perturbed = base.copy()
    perturbed[0, real_length:, :] += 10.0  # disturb padded positions only

    first = np.asarray(
        grouped_query_attention(
            jnp.asarray(base, dtype=jnp.float64), parameters, cosine, sine, bias, config
        )
    )
    second = np.asarray(
        grouped_query_attention(
            jnp.asarray(perturbed, dtype=jnp.float64), parameters, cosine, sine, bias, config
        )
    )

    assert np.allclose(
        first[0, :real_length], second[0, :real_length], atol=NUMERICAL_TOLERANCE
    ), "perturbing padded positions changed the output at real positions"


def test_regression_attention_is_causal() -> None:
    """
    Changing a later position's input must not change an earlier
    position's output.
    """
    rng = _random_generator(seed=2)
    config = _TEST_CONFIG
    parameters = make_layer_parameters(rng, config)
    activations, cosine, sine, bias = _attention_inputs(rng, config, batch=1)

    cut = 5
    perturbed = np.asarray(activations).copy()
    perturbed[0, cut:, :] += 5.0

    first = np.asarray(
        grouped_query_attention(activations, parameters, cosine, sine, bias, config)
    )
    second = np.asarray(
        grouped_query_attention(
            jnp.asarray(perturbed, dtype=jnp.float64), parameters, cosine, sine, bias, config
        )
    )

    assert np.allclose(first[0, :cut], second[0, :cut], atol=NUMERICAL_TOLERANCE), (
        "perturbing later positions changed earlier outputs, so attention is not causal"
    )


def test_regression_feedforward_gates_only_the_gate_branch() -> None:
    """
    Distinguish gating the first branch from gating both or gating the
    product.

    With the up projection set to the identity and the down projection
    to the identity, the output reduces to silu(gate(x)) times x. An
    implementation that applied the activation to both branches would
    instead give silu(gate(x)) times silu(x), which differs wherever x
    is negative.
    """
    rng = _random_generator(seed=3)
    width = 8
    config = TextEncoderConfig(
        hidden_size=width,
        intermediate_size=width,
        precision=NumericPrecision.HIGHEST,
    )

    identity = jnp.eye(width, dtype=jnp.float64)
    gate_weight = jnp.asarray(rng.standard_normal((width, width)), dtype=jnp.float64)
    parameters = {
        "mlp_gate_proj_weight": gate_weight,
        "mlp_up_proj_weight": identity,
        "mlp_down_proj_weight": identity,
    }

    activations = jnp.asarray(rng.standard_normal((1, 3, width)), dtype=jnp.float64)
    output = np.asarray(gated_feedforward(activations, parameters, config))

    gate = np.asarray(activations) @ np.asarray(gate_weight)
    silu_gate = gate / (1.0 + np.exp(-gate))
    expected = silu_gate * np.asarray(activations)

    assert np.allclose(output, expected, atol=NUMERICAL_TOLERANCE), (
        "feed-forward did not compute silu(gate) times up; the activation may be "
        "applied to the wrong branch"
    )


def test_regression_transformer_layer_is_residual() -> None:
    """
    With both output projections zeroed, each sub-layer contributes
    nothing and the layer must return its input unchanged, confirming
    both residual connections are present.
    """
    rng = _random_generator(seed=4)
    config = _TEST_CONFIG
    parameters = make_layer_parameters(rng, config)
    parameters["self_attn_o_proj_weight"] = jnp.zeros_like(parameters["self_attn_o_proj_weight"])
    parameters["mlp_down_proj_weight"] = jnp.zeros_like(parameters["mlp_down_proj_weight"])

    activations, cosine, sine, bias = _attention_inputs(rng, config, batch=1)

    output = np.asarray(
        transformer_layer(activations, parameters, cosine, sine, bias, config)
    )

    assert np.allclose(output, np.asarray(activations), atol=NUMERICAL_TOLERANCE), (
        "layer did not pass its input through when both sub-layers contributed nothing"
    )


def test_regression_transformer_layer_reports_missing_parameter() -> None:
    rng = _random_generator(seed=5)
    parameters = make_layer_parameters(rng, _TEST_CONFIG)
    del parameters["self_attn_q_norm_weight"]
    activations, cosine, sine, bias = _attention_inputs(rng, _TEST_CONFIG, batch=1)

    try:
        transformer_layer(activations, parameters, cosine, sine, bias, _TEST_CONFIG)
    except KeyError as error:
        assert "self_attn_q_norm_weight" in str(error)
        return
    raise AssertionError("Expected a missing parameter error for an incomplete group")


_TEXT_ENCODER_BLOCK_TESTS = [
    test_smoke_grouped_query_attention_preserves_shape,
    test_regression_grouped_query_heads_share_key_value_heads_in_contiguous_groups,
    test_regression_attention_respects_padding_mask,
    test_regression_attention_is_causal,
    test_regression_feedforward_gates_only_the_gate_branch,
    test_regression_transformer_layer_is_residual,
    test_regression_transformer_layer_reports_missing_parameter,
]


def run_text_encoder_block_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests against the text encoder blocks",
        len(_TEXT_ENCODER_BLOCK_TESTS),
    )
    for test_function in _TEXT_ENCODER_BLOCK_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All text encoder block tests passed")
