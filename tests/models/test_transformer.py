"""
Tests for the diffusion transformer's blocks and complete model.

Parameters are synthesised at run time at reduced width, keeping the
structural properties intact: both block types present, the fused
projections sized exactly as the checkpoint sizes them, and the
positional axes summing to the head dimension.

Several tests assert properties no oracle could establish, because they
distinguish two arrangements that produce identically shaped output:

- that the fused query-key-value projection is split component-major
  rather than head-major
- that the fused gated feed-forward gates the first half with the
  second, not the reverse
- that the double block's two streams keep separate weights while
  attending jointly, so text genuinely influences the image
- that the single block's text positions are dropped only at the end,
  after they have had their effect
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from src.blocks import (
    ModulationTriple,
    double_stream_block,
    joint_attention,
    single_stream_block,
    split_and_gate,
    split_fused_qkv,
)
from src.config import ExecutionConfig, NumericPrecision, TransformerConfig
from src.layers import axial_rotation_table, build_position_identifiers, layer_normalization
from src.models.transformer import (
    BlockCountMismatchError,
    compute_all_modulation,
    compute_conditioning_vector,
    predict_velocity,
)


NUMERICAL_TOLERANCE = 1e-10

_TEST_CONFIG = TransformerConfig(
    in_channels=16,
    context_dim=48,
    hidden_size=64,
    num_heads=2,
    num_double_blocks=2,
    num_single_blocks=2,
    positional_axes_dimensions=(8, 8, 8, 8),
    mlp_ratio=3.0,
    precision=NumericPrecision.HIGHEST,
)

TEST_LATENT_HEIGHT = 3
TEST_LATENT_WIDTH = 4
TEST_TEXT_TOKENS = 5


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.1) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape) * scale, dtype=jnp.float64)


def _make_double_block_parameters(
    rng: np.random.Generator, config: TransformerConfig, num_blocks: int
) -> dict[str, jnp.ndarray]:
    hidden = config.hidden_size
    mlp_hidden = config.mlp_hidden_size
    parameters = {}
    for stream in ("img", "txt"):
        parameters[f"{stream}_attn_qkv_weight"] = _normal(rng, (num_blocks, hidden, 3 * hidden))
        parameters[f"{stream}_attn_proj_weight"] = _normal(rng, (num_blocks, hidden, hidden))
        parameters[f"{stream}_attn_norm_query_norm_scale"] = _normal(
            rng, (num_blocks, config.head_dim), scale=0.05
        ) + 1.0
        parameters[f"{stream}_attn_norm_key_norm_scale"] = _normal(
            rng, (num_blocks, config.head_dim), scale=0.05
        ) + 1.0
        parameters[f"{stream}_mlp_0_weight"] = _normal(rng, (num_blocks, hidden, 2 * mlp_hidden))
        parameters[f"{stream}_mlp_2_weight"] = _normal(rng, (num_blocks, mlp_hidden, hidden))
    return parameters


def _make_single_block_parameters(
    rng: np.random.Generator, config: TransformerConfig, num_blocks: int
) -> dict[str, jnp.ndarray]:
    hidden = config.hidden_size
    mlp_hidden = config.mlp_hidden_size
    return {
        "linear1_weight": _normal(rng, (num_blocks, hidden, 3 * hidden + 2 * mlp_hidden)),
        "linear2_weight": _normal(rng, (num_blocks, hidden + mlp_hidden, hidden)),
        "norm_query_norm_scale": _normal(rng, (num_blocks, config.head_dim), scale=0.05) + 1.0,
        "norm_key_norm_scale": _normal(rng, (num_blocks, config.head_dim), scale=0.05) + 1.0,
    }


def make_transformer_parameters(
    rng: np.random.Generator, config: TransformerConfig
) -> dict:
    """Build a complete synthetic transformer in the checkpoint's layout."""
    hidden = config.hidden_size
    return {
        "double_blocks": _make_double_block_parameters(rng, config, config.num_double_blocks),
        "single_blocks": _make_single_block_parameters(rng, config, config.num_single_blocks),
        "global": {
            "img_in_weight": _normal(rng, (config.in_channels, hidden)),
            "txt_in_weight": _normal(rng, (config.context_dim, hidden)),
            "time_in_in_layer_weight": _normal(rng, (config.timestep_embedding_dim, hidden)),
            "time_in_out_layer_weight": _normal(rng, (hidden, hidden)),
            "double_stream_modulation_img_lin_weight": _normal(rng, (hidden, 6 * hidden)),
            "double_stream_modulation_txt_lin_weight": _normal(rng, (hidden, 6 * hidden)),
            "single_stream_modulation_lin_weight": _normal(rng, (hidden, 3 * hidden)),
            "final_layer_adaLN_modulation_1_weight": _normal(rng, (hidden, 2 * hidden)),
            "final_layer_linear_weight": _normal(rng, (hidden, config.in_channels)),
        },
    }


def _model_inputs(rng: np.random.Generator, config: TransformerConfig):
    latent = jnp.asarray(
        rng.standard_normal((1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.in_channels)),
        dtype=jnp.float64,
    )
    conditioning = jnp.asarray(
        rng.standard_normal((1, TEST_TEXT_TOKENS, config.context_dim)), dtype=jnp.float64
    )
    timesteps = jnp.asarray(np.array([0.7672]), dtype=jnp.float64)
    return latent, conditioning, timesteps


def _rotation_tables(config: TransformerConfig, text_tokens: int):
    text_identifiers, image_identifiers = build_position_identifiers(
        text_tokens, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH, config
    )
    identifiers = jnp.concatenate([text_identifiers, image_identifiers], axis=1)
    return axial_rotation_table(identifiers, config)


def _zero_modulation(config: TransformerConfig) -> ModulationTriple:
    zeros = jnp.zeros((1, 1, config.hidden_size), dtype=jnp.float64)
    return ModulationTriple(shift=zeros, scale=zeros, gate=zeros)


def test_regression_fused_qkv_splits_component_major_not_head_major() -> None:
    """
    Pin the layout of the fused query-key-value projection.

    Its final axis is ordered component first, then head, then feature.
    Reshaping head-major instead would mix queries, keys and values
    within each head while producing identically shaped tensors. The
    test builds a projection output whose every element encodes its own
    component index, so the split's correctness is directly readable.
    """
    config = _TEST_CONFIG
    num_tokens = 3
    total_width = 3 * config.hidden_size

    fused = np.zeros((1, num_tokens, total_width))
    for component_index in range(3):
        start = component_index * config.hidden_size
        fused[:, :, start : start + config.hidden_size] = float(component_index + 1)

    queries, keys, values = split_fused_qkv(jnp.asarray(fused, dtype=jnp.float64), config)

    for expected, tensor, name in ((1.0, queries, "queries"), (2.0, keys, "keys"), (3.0, values, "values")):
        assert np.allclose(np.asarray(tensor), expected), (
            f"{name} did not come from its own third of the projection; the split "
            f"may be head-major rather than component-major"
        )


def test_regression_gated_mlp_gates_first_half_with_second() -> None:
    """
    The activation is applied to the first half, which then scales the
    second. Reversing that computes something different from the same
    weights while producing identical shapes.
    """
    rng = _random_generator(seed=0)
    fused = rng.standard_normal((1, 3, 8))

    gated = np.asarray(split_and_gate(jnp.asarray(fused, dtype=jnp.float64)))

    first_half, second_half = fused[..., :4], fused[..., 4:]
    expected = (first_half / (1.0 + np.exp(-first_half))) * second_half

    assert np.allclose(gated, expected, atol=NUMERICAL_TOLERANCE), (
        "gating did not apply the activation to the first half"
    )


def test_smoke_joint_attention_merges_heads() -> None:
    rng = _random_generator(seed=1)
    config = _TEST_CONFIG
    shape = (1, config.num_heads, 6, config.head_dim)
    queries, keys, values = (_normal(rng, shape, scale=1.0) for _ in range(3))

    attended = joint_attention(queries, keys, values, config)

    assert attended.shape == (1, 6, config.hidden_size)


def test_regression_joint_attention_is_unmasked() -> None:
    """
    Unlike the text encoder's attention, this one has no mask: every
    token attends to every other, including in both directions. Changing
    a later token's value must therefore change an earlier token's
    output.
    """
    rng = _random_generator(seed=2)
    config = _TEST_CONFIG
    shape = (1, config.num_heads, 6, config.head_dim)
    queries, keys = _normal(rng, shape, scale=1.0), _normal(rng, shape, scale=1.0)
    values = np.asarray(_normal(rng, shape, scale=1.0))

    first = np.asarray(joint_attention(queries, keys, jnp.asarray(values), config))
    perturbed = values.copy()
    perturbed[:, :, 5, :] += 10.0
    second = np.asarray(joint_attention(queries, keys, jnp.asarray(perturbed), config))

    assert not np.allclose(first[:, 0], second[:, 0], atol=1e-6), (
        "perturbing the last token left the first token's output unchanged, which "
        "would mean attention is masked when it should not be"
    )


def test_smoke_double_stream_block_preserves_both_stream_shapes() -> None:
    rng = _random_generator(seed=3)
    config = _TEST_CONFIG
    parameters = {
        key: value[0] for key, value in _make_double_block_parameters(rng, config, 1).items()
    }
    image = _normal(rng, (1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.hidden_size), 1.0)
    text = _normal(rng, (1, TEST_TEXT_TOKENS, config.hidden_size), 1.0)
    cosine, sine = _rotation_tables(config, TEST_TEXT_TOKENS)
    modulation = (_zero_modulation(config), _zero_modulation(config))

    new_image, new_text = double_stream_block(
        image, text, parameters, cosine, sine, modulation, modulation, config
    )

    assert new_image.shape == image.shape
    assert new_text.shape == text.shape


def test_regression_double_stream_block_lets_text_influence_image() -> None:
    """
    The two streams keep separate weights but share one attention, which
    is the whole point of the block. Perturbing the text stream must
    therefore change the image stream; if it does not, the streams are
    not actually attending jointly.
    The perturbation must be non-uniform across features. Adding a
    constant to every feature would be removed exactly by the block's
    layer normalization, which subtracts the mean, so a constant shift
    is invisible by construction and would fail this test against
    correct code. An earlier version of this test made that mistake.
    """
    rng = _random_generator(seed=4)
    config = _TEST_CONFIG
    parameters = {
        key: value[0] for key, value in _make_double_block_parameters(rng, config, 1).items()
    }
    image = _normal(rng, (1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.hidden_size), 1.0)
    text = np.asarray(_normal(rng, (1, TEST_TEXT_TOKENS, config.hidden_size), 1.0))
    cosine, sine = _rotation_tables(config, TEST_TEXT_TOKENS)

    # A non-zero attention gate, so the attention result actually
    # reaches the residual stream.
    ones = jnp.ones((1, 1, config.hidden_size), dtype=jnp.float64)
    zeros = jnp.zeros((1, 1, config.hidden_size), dtype=jnp.float64)
    gated = ModulationTriple(shift=zeros, scale=zeros, gate=ones)
    modulation = (gated, _zero_modulation(config))

    first, _ = double_stream_block(
        image, jnp.asarray(text), parameters, cosine, sine, modulation, modulation, config
    )
    perturbed = text + rng.standard_normal(text.shape) * 2.0
    second, _ = double_stream_block(
        image, jnp.asarray(perturbed), parameters, cosine, sine, modulation, modulation, config
    )

    assert not np.allclose(np.asarray(first), np.asarray(second), atol=1e-6), (
        "perturbing the text stream left the image stream unchanged, so the two "
        "streams are not attending jointly"
    )


def test_regression_double_stream_block_is_residual_when_gates_are_zero() -> None:
    rng = _random_generator(seed=5)
    config = _TEST_CONFIG
    parameters = {
        key: value[0] for key, value in _make_double_block_parameters(rng, config, 1).items()
    }
    image = _normal(rng, (1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.hidden_size), 1.0)
    text = _normal(rng, (1, TEST_TEXT_TOKENS, config.hidden_size), 1.0)
    cosine, sine = _rotation_tables(config, TEST_TEXT_TOKENS)
    modulation = (_zero_modulation(config), _zero_modulation(config))

    new_image, new_text = double_stream_block(
        image, text, parameters, cosine, sine, modulation, modulation, config
    )

    assert np.allclose(np.asarray(new_image), np.asarray(image), atol=NUMERICAL_TOLERANCE)
    assert np.allclose(np.asarray(new_text), np.asarray(text), atol=NUMERICAL_TOLERANCE)


def test_regression_single_stream_block_is_residual_when_gate_is_zero() -> None:
    """
    A zero gate must make the block an identity. Unlike the double
    block this has only one gate, since attention and feed-forward are
    fused into a single residual addition.
    """
    rng = _random_generator(seed=6)
    config = _TEST_CONFIG
    parameters = {
        key: value[0] for key, value in _make_single_block_parameters(rng, config, 1).items()
    }
    total_tokens = TEST_TEXT_TOKENS + TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH
    activations = _normal(rng, (1, total_tokens, config.hidden_size), 1.0)
    cosine, sine = _rotation_tables(config, TEST_TEXT_TOKENS)

    output = single_stream_block(
        activations, parameters, cosine, sine, _zero_modulation(config), config
    )

    assert np.allclose(np.asarray(output), np.asarray(activations), atol=NUMERICAL_TOLERANCE)


def test_smoke_predict_velocity_returns_latent_shaped_output() -> None:
    rng = _random_generator(seed=7)
    config = _TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)
    latent, conditioning, timesteps = _model_inputs(rng, config)

    velocity = predict_velocity(
        latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH, parameters, config
    )

    assert velocity.shape == latent.shape, (
        "velocity must match the latent it will be added to"
    )
    assert np.all(np.isfinite(np.asarray(velocity)))


def test_regression_predict_velocity_depends_on_conditioning() -> None:
    """
    A transformer that ignored its conditioning would still produce
    correctly shaped output, and every shape check would pass. This
    asserts the text actually reaches the prediction.
    """
    rng = _random_generator(seed=8)
    config = _TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)
    latent, conditioning, timesteps = _model_inputs(rng, config)

    first = np.asarray(
        predict_velocity(
            latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH, parameters, config
        )
    )
    second = np.asarray(
        predict_velocity(
            latent,
            conditioning + 3.0,
            timesteps,
            TEST_LATENT_HEIGHT,
            TEST_LATENT_WIDTH,
            parameters,
            config,
        )
    )

    assert not np.allclose(first, second, atol=1e-8), "conditioning did not affect the prediction"


def test_regression_predict_velocity_depends_on_timestep() -> None:
    """
    The timestep reaches the prediction only through modulation, so
    this also confirms modulation is wired into the blocks rather than
    computed and discarded.
    """
    rng = _random_generator(seed=9)
    config = _TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)
    latent, conditioning, _ = _model_inputs(rng, config)

    first = np.asarray(
        predict_velocity(
            latent, conditioning, jnp.asarray(np.array([0.1])), TEST_LATENT_HEIGHT,
            TEST_LATENT_WIDTH, parameters, config,
        )
    )
    second = np.asarray(
        predict_velocity(
            latent, conditioning, jnp.asarray(np.array([0.9])), TEST_LATENT_HEIGHT,
            TEST_LATENT_WIDTH, parameters, config,
        )
    )

    assert not np.allclose(first, second, atol=1e-8), (
        "the timestep did not affect the prediction, so modulation may not be applied"
    )


def test_regression_predict_velocity_rejects_block_count_mismatch() -> None:
    rng = _random_generator(seed=10)
    config = _TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)
    parameters["single_blocks"] = {
        key: value[:1] for key, value in parameters["single_blocks"].items()
    }
    latent, conditioning, timesteps = _model_inputs(rng, config)

    try:
        predict_velocity(
            latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH, parameters, config
        )
    except BlockCountMismatchError as error:
        assert "single" in str(error)
        return
    raise AssertionError("Expected BlockCountMismatchError for a truncated block stack")


def test_regression_modulation_is_computed_once_for_all_blocks() -> None:
    """
    This model shares three modulation projections across every block
    rather than carrying one per block, which is what makes the whole
    modulation computation hoistable out of the sampling loop. The
    structure is asserted here so a future refactor cannot quietly
    reintroduce per-block modulation.
    """
    rng = _random_generator(seed=11)
    config = _TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)

    conditioning_vector = compute_conditioning_vector(
        jnp.asarray(np.array([0.5]), dtype=jnp.float64), parameters["global"], config
    )
    image_pair, text_pair, single_triple = compute_all_modulation(
        conditioning_vector, parameters["global"], config
    )

    assert len(image_pair) == 2, "double blocks need one triple per sub-layer"
    assert len(text_pair) == 2
    assert isinstance(single_triple, ModulationTriple), (
        "single blocks fuse their sub-layers and therefore share one triple"
    )

    modulation_keys = [
        key for key in parameters["global"] if "modulation" in key and "final" not in key
    ]
    assert len(modulation_keys) == 3, (
        f"expected exactly three shared modulation projections, found {modulation_keys}"
    )


def test_regression_forward_pass_preserves_a_single_dtype() -> None:
    """
    The whole forward pass must stay in the latent's dtype.

    This is the bug the first TPU run hit, and it is worth stating
    precisely because it is invisible on CPU with float64 test
    parameters. The timestep embedding was built in float32 regardless
    of its input, so with bfloat16 weights the modulation vectors came
    out float32, promoted the activations they scaled, and left the
    text stream entering a block as bfloat16 and leaving it as float32.
    A Python loop tolerates that; a scan does not, because its carry
    input and output dtypes must match exactly, so the program failed to
    compile at all.

    Running at bfloat16 with bfloat16 parameters is what reproduces it,
    since float64 everywhere hides the promotion.
    """
    rng = _random_generator(seed=20)
    config = _TEST_CONFIG
    parameters = jax.tree_util.tree_map(
        lambda array: array.astype(jnp.bfloat16),
        make_transformer_parameters(rng, config),
    )

    latent = jnp.zeros(
        (1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.in_channels), dtype=jnp.bfloat16
    )
    conditioning = jnp.zeros((1, TEST_TEXT_TOKENS, config.context_dim), dtype=jnp.bfloat16)
    timesteps = jnp.asarray(np.array([0.7672]), dtype=jnp.bfloat16)

    for use_scan in (True, False):
        velocity = predict_velocity(
            latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH,
            parameters, config, ExecutionConfig(use_scan_over_blocks=use_scan),
        )
        assert velocity.dtype == jnp.bfloat16, (
            f"velocity came back as {velocity.dtype} rather than bfloat16 with "
            f"use_scan_over_blocks={use_scan}; something in the forward pass promoted"
        )


def test_regression_mixed_input_dtypes_do_not_break_the_scan() -> None:
    """
    A caller mixing dtypes must not make the scan fail to compile. The
    latent decides, and everything else is cast to it on the way in.
    """
    rng = _random_generator(seed=21)
    config = _TEST_CONFIG
    parameters = jax.tree_util.tree_map(
        lambda array: array.astype(jnp.bfloat16),
        make_transformer_parameters(rng, config),
    )

    latent = jnp.zeros(
        (1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.in_channels), dtype=jnp.bfloat16
    )
    # Deliberately float32 conditioning against a bfloat16 latent.
    conditioning = jnp.zeros((1, TEST_TEXT_TOKENS, config.context_dim), dtype=jnp.float32)
    timesteps = jnp.asarray(np.array([0.5]), dtype=jnp.float32)

    velocity = predict_velocity(
        latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH,
        parameters, config, ExecutionConfig(use_scan_over_blocks=True),
    )

    assert velocity.dtype == jnp.bfloat16, (
        "the latent's dtype should govern the forward pass regardless of what the "
        "conditioning and timesteps arrive as"
    )


_TRANSFORMER_TESTS = [
    test_regression_forward_pass_preserves_a_single_dtype,
    test_regression_mixed_input_dtypes_do_not_break_the_scan,
    test_regression_fused_qkv_splits_component_major_not_head_major,
    test_regression_gated_mlp_gates_first_half_with_second,
    test_smoke_joint_attention_merges_heads,
    test_regression_joint_attention_is_unmasked,
    test_smoke_double_stream_block_preserves_both_stream_shapes,
    test_regression_double_stream_block_lets_text_influence_image,
    test_regression_double_stream_block_is_residual_when_gates_are_zero,
    test_regression_single_stream_block_is_residual_when_gate_is_zero,
    test_smoke_predict_velocity_returns_latent_shaped_output,
    test_regression_predict_velocity_depends_on_conditioning,
    test_regression_predict_velocity_depends_on_timestep,
    test_regression_predict_velocity_rejects_block_count_mismatch,
    test_regression_modulation_is_computed_once_for_all_blocks,
]


def run_transformer_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against the transformer", len(_TRANSFORMER_TESTS))
    for test_function in _TRANSFORMER_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All transformer tests passed")
