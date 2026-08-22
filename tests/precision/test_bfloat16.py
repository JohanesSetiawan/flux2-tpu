"""
Running every component at the dtype production actually uses.

The rest of the suite computes in float64, because comparing against an
oracle needs precision the oracle can be trusted at. Production does not
run in float64. It runs in bfloat16 for the transformer and the text
encoder, and float32 for the decoder, and there is a class of bug that
exists only there.

That class already cost a real failure. The timestep embedding was
built in float32 regardless of its input, so against bfloat16 weights
the modulation vectors came out float32, promoted the activations they
scaled, and left a scanned block stack with a carry whose input and
output dtypes disagreed. The program failed to compile. Every test
passed beforehand, because float64 everywhere hides a promotion to
float32.

So this module sweeps each component at its production dtype and
asserts three things that float64 testing cannot:

  dtype is preserved  no operation silently promotes, which under a
                      scan is a hard compile error rather than a slow
                      path
  values stay finite  bfloat16 has roughly three decimal digits and a
                      narrow exponent, so an accumulation that is
                      merely imprecise in float64 can overflow here
  shape is preserved  the output still tracks the float64 result in
                      correlation and scale, so a component that
                      collapses to zeros, saturates, or drifts by an
                      order of magnitude fails

What this suite deliberately does **not** assert is a precision bound,
and the reason is worth recording because it was arrived at by
experiment rather than assumption.

The intent was to catch a reduction accumulating in bfloat16 instead of
promoting, which is a mistake this codebase has made three times in
other places. That test could not be built: deliberately forcing group
normalization to accumulate in bfloat16, over a tensor large enough for
a naive sum to saturate many times over, produced output correlating
with the float64 reference at 0.99999 and matching its scale to four
decimals. XLA promotes reduction accumulators regardless of input dtype,
so the failure being tested for does not occur.

A per-element tolerance is also not meaningful here. Through
twenty-five transformer blocks, bfloat16 error reaches roughly 0.8 of
the tensor's scale at its worst element while correlation stays at
0.968 and the scale is preserved. That is ordinary accumulation in an
eight-bit mantissa, not a defect, and any threshold loose enough to
admit it would be too loose to catch anything.

So the assertions here are the ones that discriminate: dtype
preservation, which is exactly the failure that broke compilation;
finiteness; and preservation of shape and scale.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

from src.config import ExecutionConfig, TextEncoderConfig, TransformerConfig, VaeDecoderConfig, VaeLayerConfig
from src.layers import group_normalization, rms_normalization
from src.models.text_encoder import encode_prompt
from src.models.transformer import predict_velocity
from src.models.vae import decode_latent
from src.sampling import compute_sigma_schedule, denoise_latent
from src.config import SamplingConfig
from tests.models.test_text_encoder import _make_parameters as make_text_encoder_parameters
from tests.models.test_text_encoder import _TEST_CONFIG as TEXT_ENCODER_TEST_CONFIG
from tests.models.test_text_encoder import _token_inputs
from tests.models.test_transformer import (
    TEST_LATENT_HEIGHT,
    TEST_LATENT_WIDTH,
    TEST_TEXT_TOKENS,
    _TEST_CONFIG as TRANSFORMER_TEST_CONFIG,
    make_transformer_parameters,
)
from tests.models.test_vae import _make_synthetic_vae_parameters, _SYNTHETIC_CONFIG


# How closely the low-precision output must still track the reference in
# shape. Correlation answers whether the same structure survived; the
# scale ratio answers whether it survived at the same magnitude.
#
# Both are deliberately loose. They exist to catch collapse, saturation
# and drift, not to measure precision, and a component that merely
# rounds passes comfortably: through twenty-five transformer blocks,
# bfloat16 correlates at 0.968 and preserves scale to within a percent.
MINIMUM_SHAPE_CORRELATION = 0.90
MAXIMUM_SCALE_DRIFT = 0.25

# A reference this flat has no shape to compare against, so correlation
# is undefined and the check is skipped.
NEGLIGIBLE_SCALE = 1e-6


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("precision_tests")
    logger.addHandler(logging.NullHandler())
    return logger


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cast_tree(tree, dtype):
    return jax.tree_util.tree_map(lambda leaf: leaf.astype(dtype), tree)


def _shape_agreement(actual: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """
    How closely a low-precision result still tracks the reference.

    Returns correlation and the ratio of scales. Together they answer
    whether the output is still the same tensor rather than how
    precisely it matches, which is the question a low-precision run can
    actually be held to.
    """
    reference = np.asarray(reference, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()

    reference_scale = float(np.std(reference))
    if reference_scale < NEGLIGIBLE_SCALE:
        return 1.0, 1.0

    correlation = float(np.corrcoef(actual, reference)[0, 1])
    return correlation, float(np.std(actual)) / reference_scale


def _assert_usable(name: str, actual: jnp.ndarray, expected_dtype, reference: np.ndarray) -> None:
    """Check dtype preservation, finiteness, and agreement in one place."""
    assert actual.dtype == expected_dtype, (
        f"{name} returned {actual.dtype} rather than {expected_dtype}; something in "
        f"the forward pass promoted, which under a scan is a compile error"
    )

    values = np.asarray(actual, dtype=np.float64)
    assert np.all(np.isfinite(values)), (
        f"{name} produced non-finite values at {expected_dtype}; bfloat16 has a "
        f"narrow exponent, so an accumulation that merely loses precision in "
        f"float64 can overflow here"
    )

    correlation, scale_ratio = _shape_agreement(values, reference)
    assert correlation >= MINIMUM_SHAPE_CORRELATION, (
        f"{name} correlates with its float64 result at only {correlation:.3f}; the "
        f"output has lost its structure rather than merely lost precision"
    )
    assert abs(1.0 - scale_ratio) <= MAXIMUM_SCALE_DRIFT, (
        f"{name} came out at {scale_ratio:.3f} times the scale of its float64 "
        f"result; the output has collapsed, saturated, or drifted"
    )


def test_regression_rms_normalization_survives_bfloat16() -> None:
    """
    The statistic must be computed in a wider dtype and only the result
    demoted. Accumulating a sum of squares in bfloat16 saturates, which
    has already appeared three times in this codebase in other
    reductions.
    """
    rng = _random_generator(seed=0)
    width = 256
    activations = rng.standard_normal((2, 8, width))
    scale = rng.standard_normal((width,)) * 0.1 + 1.0

    reference = np.asarray(
        rms_normalization(
            jnp.asarray(activations, dtype=jnp.float64),
            jnp.asarray(scale, dtype=jnp.float64),
            1e-6,
        )
    )
    actual = rms_normalization(
        jnp.asarray(activations, dtype=jnp.bfloat16),
        jnp.asarray(scale, dtype=jnp.bfloat16),
        1e-6,
    )

    _assert_usable("rms_normalization", actual, jnp.bfloat16, reference)


def test_regression_group_normalization_survives_bfloat16() -> None:
    rng = _random_generator(seed=1)
    channels = 64
    config = VaeLayerConfig(num_groups=8)
    activations = rng.standard_normal((1, 12, 12, channels))
    scale = rng.standard_normal((channels,)) * 0.1 + 1.0
    shift = rng.standard_normal((channels,)) * 0.1

    reference = np.asarray(
        group_normalization(
            jnp.asarray(activations, dtype=jnp.float64),
            jnp.asarray(scale, dtype=jnp.float64),
            jnp.asarray(shift, dtype=jnp.float64),
            config,
        )
    )
    actual = group_normalization(
        jnp.asarray(activations, dtype=jnp.bfloat16),
        jnp.asarray(scale, dtype=jnp.bfloat16),
        jnp.asarray(shift, dtype=jnp.bfloat16),
        config,
    )

    _assert_usable("group_normalization", actual, jnp.bfloat16, reference)


def test_regression_transformer_survives_bfloat16_under_scan_and_unrolled() -> None:
    """
    The component the original failure occurred in, checked on both
    execution paths.

    A scan is stricter than a loop here: its carry dtypes must match
    exactly, so a promotion anywhere inside a block body stops the
    program compiling rather than merely slowing it.
    """
    rng = _random_generator(seed=2)
    config = TRANSFORMER_TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)

    latent = rng.standard_normal((1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.in_channels))
    conditioning = rng.standard_normal((1, TEST_TEXT_TOKENS, config.context_dim))
    timesteps = np.array([0.7672])

    reference = np.asarray(
        predict_velocity(
            jnp.asarray(latent, dtype=jnp.float64),
            jnp.asarray(conditioning, dtype=jnp.float64),
            jnp.asarray(timesteps, dtype=jnp.float64),
            TEST_LATENT_HEIGHT,
            TEST_LATENT_WIDTH,
            parameters,
            config,
        )
    )

    low_precision_parameters = _cast_tree(parameters, jnp.bfloat16)
    for use_scan in (True, False):
        actual = predict_velocity(
            jnp.asarray(latent, dtype=jnp.bfloat16),
            jnp.asarray(conditioning, dtype=jnp.bfloat16),
            jnp.asarray(timesteps, dtype=jnp.bfloat16),
            TEST_LATENT_HEIGHT,
            TEST_LATENT_WIDTH,
            low_precision_parameters,
            config,
            ExecutionConfig(use_scan_over_blocks=use_scan),
        )
        _assert_usable(
            f"predict_velocity (scan={use_scan})", actual, jnp.bfloat16, reference
        )


def test_regression_text_encoder_survives_bfloat16() -> None:
    """
    Twenty-seven layers of accumulation, each feeding the next. If any
    reduction inside runs at bfloat16 rather than promoting, the error
    compounds with depth rather than staying local.
    """
    rng = _random_generator(seed=3)
    config = TEXT_ENCODER_TEST_CONFIG
    parameters = make_text_encoder_parameters(rng, config, config.num_layers_required)
    token_ids, token_is_real = _token_inputs(rng, config, real_length=4)

    reference = np.asarray(encode_prompt(token_ids, token_is_real, parameters, config))

    low_precision_parameters = _cast_tree(parameters, jnp.bfloat16)
    for use_scan in (True, False):
        actual = encode_prompt(
            token_ids,
            token_is_real,
            low_precision_parameters,
            config,
            ExecutionConfig(use_scan_over_blocks=use_scan),
        )
        _assert_usable(
            f"encode_prompt (scan={use_scan})", actual, jnp.bfloat16, reference
        )


def test_regression_decoder_runs_at_float32_not_bfloat16() -> None:
    """
    The decoder is the one component that must not be demoted.

    The reference implementation decodes in float32, and this
    implementation follows it. A decoder quietly running in bfloat16
    would still produce an image, and the difference would show as
    banding and colour shifts rather than as an error, so the dtype is
    asserted rather than assumed.
    """
    rng = _random_generator(seed=4)
    parameters = _make_synthetic_vae_parameters(rng, num_levels=2)
    packed_channels = 2 * (_SYNTHETIC_CONFIG.latent_patch_size ** 2)
    latent = rng.standard_normal((1, 3, 3, packed_channels))

    reference = np.asarray(
        decode_latent(
            jnp.asarray(latent, dtype=jnp.float64), parameters, _SYNTHETIC_CONFIG
        )
    )

    float32_parameters = _cast_tree(parameters, jnp.float32)
    actual = decode_latent(
        jnp.asarray(latent, dtype=jnp.float32), float32_parameters, _SYNTHETIC_CONFIG
    )

    _assert_usable("decode_latent", actual, jnp.float32, reference)


def test_regression_sampler_accumulates_in_bfloat16_without_collapsing() -> None:
    """
    The latent accumulator runs in bfloat16, matching the reference.

    Four steps is few enough that the reference chose not to promote,
    and this follows it. Few is not none, so the accumulation is checked
    rather than assumed: a latent that drifts toward zero or saturates
    over four steps would produce an image with no error anywhere.
    """
    rng = _random_generator(seed=5)
    schedule = compute_sigma_schedule(4096, SamplingConfig())
    initial = rng.standard_normal((1, 16, 8))
    weight = rng.standard_normal((8, 8)) * 0.3

    def velocity_at(dtype):
        matrix = jnp.asarray(weight, dtype=dtype)

        def predict(tokens, timesteps):
            return jnp.tanh(tokens @ matrix) * timesteps[0]

        return predict

    reference = np.asarray(
        denoise_latent(
            jnp.asarray(initial, dtype=jnp.float64), schedule, velocity_at(jnp.float64)
        )
    )
    actual = denoise_latent(
        jnp.asarray(initial, dtype=jnp.bfloat16), schedule, velocity_at(jnp.bfloat16)
    )

    _assert_usable("denoise_latent", actual, jnp.bfloat16, reference)


def test_regression_bfloat16_inputs_do_not_promote_through_the_stack() -> None:
    """
    A blanket check that no component returns a wider dtype than it was
    given.

    Stated separately from the per-component tests because promotion is
    the specific failure mode that broke compilation, and it is worth
    one assertion that names it rather than only inferring it from a
    tolerance check.
    """
    rng = _random_generator(seed=6)
    config = TRANSFORMER_TEST_CONFIG
    parameters = _cast_tree(make_transformer_parameters(rng, config), jnp.bfloat16)

    velocity = predict_velocity(
        jnp.zeros((1, TEST_LATENT_HEIGHT * TEST_LATENT_WIDTH, config.in_channels), dtype=jnp.bfloat16),
        jnp.zeros((1, TEST_TEXT_TOKENS, config.context_dim), dtype=jnp.bfloat16),
        jnp.asarray(np.array([0.5]), dtype=jnp.bfloat16),
        TEST_LATENT_HEIGHT,
        TEST_LATENT_WIDTH,
        parameters,
        config,
    )

    assert velocity.dtype == jnp.bfloat16, (
        f"the transformer promoted bfloat16 input to {velocity.dtype}"
    )


_PRECISION_TESTS = [
    test_regression_rms_normalization_survives_bfloat16,
    test_regression_group_normalization_survives_bfloat16,
    test_regression_transformer_survives_bfloat16_under_scan_and_unrolled,
    test_regression_text_encoder_survives_bfloat16,
    test_regression_decoder_runs_at_float32_not_bfloat16,
    test_regression_sampler_accumulates_in_bfloat16_without_collapsing,
    test_regression_bfloat16_inputs_do_not_promote_through_the_stack,
]


def run_precision_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests at production dtypes rather than float64",
        len(_PRECISION_TESTS),
    )
    for test_function in _PRECISION_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All precision tests passed")
