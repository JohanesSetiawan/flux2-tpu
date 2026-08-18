"""
Tests for src.models.vae.

Decoder parameters here are synthesised at run time by
_make_synthetic_decoder_parameters, which builds a small but
structurally faithful decoder: the same key naming, the same
arrangement of levels and blocks, the same optional shortcut and
upsample convolutions, at a fraction of the channel count. This keeps
the suite fast and network-free while exercising the same code path the
real checkpoint takes.

The most valuable test here is
test_regression_unpack_latent_patches_inverts_packing. Patch unpacking
is a pure index permutation, and a wrong permutation produces an image
that is subtly scrambled rather than obviously broken, which is exactly
the kind of error that survives casual inspection. It is checked by
round-tripping against an independently written packing implementation.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.config import NumericPrecision, VaeDecoderConfig, VaeLayerConfig
from src.models.vae import (
    DecoderStructureError,
    apply_post_quantization_projection,
    decode_latent,
    denormalize_latent,
    discover_residual_block_indices,
    discover_upsample_level_indices,
    unpack_latent_patches,
)


NUMERICAL_TOLERANCE = 1e-10

# Small enough to run in well under a second, structurally complete
# enough to exercise shortcuts, upsampling, and attention.
SYNTHETIC_UNPACKED_LATENT_CHANNELS = 2
SYNTHETIC_STEM_CHANNELS = 8
SYNTHETIC_FINAL_CHANNELS = 4
SYNTHETIC_OUTPUT_CHANNELS = 3
SYNTHETIC_NUM_GROUPS = 2
SYNTHETIC_BLOCKS_PER_LEVEL = 2

_SYNTHETIC_CONFIG = VaeDecoderConfig(
    latent_patch_size=2,
    layer=VaeLayerConfig(
        num_groups=SYNTHETIC_NUM_GROUPS,
        precision=NumericPrecision.HIGHEST,
        attention_query_chunk_size=8,
    ),
)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _convolution_parameters(
    rng: np.random.Generator, prefix: str, in_channels: int, out_channels: int, kernel_size: int
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_weight": rng.standard_normal(
            (kernel_size, kernel_size, in_channels, out_channels)
        )
        * 0.1,
        f"{prefix}_bias": rng.standard_normal((out_channels,)) * 0.1,
    }


def _normalization_parameters(
    rng: np.random.Generator, prefix: str, channels: int
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_weight": rng.standard_normal((channels,)) * 0.1 + 1.0,
        f"{prefix}_bias": rng.standard_normal((channels,)) * 0.1,
    }


def _residual_block_parameters(
    rng: np.random.Generator, prefix: str, in_channels: int, out_channels: int
) -> dict[str, np.ndarray]:
    parameters = {}
    parameters.update(_normalization_parameters(rng, f"{prefix}_norm1", in_channels))
    parameters.update(_convolution_parameters(rng, f"{prefix}_conv1", in_channels, out_channels, 3))
    parameters.update(_normalization_parameters(rng, f"{prefix}_norm2", out_channels))
    parameters.update(
        _convolution_parameters(rng, f"{prefix}_conv2", out_channels, out_channels, 3)
    )
    if in_channels != out_channels:
        parameters.update(
            _convolution_parameters(rng, f"{prefix}_nin_shortcut", in_channels, out_channels, 1)
        )
    return parameters


def _make_synthetic_decoder_parameters(
    rng: np.random.Generator, num_levels: int
) -> dict[str, np.ndarray]:
    """
    Build a complete synthetic decoder whose structure mirrors the real
    one: a stem, a middle section with attention, `num_levels`
    upsampling levels executed from highest index down to zero, and an
    output head.

    Every level except level zero carries an upsample convolution, and
    the final level narrows the channel count and therefore carries a
    shortcut projection. This reproduces the arrangement found in the
    real checkpoint.
    """
    parameters: dict[str, np.ndarray] = {}

    parameters.update(
        _convolution_parameters(
            rng, "conv_in", SYNTHETIC_UNPACKED_LATENT_CHANNELS, SYNTHETIC_STEM_CHANNELS, 3
        )
    )

    parameters.update(
        _residual_block_parameters(
            rng, "mid_block_1", SYNTHETIC_STEM_CHANNELS, SYNTHETIC_STEM_CHANNELS
        )
    )
    parameters.update(_normalization_parameters(rng, "mid_attn_1_norm", SYNTHETIC_STEM_CHANNELS))
    for projection in ("q", "k", "v", "proj_out"):
        parameters.update(
            _convolution_parameters(
                rng,
                f"mid_attn_1_{projection}",
                SYNTHETIC_STEM_CHANNELS,
                SYNTHETIC_STEM_CHANNELS,
                1,
            )
        )
    parameters.update(
        _residual_block_parameters(
            rng, "mid_block_2", SYNTHETIC_STEM_CHANNELS, SYNTHETIC_STEM_CHANNELS
        )
    )

    for level_index in reversed(range(num_levels)):
        is_final_level = level_index == 0
        level_out_channels = (
            SYNTHETIC_FINAL_CHANNELS if is_final_level else SYNTHETIC_STEM_CHANNELS
        )
        block_in_channels = SYNTHETIC_STEM_CHANNELS

        for block_index in range(SYNTHETIC_BLOCKS_PER_LEVEL):
            parameters.update(
                _residual_block_parameters(
                    rng,
                    f"up_{level_index}_block_{block_index}",
                    block_in_channels if block_index == 0 else level_out_channels,
                    level_out_channels,
                )
            )

        if not is_final_level:
            parameters.update(
                _convolution_parameters(
                    rng,
                    f"up_{level_index}_upsample_conv",
                    level_out_channels,
                    level_out_channels,
                    3,
                )
            )

    parameters.update(_normalization_parameters(rng, "norm_out", SYNTHETIC_FINAL_CHANNELS))
    parameters.update(
        _convolution_parameters(
            rng, "conv_out", SYNTHETIC_FINAL_CHANNELS, SYNTHETIC_OUTPUT_CHANNELS, 3
        )
    )

    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in parameters.items()}


def _make_synthetic_vae_parameters(rng: np.random.Generator, num_levels: int) -> dict:
    packed_channels = SYNTHETIC_UNPACKED_LATENT_CHANNELS * (
        _SYNTHETIC_CONFIG.latent_patch_size ** 2
    )
    return {
        "decoder": _make_synthetic_decoder_parameters(rng, num_levels),
        "post_quant_conv": {
            "weight": jnp.asarray(
                rng.standard_normal(
                    (SYNTHETIC_UNPACKED_LATENT_CHANNELS, SYNTHETIC_UNPACKED_LATENT_CHANNELS)
                ),
                dtype=jnp.float64,
            ),
            "bias": jnp.asarray(
                rng.standard_normal((SYNTHETIC_UNPACKED_LATENT_CHANNELS,)), dtype=jnp.float64
            ),
        },
        "latent_denormalize": {
            "scale": jnp.asarray(
                np.abs(rng.standard_normal((packed_channels,))) + 0.5, dtype=jnp.float64
            ),
            "shift": jnp.asarray(rng.standard_normal((packed_channels,)), dtype=jnp.float64),
        },
    }


def _pack_latent_patches_oracle(unpacked: np.ndarray, patch: int) -> np.ndarray:
    """
    Independently written spatial-to-channel packing, assigning each
    output element by explicit index arithmetic. Test-only; this is the
    operation unpack_latent_patches must invert.
    """
    batch, height, width, channels = unpacked.shape
    packed = np.zeros(
        (batch, height // patch, width // patch, channels * patch * patch), dtype=unpacked.dtype
    )
    for y, x, c, patch_row, patch_column in itertools.product(
        range(height // patch),
        range(width // patch),
        range(channels),
        range(patch),
        range(patch),
    ):
        packed_channel = c * patch * patch + patch_row * patch + patch_column
        packed[:, y, x, packed_channel] = unpacked[
            :, y * patch + patch_row, x * patch + patch_column, c
        ]
    return packed


def test_regression_unpack_latent_patches_inverts_packing() -> None:
    for patch, channels, (height, width) in itertools.product(
        (2, 3), (1, 2, 3), ((4, 6), (6, 6))
    ):
        rng = _random_generator(seed=patch * 100 + channels * 10 + height)
        config = VaeDecoderConfig(latent_patch_size=patch)

        unpacked_original = rng.standard_normal(
            (2, height * patch, width * patch, channels)
        )
        packed = _pack_latent_patches_oracle(unpacked_original, patch)

        unpacked_again = np.asarray(
            unpack_latent_patches(jnp.asarray(packed, dtype=jnp.float64), config)
        )

        assert unpacked_again.shape == unpacked_original.shape, (
            f"shape mismatch at patch={patch} channels={channels}"
        )
        assert np.allclose(unpacked_again, unpacked_original, atol=NUMERICAL_TOLERANCE), (
            f"unpack did not invert packing at patch={patch} channels={channels}"
        )


def test_regression_unpack_latent_patches_rejects_indivisible_channels() -> None:
    config = VaeDecoderConfig(latent_patch_size=2)
    latent = jnp.zeros((1, 2, 2, 7), dtype=jnp.float32)

    try:
        unpack_latent_patches(latent, config)
    except ValueError as error:
        assert "divisible" in str(error)
        return
    raise AssertionError("Expected ValueError for an indivisible packed channel count")


def test_smoke_discover_upsample_levels_returns_descending_execution_order() -> None:
    rng = _random_generator(seed=1)
    parameters = _make_synthetic_decoder_parameters(rng, num_levels=3)

    assert discover_upsample_level_indices(parameters) == [2, 1, 0]


def test_smoke_discover_residual_blocks_returns_ascending_order() -> None:
    rng = _random_generator(seed=2)
    parameters = _make_synthetic_decoder_parameters(rng, num_levels=2)

    for level_index in (0, 1):
        assert discover_residual_block_indices(parameters, level_index) == list(
            range(SYNTHETIC_BLOCKS_PER_LEVEL)
        )


def test_regression_discover_upsample_levels_rejects_empty_parameters() -> None:
    try:
        discover_upsample_level_indices({"conv_in_weight": np.zeros((1,))})
    except DecoderStructureError as error:
        assert "No upsampling levels" in str(error)
        return
    raise AssertionError("Expected DecoderStructureError when no levels are present")


def test_regression_discover_upsample_levels_rejects_non_contiguous_indices() -> None:
    parameters = {"up_0_block_0_conv1_weight": np.zeros((1,)), "up_2_block_0_conv1_weight": np.zeros((1,))}

    try:
        discover_upsample_level_indices(parameters)
    except DecoderStructureError as error:
        assert "contiguous" in str(error)
        return
    raise AssertionError("Expected DecoderStructureError for non-contiguous level indices")


def test_regression_discover_residual_blocks_rejects_empty_level() -> None:
    rng = _random_generator(seed=3)
    parameters = _make_synthetic_decoder_parameters(rng, num_levels=2)

    try:
        discover_residual_block_indices(parameters, level_index=99)
    except DecoderStructureError as error:
        assert "no residual blocks" in str(error)
        return
    raise AssertionError("Expected DecoderStructureError for a level with no blocks")


def test_regression_denormalize_latent_applies_affine_transform() -> None:
    rng = _random_generator(seed=4)
    channels = 8
    latent = rng.standard_normal((2, 3, 3, channels))
    scale = rng.standard_normal((channels,))
    shift = rng.standard_normal((channels,))

    expected = latent * scale + shift
    actual = np.asarray(
        denormalize_latent(
            jnp.asarray(latent, dtype=jnp.float64),
            {
                "scale": jnp.asarray(scale, dtype=jnp.float64),
                "shift": jnp.asarray(shift, dtype=jnp.float64),
            },
        )
    )

    assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE)


def test_regression_post_quantization_projection_matches_per_pixel_matmul() -> None:
    rng = _random_generator(seed=5)
    in_channels, out_channels = 4, 6
    activations = rng.standard_normal((2, 3, 5, in_channels))
    weight = rng.standard_normal((in_channels, out_channels))
    bias = rng.standard_normal((out_channels,))

    expected = np.zeros((2, 3, 5, out_channels))
    for n, y, x in itertools.product(range(2), range(3), range(5)):
        expected[n, y, x] = activations[n, y, x] @ weight + bias

    actual = np.asarray(
        apply_post_quantization_projection(
            jnp.asarray(activations, dtype=jnp.float64),
            {
                "weight": jnp.asarray(weight, dtype=jnp.float64),
                "bias": jnp.asarray(bias, dtype=jnp.float64),
            },
        )
    )

    assert np.allclose(actual, expected, atol=NUMERICAL_TOLERANCE)


def test_regression_decode_scales_spatial_dimensions_by_expected_factor() -> None:
    """
    The total spatial scale factor is the patch size multiplied by two
    raised to the number of upsampling levels that carry an upsample
    convolution, which is every level except the last. Sweeping the
    level count confirms the decoder honours the structure it discovers
    rather than assuming a fixed depth.
    """
    for num_levels, (latent_height, latent_width) in itertools.product(
        (2, 3), ((3, 4), (4, 4))
    ):
        rng = _random_generator(seed=num_levels * 10 + latent_height)
        vae_parameters = _make_synthetic_vae_parameters(rng, num_levels)

        packed_channels = SYNTHETIC_UNPACKED_LATENT_CHANNELS * (
            _SYNTHETIC_CONFIG.latent_patch_size ** 2
        )
        latent = jnp.asarray(
            rng.standard_normal((1, latent_height, latent_width, packed_channels)),
            dtype=jnp.float64,
        )

        image = decode_latent(latent, vae_parameters, _SYNTHETIC_CONFIG)

        upsampling_levels = num_levels - 1
        expected_scale = _SYNTHETIC_CONFIG.latent_patch_size * (2 ** upsampling_levels)
        assert image.shape == (
            1,
            latent_height * expected_scale,
            latent_width * expected_scale,
            SYNTHETIC_OUTPUT_CHANNELS,
        ), f"unexpected output shape at num_levels={num_levels}"

        assert np.all(np.isfinite(np.asarray(image))), (
            f"decoder produced non-finite values at num_levels={num_levels}"
        )


def test_regression_decode_is_deterministic() -> None:
    """
    Decoding is a pure function, so the same latent and parameters must
    produce bit-identical output across calls. A failure here would
    indicate accidental nondeterminism, such as uninitialised state.
    """
    rng = _random_generator(seed=700)
    vae_parameters = _make_synthetic_vae_parameters(rng, num_levels=2)
    packed_channels = SYNTHETIC_UNPACKED_LATENT_CHANNELS * (
        _SYNTHETIC_CONFIG.latent_patch_size ** 2
    )
    latent = jnp.asarray(rng.standard_normal((1, 3, 3, packed_channels)), dtype=jnp.float64)

    first = np.asarray(decode_latent(latent, vae_parameters, _SYNTHETIC_CONFIG))
    second = np.asarray(decode_latent(latent, vae_parameters, _SYNTHETIC_CONFIG))

    assert np.array_equal(first, second)


_VAE_TESTS = [
    test_regression_unpack_latent_patches_inverts_packing,
    test_regression_unpack_latent_patches_rejects_indivisible_channels,
    test_smoke_discover_upsample_levels_returns_descending_execution_order,
    test_smoke_discover_residual_blocks_returns_ascending_order,
    test_regression_discover_upsample_levels_rejects_empty_parameters,
    test_regression_discover_upsample_levels_rejects_non_contiguous_indices,
    test_regression_discover_residual_blocks_rejects_empty_level,
    test_regression_denormalize_latent_applies_affine_transform,
    test_regression_post_quantization_projection_matches_per_pixel_matmul,
    test_regression_decode_scales_spatial_dimensions_by_expected_factor,
    test_regression_decode_is_deterministic,
]


def run_vae_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against vae.py", len(_VAE_TESTS))
    for test_function in _VAE_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All VAE tests passed")
