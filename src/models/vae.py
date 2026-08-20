"""
Full autoencoder decoder.

Turns a packed latent into an image: reverse the stored normalization,
unpack the 2x2 spatial patches out of the channel axis, then run the
convolutional decoder (stem, middle section with attention, a stack of
upsampling levels, output head).

The decoder's depth and channel counts are **discovered from the
checkpoint** rather than written down here. Levels are found by
scanning parameter keys, and every channel count follows from the
kernels themselves. A checkpoint with a different number of levels or
blocks would therefore decode correctly, and a checkpoint that
disagrees with this code's structural assumptions fails with a clear
error rather than a silent shape mismatch deep inside a convolution.

The reference implementation runs this decoder entirely in float32,
which is preserved here: this module never casts to a lower precision,
and the precision of the underlying matrix multiplications is
controlled through VaeLayerConfig.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np

from ..blocks import attention_block, residual_block
from ..checkpoint import require_parameter, select_parameter_group
from ..telemetry.tracing import trace_tensor
from ..config import VaeDecoderConfig
from ..layers import (
    convolution_2d,
    group_normalization,
    nearest_neighbor_upsample_2d,
    sigmoid_linear_unit,
)


# Key prefixes within the decoder's flat parameter dictionary.
STEM_CONVOLUTION_PREFIX = "conv_in"
MIDDLE_FIRST_BLOCK_PREFIX = "mid_block_1"
MIDDLE_ATTENTION_PREFIX = "mid_attn_1"
MIDDLE_SECOND_BLOCK_PREFIX = "mid_block_2"
OUTPUT_NORM_PREFIX = "norm_out"
OUTPUT_CONVOLUTION_PREFIX = "conv_out"
UPSAMPLE_CONVOLUTION_SUFFIX = "upsample_conv"

WEIGHT_SUFFIX = "weight"
BIAS_SUFFIX = "bias"

# Patterns used to discover structure. Level and block indices are read
# from the keys themselves so that depth is never assumed.
UPSAMPLE_LEVEL_PATTERN = re.compile(r"^up_(\d+)_")
RESIDUAL_BLOCK_PATTERN_TEMPLATE = r"^up_{level}_block_(\d+)_"


class DecoderStructureError(Exception):
    """
    Raised when the restored parameters do not form a decoder this code
    can run: no upsampling levels found, non-contiguous level indices,
    or a level with no residual blocks. Failing here, with a message
    naming what was found, is far more useful than failing later inside
    a convolution with a shape mismatch.
    """


def discover_upsample_level_indices(decoder_parameters: dict[str, np.ndarray]) -> list[int]:
    """
    Find every upsampling level present in the checkpoint, verifying the
    indices form a contiguous range starting at zero.

    Returns them in **descending** order, which is the order the decoder
    executes: the reference implementation builds its level list from
    lowest to highest resolution but iterates it in reverse, so the
    highest-numbered level runs first, closest to the latent.
    """
    level_indices = {
        int(match.group(1))
        for key in decoder_parameters
        if (match := UPSAMPLE_LEVEL_PATTERN.match(key)) is not None
    }

    if not level_indices:
        raise DecoderStructureError(
            "No upsampling levels found in the decoder parameters. Expected keys "
            "matching the pattern 'up_<level>_...'."
        )

    expected_indices = set(range(len(level_indices)))
    if level_indices != expected_indices:
        raise DecoderStructureError(
            f"Upsampling level indices are not a contiguous range starting at zero. "
            f"Found: {sorted(level_indices)}"
        )

    return sorted(level_indices, reverse=True)


def discover_residual_block_indices(
    decoder_parameters: dict[str, np.ndarray], level_index: int
) -> list[int]:
    """
    Find every residual block within one upsampling level, verifying the
    indices form a contiguous range starting at zero. Returned in
    ascending execution order.
    """
    pattern = re.compile(RESIDUAL_BLOCK_PATTERN_TEMPLATE.format(level=level_index))
    block_indices = {
        int(match.group(1))
        for key in decoder_parameters
        if (match := pattern.match(key)) is not None
    }

    if not block_indices:
        raise DecoderStructureError(
            f"Upsampling level {level_index} contains no residual blocks."
        )

    expected_indices = set(range(len(block_indices)))
    if block_indices != expected_indices:
        raise DecoderStructureError(
            f"Residual block indices within upsampling level {level_index} are not a "
            f"contiguous range starting at zero. Found: {sorted(block_indices)}"
        )

    return sorted(block_indices)


def denormalize_latent(
    latent: jnp.ndarray, denormalize_parameters: dict[str, np.ndarray]
) -> jnp.ndarray:
    """
    Undo the channel-wise normalization applied when the latent was
    produced.

    The reference implementation performs this with a frozen,
    affine-free BatchNorm in evaluation mode; the conversion collapsed
    that into the two constant vectors used here, so this is a plain
    affine transform rather than a normalization layer.

    Note that this operates on the **packed** latent, before spatial
    unpacking, so the statistics are per packed channel.
    """
    scale = require_parameter(denormalize_parameters, "scale", "denormalize_latent")
    shift = require_parameter(denormalize_parameters, "shift", "denormalize_latent")
    return latent * scale + shift


def unpack_latent_patches(latent: jnp.ndarray, config: VaeDecoderConfig) -> jnp.ndarray:
    """
    Move each 2x2 spatial patch back out of the channel axis, undoing
    the encoder's space-to-depth packing.

    The packing interleaved channels as (channel, patch_row,
    patch_column), so an input channel index decomposes in that order.
    Getting this ordering wrong produces an image that is subtly
    scrambled at the pixel level rather than obviously broken, so the
    decomposition is written out explicitly here and asserted in the
    tests by round-tripping against an independent packing
    implementation.

    Parameters
    ----------
    latent:
        Packed latent, shape (batch, height, width, packed_channels).
    config:
        Supplies the patch size.

    Returns
    -------
    Unpacked latent, shape
    (batch, height * patch, width * patch, packed_channels / patch^2).
    """
    patch = config.latent_patch_size
    batch, height, width, packed_channels = latent.shape

    patch_area = patch * patch
    if packed_channels % patch_area != 0:
        raise ValueError(
            f"Packed channel count {packed_channels} is not divisible by the square "
            f"of the patch size ({patch_area})"
        )
    unpacked_channels = packed_channels // patch_area

    # (batch, height, width, channel, patch_row, patch_column)
    reshaped = latent.reshape(batch, height, width, unpacked_channels, patch, patch)
    # Interleave each patch axis with its spatial axis:
    # (batch, height, patch_row, width, patch_column, channel)
    transposed = jnp.transpose(reshaped, (0, 1, 4, 2, 5, 3))
    return transposed.reshape(batch, height * patch, width * patch, unpacked_channels)


def apply_post_quantization_projection(
    activations: jnp.ndarray, projection_parameters: dict[str, np.ndarray]
) -> jnp.ndarray:
    """
    Apply the per-pixel linear projection that precedes the
    convolutional decoder.

    This was a 1x1 convolution in the reference implementation. Because
    a 1x1 convolution has no spatial receptive field, the conversion
    stored it as a plain (in_channels, out_channels) matrix, and it is
    applied here as a per-pixel matrix multiply. That is exactly
    equivalent and, unlike folding it into the following convolution,
    remains exact at the image border.
    """
    weight = require_parameter(
        projection_parameters, WEIGHT_SUFFIX, "apply_post_quantization_projection"
    )
    bias = require_parameter(
        projection_parameters, BIAS_SUFFIX, "apply_post_quantization_projection"
    )
    return jnp.einsum("bhwi,io->bhwo", activations, weight) + bias


def _apply_named_convolution(
    activations: jnp.ndarray,
    decoder_parameters: dict[str, np.ndarray],
    prefix: str,
    config: VaeDecoderConfig,
) -> jnp.ndarray:
    """Look up a convolution's kernel and bias by prefix and apply it."""
    kernel = require_parameter(decoder_parameters, f"{prefix}_{WEIGHT_SUFFIX}", prefix)
    bias = require_parameter(decoder_parameters, f"{prefix}_{BIAS_SUFFIX}", prefix)
    return convolution_2d(activations, kernel, bias, config.layer)


def _run_middle_section(
    activations: jnp.ndarray,
    decoder_parameters: dict[str, np.ndarray],
    config: VaeDecoderConfig,
) -> jnp.ndarray:
    """
    Run the decoder's middle section: residual block, self-attention,
    residual block, all at latent resolution.

    This is where the decoder's attention lives, and where its peak
    memory usage occurs. Nothing here changes spatial size or channel
    count.
    """
    activations = residual_block(
        activations,
        select_parameter_group(decoder_parameters, MIDDLE_FIRST_BLOCK_PREFIX),
        config.layer,
    )
    activations = attention_block(
        activations,
        select_parameter_group(decoder_parameters, MIDDLE_ATTENTION_PREFIX),
        config.layer,
    )
    activations = residual_block(
        activations,
        select_parameter_group(decoder_parameters, MIDDLE_SECOND_BLOCK_PREFIX),
        config.layer,
    )
    return activations


def _run_upsample_levels(
    activations: jnp.ndarray,
    decoder_parameters: dict[str, np.ndarray],
    config: VaeDecoderConfig,
) -> jnp.ndarray:
    """
    Run every upsampling level in execution order.

    Each level applies its residual blocks in sequence, then upsamples
    if it carries an upsample convolution. The final level does not: the
    decoder reaches full resolution at the end of the second-to-last
    upsample, and the last level only refines. Whether to upsample is
    therefore decided by the presence of the convolution in the
    checkpoint rather than by comparing the level index against zero,
    keeping the code's behaviour tied to the weights it was given.
    """
    for level_index in discover_upsample_level_indices(decoder_parameters):
        for block_index in discover_residual_block_indices(decoder_parameters, level_index):
            block_parameters = select_parameter_group(
                decoder_parameters, f"up_{level_index}_block_{block_index}"
            )
            activations = residual_block(activations, block_parameters, config.layer)

        upsample_prefix = f"up_{level_index}_{UPSAMPLE_CONVOLUTION_SUFFIX}"
        if f"{upsample_prefix}_{WEIGHT_SUFFIX}" in decoder_parameters:
            activations = nearest_neighbor_upsample_2d(activations, config.layer)
            activations = _apply_named_convolution(
                activations, decoder_parameters, upsample_prefix, config
            )

    return activations


def decode_latent(
    latent: jnp.ndarray,
    vae_parameters: dict,
    config: VaeDecoderConfig,
) -> jnp.ndarray:
    """
    Decode a packed latent into an image.

    Parameters
    ----------
    latent:
        Packed latent in NHWC layout, shape
        (batch, height, width, packed_channels). For a 1024x1024 output
        this is (batch, 64, 64, 128).
    vae_parameters:
        The restored VAE component, containing "decoder",
        "post_quant_conv" and "latent_denormalize" groups.
    config:
        Structural and numerical settings.

    Returns
    -------
    Image tensor in NHWC layout, shape
    (batch, height * 16, width * 16, 3), with values in the model's
    output range rather than clamped to a display range; conversion to
    an image is the caller's responsibility.
    """
    decoder_parameters = vae_parameters["decoder"]

    trace_tensor("vae.input.latent", latent)

    activations = denormalize_latent(latent, vae_parameters["latent_denormalize"])
    activations = trace_tensor("vae.denormalized", activations)

    activations = unpack_latent_patches(activations, config)
    activations = apply_post_quantization_projection(
        activations, vae_parameters["post_quant_conv"]
    )
    activations = trace_tensor("vae.unpacked", activations)

    activations = _apply_named_convolution(
        activations, decoder_parameters, STEM_CONVOLUTION_PREFIX, config
    )
    activations = trace_tensor("vae.stem", activations)

    activations = trace_tensor(
        "vae.middle", _run_middle_section(activations, decoder_parameters, config)
    )
    activations = trace_tensor(
        "vae.upsampled", _run_upsample_levels(activations, decoder_parameters, config)
    )

    output_scale = require_parameter(
        decoder_parameters, f"{OUTPUT_NORM_PREFIX}_{WEIGHT_SUFFIX}", OUTPUT_NORM_PREFIX
    )
    output_shift = require_parameter(
        decoder_parameters, f"{OUTPUT_NORM_PREFIX}_{BIAS_SUFFIX}", OUTPUT_NORM_PREFIX
    )
    activations = group_normalization(activations, output_scale, output_shift, config.layer)
    activations = sigmoid_linear_unit(activations)

    return trace_tensor(
        "vae.output.image",
        _apply_named_convolution(
            activations, decoder_parameters, OUTPUT_CONVOLUTION_PREFIX, config
        ),
    )
