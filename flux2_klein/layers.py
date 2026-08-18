"""
Layer primitives for the autoencoder decoder.

Every function here is a pure JAX function: it takes arrays and a
configuration value, returns an array, and holds no state. None of them
log. This is deliberate and worth understanding before adding to this
module: these functions run inside jax.jit-compiled regions, where
Python-level logging executes once at trace time and never again during
the many actual invocations, making it actively misleading rather than
merely useless. Observability for this codebase lives at the
orchestration layer (checkpoint loading, pipeline stages, test runs),
not inside traced numerics.

Layout conventions, applied consistently throughout:

- Activations are NHWC (batch, height, width, channels). TPU's
  convolution and reduction kernels are laid out for a trailing channel
  axis; NCHW would force a transpose on every operation.
- Convolution kernels are HWIO (height, width, in_channels,
  out_channels). The checkpoint conversion already reordered every
  kernel from PyTorch's OIHW into this layout, so no transpose happens
  at inference time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import VaeLayerConfig


# jax.lax.conv_general_dilated needs the axis order of its three
# operands spelled out. These correspond to the layout conventions
# described in the module docstring: NHWC activations in, HWIO kernel,
# NHWC activations out.
CONVOLUTION_DIMENSION_NUMBERS = ("NHWC", "HWIO", "NHWC")

# Minimum dtype for reductions inside normalization. Accumulating a sum
# of squares over tens of thousands of elements in bfloat16 loses
# precision quickly enough to be visible in decoded output, and the cost
# of promoting for the reduction alone is negligible against the
# surrounding convolutions.
#
# This is a floor, not a fixed target: it is applied through
# jnp.promote_types, so a bfloat16 or float16 input is promoted up to
# float32 while a float64 input is left alone. Casting unconditionally
# to float32 would silently reduce precision for callers that
# deliberately work in float64, which is exactly what the regression
# tests do when comparing against their float64 oracles.
MINIMUM_NORMALIZATION_ACCUMULATION_DTYPE = jnp.float32


def convolution_2d(
    activations: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None,
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Apply a stride-1 2D convolution with zero padding sized to preserve
    spatial dimensions.

    Padding is derived from the kernel's own spatial size rather than
    passed in: every convolution in this decoder is stride-1 and
    shape-preserving, so a 3x3 kernel always pads by 1 and a 1x1 kernel
    always pads by 0. Deriving it removes a parameter that could be
    passed inconsistently with the kernel it accompanies.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, in_channels).
    kernel:
        Convolution kernel in HWIO layout, shape
        (kernel_height, kernel_width, in_channels, out_channels).
    bias:
        Optional per-output-channel bias, shape (out_channels,). Passed
        as None for the convolutions in this decoder that have no bias.
    config:
        Supplies the precision level.

    Returns
    -------
    Output of shape (batch, height, width, out_channels), with height
    and width unchanged from the input.
    """
    kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
    padding = (
        (kernel_height // 2, kernel_height // 2),
        (kernel_width // 2, kernel_width // 2),
    )

    output = jax.lax.conv_general_dilated(
        lhs=activations,
        rhs=kernel,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=CONVOLUTION_DIMENSION_NUMBERS,
        precision=config.precision.to_jax_precision(),
    )

    if bias is not None:
        output = output + bias

    return output


def group_normalization(
    activations: jnp.ndarray,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Normalize activations over spatial dimensions within channel groups,
    then apply a learned per-channel affine transform.

    Channels are partitioned into config.num_groups contiguous groups.
    Each group's mean and variance are computed across height, width and
    that group's channels, independently per batch element. Variance
    uses the biased (population) estimator, matching the reference
    implementation.

    The reduction runs in at least float32 (see
    MINIMUM_NORMALIZATION_ACCUMULATION_DTYPE above) and the result is
    cast back to the input dtype.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, channels). channels must be
        divisible by config.num_groups.
    scale:
        Per-channel multiplicative parameter, shape (channels,).
    shift:
        Per-channel additive parameter, shape (channels,).
    config:
        Supplies the group count and the epsilon added to variance
        before the reciprocal square root.
    """
    batch, height, width, channels = activations.shape
    if channels % config.num_groups != 0:
        raise ValueError(
            f"Channel count {channels} is not divisible by the configured "
            f"group count {config.num_groups}"
        )

    input_dtype = activations.dtype
    channels_per_group = channels // config.num_groups
    accumulation_dtype = jnp.promote_types(
        input_dtype, MINIMUM_NORMALIZATION_ACCUMULATION_DTYPE
    )

    grouped = activations.astype(accumulation_dtype).reshape(
        batch, height, width, config.num_groups, channels_per_group
    )

    # Reduce over height, width and within-group channels, keeping the
    # batch and group axes so the statistics broadcast back correctly.
    reduction_axes = (1, 2, 4)
    mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
    variance = jnp.mean(jnp.square(grouped - mean), axis=reduction_axes, keepdims=True)

    normalized = (grouped - mean) * jax.lax.rsqrt(variance + config.normalization_epsilon)
    normalized = normalized.reshape(batch, height, width, channels).astype(input_dtype)

    return normalized * scale + shift


def nearest_neighbor_upsample_2d(
    activations: jnp.ndarray,
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Upsample spatially by an integer factor, repeating each input pixel
    into a square block of output pixels.

    This is implemented as two repeats rather than a resize operation
    because nearest-neighbour upsampling by an integer factor is exactly
    a repeat: every output pixel takes the value of exactly one input
    pixel, with no interpolation, no sampling-grid convention, and
    therefore no opportunity for an off-by-half-pixel mismatch against
    the reference implementation.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, channels).
    config:
        Supplies the integer scale factor.

    Returns
    -------
    Output of shape
    (batch, height * factor, width * factor, channels).
    """
    factor = config.upsample_scale_factor
    upsampled = jnp.repeat(activations, repeats=factor, axis=1)
    upsampled = jnp.repeat(upsampled, repeats=factor, axis=2)
    return upsampled


def sigmoid_linear_unit(activations: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the SiLU (also called swish) activation, x * sigmoid(x).

    Wrapped rather than called directly at each use site so that the
    decoder's activation function is named in one place; the reference
    implementation uses SiLU throughout the decoder.
    """
    return jax.nn.silu(activations)
