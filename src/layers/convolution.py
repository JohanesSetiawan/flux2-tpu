"""Convolution primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import VaeLayerConfig


# jax.lax.conv_general_dilated needs the axis order of its three
# operands spelled out. These match the package's layout conventions:
# NHWC activations in, HWIO kernel, NHWC activations out.
CONVOLUTION_DIMENSION_NUMBERS = ("NHWC", "HWIO", "NHWC")


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
    passed in. Every convolution in this decoder is stride-1 and
    shape-preserving, so a 3x3 kernel always pads by 1 and a 1x1 kernel
    always pads by 0; deriving it removes a parameter that could be
    passed inconsistently with the kernel it accompanies.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, in_channels).
    kernel:
        Kernel in HWIO layout, shape
        (kernel_height, kernel_width, in_channels, out_channels).
    bias:
        Optional per-output-channel bias. None for bias-free
        convolutions.
    config:
        Supplies the precision level.

    Returns
    -------
    Output of shape (batch, height, width, out_channels), spatially
    unchanged from the input.
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
