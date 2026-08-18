"""The decoder's basic repeated unit."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..checkpoint import has_parameter_group
from ..config import VaeLayerConfig
from ..layers import convolution_2d, group_normalization, sigmoid_linear_unit
from ._parameter_access import convolution_parameters, normalization_parameters


# Parameter key prefixes within a residual block's group.
RESIDUAL_FIRST_NORM_PREFIX = "norm1"
RESIDUAL_SECOND_NORM_PREFIX = "norm2"
RESIDUAL_FIRST_CONVOLUTION_PREFIX = "conv1"
RESIDUAL_SECOND_CONVOLUTION_PREFIX = "conv2"
RESIDUAL_SHORTCUT_PREFIX = "nin_shortcut"


def residual_block(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Apply one residual block: two normalize-activate-convolve stages
    added to a shortcut path.

    The shortcut is the identity when the block preserves channel count,
    and a learned 1x1 projection when it does not. Which case applies is
    determined by whether the checkpoint contains shortcut weights for
    this block, rather than by comparing channel counts, because the
    checkpoint is the authority on what the trained model actually does.
    A block that changes channel count without shortcut weights present
    is a structural mismatch and will fail loudly at the addition rather
    than being silently papered over.

    Note the ordering: normalization comes before the activation and the
    convolution, not after, and the shortcut is added after the second
    convolution with no activation applied to the sum. This matches the
    reference implementation exactly.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, in_channels).
    parameters:
        Parameter group for this block, containing norm1, conv1, norm2,
        conv2, and optionally nin_shortcut entries.
    config:
        Normalization and precision settings.
    """
    context = "residual_block"

    first_norm_scale, first_norm_shift = normalization_parameters(
        parameters, RESIDUAL_FIRST_NORM_PREFIX, context
    )
    hidden = group_normalization(activations, first_norm_scale, first_norm_shift, config)
    hidden = sigmoid_linear_unit(hidden)
    first_kernel, first_bias = convolution_parameters(
        parameters, RESIDUAL_FIRST_CONVOLUTION_PREFIX, context
    )
    hidden = convolution_2d(hidden, first_kernel, first_bias, config)

    second_norm_scale, second_norm_shift = normalization_parameters(
        parameters, RESIDUAL_SECOND_NORM_PREFIX, context
    )
    hidden = group_normalization(hidden, second_norm_scale, second_norm_shift, config)
    hidden = sigmoid_linear_unit(hidden)
    second_kernel, second_bias = convolution_parameters(
        parameters, RESIDUAL_SECOND_CONVOLUTION_PREFIX, context
    )
    hidden = convolution_2d(hidden, second_kernel, second_bias, config)

    shortcut = activations
    if has_parameter_group(parameters, RESIDUAL_SHORTCUT_PREFIX):
        shortcut_kernel, shortcut_bias = convolution_parameters(
            parameters, RESIDUAL_SHORTCUT_PREFIX, context
        )
        shortcut = convolution_2d(activations, shortcut_kernel, shortcut_bias, config)

    return shortcut + hidden
