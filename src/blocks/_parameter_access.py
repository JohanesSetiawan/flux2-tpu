"""
Parameter lookup helpers shared by the decoder's composite blocks.

Both block types address their weights the same way: a normalization
layer is a scale and a shift, a convolution is a kernel and a bias.
Naming those two patterns once here keeps the block implementations
free of repeated string concatenation.
"""

from __future__ import annotations

import numpy as np

from ..checkpoint import require_parameter


WEIGHT_SUFFIX = "weight"
BIAS_SUFFIX = "bias"


def normalization_parameters(
    parameters: dict[str, np.ndarray], prefix: str, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the scale and shift of a normalization layer as a pair."""
    scale = require_parameter(parameters, f"{prefix}_{WEIGHT_SUFFIX}", context)
    shift = require_parameter(parameters, f"{prefix}_{BIAS_SUFFIX}", context)
    return scale, shift


def convolution_parameters(
    parameters: dict[str, np.ndarray], prefix: str, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the kernel and bias of a convolution as a pair."""
    kernel = require_parameter(parameters, f"{prefix}_{WEIGHT_SUFFIX}", context)
    bias = require_parameter(parameters, f"{prefix}_{BIAS_SUFFIX}", context)
    return kernel, bias
