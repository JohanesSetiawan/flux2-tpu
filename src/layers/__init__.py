"""
Individual mathematical primitives.

Every function here is pure: it takes arrays and a configuration
object, returns an array, and holds no state. None of them log, because
they run inside jit-compiled regions where logging reports tracing
rather than execution.

Layout conventions, applied consistently throughout the package:

- Activations are NHWC (batch, height, width, channels). TPU's
  convolution and reduction kernels are laid out for a trailing channel
  axis; NCHW would force a transpose on every operation.
- Convolution kernels are HWIO (height, width, in_channels,
  out_channels). The checkpoint conversion already reordered every
  kernel from PyTorch's OIHW into this layout, so no transpose happens
  at inference time.
"""

from .activation import sigmoid_linear_unit
from .axial_positional import (
    apply_axial_rotation,
    axial_rotation_table,
    build_position_identifiers,
)
from .embedding import timestep_embedding
from .convolution import convolution_2d
from .masking import causal_padding_mask
from .normalization import group_normalization, rms_normalization
from .positional import apply_rotary_embedding, rotary_frequency_table
from .resampling import nearest_neighbor_upsample_2d

__all__ = [
    "apply_axial_rotation",
    "apply_rotary_embedding",
    "axial_rotation_table",
    "build_position_identifiers",
    "causal_padding_mask",
    "convolution_2d",
    "group_normalization",
    "nearest_neighbor_upsample_2d",
    "rms_normalization",
    "rotary_frequency_table",
    "sigmoid_linear_unit",
    "timestep_embedding",
]
