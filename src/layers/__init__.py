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
from .convolution import convolution_2d
from .normalization import group_normalization
from .resampling import nearest_neighbor_upsample_2d

__all__ = [
    "convolution_2d",
    "group_normalization",
    "nearest_neighbor_upsample_2d",
    "sigmoid_linear_unit",
]
