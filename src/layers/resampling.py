"""Spatial resampling primitives."""

from __future__ import annotations

import jax.numpy as jnp

from ..config import VaeLayerConfig


def nearest_neighbor_upsample_2d(
    activations: jnp.ndarray,
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Upsample spatially by an integer factor, repeating each input pixel
    into a square block of output pixels.

    Implemented as two repeats rather than a resize operation because
    nearest-neighbour upsampling by an integer factor is exactly a
    repeat: every output pixel takes the value of exactly one input
    pixel, with no interpolation, no sampling-grid convention, and
    therefore no opportunity for an off-by-half-pixel mismatch against
    the reference implementation.

    Returns
    -------
    Output of shape (batch, height * factor, width * factor, channels).
    """
    factor = config.upsample_scale_factor
    upsampled = jnp.repeat(activations, repeats=factor, axis=1)
    return jnp.repeat(upsampled, repeats=factor, axis=2)
