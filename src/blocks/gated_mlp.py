"""Fused gated feed-forward used by the diffusion transformer."""

from __future__ import annotations

import jax.numpy as jnp

from ..layers import sigmoid_linear_unit


def split_and_gate(fused: jnp.ndarray) -> jnp.ndarray:
    """
    Halve a fused projection and gate one half with the other.

    The activation is applied to the **first** half, which then scales
    the second. Gating the other way round, or applying the activation
    to both, computes something different from the same weights while
    producing identical shapes.

    This differs from the text encoder's feed-forward only in packaging:
    there the two halves come from two separate projections, here from
    one wide projection that is split. The arithmetic is the same.
    """
    first_half, second_half = jnp.split(fused, 2, axis=-1)
    return sigmoid_linear_unit(first_half) * second_half
