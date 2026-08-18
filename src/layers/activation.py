"""Activation functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sigmoid_linear_unit(activations: jnp.ndarray) -> jnp.ndarray:
    """
    Apply the SiLU activation, x times sigmoid of x.

    Wrapped rather than called directly at each use site so the
    decoder's activation function is named in one place; the reference
    implementation uses SiLU throughout.
    """
    return jax.nn.silu(activations)
