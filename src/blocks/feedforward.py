"""Gated feed-forward network."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..checkpoint import require_parameter
from ..config import TextEncoderConfig
from ..layers import sigmoid_linear_unit


GATE_PROJECTION_KEY = "mlp_gate_proj_weight"
UP_PROJECTION_KEY = "mlp_up_proj_weight"
DOWN_PROJECTION_KEY = "mlp_down_proj_weight"


def gated_feedforward(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    config: TextEncoderConfig,
) -> jnp.ndarray:
    """
    Apply the SwiGLU feed-forward network used by Qwen3.

    Two projections read the same input: one is passed through SiLU and
    acts as a gate, the other passes through unchanged. Their elementwise
    product is projected back down. Note that the activation is applied
    to the gate branch only; applying it to both, or to the product,
    computes something different from the same weights.

    None of these projections has a bias, matching the checkpoint.

    Parameters
    ----------
    activations:
        Shape (batch, sequence_length, hidden_size).
    parameters:
        One layer's parameter group, containing the three projections.
    config:
        Supplies the precision level.
    """
    context = "gated_feedforward"
    precision = config.precision.to_jax_precision()

    gate_weight = require_parameter(parameters, GATE_PROJECTION_KEY, context)
    up_weight = require_parameter(parameters, UP_PROJECTION_KEY, context)
    down_weight = require_parameter(parameters, DOWN_PROJECTION_KEY, context)

    gate = jnp.matmul(activations, gate_weight, precision=precision)
    up = jnp.matmul(activations, up_weight, precision=precision)

    return jnp.matmul(sigmoid_linear_unit(gate) * up, down_weight, precision=precision)
