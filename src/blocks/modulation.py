"""
Adaptive modulation for the diffusion transformer.

Modulation converts the timestep embedding into per-block shift, scale
and gate vectors. A structural detail worth knowing before reading
further: this model carries only three modulation modules in total, not
one per block. Every double block shares the same pair of image and
text modulation outputs, and every single block shares one more. This
differs from FLUX.1, which modulates per block, and it is what makes
the entire modulation computation hoistable out of the sampling loop:
it depends on the timestep alone, never on the evolving latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..checkpoint import require_parameter
from ..config import TransformerConfig
from ..layers import sigmoid_linear_unit


MODULATION_WEIGHT_SUFFIX = "lin_weight"

# A modulation output is consumed as three vectors: a shift and scale
# applied around a normalization, and a gate applied to the sub-layer's
# contribution before it rejoins the residual stream.
COMPONENTS_PER_MODULATION = 3

# Double-stream blocks need two such triples, one for the attention
# sub-layer and one for the feed-forward sub-layer. Single-stream blocks
# fuse both sub-layers and therefore need only one.
TRIPLES_PER_DOUBLE_BLOCK = 2
TRIPLES_PER_SINGLE_BLOCK = 1


@dataclass(frozen=True)
class ModulationTriple:
    """
    One sub-layer's modulation.

    `shift` and `scale` are applied around the sub-layer's input
    normalization; `gate` scales its output before the residual
    addition, which lets a block contribute nothing at all when the
    gate is zero.
    """

    shift: jnp.ndarray
    scale: jnp.ndarray
    gate: jnp.ndarray


def _split_into_triples(
    modulation_output: jnp.ndarray, num_triples: int, hidden_size: int
) -> list[ModulationTriple]:
    """
    Split a modulation projection's output into consecutive triples.

    The ordering within the projection is shift, scale, gate for the
    first sub-layer, then shift, scale, gate for the second. The trained
    weights assume that arrangement, so reordering here would apply a
    gate where a shift belongs while keeping every shape valid.
    """
    expected_width = num_triples * COMPONENTS_PER_MODULATION * hidden_size
    if modulation_output.shape[-1] != expected_width:
        raise ValueError(
            f"Modulation output has width {modulation_output.shape[-1]}, expected "
            f"{expected_width} for {num_triples} triples at hidden size {hidden_size}"
        )

    chunks = jnp.split(modulation_output, num_triples * COMPONENTS_PER_MODULATION, axis=-1)
    return [
        ModulationTriple(
            shift=chunks[index * COMPONENTS_PER_MODULATION],
            scale=chunks[index * COMPONENTS_PER_MODULATION + 1],
            gate=chunks[index * COMPONENTS_PER_MODULATION + 2],
        )
        for index in range(num_triples)
    ]


def compute_modulation(
    conditioning_vector: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    parameter_prefix: str,
    num_triples: int,
    config: TransformerConfig,
) -> list[ModulationTriple]:
    """
    Project the conditioning vector into modulation triples.

    The activation precedes the projection rather than following it,
    matching the reference. Applying it afterwards would change what the
    trained projection receives.

    Parameters
    ----------
    conditioning_vector:
        Shape (batch, hidden_size), derived from the timestep.
    parameters:
        The transformer's global parameter group.
    parameter_prefix:
        Which modulation projection to use, for instance
        "double_stream_modulation_img".
    num_triples:
        How many triples the projection produces.
    config:
        Supplies hidden size and precision.

    Returns
    -------
    A list of `num_triples` triples, each carrying vectors of shape
    (batch, 1, hidden_size). The singleton axis broadcasts across
    tokens, since modulation is uniform over the sequence.
    """
    weight = require_parameter(
        parameters, f"{parameter_prefix}_{MODULATION_WEIGHT_SUFFIX}", "compute_modulation"
    )

    activated = sigmoid_linear_unit(conditioning_vector)
    projected = jnp.matmul(activated, weight, precision=config.precision.to_jax_precision())

    # Insert a token axis so each vector broadcasts across the sequence.
    projected = projected[:, None, :]

    return _split_into_triples(projected, num_triples, config.hidden_size)


def apply_modulated_normalization(
    normalized: jnp.ndarray,
    triple: ModulationTriple,
) -> jnp.ndarray:
    """
    Apply a modulation triple's shift and scale to already normalized
    activations.

    The scale is applied as one plus the modulation value rather than
    the value alone, so a zero modulation leaves the normalized
    activations unchanged. That offset is what lets the model express an
    identity transform; dropping it would make a zero modulation
    annihilate the signal instead.
    """
    return normalized * (1.0 + triple.scale) + triple.shift
