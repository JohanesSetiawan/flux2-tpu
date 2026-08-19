"""
Multi-axis rotary position embedding for the diffusion transformer.

Kept separate from `positional.py` deliberately. The two modules
implement rotary embedding under **different and incompatible pairing
conventions**, and merging them would invite applying one model's
convention to the other's weights:

  positional.py       text encoder, half-split pairing: feature i
                      rotates against feature i + head_dim/2
  axial_positional.py diffusion transformer, interleaved pairing:
                      feature 2i rotates against feature 2i+1

Both are self-consistent, both produce correctly shaped output, and
each matches only its own checkpoint. There is no single correct
convention to unify on.

This module also differs in carrying several independent position axes.
Each token has a position along every axis, and the head dimension is
partitioned between them, so one part of a head encodes vertical
position, another horizontal, and so on.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..config import TransformerConfig


# Rotation tables are built in float32 for the same reason the text
# encoder's are: the angles must stay distinguishable between adjacent
# positions.
ROTATION_TABLE_DTYPE = jnp.float32

# Positions are small non-negative integers, so 32 bits is ample. The
# dtype is named rather than written inline so every array that holds
# or is scattered into position identifiers agrees; a mismatch produces
# an unsafe-cast warning from JAX rather than a hard failure, which is
# easy to overlook.
POSITION_IDENTIFIER_DTYPE = jnp.int32


def _single_axis_rotation(
    positions: jnp.ndarray, axis_dim: int, theta: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build cosine and sine factors for one position axis.

    Returns a pair each of shape (..., axis_dim / 2), one entry per
    rotation pair rather than per feature, because both members of a
    pair rotate by the same angle.
    """
    if axis_dim % 2 != 0:
        raise ValueError(f"Axis dimension {axis_dim} must be even to form rotation pairs")

    pair_offsets = jnp.arange(0, axis_dim, 2, dtype=ROTATION_TABLE_DTYPE) / axis_dim
    inverse_frequencies = 1.0 / (theta ** pair_offsets)

    angles = positions.astype(ROTATION_TABLE_DTYPE)[..., None] * inverse_frequencies
    return jnp.cos(angles), jnp.sin(angles)


def axial_rotation_table(
    position_identifiers: jnp.ndarray,
    config: TransformerConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the rotation tables for every token from its position along
    each axis.

    The head dimension is partitioned between the axes in the order
    given by config.positional_axes_dimensions, and each axis is
    encoded independently over its own slice. Concatenating those
    slices gives every token a single rotation covering all its
    positions at once.

    Parameters
    ----------
    position_identifiers:
        Shape (batch, num_tokens, num_axes), integer positions along
        each axis.
    config:
        Supplies the axis widths and the rotation base.

    Returns
    -------
    A (cosine, sine) pair, each of shape
    (batch, 1, num_tokens, head_dim / 2). The singleton axis broadcasts
    across attention heads, since every head rotates identically.
    """
    if position_identifiers.shape[-1] != config.num_positional_axes:
        raise ValueError(
            f"Position identifiers carry {position_identifiers.shape[-1]} axes but the "
            f"configuration declares {config.num_positional_axes}"
        )

    cosine_parts = []
    sine_parts = []
    for axis_index, axis_dim in enumerate(config.positional_axes_dimensions):
        cosine, sine = _single_axis_rotation(
            position_identifiers[..., axis_index], axis_dim, config.rope_theta
        )
        cosine_parts.append(cosine)
        sine_parts.append(sine)

    cosine_table = jnp.concatenate(cosine_parts, axis=-1)[:, None, :, :]
    sine_table = jnp.concatenate(sine_parts, axis=-1)[:, None, :, :]
    return cosine_table, sine_table


def apply_axial_rotation(
    vectors: jnp.ndarray,
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rotate query or key vectors under the interleaved pairing
    convention.

    Adjacent features form each pair: feature 2i with feature 2i+1. The
    pair is rotated as a two dimensional vector, which is why the tables
    carry one entry per pair rather than per feature.

    Parameters
    ----------
    vectors:
        Shape (batch, num_heads, num_tokens, head_dim).
    cosine_table, sine_table:
        Shape (batch, 1, num_tokens, head_dim / 2), as returned by
        axial_rotation_table.

    Returns
    -------
    Rotated vectors, same shape and dtype as the input.
    """
    input_dtype = vectors.dtype
    batch, num_heads, num_tokens, head_dim = vectors.shape

    # Split the feature axis into (pair, member) so the two members of
    # each pair sit on their own axis and can be rotated together.
    paired = vectors.astype(cosine_table.dtype).reshape(
        batch, num_heads, num_tokens, head_dim // 2, 2
    )
    first_member = paired[..., 0]
    second_member = paired[..., 1]

    rotated_first = first_member * cosine_table - second_member * sine_table
    rotated_second = first_member * sine_table + second_member * cosine_table

    rotated = jnp.stack([rotated_first, rotated_second], axis=-1)
    return rotated.reshape(batch, num_heads, num_tokens, head_dim).astype(input_dtype)


def build_position_identifiers(
    text_length: int,
    image_height: int,
    image_width: int,
    config: TransformerConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the per-token position identifiers for text-to-image
    generation.

    Text and image tokens occupy disjoint position axes. Text tokens
    carry their sequence index on the final axis and zero elsewhere;
    image tokens carry their row and column on the middle two axes and
    zero elsewhere. The first axis is unused in this mode and stays
    zero throughout: it exists for other generation modes this
    implementation does not cover.

    Because the two groups use different axes, a text token and an image
    token never share a position encoding despite both being present in
    the same attention sequence.

    Image positions are laid out in row-major order, which is what makes
    the eventual unpacking of the latent a plain reshape rather than a
    scatter.

    Returns
    -------
    A (text_identifiers, image_identifiers) pair with shapes
    (1, text_length, num_axes) and
    (1, image_height * image_width, num_axes).
    """
    num_axes = config.num_positional_axes

    text_identifiers = jnp.zeros((1, text_length, num_axes), dtype=POSITION_IDENTIFIER_DTYPE)
    text_identifiers = text_identifiers.at[0, :, num_axes - 1].set(
        jnp.arange(text_length, dtype=POSITION_IDENTIFIER_DTYPE)
    )

    rows = jnp.repeat(jnp.arange(image_height, dtype=POSITION_IDENTIFIER_DTYPE), image_width)
    columns = jnp.tile(jnp.arange(image_width, dtype=POSITION_IDENTIFIER_DTYPE), image_height)

    image_identifiers = jnp.zeros(
        (1, image_height * image_width, num_axes), dtype=POSITION_IDENTIFIER_DTYPE
    )
    image_identifiers = image_identifiers.at[0, :, 1].set(rows)
    image_identifiers = image_identifiers.at[0, :, 2].set(columns)

    return text_identifiers, image_identifiers
