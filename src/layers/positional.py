"""Rotary position embedding."""

from __future__ import annotations

import jax.numpy as jnp


# Rotary tables are built in float32 regardless of the activation dtype.
# The angles grow linearly with position, so at the sequence lengths
# involved a bfloat16 table quantises adjacent positions to the same
# angle, which is a real loss of positional resolution rather than a
# rounding nicety.
ROTARY_TABLE_DTYPE = jnp.float32


def rotary_frequency_table(
    sequence_length: int,
    head_dim: int,
    theta: float,
    dtype: jnp.dtype = ROTARY_TABLE_DTYPE,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build the cosine and sine tables for rotary position embedding.

    Each pair of feature dimensions is rotated by an angle proportional
    to its position, with the rate of rotation decreasing geometrically
    across the head dimension. Low-index pairs rotate quickly and encode
    fine positional detail; high-index pairs rotate slowly and encode
    coarse position.

    The returned tables are full width rather than half width: each
    angle appears twice, once for each element of the pair it rotates.
    This matches the convention used by the reference implementation and
    lets `apply_rotary_embedding` be a plain elementwise expression.

    Because the tables depend only on sequence length and head geometry,
    never on the data, a caller should build them once and reuse them
    across every layer and every sampling step rather than recomputing
    them.

    The dtype defaults to float32, which is what the reference uses and
    what inference runs at. It is a parameter rather than a constant so
    that tests comparing against a float64 oracle can build the table at
    matching precision; without that, a float32 table would put a floor
    of roughly 1e-8 under any parity measurement and mask smaller real
    differences.

    Returns
    -------
    A (cosine, sine) pair, each of shape (sequence_length, head_dim).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension {head_dim} must be even to form rotation pairs")

    pair_indices = jnp.arange(0, head_dim, 2, dtype=dtype)
    inverse_frequencies = 1.0 / (theta ** (pair_indices / head_dim))

    positions = jnp.arange(sequence_length, dtype=dtype)
    angles = jnp.outer(positions, inverse_frequencies)

    # Duplicate rather than interleave. See apply_rotary_embedding for
    # why this specific arrangement is required.
    duplicated_angles = jnp.concatenate([angles, angles], axis=-1)
    return jnp.cos(duplicated_angles), jnp.sin(duplicated_angles)


def _rotate_half(vectors: jnp.ndarray) -> jnp.ndarray:
    """
    Pair each element in the first half of the final axis with the
    element at the same offset in the second half, and rotate that pair
    by ninety degrees.

    This is the half-split (sometimes called NeoX) pairing: element i is
    paired with element i + head_dim/2. It is not the interleaved
    pairing where element 2i is paired with element 2i+1. The two
    conventions produce different results from the same weights, and
    choosing the wrong one yields output that is plausible but subtly
    incorrect rather than obviously broken, so it must match the
    checkpoint's own convention. Qwen3 uses the half-split form.
    """
    first_half = vectors[..., : vectors.shape[-1] // 2]
    second_half = vectors[..., vectors.shape[-1] // 2 :]
    return jnp.concatenate([-second_half, first_half], axis=-1)


def apply_rotary_embedding(
    vectors: jnp.ndarray,
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rotate query or key vectors according to their positions.

    Parameters
    ----------
    vectors:
        Shape (batch, num_heads, sequence_length, head_dim).
    cosine_table, sine_table:
        Shape (sequence_length, head_dim), as returned by
        rotary_frequency_table. They are broadcast across batch and
        heads, since every head at a given position rotates by the same
        angle.

    Returns
    -------
    Rotated vectors, same shape and dtype as the input.
    """
    input_dtype = vectors.dtype
    promoted = vectors.astype(cosine_table.dtype)

    rotated = promoted * cosine_table + _rotate_half(promoted) * sine_table
    return rotated.astype(input_dtype)
