"""Sinusoidal timestep embedding."""

from __future__ import annotations

import jax.numpy as jnp

from ..config import TransformerConfig


# Built in float32 regardless of the surrounding dtype: the frequencies
# span several orders of magnitude, and a lower-precision table would
# collapse the slowest of them onto identical values.
TIMESTEP_EMBEDDING_DTYPE = jnp.float32


def timestep_embedding(
    timesteps: jnp.ndarray,
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Encode continuous timesteps as a sinusoidal feature vector.

    Timesteps arrive in the sampler's own range, roughly zero to one,
    and are multiplied by a scale factor before encoding so that they
    span a range the frequency spectrum can resolve. Omitting that
    factor would compress every timestep into the flattest part of the
    lowest frequency.

    The cosine block precedes the sine block in the output. That
    ordering is arbitrary in principle but not in practice: the
    projection that consumes this embedding was trained against one
    specific arrangement, so reversing it would feed the trained weights
    a permuted input.

    Parameters
    ----------
    timesteps:
        Shape (batch,), continuous values.
    config:
        Supplies the embedding width, maximum period and scale factor.

    Returns
    -------
    Shape (batch, timestep_embedding_dim).
    """
    half_dim = config.timestep_embedding_dim // 2
    if config.timestep_embedding_dim != half_dim * 2:
        raise ValueError(
            f"Timestep embedding dimension {config.timestep_embedding_dim} must be even"
        )

    scaled = timesteps.astype(TIMESTEP_EMBEDDING_DTYPE) * config.timestep_scale_factor

    exponents = jnp.arange(half_dim, dtype=TIMESTEP_EMBEDDING_DTYPE) / half_dim
    frequencies = jnp.exp(-jnp.log(config.timestep_max_period) * exponents)

    angles = scaled[:, None] * frequencies[None, :]
    return jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
