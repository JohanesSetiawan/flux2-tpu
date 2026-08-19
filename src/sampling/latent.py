"""
Converting between the latent's spatial form and the token sequence the
transformer consumes.
"""

from __future__ import annotations

import jax.numpy as jnp


def pack_latent_to_tokens(latent: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten a spatial latent into a token sequence in row-major order.

    Parameters
    ----------
    latent:
        Shape (batch, height, width, channels), NHWC as used throughout
        this package.

    Returns
    -------
    Shape (batch, height * width, channels).
    """
    batch, height, width, channels = latent.shape
    return latent.reshape(batch, height * width, channels)


def unpack_tokens_to_latent(
    tokens: jnp.ndarray, height: int, width: int
) -> jnp.ndarray:
    """
    Restore a token sequence to its spatial form.

    This is a plain reshape rather than a scatter, and that is worth
    justifying because the reference implements it as a scatter.

    The reference builds position identifiers by taking a cartesian
    product over the axes, then scatters tokens into place using those
    identifiers. For text-to-image the identifiers come out in exactly
    row-major order, so the permutation the scatter performs is the
    identity, and a reshape reproduces it exactly. This was verified
    numerically against the reference's construction rather than
    assumed.

    The generality the scatter provides is needed for the reference's
    other generation modes, which interleave tokens from several sources
    and therefore genuinely permute. This implementation covers only
    text-to-image, so it does not need it.

    Parameters
    ----------
    tokens:
        Shape (batch, height * width, channels).
    height, width:
        Target spatial dimensions. Passed explicitly rather than
        inferred, since many pairs share a product and inferring would
        silently accept a transposed result.
    """
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(
            f"Token count {num_tokens} does not match the requested spatial size "
            f"{height} by {width}, which needs {height * width}"
        )
    return tokens.reshape(batch, height, width, channels)
