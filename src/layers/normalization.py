"""Normalization primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import VaeLayerConfig


# Minimum dtype for reductions inside normalization. Accumulating a sum
# of squares over tens of thousands of elements in bfloat16 loses
# precision quickly enough to be visible in decoded output, and the cost
# of promoting for the reduction alone is negligible against the
# surrounding convolutions.
#
# This is a floor, not a fixed target: it is applied through
# jnp.promote_types, so a bfloat16 input is promoted up to float32 while
# a float64 input is left alone. Casting unconditionally to float32
# would silently reduce precision for callers deliberately working in
# float64, which is exactly what the regression tests do when comparing
# against float64 oracles. An earlier version made that mistake.
MINIMUM_NORMALIZATION_ACCUMULATION_DTYPE = jnp.float32


def group_normalization(
    activations: jnp.ndarray,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Normalize activations over spatial dimensions within channel groups,
    then apply a learned per-channel affine transform.

    Channels are partitioned into config.num_groups contiguous groups.
    Each group's mean and variance are computed across height, width and
    that group's channels, independently per batch element. Variance
    uses the biased (population) estimator, matching the reference.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, channels). The channel count
        must be divisible by config.num_groups.
    scale:
        Per-channel multiplicative parameter, shape (channels,).
    shift:
        Per-channel additive parameter, shape (channels,).
    config:
        Supplies the group count and the epsilon added to variance
        before the reciprocal square root.
    """
    batch, height, width, channels = activations.shape
    if channels % config.num_groups != 0:
        raise ValueError(
            f"Channel count {channels} is not divisible by the configured "
            f"group count {config.num_groups}"
        )

    input_dtype = activations.dtype
    channels_per_group = channels // config.num_groups
    accumulation_dtype = jnp.promote_types(
        input_dtype, MINIMUM_NORMALIZATION_ACCUMULATION_DTYPE
    )

    grouped = activations.astype(accumulation_dtype).reshape(
        batch, height, width, config.num_groups, channels_per_group
    )

    # Reduce over height, width and within-group channels, keeping the
    # batch and group axes so the statistics broadcast back correctly.
    reduction_axes = (1, 2, 4)
    mean = jnp.mean(grouped, axis=reduction_axes, keepdims=True)
    variance = jnp.mean(jnp.square(grouped - mean), axis=reduction_axes, keepdims=True)

    normalized = (grouped - mean) * jax.lax.rsqrt(variance + config.normalization_epsilon)
    normalized = normalized.reshape(batch, height, width, channels).astype(input_dtype)

    return normalized * scale + shift
