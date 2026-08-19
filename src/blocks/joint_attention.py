"""
Joint attention over text and image tokens.

This is the third and last attention implementation in the codebase,
and like the other two it is not interchangeable with them. The
autoencoder's is a single unmasked head over spatial positions; the
text encoder's is masked, causal and grouped-query. This one is
multi-head, fully bidirectional, and carries **no mask at all**.

The absence of a mask is deliberate and matches the reference. In the
text-to-image path every token, text and image alike, attends to every
other, including the text encoder's padding positions. Adding a mask
here to "fix" that would diverge from the trained model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import TransformerConfig
from ..layers import apply_axial_rotation, rms_normalization


# Softmax is exponentiated, so scores accumulate in at least float32,
# applied as a floor via jnp.promote_types rather than a hard cast.
MINIMUM_ATTENTION_ACCUMULATION_DTYPE = jnp.float32


def split_fused_qkv(
    fused: jnp.ndarray, config: TransformerConfig
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Split a fused query-key-value projection into three head-major
    tensors.

    The fused output is laid out as (component, head, feature) along its
    final axis, so queries occupy the first third, keys the second and
    values the third, with heads nested inside each. Reshaping in a
    different order, for instance treating heads as the outer grouping,
    would mix components across heads while keeping every shape valid.

    Returns three tensors of shape
    (batch, num_heads, num_tokens, head_dim).
    """
    batch, num_tokens, _ = fused.shape
    reshaped = fused.reshape(
        batch, num_tokens, 3, config.num_heads, config.head_dim
    )
    # (component, batch, heads, tokens, head_dim)
    transposed = jnp.transpose(reshaped, (2, 0, 3, 1, 4))
    return transposed[0], transposed[1], transposed[2]


def normalize_and_rotate(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    query_scale: jnp.ndarray,
    key_scale: jnp.ndarray,
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
    config: TransformerConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply per-head RMS normalization to queries and keys, then rotate
    them by position.

    Normalization precedes rotation, and the order is not
    interchangeable: rotating first would normalize an already rotated
    vector, changing what the trained scales act on. Values are left
    untouched, which is why they are not passed here.
    """
    queries = rms_normalization(queries, query_scale, config.rms_norm_epsilon)
    keys = rms_normalization(keys, key_scale, config.rms_norm_epsilon)

    queries = apply_axial_rotation(queries, cosine_table, sine_table)
    keys = apply_axial_rotation(keys, cosine_table, sine_table)

    return queries, keys


def joint_attention(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Unmasked multi-head scaled dot-product attention.

    Parameters
    ----------
    queries, keys, values:
        Shape (batch, num_heads, num_tokens, head_dim).
    config:
        Supplies head dimension and precision.

    Returns
    -------
    Shape (batch, num_tokens, num_heads * head_dim), with heads already
    merged back into a single feature axis ready for the output
    projection.
    """
    batch, num_heads, num_tokens, head_dim = queries.shape
    precision = config.precision.to_jax_precision()

    accumulation_dtype = jnp.promote_types(
        queries.dtype, MINIMUM_ATTENTION_ACCUMULATION_DTYPE
    )
    scale = jax.lax.rsqrt(jnp.asarray(head_dim, dtype=accumulation_dtype))

    scores = jnp.einsum(
        "bhqd,bhkd->bhqk",
        queries.astype(accumulation_dtype),
        keys.astype(accumulation_dtype),
        precision=precision,
    ) * scale
    weights = jax.nn.softmax(scores, axis=-1)

    attended = jnp.einsum(
        "bhqk,bhkd->bhqd", weights, values.astype(accumulation_dtype), precision=precision
    )

    merged = jnp.swapaxes(attended, 1, 2).reshape(batch, num_tokens, num_heads * head_dim)
    return merged.astype(queries.dtype)
