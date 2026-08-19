"""
Grouped-query self-attention as used by the Qwen3 text encoder.

Distinct from the VAE's attention block in `attention.py`: that one is
a single head over spatial positions with no mask, this one is
multi-head with fewer key/value heads than query heads, applies
per-head normalization to queries and keys, rotates them by position,
and is masked.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..checkpoint import require_parameter
from ..config import TextEncoderConfig
from ..layers import apply_rotary_embedding, rms_normalization


QUERY_PROJECTION_KEY = "self_attn_q_proj_weight"
KEY_PROJECTION_KEY = "self_attn_k_proj_weight"
VALUE_PROJECTION_KEY = "self_attn_v_proj_weight"
OUTPUT_PROJECTION_KEY = "self_attn_o_proj_weight"
QUERY_NORM_KEY = "self_attn_q_norm_weight"
KEY_NORM_KEY = "self_attn_k_norm_weight"

# Softmax is exponentiated, so scores accumulate in at least float32 for
# the same reason normalization statistics do. Applied as a floor via
# jnp.promote_types so float64 callers keep their precision.
MINIMUM_ATTENTION_ACCUMULATION_DTYPE = jnp.float32


def _project_to_heads(
    activations: jnp.ndarray,
    weight: np.ndarray,
    num_heads: int,
    head_dim: int,
    precision,
) -> jnp.ndarray:
    """
    Project and split into heads, returning
    (batch, num_heads, sequence_length, head_dim).

    Note that num_heads times head_dim need not equal the hidden size:
    for this model the query projection widens from 2560 to 4096. The
    head count and head dimension are therefore taken from the
    configuration rather than derived from the activation width.
    """
    batch, sequence_length, _ = activations.shape
    projected = jnp.matmul(activations, weight, precision=precision)
    reshaped = projected.reshape(batch, sequence_length, num_heads, head_dim)
    return jnp.swapaxes(reshaped, 1, 2)


def _repeat_key_value_heads(
    key_or_value: jnp.ndarray, repeats: int
) -> jnp.ndarray:
    """
    Expand key or value heads so each is shared by several query heads.

    Grouped-query attention stores fewer key/value heads than query
    heads; each key/value head serves a contiguous group of query heads.
    The repeat must therefore place copies adjacent to one another, so
    that query heads 0 through 3 all see key/value head 0. Interleaving
    instead would pair each query head with the wrong key/value head
    while producing identically shaped output.
    """
    return jnp.repeat(key_or_value, repeats=repeats, axis=1)


def grouped_query_attention(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    rotary_cosine: jnp.ndarray,
    rotary_sine: jnp.ndarray,
    attention_bias: jnp.ndarray,
    config: TextEncoderConfig,
) -> jnp.ndarray:
    """
    Apply one grouped-query self-attention layer.

    The order of operations follows the reference exactly and is not
    interchangeable: project, split into heads, normalize each head's
    queries and keys, rotate by position, expand key/value heads, then
    attend. Normalizing after rotation, or rotating before splitting
    into heads, computes something different from the same weights.

    Parameters
    ----------
    activations:
        Shape (batch, sequence_length, hidden_size).
    parameters:
        One layer's parameter group.
    rotary_cosine, rotary_sine:
        Rotary tables of shape (sequence_length, head_dim). Built once
        by the caller and reused across every layer, since they depend
        only on position and head geometry.
    attention_bias:
        Additive mask of shape (batch, 1, sequence_length,
        sequence_length), as built by `causal_padding_mask`.
    config:
        Supplies head geometry, normalization epsilon and precision.
    """
    context = "grouped_query_attention"
    precision = config.precision.to_jax_precision()
    batch, sequence_length, _ = activations.shape

    queries = _project_to_heads(
        activations,
        require_parameter(parameters, QUERY_PROJECTION_KEY, context),
        config.num_attention_heads,
        config.head_dim,
        precision,
    )
    keys = _project_to_heads(
        activations,
        require_parameter(parameters, KEY_PROJECTION_KEY, context),
        config.num_key_value_heads,
        config.head_dim,
        precision,
    )
    values = _project_to_heads(
        activations,
        require_parameter(parameters, VALUE_PROJECTION_KEY, context),
        config.num_key_value_heads,
        config.head_dim,
        precision,
    )

    # Normalization acts on each head's own feature vector, which is why
    # it comes after the split into heads rather than before.
    queries = rms_normalization(
        queries,
        require_parameter(parameters, QUERY_NORM_KEY, context),
        config.rms_norm_epsilon,
    )
    keys = rms_normalization(
        keys,
        require_parameter(parameters, KEY_NORM_KEY, context),
        config.rms_norm_epsilon,
    )

    queries = apply_rotary_embedding(queries, rotary_cosine, rotary_sine)
    keys = apply_rotary_embedding(keys, rotary_cosine, rotary_sine)

    repeats = config.query_heads_per_key_value_head
    keys = _repeat_key_value_heads(keys, repeats)
    values = _repeat_key_value_heads(values, repeats)

    accumulation_dtype = jnp.promote_types(
        activations.dtype, MINIMUM_ATTENTION_ACCUMULATION_DTYPE
    )
    scale = jax.lax.rsqrt(jnp.asarray(config.head_dim, dtype=accumulation_dtype))

    scores = jnp.einsum(
        "bhqd,bhkd->bhqk",
        queries.astype(accumulation_dtype),
        keys.astype(accumulation_dtype),
        precision=precision,
    )
    scores = scores * scale + attention_bias.astype(accumulation_dtype)
    weights = jax.nn.softmax(scores, axis=-1)

    attended = jnp.einsum(
        "bhqk,bhkd->bhqd", weights, values.astype(accumulation_dtype), precision=precision
    )

    # Merge heads back into a single feature axis before the output
    # projection, reversing the split above.
    merged = jnp.swapaxes(attended, 1, 2).reshape(
        batch, sequence_length, config.num_attention_heads * config.head_dim
    )

    return jnp.matmul(
        merged.astype(activations.dtype),
        require_parameter(parameters, OUTPUT_PROJECTION_KEY, context),
        precision=precision,
    )
