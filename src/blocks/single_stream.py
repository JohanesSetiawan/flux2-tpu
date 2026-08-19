"""
Single-stream block: attention and feed-forward computed in parallel
from one fused projection.

Text and image tokens have been concatenated into one sequence by this
point and share every weight, which is why there is one stream rather
than two. The block is also structurally different from the
double-stream one: rather than running attention and then feeding its
output into a feed-forward, both are computed from the same normalized
input and their results concatenated before a single output
projection. There is one residual addition, not two.

That fusion is already present in the checkpoint's own weights: the
first projection produces queries, keys, values and the feed-forward
expansion in a single matrix, and the second consumes the attention
output and the gated feed-forward together. No further fusion is
available here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..checkpoint import require_parameter
from ..config import TransformerConfig
from ..layers import layer_normalization
from .gated_mlp import split_and_gate
from .joint_attention import joint_attention, normalize_and_rotate, split_fused_qkv
from .modulation import ModulationTriple, apply_modulated_normalization


FUSED_INPUT_PROJECTION_KEY = "linear1_weight"
FUSED_OUTPUT_PROJECTION_KEY = "linear2_weight"
QUERY_NORM_KEY = "norm_query_norm_scale"
KEY_NORM_KEY = "norm_key_norm_scale"


def single_stream_block(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
    modulation: ModulationTriple,
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Run one single-stream block.

    Parameters
    ----------
    activations:
        Shape (batch, num_tokens, hidden_size), text and image tokens
        already concatenated.
    parameters:
        This block's parameter group.
    cosine_table, sine_table:
        Rotation tables covering the full concatenated sequence.
    modulation:
        One triple, shared by both sub-layers since they are fused.
    config:
        Architecture and precision settings.
    """
    context = "single_stream_block"
    precision = config.precision.to_jax_precision()

    normalized = layer_normalization(activations, config.layer_norm_epsilon)
    modulated = apply_modulated_normalization(normalized, modulation)

    fused = jnp.matmul(
        modulated,
        require_parameter(parameters, FUSED_INPUT_PROJECTION_KEY, context),
        precision=precision,
    )

    # The projection's output holds attention inputs followed by the
    # feed-forward expansion. The split point is the width of three
    # head-major projections; everything after it belongs to the
    # feed-forward, which is twice the mlp hidden size because it will
    # be halved again by gating.
    attention_width = 3 * config.hidden_size
    attention_part = fused[..., :attention_width]
    feedforward_part = fused[..., attention_width:]

    queries, keys, values = split_fused_qkv(attention_part, config)
    queries, keys = normalize_and_rotate(
        queries,
        keys,
        require_parameter(parameters, QUERY_NORM_KEY, context),
        require_parameter(parameters, KEY_NORM_KEY, context),
        cosine_table,
        sine_table,
        config,
    )

    attended = joint_attention(queries, keys, values, config)
    gated = split_and_gate(feedforward_part)

    combined = jnp.concatenate([attended, gated], axis=-1)
    projected = jnp.matmul(
        combined,
        require_parameter(parameters, FUSED_OUTPUT_PROJECTION_KEY, context),
        precision=precision,
    )

    return activations + modulation.gate * projected
