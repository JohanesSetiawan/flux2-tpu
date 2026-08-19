"""
Double-stream block: image and text carry separate weights but attend
jointly.

Each stream has its own projections, its own feed-forward, and its own
modulation, so the two are processed independently everywhere except
inside attention, where their queries, keys and values are concatenated
and every token attends to every other. That is what lets text
condition the image without the two ever sharing parameters.

Token order inside attention is text first, then image. The order is
not cosmetic: it determines which slice of the result belongs to which
stream, and it must match the order the position identifiers were
concatenated in.
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


IMAGE_STREAM_PREFIX = "img"
TEXT_STREAM_PREFIX = "txt"


def _stream_keys(stream_prefix: str) -> dict[str, str]:
    """Parameter key names for one stream of a double block."""
    return {
        "qkv": f"{stream_prefix}_attn_qkv_weight",
        "projection": f"{stream_prefix}_attn_proj_weight",
        "query_norm": f"{stream_prefix}_attn_norm_query_norm_scale",
        "key_norm": f"{stream_prefix}_attn_norm_key_norm_scale",
        "mlp_in": f"{stream_prefix}_mlp_0_weight",
        "mlp_out": f"{stream_prefix}_mlp_2_weight",
    }


def _prepare_stream(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    stream_prefix: str,
    attention_modulation: ModulationTriple,
    config: TransformerConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Normalize, modulate and project one stream into queries, keys and
    values, without yet applying rotation.

    Rotation is deferred to the caller because the two streams use
    different position identifiers but must be rotated with a single
    concatenated table, in the same order they are concatenated for
    attention.
    """
    keys = _stream_keys(stream_prefix)
    context = f"double_stream_block[{stream_prefix}]"

    normalized = layer_normalization(activations, config.layer_norm_epsilon)
    modulated = apply_modulated_normalization(normalized, attention_modulation)

    fused = jnp.matmul(
        modulated,
        require_parameter(parameters, keys["qkv"], context),
        precision=config.precision.to_jax_precision(),
    )
    return split_fused_qkv(fused, config)


def _apply_stream_residuals(
    activations: jnp.ndarray,
    attended: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    stream_prefix: str,
    attention_modulation: ModulationTriple,
    feedforward_modulation: ModulationTriple,
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Project the attention result back, add it through its gate, then
    run the gated feed-forward and add that through its own gate.

    Note that the feed-forward reads the activations *after* the
    attention residual has been added, not the block's original input.
    Reading the original input instead would turn the two sub-layers
    into parallel branches rather than a sequence.
    """
    keys = _stream_keys(stream_prefix)
    context = f"double_stream_block[{stream_prefix}]"
    precision = config.precision.to_jax_precision()

    projected = jnp.matmul(
        attended, require_parameter(parameters, keys["projection"], context), precision=precision
    )
    activations = activations + attention_modulation.gate * projected

    normalized = layer_normalization(activations, config.layer_norm_epsilon)
    modulated = apply_modulated_normalization(normalized, feedforward_modulation)

    expanded = jnp.matmul(
        modulated, require_parameter(parameters, keys["mlp_in"], context), precision=precision
    )
    gated = split_and_gate(expanded)
    contracted = jnp.matmul(
        gated, require_parameter(parameters, keys["mlp_out"], context), precision=precision
    )

    return activations + feedforward_modulation.gate * contracted


def double_stream_block(
    image_activations: jnp.ndarray,
    text_activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
    image_modulation: tuple[ModulationTriple, ModulationTriple],
    text_modulation: tuple[ModulationTriple, ModulationTriple],
    config: TransformerConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run one double-stream block.

    Parameters
    ----------
    image_activations, text_activations:
        Shape (batch, num_tokens, hidden_size) each, with independent
        token counts.
    parameters:
        This block's parameter group.
    cosine_table, sine_table:
        Rotation tables covering text tokens followed by image tokens,
        in that order, of shape
        (batch, 1, text_tokens + image_tokens, head_dim / 2).
    image_modulation, text_modulation:
        Each a pair of triples: the first for the attention sub-layer,
        the second for the feed-forward.
    config:
        Architecture and precision settings.

    Returns
    -------
    Updated (image_activations, text_activations).
    """
    image_attention_modulation, image_feedforward_modulation = image_modulation
    text_attention_modulation, text_feedforward_modulation = text_modulation

    image_queries, image_keys, image_values = _prepare_stream(
        image_activations, parameters, IMAGE_STREAM_PREFIX, image_attention_modulation, config
    )
    text_queries, text_keys, text_values = _prepare_stream(
        text_activations, parameters, TEXT_STREAM_PREFIX, text_attention_modulation, config
    )

    # Each stream normalizes with its own learned scales, so this happens
    # per stream before concatenation rather than once afterwards. Each
    # also takes its own slice of the rotation table, which covers text
    # tokens first and image tokens second.
    num_text_tokens = text_activations.shape[1]
    text_context = "double_stream_block[txt]"
    image_context = "double_stream_block[img]"
    text_keys_names = _stream_keys(TEXT_STREAM_PREFIX)
    image_keys_names = _stream_keys(IMAGE_STREAM_PREFIX)

    text_queries, text_keys = normalize_and_rotate(
        text_queries,
        text_keys,
        require_parameter(parameters, text_keys_names["query_norm"], text_context),
        require_parameter(parameters, text_keys_names["key_norm"], text_context),
        cosine_table[:, :, :num_text_tokens, :],
        sine_table[:, :, :num_text_tokens, :],
        config,
    )
    image_queries, image_keys = normalize_and_rotate(
        image_queries,
        image_keys,
        require_parameter(parameters, image_keys_names["query_norm"], image_context),
        require_parameter(parameters, image_keys_names["key_norm"], image_context),
        cosine_table[:, :, num_text_tokens:, :],
        sine_table[:, :, num_text_tokens:, :],
        config,
    )

    # Text first, then image, matching the order the rotation tables and
    # position identifiers were built in.
    queries = jnp.concatenate([text_queries, image_queries], axis=2)
    keys = jnp.concatenate([text_keys, image_keys], axis=2)
    values = jnp.concatenate([text_values, image_values], axis=2)

    attended = joint_attention(queries, keys, values, config)

    text_attended = attended[:, :num_text_tokens]
    image_attended = attended[:, num_text_tokens:]

    image_activations = _apply_stream_residuals(
        image_activations,
        image_attended,
        parameters,
        IMAGE_STREAM_PREFIX,
        image_attention_modulation,
        image_feedforward_modulation,
        config,
    )
    text_activations = _apply_stream_residuals(
        text_activations,
        text_attended,
        parameters,
        TEXT_STREAM_PREFIX,
        text_attention_modulation,
        text_feedforward_modulation,
        config,
    )

    return image_activations, text_activations
