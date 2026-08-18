"""
Composite blocks of the autoencoder decoder.

Two blocks are defined here, both assembled purely from the primitives
in layers.py and both pure functions of their inputs:

- residual_block: the decoder's basic unit, repeated throughout the
  middle section and every upsampling level.
- attention_block: a single self-attention layer applied once, in the
  decoder's middle section, at latent resolution.

Both take a parameter group already sliced out of the flat checkpoint
dictionary by parameters.select_parameter_group, so neither knows about
the checkpoint's flat naming convention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .config import VaeLayerConfig
from .layers import convolution_2d, group_normalization, sigmoid_linear_unit
from .parameters import has_parameter_group, require_parameter, select_parameter_group


# Parameter key names within a residual block's group. Named here rather
# than written inline so the checkpoint's vocabulary appears in exactly
# one place per block type.
RESIDUAL_FIRST_NORM_PREFIX = "norm1"
RESIDUAL_SECOND_NORM_PREFIX = "norm2"
RESIDUAL_FIRST_CONVOLUTION_PREFIX = "conv1"
RESIDUAL_SECOND_CONVOLUTION_PREFIX = "conv2"
RESIDUAL_SHORTCUT_PREFIX = "nin_shortcut"

# Parameter key names within an attention block's group.
ATTENTION_NORM_PREFIX = "norm"
ATTENTION_QUERY_PREFIX = "q"
ATTENTION_KEY_PREFIX = "k"
ATTENTION_VALUE_PREFIX = "v"
ATTENTION_OUTPUT_PREFIX = "proj_out"

WEIGHT_SUFFIX = "weight"
BIAS_SUFFIX = "bias"

# Attention scores are exponentiated, so they are accumulated in at
# least float32 for the same reason normalization statistics are: a
# softmax computed in bfloat16 over thousands of positions loses
# resolution in the tail of the distribution. Applied as a floor via
# jnp.promote_types, never as an unconditional cast, so float64 callers
# (the regression tests) keep their precision.
MINIMUM_ATTENTION_ACCUMULATION_DTYPE = jnp.float32


def _normalization_parameters(
    parameters: dict[str, np.ndarray], prefix: str, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the scale and shift of a normalization layer as a pair."""
    scale = require_parameter(parameters, f"{prefix}_{WEIGHT_SUFFIX}", context)
    shift = require_parameter(parameters, f"{prefix}_{BIAS_SUFFIX}", context)
    return scale, shift


def _convolution_parameters(
    parameters: dict[str, np.ndarray], prefix: str, context: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch the kernel and bias of a convolution as a pair."""
    kernel = require_parameter(parameters, f"{prefix}_{WEIGHT_SUFFIX}", context)
    bias = require_parameter(parameters, f"{prefix}_{BIAS_SUFFIX}", context)
    return kernel, bias


def residual_block(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Apply one residual block: two normalize-activate-convolve stages
    added to a shortcut path.

    The shortcut is the identity when the block preserves channel count,
    and a learned 1x1 projection when it does not. Which case applies is
    determined by whether the checkpoint contains shortcut weights for
    this block, rather than by comparing channel counts, because the
    checkpoint is the authority on what the trained model actually does.
    A block that changes channel count without shortcut weights present
    is a structural mismatch and will fail loudly at the addition rather
    than being silently papered over.

    Note the ordering: normalization comes before the activation and the
    convolution, not after, and the shortcut is added after the second
    convolution with no activation applied to the sum. This matches the
    reference implementation exactly.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, in_channels).
    parameters:
        Parameter group for this block, containing norm1, conv1, norm2,
        conv2, and optionally nin_shortcut entries.
    config:
        Normalization and precision settings.
    """
    context = "residual_block"

    first_norm_scale, first_norm_shift = _normalization_parameters(
        parameters, RESIDUAL_FIRST_NORM_PREFIX, context
    )
    hidden = group_normalization(activations, first_norm_scale, first_norm_shift, config)
    hidden = sigmoid_linear_unit(hidden)
    first_kernel, first_bias = _convolution_parameters(
        parameters, RESIDUAL_FIRST_CONVOLUTION_PREFIX, context
    )
    hidden = convolution_2d(hidden, first_kernel, first_bias, config)

    second_norm_scale, second_norm_shift = _normalization_parameters(
        parameters, RESIDUAL_SECOND_NORM_PREFIX, context
    )
    hidden = group_normalization(hidden, second_norm_scale, second_norm_shift, config)
    hidden = sigmoid_linear_unit(hidden)
    second_kernel, second_bias = _convolution_parameters(
        parameters, RESIDUAL_SECOND_CONVOLUTION_PREFIX, context
    )
    hidden = convolution_2d(hidden, second_kernel, second_bias, config)

    shortcut = activations
    if has_parameter_group(parameters, RESIDUAL_SHORTCUT_PREFIX):
        shortcut_kernel, shortcut_bias = _convolution_parameters(
            parameters, RESIDUAL_SHORTCUT_PREFIX, context
        )
        shortcut = convolution_2d(activations, shortcut_kernel, shortcut_bias, config)

    return shortcut + hidden


def _chunked_self_attention(
    queries: jnp.ndarray,
    keys: jnp.ndarray,
    values: jnp.ndarray,
    query_chunk_size: int,
) -> jnp.ndarray:
    """
    Single-head scaled dot-product self-attention, computing the output
    in chunks along the query axis to bound peak memory.

    Chunking is exact, not an approximation. Each query position's
    output depends only on that query and the full key and value
    sequences, never on other queries, so partitioning the query axis
    and concatenating the results reproduces the unchunked computation
    exactly. The test suite asserts this equivalence directly rather
    than taking it on trust.

    The query axis is zero-padded up to a whole number of chunks and the
    padding sliced away afterwards. Padded query rows produce their own
    (meaningless) outputs which are discarded; because attention is
    independent per query, they cannot influence the real rows.

    Parameters
    ----------
    queries, keys, values:
        Shape (batch, sequence_length, channels). Single head, so the
        head dimension is the full channel count.
    query_chunk_size:
        Number of query positions per chunk.
    """
    batch, sequence_length, channels = queries.shape

    accumulation_dtype = jnp.promote_types(
        queries.dtype, MINIMUM_ATTENTION_ACCUMULATION_DTYPE
    )
    scale = jax.lax.rsqrt(jnp.asarray(channels, dtype=accumulation_dtype))

    queries_accumulated = queries.astype(accumulation_dtype)
    keys_accumulated = keys.astype(accumulation_dtype)
    values_accumulated = values.astype(accumulation_dtype)

    padded_length = -(-sequence_length // query_chunk_size) * query_chunk_size
    padding_amount = padded_length - sequence_length
    if padding_amount:
        queries_accumulated = jnp.pad(
            queries_accumulated, ((0, 0), (0, padding_amount), (0, 0))
        )

    num_chunks = padded_length // query_chunk_size
    chunked_queries = queries_accumulated.reshape(
        batch, num_chunks, query_chunk_size, channels
    )
    # Move the chunk axis to the front so lax.map iterates over chunks,
    # keeping only one chunk's score matrix live at a time.
    chunked_queries = jnp.swapaxes(chunked_queries, 0, 1)

    def attend_one_chunk(query_chunk: jnp.ndarray) -> jnp.ndarray:
        scores = jnp.einsum("bqc,bkc->bqk", query_chunk, keys_accumulated) * scale
        weights = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("bqk,bkc->bqc", weights, values_accumulated)

    chunk_outputs = jax.lax.map(attend_one_chunk, chunked_queries)

    outputs = jnp.swapaxes(chunk_outputs, 0, 1).reshape(batch, padded_length, channels)
    return outputs[:, :sequence_length, :].astype(queries.dtype)


def attention_block(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    config: VaeLayerConfig,
) -> jnp.ndarray:
    """
    Apply the decoder's self-attention block: normalize, project to
    query, key and value, attend across every spatial position, project
    back, and add to the input.

    This is a single attention head whose head dimension is the full
    channel count, not a multi-head layer. Splitting the channels into
    heads would be a different operation producing different numbers,
    so it must not be "optimized" into multi-head form.

    Every spatial position attends to every other, so the sequence
    length is height times width. At the resolutions this decoder runs
    at that is large enough for the score matrix to dominate memory
    usage, which is why the attention itself is chunked.

    Parameters
    ----------
    activations:
        Input, shape (batch, height, width, channels).
    parameters:
        Parameter group containing norm, q, k, v and proj_out entries.
    config:
        Normalization, precision and chunk size settings.
    """
    context = "attention_block"
    batch, height, width, channels = activations.shape

    norm_scale, norm_shift = _normalization_parameters(
        parameters, ATTENTION_NORM_PREFIX, context
    )
    normalized = group_normalization(activations, norm_scale, norm_shift, config)

    projections = []
    for prefix in (ATTENTION_QUERY_PREFIX, ATTENTION_KEY_PREFIX, ATTENTION_VALUE_PREFIX):
        kernel, bias = _convolution_parameters(parameters, prefix, context)
        projected = convolution_2d(normalized, kernel, bias, config)
        projections.append(projected.reshape(batch, height * width, channels))

    queries, keys, values = projections

    attended = _chunked_self_attention(
        queries, keys, values, config.attention_query_chunk_size
    )
    attended = attended.reshape(batch, height, width, channels)

    output_kernel, output_bias = _convolution_parameters(
        parameters, ATTENTION_OUTPUT_PREFIX, context
    )
    projected_output = convolution_2d(attended, output_kernel, output_bias, config)

    return activations + projected_output
