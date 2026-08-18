"""
The decoder's self-attention block, with memory-bounded attention.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..config import VaeLayerConfig
from ..layers import convolution_2d, group_normalization
from ._parameter_access import convolution_parameters, normalization_parameters


# Parameter key prefixes within an attention block's group.
ATTENTION_NORM_PREFIX = "norm"
ATTENTION_QUERY_PREFIX = "q"
ATTENTION_KEY_PREFIX = "k"
ATTENTION_VALUE_PREFIX = "v"
ATTENTION_OUTPUT_PREFIX = "proj_out"

# Attention scores are exponentiated, so they are accumulated in at
# least float32 for the same reason normalization statistics are: a
# softmax computed in bfloat16 over thousands of positions loses
# resolution in the tail of the distribution. Applied as a floor via
# jnp.promote_types, never as an unconditional cast, so float64 callers
# keep their precision.
MINIMUM_ATTENTION_ACCUMULATION_DTYPE = jnp.float32


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

    norm_scale, norm_shift = normalization_parameters(
        parameters, ATTENTION_NORM_PREFIX, context
    )
    normalized = group_normalization(activations, norm_scale, norm_shift, config)

    projections = []
    for prefix in (ATTENTION_QUERY_PREFIX, ATTENTION_KEY_PREFIX, ATTENTION_VALUE_PREFIX):
        kernel, bias = convolution_parameters(parameters, prefix, context)
        projected = convolution_2d(normalized, kernel, bias, config)
        projections.append(projected.reshape(batch, height * width, channels))

    queries, keys, values = projections

    attended = _chunked_self_attention(
        queries, keys, values, config.attention_query_chunk_size
    )
    attended = attended.reshape(batch, height, width, channels)

    output_kernel, output_bias = convolution_parameters(
        parameters, ATTENTION_OUTPUT_PREFIX, context
    )
    projected_output = convolution_2d(attended, output_kernel, output_bias, config)

    return activations + projected_output
