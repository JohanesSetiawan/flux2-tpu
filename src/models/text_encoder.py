"""
The Qwen3 text encoder, producing FLUX.2 Klein conditioning.

This is not a language model: nothing here generates tokens, and the
checkpoint carries no output head. It runs a stack of transformer
layers over a padded prompt and returns a conditioning tensor built by
concatenating three intermediate hidden states.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..blocks import transformer_layer
from ..checkpoint import require_parameter
from ..config import TextEncoderConfig
from ..layers import causal_padding_mask, rotary_frequency_table


EMBEDDING_KEY = "weight"
EMBEDDING_GROUP = "embed_tokens"
LAYER_GROUP = "layers"


class LayerCountMismatchError(Exception):
    """
    Raised when the checkpoint carries a different number of layers than
    the configuration requires to produce its requested hidden states.

    Running fewer layers than required would silently return a
    conditioning tensor built from the wrong depth, so this fails loudly
    instead.
    """


def embed_tokens(
    token_ids: jnp.ndarray, parameters: dict, config: TextEncoderConfig
) -> jnp.ndarray:
    """
    Look up token embeddings.

    The embedding table keeps its original (vocabulary, hidden) row
    layout because it is read by gather rather than by matrix
    multiplication, so unlike every projection weight it was not
    transposed during conversion.
    """
    table = require_parameter(parameters[EMBEDDING_GROUP], EMBEDDING_KEY, "embed_tokens")
    if table.shape[0] != config.vocab_size:
        raise ValueError(
            f"Embedding table has {table.shape[0]} rows but the configuration "
            f"declares a vocabulary of {config.vocab_size}"
        )
    return jnp.take(table, token_ids, axis=0)


def _layer_parameters_at(stacked_layers: dict, layer_index: int) -> dict:
    """
    Slice one layer's parameters out of the stacked arrays.

    The conversion stacked every layer's tensors along a new leading
    axis so the stack can eventually be driven by a scan. Until that
    optimisation is in place, indexing that axis gives an ordinary
    per-layer parameter group.
    """
    return {key: value[layer_index] for key, value in stacked_layers.items()}


def encode_prompt(
    token_ids: jnp.ndarray,
    token_is_real: jnp.ndarray,
    parameters: dict,
    config: TextEncoderConfig,
) -> jnp.ndarray:
    """
    Encode a tokenized prompt into the conditioning tensor consumed by
    the diffusion transformer.

    Conditioning is formed by concatenating the hidden states after the
    layers named in config.hidden_states_output_layers. Those states are
    taken raw, with no final normalization applied: in the full upstream
    model a final norm exists but is applied only after the last layer,
    so an intermediate hidden state never passes through it. The
    conversion dropped that norm accordingly, and applying one here
    would diverge from the reference.

    Position indices run over the full padded length, including padded
    positions. This matches the reference, which derives positions from
    a plain arange rather than from a cumulative sum of the attention
    mask, so padding still advances the position counter. Do not
    "correct" this.

    Parameters
    ----------
    token_ids:
        Shape (batch, sequence_length), integer token identifiers.
    token_is_real:
        Shape (batch, sequence_length), non-zero for real tokens and
        zero for padding.
    parameters:
        The restored text encoder component.
    config:
        Architecture and precision settings.

    Returns
    -------
    Conditioning of shape
    (batch, sequence_length, hidden_size * number_of_selected_layers).
    """
    batch, sequence_length = token_ids.shape

    stacked_layers = parameters[LAYER_GROUP]
    available_layers = next(iter(stacked_layers.values())).shape[0]
    if available_layers < config.num_layers_required:
        raise LayerCountMismatchError(
            f"Checkpoint carries {available_layers} layers but the configured "
            f"hidden state selection {config.hidden_states_output_layers} requires "
            f"{config.num_layers_required}"
        )

    rotary_cosine, rotary_sine = rotary_frequency_table(
        sequence_length, config.head_dim, config.rope_theta
    )
    attention_bias = causal_padding_mask(token_is_real, sequence_length)

    activations = embed_tokens(token_ids, parameters, config)

    # Hidden state k is the value after k layers have run, so the
    # embedding output is state zero. Recording states in a dictionary
    # keyed by depth, rather than appending to a list, keeps the mapping
    # from configured index to captured tensor explicit.
    selected_states: dict[int, jnp.ndarray] = {}
    if 0 in config.hidden_states_output_layers:
        selected_states[0] = activations

    for layer_index in range(config.num_layers_required):
        activations = transformer_layer(
            activations,
            _layer_parameters_at(stacked_layers, layer_index),
            rotary_cosine,
            rotary_sine,
            attention_bias,
            config,
        )
        depth = layer_index + 1
        if depth in config.hidden_states_output_layers:
            selected_states[depth] = activations

    missing_depths = set(config.hidden_states_output_layers) - set(selected_states)
    if missing_depths:
        raise LayerCountMismatchError(
            f"Hidden states at depths {sorted(missing_depths)} were never captured"
        )

    ordered_states = [
        selected_states[depth] for depth in config.hidden_states_output_layers
    ]
    return jnp.concatenate(ordered_states, axis=-1)
