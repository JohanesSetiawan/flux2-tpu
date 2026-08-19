"""One Qwen3 transformer layer."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..checkpoint import require_parameter
from ..config import TextEncoderConfig
from ..layers import rms_normalization
from .feedforward import gated_feedforward
from .grouped_query_attention import grouped_query_attention


INPUT_NORM_KEY = "input_layernorm_weight"
POST_ATTENTION_NORM_KEY = "post_attention_layernorm_weight"


def transformer_layer(
    activations: jnp.ndarray,
    parameters: dict[str, np.ndarray],
    rotary_cosine: jnp.ndarray,
    rotary_sine: jnp.ndarray,
    attention_bias: jnp.ndarray,
    config: TextEncoderConfig,
) -> jnp.ndarray:
    """
    Apply attention and feed-forward, each wrapped in a pre-norm
    residual.

    Pre-norm means normalization is applied to the input of each
    sub-layer, not to its output, and the residual carries the
    un-normalized value forward. Post-norm ordering would produce
    different results from the same weights.

    Despite the checkpoint's key names ending in "layernorm", these are
    RMS normalizations: the names are inherited from the upstream
    implementation, which kept the original attribute names when it
    changed the operation. Reading the names literally and applying
    layer normalization here would be wrong.
    """
    context = "transformer_layer"

    normalized = rms_normalization(
        activations,
        require_parameter(parameters, INPUT_NORM_KEY, context),
        config.rms_norm_epsilon,
    )
    activations = activations + grouped_query_attention(
        normalized, parameters, rotary_cosine, rotary_sine, attention_bias, config
    )

    normalized = rms_normalization(
        activations,
        require_parameter(parameters, POST_ATTENTION_NORM_KEY, context),
        config.rms_norm_epsilon,
    )
    return activations + gated_feedforward(normalized, parameters, config)
