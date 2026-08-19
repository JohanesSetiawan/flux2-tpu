"""
Composite blocks assembled from layer primitives.

A block may import from layers, config and checkpoint, but never from
models, keeping the dependency direction strictly downward.

Two attention implementations live here and are not interchangeable.
`attention_block` is the autoencoder's: a single head over spatial
positions, unmasked, with query chunking for memory. 
`grouped_query_attention` is the text encoder's: multi-head with fewer
key/value heads than query heads, per-head normalization, rotary
position embedding, and masking.
"""

from .attention import attention_block
from .feedforward import gated_feedforward
from .grouped_query_attention import grouped_query_attention
from .modulation import (
    ModulationTriple,
    apply_modulated_normalization,
    compute_modulation,
)
from .residual import residual_block
from .transformer_layer import transformer_layer

__all__ = [
    "ModulationTriple",
    "apply_modulated_normalization",
    "attention_block",
    "compute_modulation",
    "gated_feedforward",
    "grouped_query_attention",
    "residual_block",
    "transformer_layer",
]
