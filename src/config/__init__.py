"""
Configuration for the inference package.

Every value that drives behaviour lives in one of the modules here as a
dataclass field or a documented constant. Nothing outside this package
contains a literal that changes what the model computes.

This module re-exports the full public surface so that callers write
`from src.config import VaeDecoderConfig` without needing to know which
file within the package defines it.
"""

from .checkpoint import CheckpointSourceConfig, InferenceConfig
from .precision import NumericPrecision
from .text_encoder import TextEncoderConfig
from .transformer import TransformerConfig
from .runtime import (
    MAXIMUM_RECOMMENDED_IMAGE_TOKENS,
    STANDARD_RESOLUTION_BUCKETS,
    MemoryResidencyStrategy,
    ResolutionBucket,
    resolve_residency_strategy,
)
from .vae import VaeDecoderConfig, VaeLayerConfig

__all__ = [
    "CheckpointSourceConfig",
    "InferenceConfig",
    "MAXIMUM_RECOMMENDED_IMAGE_TOKENS",
    "MemoryResidencyStrategy",
    "NumericPrecision",
    "ResolutionBucket",
    "STANDARD_RESOLUTION_BUCKETS",
    "TextEncoderConfig",
    "TransformerConfig",
    "VaeDecoderConfig",
    "VaeLayerConfig",
    "resolve_residency_strategy",
]
