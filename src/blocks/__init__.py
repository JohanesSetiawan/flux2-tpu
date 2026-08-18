"""
Composite blocks assembled from layer primitives.

A block may import from layers, config and checkpoint, but never from
models, keeping the dependency direction strictly downward.
"""

from .attention import attention_block
from .residual import residual_block

__all__ = ["attention_block", "residual_block"]
