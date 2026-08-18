"""Configuration for the autoencoder decoder."""

from __future__ import annotations

from dataclasses import dataclass, field

from .precision import NumericPrecision


@dataclass(frozen=True)
class VaeLayerConfig:
    """
    Numerical parameters for the decoder's individual layers.

    The group count and epsilon are properties of the trained
    checkpoint, not free choices: changing them produces different
    outputs from the same weights. They live here rather than inline so
    that where they are defined is also where they are documented.
    """

    num_groups: int = 32
    normalization_epsilon: float = 1e-6
    upsample_scale_factor: int = 2
    precision: NumericPrecision = NumericPrecision.HIGHEST

    # Number of query positions processed per chunk in the decoder's
    # middle attention block.
    #
    # That block is the decoder's heaviest memory consumer: a single
    # attention head whose head dimension equals the channel count,
    # attending over every spatial position of the latent. At 1024x1024
    # output the latent is 128x128, so a fully materialized float32
    # score matrix is slightly over one gigabyte for one block.
    # Chunking bounds that to chunk_size by sequence_length.
    #
    # This is purely a memory and throughput tradeoff. Chunked and
    # unchunked attention are numerically equivalent, which the test
    # suite asserts directly.
    attention_query_chunk_size: int = 2048


@dataclass(frozen=True)
class VaeDecoderConfig:
    """
    Structural parameters of the decoder.

    Only values that cannot be discovered from the checkpoint appear
    here. The number of upsampling levels, the number of residual blocks
    per level, and every channel count are deliberately absent: they are
    read from the restored parameters at run time, so a checkpoint with
    a different depth is handled correctly rather than silently
    mismatched against a hardcoded assumption.
    """

    # The encoder packs each 2x2 spatial block into the channel axis
    # before the latent is stored, so decoding begins by reversing that.
    latent_patch_size: int = 2

    layer: VaeLayerConfig = field(default_factory=VaeLayerConfig)
