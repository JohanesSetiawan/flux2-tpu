"""Configuration for locating and loading the weight bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .runtime import (
    STANDARD_RESOLUTION_BUCKETS,
    MemoryResidencyStrategy,
    ResolutionBucket,
)


@dataclass(frozen=True)
class CheckpointSourceConfig:
    """
    Where the converted JAX-native checkpoint bundle is downloaded from.

    The repository is public, so the token is optional and used only to
    raise the Hub's anonymous rate limit.
    """

    huggingface_repo_id: str = "johaness14/flux2-klein-4b-jax"
    huggingface_revision: str | None = None
    huggingface_token_environment_variable: str = "HF_TOKEN"
    local_cache_directory: Path = field(
        default_factory=lambda: Path("/kaggle/temp/flux2_klein_checkpoint_cache")
    )


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level container bundling every configuration group together."""

    checkpoint_source: CheckpointSourceConfig = field(default_factory=CheckpointSourceConfig)
    residency_strategy: MemoryResidencyStrategy = MemoryResidencyStrategy.AUTO
    resolution_buckets: tuple[ResolutionBucket, ...] = STANDARD_RESOLUTION_BUCKETS
    log_file_path: Path = field(
        default_factory=lambda: Path("/kaggle/working/flux2_klein_inference_log.txt")
    )
