"""
Configuration for the flux2_klein inference package.

As with the converter, every value that drives behaviour lives here as
a dataclass field or a named constant, not as a literal inline in the
logic that uses it. Where one value is mechanically derivable from
another (for example, image token count from resolution and patch
size), it is exposed as a property instead of being duplicated.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class MemoryResidencyStrategy(enum.Enum):
    """
    How the three model components (text encoder, transformer, VAE
    decoder) are held in accelerator memory across a generation call.

    FULLY_RESIDENT: all three components stay resident in HBM for the
        lifetime of the session. Requires either enough aggregate HBM
        to hold all three at once (true of multi-chip TPU pods such as
        Kaggle's v5e-8), or a reduced-memory text encoder. Zero
        per-request swap latency.

    SWAPPED: the transformer and VAE decoder stay resident; the text
        encoder lives in host RAM and is copied into HBM only while
        encoding a prompt, then evicted. Fits single-chip 16 GB
        accelerators such as Colab's free-tier v5e-1 without reducing
        any component's precision, at the cost of a host-to-HBM
        transfer on every prompt change.

    AUTO: resolved to FULLY_RESIDENT or SWAPPED based on the number of
        visible JAX devices at runtime, via resolve_residency_strategy.
    """

    FULLY_RESIDENT = "fully_resident"
    SWAPPED = "swapped"
    AUTO = "auto"


def resolve_residency_strategy(
    configured_strategy: MemoryResidencyStrategy,
    visible_device_count: int,
) -> MemoryResidencyStrategy:
    """
    Resolve AUTO into a concrete strategy based on how many JAX devices
    are visible. A single visible device is treated as a single-chip,
    16 GB-class accelerator (Colab's free-tier v5e-1) and resolves to
    SWAPPED; more than one device is treated as a multi-chip pod with
    enough aggregate HBM to hold everything at once, and resolves to
    FULLY_RESIDENT.

    A strategy that is not AUTO is returned unchanged: an explicit
    choice always overrides the heuristic.
    """
    if configured_strategy is not MemoryResidencyStrategy.AUTO:
        return configured_strategy
    return (
        MemoryResidencyStrategy.FULLY_RESIDENT
        if visible_device_count > 1
        else MemoryResidencyStrategy.SWAPPED
    )


@dataclass(frozen=True)
class ResolutionBucket:
    """
    One supported output resolution. Image token count is derived from
    width and height rather than stored separately, so the two values
    cannot drift out of sync.
    """

    width: int
    height: int
    patch_size: int = 16

    @property
    def image_tokens(self) -> int:
        return (self.width // self.patch_size) * (self.height // self.patch_size)

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


# The three resolutions below all sit at or under the 4300 image-token
# threshold past which the reference sampling schedule switches to a
# formula this checkpoint's 4-step distillation was not tuned against.
# See the checkpoint bundle's README for the full derivation.
STANDARD_RESOLUTION_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(width=1024, height=1024),
    ResolutionBucket(width=1360, height=768),
    ResolutionBucket(width=768, height=1360),
)

MAXIMUM_RECOMMENDED_IMAGE_TOKENS = 4300


@dataclass(frozen=True)
class CheckpointSourceConfig:
    """Where the converted JAX-native checkpoint bundle is downloaded from."""

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
