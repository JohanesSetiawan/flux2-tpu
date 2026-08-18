"""
Runtime execution settings: where model components live in memory, and
which output resolutions are supported.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class MemoryResidencyStrategy(enum.Enum):
    """
    How the three model components (text encoder, transformer, VAE
    decoder) are held in accelerator memory across a generation call.

    FULLY_RESIDENT: all three stay resident in accelerator memory for
        the lifetime of the session. Requires either enough aggregate
        memory to hold all three at once, which is true of multi-chip
        TPU pods, or a reduced-memory text encoder. Zero per-request
        swap latency.

    SWAPPED: the transformer and VAE decoder stay resident; the text
        encoder lives in host RAM and is copied into accelerator memory
        only while encoding a prompt, then evicted. Fits single-chip
        16 GB accelerators without reducing any component's precision,
        at the cost of a host transfer on every prompt change.

    AUTO: resolved to one of the above from the visible device count.
    """

    FULLY_RESIDENT = "fully_resident"
    SWAPPED = "swapped"
    AUTO = "auto"


def resolve_residency_strategy(
    configured_strategy: MemoryResidencyStrategy,
    visible_device_count: int,
) -> MemoryResidencyStrategy:
    """
    Resolve AUTO into a concrete strategy from the number of visible
    JAX devices.

    A single visible device is treated as a single-chip, 16 GB-class
    accelerator and resolves to SWAPPED; more than one device is treated
    as a pod with enough aggregate memory for everything at once, and
    resolves to FULLY_RESIDENT.

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
    One supported output resolution.

    Image token count is derived from width and height rather than
    stored, so the two cannot drift out of sync.
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


# Above this token count the reference sampling schedule switches to a
# formula that ignores the step count entirely, which this checkpoint's
# 4-step distillation was not tuned against. See AGENTS.md for the full
# derivation and the measured discontinuity.
MAXIMUM_RECOMMENDED_IMAGE_TOKENS = 4300

STANDARD_RESOLUTION_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(width=1024, height=1024),
    ResolutionBucket(width=1360, height=768),
    ResolutionBucket(width=768, height=1360),
)
