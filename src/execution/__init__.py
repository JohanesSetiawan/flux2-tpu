"""
How work is placed and compiled.

Nothing here changes what the model computes. These modules decide
where parameters live, how they are split across devices, and whether
compiled programs persist between sessions.
"""

from .compilation import configure_compilation_cache
from .residency import (
    ComponentResidency,
    evict_to_host,
    move_to_accelerator,
    plan_component_residency,
)
from .sharding import (
    SPLITTABLE_GROUPS_BY_COMPONENT,
    build_device_mesh,
    place_component,
    replicate_parameters,
    shard_stacked_blocks,
)

__all__ = [
    "ComponentResidency",
    "SPLITTABLE_GROUPS_BY_COMPONENT",
    "build_device_mesh",
    "place_component",
    "configure_compilation_cache",
    "evict_to_host",
    "move_to_accelerator",
    "plan_component_residency",
    "replicate_parameters",
    "shard_stacked_blocks",
]
