"""
Interactive front ends.

These modules contain no generation logic. Input handling lives in
`session`, which both front ends share, and generation lives in the
pipeline. A front end here should be readable as wiring and nothing
else.

Both toolkits are optional dependencies, imported inside the functions
that need them so that using the pipeline from a script does not
require a notebook or a web framework to be installed.
"""

from .session import (
    GenerationOutcome,
    RANDOM_SEED_SENTINEL,
    UnknownResolutionError,
    build_request,
    describe_outcome,
    resolution_labels,
    resolve_resolution,
    resolve_seed,
    to_display_image,
)

__all__ = [
    "GenerationOutcome",
    "RANDOM_SEED_SENTINEL",
    "UnknownResolutionError",
    "build_request",
    "describe_outcome",
    "resolution_labels",
    "resolve_resolution",
    "resolve_seed",
    "to_display_image",
]
