"""
The logic behind the interactive interfaces, with no interface in it.

Both the widget and the browser front ends need the same things: turn a
resolution label back into a bucket, turn a seed field into a number,
build a request, and convert the result into something displayable.
None of that involves a widget, so none of it lives in a widget module.

Keeping it here is what makes the interfaces testable. A front end
built directly on the pipeline would have its input handling verifiable
only by clicking, and clicking is not a test.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..config import ResolutionBucket
from ..pipeline import GenerationRequest


# Seeds are drawn from this range when the caller asks for a random one.
# Bounded well below the platform integer limit so the value round-trips
# through a text field and a JSON payload without surprises.
RANDOM_SEED_MINIMUM = 0
RANDOM_SEED_MAXIMUM = 2**31 - 1

# Sentinel a front end can pass instead of a seed to request a fresh
# one. Distinct from any valid seed, and negative so it cannot collide
# with the range above.
RANDOM_SEED_SENTINEL = -1

DISPLAY_IMAGE_MAXIMUM = 255


class UnknownResolutionError(ValueError):
    """
    Raised when a resolution label does not match any configured bucket.

    A front end offering a fixed set of choices should never trigger
    this, so reaching it means the front end and the configuration have
    drifted apart, which is worth failing on rather than defaulting.
    """


@dataclass(frozen=True)
class GenerationOutcome:
    """
    What a front end needs to display after a request completes.

    Carries the resolved seed rather than the requested one, so that a
    randomly seeded result can be reproduced later. Returning only the
    image would make a good result impossible to recover.
    """

    image: np.ndarray
    seed: int
    resolution_label: str


def resolution_labels(buckets: tuple[ResolutionBucket, ...]) -> list[str]:
    """List the labels a front end should offer, in configuration order."""
    return [bucket.label for bucket in buckets]


def resolve_resolution(
    label: str, buckets: tuple[ResolutionBucket, ...]
) -> ResolutionBucket:
    """Look a bucket up by its label."""
    for bucket in buckets:
        if bucket.label == label:
            return bucket
    raise UnknownResolutionError(
        f"No configured resolution matches '{label}'. Available: "
        f"{resolution_labels(buckets)}"
    )


def resolve_seed(requested_seed: int, random_source: random.Random | None = None) -> int:
    """
    Turn a requested seed into a concrete one, drawing a random value
    for the sentinel.

    The random source is injectable so tests can make the draw
    deterministic. Defaulting to the module-level generator keeps the
    common call site simple.
    """
    if requested_seed != RANDOM_SEED_SENTINEL:
        return requested_seed

    source = random_source or random.Random()
    return source.randint(RANDOM_SEED_MINIMUM, RANDOM_SEED_MAXIMUM)


def build_request(
    prompt: str,
    resolution_label: str,
    requested_seed: int,
    buckets: tuple[ResolutionBucket, ...],
    random_source: random.Random | None = None,
) -> GenerationRequest:
    """
    Assemble a request from raw front-end inputs.

    The prompt is stripped, because trailing whitespace from a text box
    would otherwise make two visually identical prompts miss the
    conditioning cache.
    """
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        raise ValueError("Prompt is empty")

    return GenerationRequest(
        prompt=cleaned_prompt,
        resolution=resolve_resolution(resolution_label, buckets),
        seed=resolve_seed(requested_seed, random_source),
    )


def to_display_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a unit-range float image into eight-bit values.

    Rounds rather than truncating: truncation would darken the image
    slightly and systematically, since it always moves values down.
    """
    scaled = np.round(image * DISPLAY_IMAGE_MAXIMUM)
    return np.clip(scaled, 0, DISPLAY_IMAGE_MAXIMUM).astype(np.uint8)


def describe_outcome(outcome: GenerationOutcome) -> str:
    """
    One line summarising a completed generation.

    Includes the seed prominently, since that is the only part of the
    outcome a person needs in order to reproduce it.
    """
    return (
        f"{outcome.resolution_label} generated with seed {outcome.seed}. "
        f"Reuse that seed to reproduce this image."
    )
