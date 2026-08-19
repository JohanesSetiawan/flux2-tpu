"""
Tests for the interface layer.

The front ends themselves are wiring and are not unit tested by
clicking them. What is tested is everything they delegate to, which is
where their behaviour actually lives, plus the one property of the
front ends worth asserting mechanically: that importing them does not
require their optional toolkit, so a script user is not forced to
install a notebook or a web framework.

Two tests deserve their reasoning stated. Seed resolution is checked
with an injected random source rather than by observing that two calls
differ, because two random draws can coincide and a flaky test is worse
than no test. Image conversion is checked for rounding rather than
truncation, because truncation is the natural implementation and it
darkens every image slightly and systematically.
"""

from __future__ import annotations

import importlib
import logging
import random

import numpy as np

from src.config import STANDARD_RESOLUTION_BUCKETS, ResolutionBucket
from src.interfaces.session import (
    RANDOM_SEED_MAXIMUM,
    RANDOM_SEED_MINIMUM,
    RANDOM_SEED_SENTINEL,
    GenerationOutcome,
    UnknownResolutionError,
    build_request,
    describe_outcome,
    resolution_labels,
    resolve_resolution,
    resolve_seed,
    to_display_image,
)


FRONT_END_MODULES = ("src.interfaces.widgets", "src.interfaces.browser")


def test_smoke_resolution_labels_match_configured_buckets() -> None:
    labels = resolution_labels(STANDARD_RESOLUTION_BUCKETS)

    assert len(labels) == len(STANDARD_RESOLUTION_BUCKETS)
    assert labels[0] == STANDARD_RESOLUTION_BUCKETS[0].label


def test_regression_resolve_resolution_round_trips_every_bucket() -> None:
    """
    Every label a front end offers must resolve back to the bucket it
    came from. A front end builds its choices from these labels, so a
    label that does not resolve would be an option a person can select
    and never use.
    """
    for bucket in STANDARD_RESOLUTION_BUCKETS:
        resolved = resolve_resolution(bucket.label, STANDARD_RESOLUTION_BUCKETS)

        assert resolved is bucket, f"label {bucket.label} did not resolve to its own bucket"


def test_regression_resolve_resolution_rejects_unknown_label() -> None:
    try:
        resolve_resolution("640x480", STANDARD_RESOLUTION_BUCKETS)
    except UnknownResolutionError as error:
        assert "640x480" in str(error)
        assert "1024x1024" in str(error), "the error should list the available choices"
        return
    raise AssertionError("Expected UnknownResolutionError for a label with no bucket")


def test_regression_explicit_seed_is_passed_through_unchanged() -> None:
    for seed in (0, 1, 12345, RANDOM_SEED_MAXIMUM):
        assert resolve_seed(seed) == seed


def test_regression_sentinel_draws_a_seed_in_range() -> None:
    """
    Checked with an injected source rather than by comparing two draws.
    Two random values can legitimately coincide, so a difference-based
    test would fail occasionally against correct code.
    """
    source = random.Random(20260819)

    drawn = resolve_seed(RANDOM_SEED_SENTINEL, source)

    assert RANDOM_SEED_MINIMUM <= drawn <= RANDOM_SEED_MAXIMUM
    assert drawn != RANDOM_SEED_SENTINEL

    # The same seeded source must reproduce the same draw, which is what
    # makes a randomly seeded generation reproducible after the fact.
    assert resolve_seed(RANDOM_SEED_SENTINEL, random.Random(20260819)) == drawn


def test_regression_build_request_strips_surrounding_whitespace() -> None:
    """
    Trailing whitespace from a text box would make two visually
    identical prompts miss the conditioning cache, silently paying for
    an extra encode.
    """
    request = build_request(
        "  a quiet street  ", "1024x1024", 7, STANDARD_RESOLUTION_BUCKETS
    )

    assert request.prompt == "a quiet street"


def test_regression_build_request_rejects_an_empty_prompt() -> None:
    for empty in ("", "   ", "\n\t"):
        try:
            build_request(empty, "1024x1024", 1, STANDARD_RESOLUTION_BUCKETS)
        except ValueError as error:
            assert "empty" in str(error).lower()
            continue
        raise AssertionError(f"Expected ValueError for prompt {empty!r}")


def test_regression_build_request_resolves_the_sentinel_seed() -> None:
    request = build_request(
        "a prompt",
        "1360x768",
        RANDOM_SEED_SENTINEL,
        STANDARD_RESOLUTION_BUCKETS,
        random.Random(1),
    )

    assert request.seed != RANDOM_SEED_SENTINEL
    assert RANDOM_SEED_MINIMUM <= request.seed <= RANDOM_SEED_MAXIMUM
    assert request.resolution.label == "1360x768"


def test_regression_display_image_rounds_rather_than_truncates() -> None:
    """
    Truncation is the natural implementation and is wrong: it always
    moves values down, darkening every image slightly. A value just
    below a boundary must round up.
    """
    values = np.array([0.0, 0.5, 0.999, 1.0])

    converted = to_display_image(values)

    assert converted.dtype == np.uint8
    assert converted[0] == 0
    assert converted[1] == 128, "the midpoint should round to 128, not truncate to 127"
    assert converted[2] == 255, "a value just below one should round up, not down to 254"
    assert converted[3] == 255


def test_regression_display_image_clips_out_of_range_values() -> None:
    """
    Values outside unit range would wrap when cast to eight bits,
    turning a bright pixel dark. Clipping happens before the cast.
    """
    values = np.array([-0.5, 1.5])

    converted = to_display_image(values)

    assert converted[0] == 0
    assert converted[1] == 255


def test_regression_outcome_description_includes_the_seed() -> None:
    """
    The seed is the only part of an outcome needed to reproduce it, so
    it must be visible rather than merely returned.
    """
    outcome = GenerationOutcome(
        image=np.zeros((2, 2, 3)), seed=4242, resolution_label="1024x1024"
    )

    description = describe_outcome(outcome)

    assert "4242" in description
    assert "1024x1024" in description


def test_regression_front_ends_import_without_their_toolkits() -> None:
    """
    Both front ends import their toolkit inside the function that needs
    it, so someone using the pipeline from a script is not required to
    install ipywidgets or Gradio. Neither is installed in this
    environment, so a module-scope import would fail here.
    """
    for module_name in FRONT_END_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, f"{module_name} failed to import"


def test_regression_front_ends_report_a_missing_toolkit_clearly() -> None:
    """
    When the toolkit really is absent, the error must name it and say
    how to install it, rather than surfacing a bare ImportError from
    deep inside the module.
    """
    from src.interfaces.browser import build_interface as build_browser
    from src.interfaces.widgets import build_control_panel

    logger = logging.getLogger("interface_tests")
    logger.addHandler(logging.NullHandler())

    for builder, expected_name in (
        (build_browser, "gradio"),
        (build_control_panel, "ipywidgets"),
    ):
        try:
            builder(None, logger)
        except ImportError as error:
            assert expected_name in str(error).lower(), (
                f"the error should name {expected_name}: {error}"
            )
            assert "install" in str(error).lower(), (
                "the error should say how to install the missing toolkit"
            )
            continue
        except Exception as error:
            raise AssertionError(
                f"expected an ImportError naming {expected_name}, got {type(error).__name__}"
            ) from error
        raise AssertionError(f"expected an ImportError for the missing {expected_name}")


def test_regression_custom_buckets_are_respected() -> None:
    """
    A front end offers whatever the configuration holds, not a fixed
    list. Hardcoding the standard resolutions anywhere in this layer
    would break a caller who configured their own.
    """
    custom = (ResolutionBucket(width=512, height=512),)

    assert resolution_labels(custom) == ["512x512"]

    request = build_request("a prompt", "512x512", 3, custom)
    assert request.resolution.width == 512

    try:
        build_request("a prompt", "1024x1024", 3, custom)
    except UnknownResolutionError:
        return
    raise AssertionError("a label outside the custom buckets should not resolve")


_INTERFACE_TESTS = [
    test_smoke_resolution_labels_match_configured_buckets,
    test_regression_resolve_resolution_round_trips_every_bucket,
    test_regression_resolve_resolution_rejects_unknown_label,
    test_regression_explicit_seed_is_passed_through_unchanged,
    test_regression_sentinel_draws_a_seed_in_range,
    test_regression_build_request_strips_surrounding_whitespace,
    test_regression_build_request_rejects_an_empty_prompt,
    test_regression_build_request_resolves_the_sentinel_seed,
    test_regression_display_image_rounds_rather_than_truncates,
    test_regression_display_image_clips_out_of_range_values,
    test_regression_outcome_description_includes_the_seed,
    test_regression_front_ends_import_without_their_toolkits,
    test_regression_front_ends_report_a_missing_toolkit_clearly,
    test_regression_custom_buckets_are_respected,
]


def run_interface_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against the interface layer", len(_INTERFACE_TESTS))
    for test_function in _INTERFACE_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All interface tests passed")
