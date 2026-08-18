"""
Unit tests for config.py.
"""

from __future__ import annotations

import logging

from src.config import (
    MAXIMUM_RECOMMENDED_IMAGE_TOKENS,
    MemoryResidencyStrategy,
    ResolutionBucket,
    STANDARD_RESOLUTION_BUCKETS,
    resolve_residency_strategy,
)


def test_resolve_residency_strategy_single_device_resolves_to_swapped() -> None:
    result = resolve_residency_strategy(MemoryResidencyStrategy.AUTO, visible_device_count=1)
    assert result is MemoryResidencyStrategy.SWAPPED


def test_resolve_residency_strategy_multi_device_resolves_to_fully_resident() -> None:
    result = resolve_residency_strategy(MemoryResidencyStrategy.AUTO, visible_device_count=8)
    assert result is MemoryResidencyStrategy.FULLY_RESIDENT


def test_resolve_residency_strategy_explicit_choice_is_never_overridden() -> None:
    result = resolve_residency_strategy(MemoryResidencyStrategy.SWAPPED, visible_device_count=8)
    assert result is MemoryResidencyStrategy.SWAPPED

    result = resolve_residency_strategy(MemoryResidencyStrategy.FULLY_RESIDENT, visible_device_count=1)
    assert result is MemoryResidencyStrategy.FULLY_RESIDENT


def test_resolution_bucket_image_tokens_derived_correctly() -> None:
    bucket = ResolutionBucket(width=1024, height=1024)
    assert bucket.image_tokens == 4096
    assert bucket.label == "1024x1024"

    bucket = ResolutionBucket(width=1360, height=768)
    assert bucket.image_tokens == 4080


def test_standard_resolution_buckets_are_all_under_the_schedule_threshold() -> None:
    for bucket in STANDARD_RESOLUTION_BUCKETS:
        assert bucket.image_tokens <= MAXIMUM_RECOMMENDED_IMAGE_TOKENS, (
            f"{bucket.label} exceeds the recommended schedule threshold"
        )


_CONFIG_TESTS = [
    test_resolve_residency_strategy_single_device_resolves_to_swapped,
    test_resolve_residency_strategy_multi_device_resolves_to_fully_resident,
    test_resolve_residency_strategy_explicit_choice_is_never_overridden,
    test_resolution_bucket_image_tokens_derived_correctly,
    test_standard_resolution_buckets_are_all_under_the_schedule_threshold,
]


def run_config_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against config.py", len(_CONFIG_TESTS))
    for test_function in _CONFIG_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All config tests passed")
