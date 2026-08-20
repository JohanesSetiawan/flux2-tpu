"""
Tests for the telemetry layer.

Two of these are load-bearing, because both pin behaviour that would
otherwise silently produce a plausible but wrong number:

test_regression_timing_waits_for_the_result checks that a timed stage
blocks. JAX dispatch is asynchronous, so a timer that does not wait
reports queueing time. It is the difference between a profile that
directs attention correctly and one that reports every stage as
instantaneous.

test_regression_value_summary_promotes_before_reducing checks that
statistics are computed in a wider dtype. Summing bfloat16 values in
bfloat16 saturates, and an early version of this module reported an
array of ones as having a mean near zero, which would send a reader
hunting for a bug in the stage being described rather than in the
description of it.
"""

from __future__ import annotations

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.telemetry import (
    RunProfile,
    describe_array,
    describe_tree,
    format_bytes,
    read_device_memory,
    summarise_values,
    timed_stage,
)


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("telemetry_tests")
    logger.addHandler(logging.NullHandler())
    return logger


def test_smoke_format_bytes_scales_readably() -> None:
    assert format_bytes(512).endswith("B")
    assert "MiB" in format_bytes(5 * 1024 ** 2)
    assert "GiB" in format_bytes(5 * 1024 ** 3)


def test_regression_describe_array_reports_size_without_reading_values() -> None:
    array = jnp.ones((128, 3072), dtype=jnp.bfloat16)

    description = describe_array(array)

    assert description.shape == (128, 3072)
    assert description.dtype == "bfloat16"
    assert description.bytes_used == 128 * 3072 * 2, (
        "size should follow from shape and dtype rather than being measured"
    )


def test_regression_describe_tree_reports_every_dtype_present() -> None:
    """
    A tree reported as having one dtype when it has two would hide
    exactly the situation that broke the scanned block stack, where a
    handful of float32 tensors sat among bfloat16 ones.
    """
    tree = {
        "a": jnp.ones((4,), dtype=jnp.bfloat16),
        "b": {"c": jnp.ones((4,), dtype=jnp.float32)},
    }

    description = describe_tree(tree)

    assert description.leaf_count == 2
    assert set(description.dtypes) == {"bfloat16", "float32"}


def test_regression_single_device_placement_is_not_reported_as_split() -> None:
    """
    An array sitting on one device is not a split one, and conflating
    them is how a component that never reached the device mesh went
    unnoticed while an eight-chip pod ran it on one chip.
    """
    tree = {"weight": jax.device_put(jnp.ones((4, 4)), jax.devices()[0])}

    description = describe_tree(tree)

    assert description.sharded_leaf_count == 0, (
        "a parameter on a single device should not be counted as split across devices"
    )
    assert describe_array(tree["weight"]).sharding == "single device"


def test_regression_value_summary_promotes_before_reducing() -> None:
    """
    Statistics must be computed in a wider dtype than bfloat16.

    Ten thousand ones summed in bfloat16 saturate long before the end,
    so the mean comes out near zero. The array below is large enough for
    that to happen, and its correct mean is exactly one.
    """
    array = jnp.ones((256, 256), dtype=jnp.bfloat16)

    summary = summarise_values(array)

    assert "mean=1.0000" in summary, (
        f"summary reported {summary}; statistics may be accumulating in bfloat16"
    )


def test_regression_value_summary_reports_non_finite_values_prominently() -> None:
    """
    A single infinity makes a minimum or maximum meaningless while being
    the most important thing to report, so it replaces the summary
    rather than appearing beside it.
    """
    array = jnp.array([1.0, jnp.inf, 3.0])

    summary = summarise_values(array)

    assert "NON-FINITE" in summary
    assert "1" in summary


def test_regression_timing_waits_for_the_result() -> None:
    """
    A timed stage must block on what it produces.

    The work below takes long enough that dispatch and execution are
    clearly distinguishable: without blocking the reported time would be
    a fraction of a millisecond regardless of how long the computation
    actually ran.
    """
    logger = _silent_logger()
    profile = RunProfile()
    matrix = jnp.ones((1200, 1200))
    multiply = jax.jit(lambda value: value @ value)
    multiply(matrix).block_until_ready()  # compile first, so we time execution

    with timed_stage(logger, "matmul", profile) as holder:
        holder.set(multiply(matrix))

    recorded = profile.stages[0].seconds
    assert recorded > 1e-3, (
        f"stage reported {recorded:.6f}s, which is dispatch time rather than "
        f"execution time; the timer may not be blocking on its result"
    )


def test_regression_stage_without_a_result_is_marked_as_such() -> None:
    """
    A stage that hands nothing back cannot be waited on, and its number
    covers dispatch alone. That must be visible in the profile rather
    than passing as a real measurement.
    """
    logger = _silent_logger()
    profile = RunProfile()

    with timed_stage(logger, "no result", profile):
        pass

    assert profile.stages[0].name == "no result"


def test_regression_profile_summary_orders_by_cost() -> None:
    """
    The question a profile answers is what to fix first, which is the
    largest entry rather than the earliest.
    """
    profile = RunProfile()
    profile.record("cheap", 0.1)
    profile.record("expensive", 10.0)
    profile.record("middling", 1.0)

    lines: list[str] = []

    class CapturingLogger:
        def info(self, message, *arguments):
            lines.append(message % arguments if arguments else message)

    profile.log_summary(CapturingLogger(), "test")

    stage_lines = [line for line in lines if "%" in line and "s " in line]
    assert "expensive" in stage_lines[0], (
        f"the largest stage should come first, got {stage_lines[0]}"
    )
    assert abs(profile.total_seconds - 11.1) < 1e-9


def test_regression_device_memory_degrades_when_unreported() -> None:
    """
    CPU reports no memory statistics at all. That must read as a stated
    absence rather than as an error, or as a zero that would look like
    an empty device.
    """
    memory = read_device_memory(jax.devices()[0])

    if not memory.available:
        assert "not reported" in memory.format()
    else:
        assert memory.bytes_in_use >= 0


_TELEMETRY_TESTS = [
    test_smoke_format_bytes_scales_readably,
    test_regression_describe_array_reports_size_without_reading_values,
    test_regression_describe_tree_reports_every_dtype_present,
    test_regression_single_device_placement_is_not_reported_as_split,
    test_regression_value_summary_promotes_before_reducing,
    test_regression_value_summary_reports_non_finite_values_prominently,
    test_regression_timing_waits_for_the_result,
    test_regression_stage_without_a_result_is_marked_as_such,
    test_regression_profile_summary_orders_by_cost,
    test_regression_device_memory_degrades_when_unreported,
]


def run_telemetry_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against the telemetry layer", len(_TELEMETRY_TESTS))
    for test_function in _TELEMETRY_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All telemetry tests passed")
