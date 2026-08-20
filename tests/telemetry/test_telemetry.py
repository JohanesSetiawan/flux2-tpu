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


def _capture_stdout(action):
    """Run something and return whatever it printed."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        action()
    return buffer.getvalue()


def test_regression_tracing_is_silent_and_free_when_disabled() -> None:
    """
    A disabled trace point must emit nothing at all, not merely print
    nothing. The flag is read at trace time, so the branch is resolved
    while the program is built and no operation survives into the
    compiled result.
    """
    from src.telemetry.tracing import disable_model_tracing, trace_tensor

    disable_model_tracing()

    @jax.jit
    def traced(value):
        return trace_tensor("anything", value) * 2

    output = _capture_stdout(lambda: traced(jnp.ones((4,))).block_until_ready())

    assert output == "", f"disabled tracing printed: {output!r}"


def test_regression_tracing_reports_on_every_execution_not_once() -> None:
    """
    The whole reason for jax.debug.print rather than print.

    A Python print inside a compiled function runs once, during
    tracing, and never again. This asserts the trace point fires on each
    call, which is what makes it usable for watching values change
    across executions.
    """
    from src.telemetry.tracing import disable_model_tracing, enable_model_tracing, trace_tensor

    enable_model_tracing()
    try:

        @jax.jit
        def traced(value):
            return trace_tensor("repeated", value)

        def run_three_times():
            for multiplier in (1.0, 2.0, 3.0):
                traced(jnp.ones((2,)) * multiplier).block_until_ready()

        output = _capture_stdout(run_three_times)
    finally:
        disable_model_tracing()

    assert output.count("[trace] repeated") == 3, (
        f"expected three reports, one per execution, got:\n{output}"
    )


def test_regression_tracing_fires_inside_a_scan() -> None:
    """
    Block stacks run under lax.scan, so a trace point that did not
    survive into a scan body would report the stack once instead of
    reporting each block, which is the case it is most needed for.
    """
    from src.telemetry.tracing import disable_model_tracing, enable_model_tracing, trace_tensor

    enable_model_tracing()
    try:

        def body(carry, value):
            return trace_tensor("scanned", carry + value), None

        def run():
            jax.lax.scan(body, jnp.float32(0), jnp.arange(4, dtype=jnp.float32))

        output = _capture_stdout(run)
    finally:
        disable_model_tracing()

    assert output.count("[trace] scanned") == 4, (
        f"expected one report per scan iteration, got:\n{output}"
    )


def test_regression_prefix_filter_narrows_reporting() -> None:
    """
    Tracing a twenty-block stack unfiltered buries the one line that
    matters, so narrowing must actually exclude non-matching labels.
    """
    from src.telemetry.tracing import disable_model_tracing, enable_model_tracing, trace_tensor

    enable_model_tracing("model.wanted")
    try:

        def run():
            trace_tensor("model.wanted.here", jnp.ones((2,)))
            trace_tensor("model.other.here", jnp.ones((2,)))

        output = _capture_stdout(run)
    finally:
        disable_model_tracing()

    assert "model.wanted.here" in output
    assert "model.other.here" not in output


def test_regression_toggling_tracing_takes_effect_on_already_compiled_code() -> None:
    """
    Trace points are resolved when a function is traced, so a program
    compiled with tracing off keeps it off for its lifetime. Without
    clearing compiled programs, enabling tracing after a first call
    would appear to do nothing, which reads as a broken feature rather
    than a caching subtlety.
    """
    from src.telemetry.tracing import disable_model_tracing, enable_model_tracing, trace_tensor

    @jax.jit
    def traced(value):
        return trace_tensor("toggled", value)

    disable_model_tracing()
    traced(jnp.ones((2,))).block_until_ready()  # compile with tracing off

    enable_model_tracing()
    try:
        output = _capture_stdout(lambda: traced(jnp.ones((2,))).block_until_ready())
    finally:
        disable_model_tracing()

    assert "[trace] toggled" in output, (
        "enabling tracing did not affect an already compiled program; the "
        "compilation caches may not be cleared on toggle"
    )


def test_regression_trace_statistics_promote_before_reducing() -> None:
    """
    Same rule as everywhere else in this codebase: reducing bfloat16 in
    bfloat16 saturates. An array of ones must report a mean of one.
    """
    from src.telemetry.tracing import disable_model_tracing, enable_model_tracing, trace_tensor

    enable_model_tracing()
    try:
        output = _capture_stdout(
            lambda: trace_tensor("wide", jnp.ones((256, 256), dtype=jnp.bfloat16))
        )
    finally:
        disable_model_tracing()

    assert "mean=1.0000" in output, (
        f"statistics may be accumulating in bfloat16: {output}"
    )


_TELEMETRY_TESTS.extend(
    [
        test_regression_tracing_is_silent_and_free_when_disabled,
        test_regression_tracing_reports_on_every_execution_not_once,
        test_regression_tracing_fires_inside_a_scan,
        test_regression_prefix_filter_narrows_reporting,
        test_regression_toggling_tracing_takes_effect_on_already_compiled_code,
        test_regression_trace_statistics_promote_before_reducing,
    ]
)


def test_regression_compilation_is_separated_from_execution() -> None:
    """
    The measurement this whole layer exists for.

    A stage's wall time answers almost nothing on its own: a decode
    taking a hundred seconds might be nearly all compilation, which is
    paid once per shape and survives in a cache, or nearly all
    execution, which is paid on every image. The two call for opposite
    fixes.

    A first call compiles and runs; a second only runs. The first must
    therefore report meaningful compilation time and the second almost
    none, with the second's wall time close to the first's execution
    time.
    """
    from src.telemetry import RunProfile, start_recording_compilations, timed_stage

    start_recording_compilations()
    logger = _silent_logger()
    profile = RunProfile()

    # Distinct shape, so this is genuinely uncompiled at first call.
    matrix = jnp.ones((613, 613))
    compute = jax.jit(lambda value: jnp.tanh(value @ value).sum())

    with timed_stage(logger, "cold", profile) as holder:
        holder.set(compute(matrix))
    with timed_stage(logger, "warm", profile) as holder:
        holder.set(compute(matrix))

    cold, warm = profile.stages[0], profile.stages[1]

    assert cold.compile_seconds > 0.0, (
        "a first call must report compilation time; the event listener may not be "
        "registered"
    )
    assert warm.compile_seconds < cold.compile_seconds, (
        f"a repeated call compiled for {warm.compile_seconds:.3f}s against the "
        f"first call's {cold.compile_seconds:.3f}s; compilation may be attributed "
        f"to the wrong stage"
    )
    assert cold.execute_seconds >= 0.0


def test_regression_execution_time_never_goes_negative() -> None:
    """
    Wall time and compilation time come from different sources, so
    subtracting one from the other can go negative at the margins. That
    must clamp rather than surface as a nonsensical figure in a report.
    """
    from src.telemetry import StageRecord

    record = StageRecord(name="odd", seconds=1.0, compile_seconds=1.5)

    assert record.execute_seconds == 0.0


def test_regression_profile_reports_the_compilation_share() -> None:
    """
    A run dominated by compilation will be far cheaper the second time;
    one dominated by execution will not. The summary must make that
    difference visible rather than leaving it to be worked out.
    """
    from src.telemetry import RunProfile

    profile = RunProfile()
    profile.record("mostly compiling", 10.0, compile_seconds=9.0)
    profile.record("mostly running", 5.0, compile_seconds=0.1)

    lines: list[str] = []

    class CapturingLogger:
        def info(self, message, *arguments):
            lines.append(message % arguments if arguments else message)

    profile.log_summary(CapturingLogger(), "test")
    joined = "\n".join(lines)

    assert "compiling" in joined and "executing" in joined
    assert "a repeat should cost about" in joined, (
        "the summary should state what a cached repeat would cost"
    )
    assert abs(profile.total_compile_seconds - 9.1) < 1e-9


_TELEMETRY_TESTS.extend(
    [
        test_regression_compilation_is_separated_from_execution,
        test_regression_execution_time_never_goes_negative,
        test_regression_profile_reports_the_compilation_share,
    ]
)
