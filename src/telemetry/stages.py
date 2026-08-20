"""
Timing the stages of a run.

One detail governs everything here: JAX dispatch is asynchronous. A
timer that starts before a call and stops after it measures how long
the call took to queue, not how long the work took, and the two differ
by orders of magnitude. Measured on this project's own code, a matrix
multiply that takes 186 milliseconds reports 0.14 if the timer does not
wait for it.

So every stage timed here blocks on its result before stopping the
clock. That does change behaviour: it removes the overlap JAX would
otherwise get between stages. The trade is deliberate and worth
stating, because a profile that cannot be trusted is worse than no
profile, and the overlap lost between a handful of coarse stages is
small next to the stages themselves.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import jax

from .compilation import discard_pending_compilations, log_compilations


@dataclass
class StageRecord:
    """
    One completed stage, with compilation separated from execution.

    The split is the point. A stage's wall time answers almost nothing
    on its own: a decode taking 108 seconds might be 105 of compilation
    and 3 of work, or the reverse, and those call for opposite fixes.
    Compilation is paid once per output shape and survives in a cache;
    execution is paid on every image.
    """

    name: str
    seconds: float
    detail: str = ""
    compile_seconds: float = 0.0

    @property
    def execute_seconds(self) -> float:
        """
        Wall time not spent compiling.

        Clamped at zero because the two measurements come from different
        sources: wall time from a clock around the stage, compilation
        from JAX's own events, which may include work attributed
        slightly outside the window.
        """
        return max(0.0, self.seconds - self.compile_seconds)


@dataclass
class RunProfile:
    """
    Every stage of one run, in order.

    Accumulated rather than only logged, so a summary can name the
    dominant stage at the end. Reading a profile out of interleaved log
    lines is exactly the manual work this exists to remove.
    """

    stages: list[StageRecord] = field(default_factory=list)

    def record(
        self, name: str, seconds: float, detail: str = "", compile_seconds: float = 0.0
    ) -> None:
        self.stages.append(
            StageRecord(
                name=name,
                seconds=seconds,
                detail=detail,
                compile_seconds=compile_seconds,
            )
        )

    @property
    def total_seconds(self) -> float:
        return sum(stage.seconds for stage in self.stages)

    @property
    def total_compile_seconds(self) -> float:
        return sum(stage.compile_seconds for stage in self.stages)

    def log_summary(self, logger: logging.Logger, title: str) -> None:
        """
        Write the profile as a table, sorted by cost.

        Sorted rather than chronological on purpose: the question a
        profile answers is what to fix first, and that is the largest
        entry, not the earliest.
        """
        if not self.stages:
            return

        total = self.total_seconds
        compile_total = self.total_compile_seconds

        logger.info("=" * 78)
        logger.info(
            "%s: %.2fs total (%.2fs compiling, %.2fs executing)",
            title,
            total,
            compile_total,
            total - compile_total,
        )
        logger.info("-" * 78)
        logger.info(
            "  %-28s %9s %9s %7s  %s", "stage", "total", "compile", "share", "detail"
        )

        for stage in sorted(self.stages, key=lambda entry: entry.seconds, reverse=True):
            share = 100.0 * stage.seconds / total if total else 0.0
            logger.info(
                "  %-28s %8.2fs %8.2fs %6.1f%%  %s",
                stage.name,
                stage.seconds,
                stage.compile_seconds,
                share,
                stage.detail,
            )

        logger.info("-" * 78)
        if compile_total > 0.0:
            # The distinction a reader most needs: compilation is paid
            # once per shape and survives in a persistent cache, so a
            # run dominated by it will be far cheaper the second time,
            # while one dominated by execution will not.
            compile_share = 100.0 * compile_total / total if total else 0.0
            logger.info(
                "  compilation is %.0f%% of this run and is cached; "
                "a repeat should cost about %.2fs",
                compile_share,
                total - compile_total,
            )
        logger.info("=" * 78)


def _block_on(result):
    """
    Wait for every array in a result to be computed.

    Accepts a single array, a tuple, or a pytree, since stages return
    all three. Anything without arrays passes through untouched.
    """
    for leaf in jax.tree_util.tree_leaves(result):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return result


@contextmanager
def timed_stage(
    logger: logging.Logger,
    name: str,
    profile: RunProfile | None = None,
    detail: str = "",
):
    """
    Time a stage, blocking on whatever it produces.

    The stage's result is reported back through the yielded holder:
    assign to `holder.result` inside the block, and the timer will wait
    for it before stopping. A stage that assigns nothing is still timed,
    but only its dispatch, which is stated in the log so the number is
    not mistaken for real work.
    """

    class ResultHolder:
        result = None
        assigned = False

        def set(self, value):
            self.result = value
            self.assigned = True
            return value

    holder = ResultHolder()

    # Compilation events are recorded process-wide and read at the end
    # of each stage, so anything still pending belongs to whatever ran
    # before this stage started. Left in place it would be attributed
    # here, which is how a stage came to report more compilation time
    # than wall time: an impossibility that made the whole profile
    # suspect.
    discard_pending_compilations()

    logger.info("[stage] %s: starting%s", name, f" ({detail})" if detail else "")
    started = time.perf_counter()

    try:
        yield holder
    finally:
        if holder.assigned:
            _block_on(holder.result)
            qualifier = ""
        else:
            # No result was handed back, so this timer could not wait for
            # one. The stage may still have waited internally, which is
            # why the wording says the timer did not wait rather than
            # claiming the work was never awaited: an earlier version
            # read as the latter and misdescribed stages that block on
            # their own.
            qualifier = " (timer did not wait; stage may have blocked internally)"

        elapsed = time.perf_counter() - started

        # Read compilation events now, before anything else compiles, so
        # whatever JAX reports is attributable to this stage. Paired
        # with the discard at the start, this bounds attribution to the
        # stage's own window on both sides.
        compile_seconds = log_compilations(logger, name)

        # Compilation is measured by JAX's own clock and the stage by
        # this one, so at the margins the reported compilation can
        # slightly exceed the measured wall time. Clamping keeps the
        # reported split arithmetically consistent rather than showing
        # a negative execution time.
        compile_seconds = min(compile_seconds, elapsed)

        if compile_seconds > 0.0:
            logger.info(
                "[stage] %s: %.3fs total, %.3fs compiling, %.3fs executing%s",
                name,
                elapsed,
                compile_seconds,
                max(0.0, elapsed - compile_seconds),
                qualifier,
            )
        else:
            logger.info("[stage] %s: %.3fs%s", name, elapsed, qualifier)

        if profile is not None:
            profile.record(name, elapsed, detail, compile_seconds)
