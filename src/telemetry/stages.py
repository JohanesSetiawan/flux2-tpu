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


@dataclass
class StageRecord:
    """One completed stage."""

    name: str
    seconds: float
    detail: str = ""


@dataclass
class RunProfile:
    """
    Every stage of one run, in order.

    Accumulated rather than only logged, so a summary can name the
    dominant stage at the end. Reading a profile out of interleaved log
    lines is exactly the manual work this exists to remove.
    """

    stages: list[StageRecord] = field(default_factory=list)

    def record(self, name: str, seconds: float, detail: str = "") -> None:
        self.stages.append(StageRecord(name=name, seconds=seconds, detail=detail))

    @property
    def total_seconds(self) -> float:
        return sum(stage.seconds for stage in self.stages)

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
        logger.info("=" * 68)
        logger.info("%s: %.2fs total", title, total)
        logger.info("-" * 68)

        for stage in sorted(self.stages, key=lambda entry: entry.seconds, reverse=True):
            share = 100.0 * stage.seconds / total if total else 0.0
            suffix = f"  {stage.detail}" if stage.detail else ""
            logger.info("  %-34s %8.2fs  %5.1f%%%s", stage.name, stage.seconds, share, suffix)

        logger.info("=" * 68)


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
    logger.info("[stage] %s: starting%s", name, f" ({detail})" if detail else "")
    started = time.perf_counter()

    try:
        yield holder
    finally:
        if holder.assigned:
            _block_on(holder.result)
            qualifier = ""
        else:
            # No result was handed back, so there is nothing to wait for
            # and the figure covers dispatch alone.
            qualifier = " (dispatch only, not waited on)"

        elapsed = time.perf_counter() - started
        logger.info("[stage] %s: %.3fs%s", name, elapsed, qualifier)

        if profile is not None:
            profile.record(name, elapsed, detail)
