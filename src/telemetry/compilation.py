"""
Separating compilation from execution.

This exists because of a question the stage timings could not answer.
A warm-up run reported the autoencoder decode taking 108 seconds, and
that number is useless on its own: it might be 105 seconds of
compilation and 3 of work, or the reverse, and the two call for
opposite fixes. Compilation is paid once per output shape and can be
cached; execution is paid every image and can only be made faster.

JAX emits an event for every compilation it performs, carrying the
function's name and how long each phase took. Listening to those events
gives the breakdown directly, rather than inferring it by running
something twice and subtracting.

The three phases reported are worth telling apart:

  jaxpr trace     turning Python into JAX's own representation. Grows
                  with how much Python runs, so an unrolled loop pays
                  here and a scan does not.
  lowering        turning that into MLIR. Grows with program size.
  backend compile XLA producing machine code. Usually the largest, and
                  the one a persistent cache eliminates on later runs.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import jax

from .arrays import format_bytes


# Event names JAX emits. Each carries a duration in seconds and the name
# of the function being compiled.
TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
LOWERING_EVENT = "/jax/core/compile/jaxpr_to_mlir_module_duration"
BACKEND_EVENT = "/jax/core/compile/backend_compile_duration"

PHASE_LABELS = {
    TRACE_EVENT: "trace",
    LOWERING_EVENT: "lower",
    BACKEND_EVENT: "backend",
}

# Compilations below this are noise: JAX compiles many tiny helper
# programs for individual operations, and listing them buries the
# handful that matter.
REPORTABLE_THRESHOLD_SECONDS = 0.05


@dataclass
class CompilationRecord:
    """Every phase of compiling one function."""

    function_name: str
    phase_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def total_seconds(self) -> float:
        return sum(self.phase_seconds.values())

    def format(self) -> str:
        phases = " ".join(
            f"{label}={self.phase_seconds[label]:.2f}s"
            for label in ("trace", "lower", "backend")
            if self.phase_seconds.get(label)
        )
        return f"{self.function_name}: {self.total_seconds:.2f}s ({phases})"


class CompilationRecorder:
    """
    Accumulates every compilation JAX performs.

    Registered once for the process, since JAX's listener registration
    has no removal. Recording is therefore always on; what a caller
    controls is when to read and reset it, which is how a per-stage
    breakdown is obtained from a process-wide stream of events.

    The lock matters: JAX may compile on more than one thread, and an
    unlocked accumulation would lose events rather than fail visibly.
    """

    def __init__(self) -> None:
        self._records: dict[str, CompilationRecord] = {}
        self._lock = threading.Lock()
        self._registered = False

    def register(self) -> None:
        """
        Begin listening. Safe to call more than once; only the first
        call registers, since JAX offers no way to unregister and a
        second listener would double every duration.
        """
        if self._registered:
            return
        jax.monitoring.register_event_duration_secs_listener(self._on_event)
        self._registered = True

    def _on_event(self, event_name: str, duration: float, **details) -> None:
        phase = PHASE_LABELS.get(event_name)
        if phase is None:
            return

        function_name = details.get("fun_name", "unknown")
        with self._lock:
            record = self._records.get(function_name)
            if record is None:
                record = CompilationRecord(function_name=function_name)
                self._records[function_name] = record
            record.phase_seconds[phase] += duration

    def take(self) -> list[CompilationRecord]:
        """
        Return everything recorded so far and clear the buffer.

        Taking rather than reading is what allows a per-stage
        attribution: read immediately after a stage and whatever comes
        back was compiled during it.
        """
        with self._lock:
            records = list(self._records.values())
            self._records.clear()
        return records

    @property
    def total_seconds(self) -> float:
        with self._lock:
            return sum(record.total_seconds for record in self._records.values())


_RECORDER = CompilationRecorder()


def start_recording_compilations() -> CompilationRecorder:
    """Begin recording, returning the process-wide recorder."""
    _RECORDER.register()
    return _RECORDER


def log_compilations(logger: logging.Logger, stage_name: str) -> float:
    """
    Report and clear whatever has been compiled since the last call.

    Returns the total seconds spent compiling, so a caller can subtract
    it from a stage's wall time and see what the work itself cost.

    Compilations below the reporting threshold are counted but not
    listed individually, since JAX compiles many small helper programs
    whose names would bury the ones that matter.
    """
    records = _RECORDER.take()
    if not records:
        return 0.0

    total = sum(record.total_seconds for record in records)
    reportable = [
        record for record in records if record.total_seconds >= REPORTABLE_THRESHOLD_SECONDS
    ]
    reportable.sort(key=lambda record: record.total_seconds, reverse=True)

    logger.info(
        "  compiled during %s: %.2fs across %d program(s)",
        stage_name,
        total,
        len(records),
    )
    for record in reportable:
        logger.info("    %s", record.format())

    hidden = len(records) - len(reportable)
    if hidden:
        logger.info(
            "    plus %d program(s) under %.2fs each",
            hidden,
            REPORTABLE_THRESHOLD_SECONDS,
        )

    return total


def describe_compiled_program(
    logger: logging.Logger, label: str, function, *arguments
) -> None:
    """
    Report what a compiled program costs before running it.

    Reports the arithmetic XLA believes the program performs and the
    memory it will need, which together answer whether a slow stage is
    slow because it does a great deal of work or because it is executing
    that work badly. The distinction is not visible from a wall time
    alone.

    Compiling here is not wasted: JAX caches the result, so the
    subsequent real call reuses it.
    """
    try:
        lowered = jax.jit(function).lower(*arguments)
        compiled = lowered.compile()
    except Exception as error:
        logger.info("  %s: could not analyse program (%s)", label, type(error).__name__)
        return

    parts = []

    try:
        cost = compiled.cost_analysis()
        flops = cost.get("flops") if isinstance(cost, dict) else None
        if flops:
            parts.append(f"{flops / 1e9:.1f} GFLOP")
    except Exception:
        pass

    try:
        memory = compiled.memory_analysis()
        parts.append(
            f"temp {format_bytes(memory.temp_size_in_bytes)}, "
            f"args {format_bytes(memory.argument_size_in_bytes)}, "
            f"out {format_bytes(memory.output_size_in_bytes)}"
        )
    except Exception:
        pass

    try:
        parts.append(f"HLO {len(lowered.as_text()) / 1024:.0f} KiB")
    except Exception:
        pass

    if parts:
        logger.info("  %s: %s", label, " | ".join(parts))
