"""
Tracing tensors from inside the model.

The stage timing in `stages.py` reports what happens between the model's
components. This reports what happens inside them: the value of every
intermediate tensor, at every layer, on every execution.

Why a plain print does not work here
------------------------------------
Numeric code in this package runs inside jax.jit. A Python `print`
placed there executes once, while the function is being traced into a
compiled program, and never again during the thousands of times that
program actually runs. It appears to report per-call behaviour while
reporting a single trace, which is worse than silence.

`jax.debug.print` is the mechanism designed for this. It emits a real
operation into the compiled program that calls back to the host on every
execution, and it works inside `lax.scan`, so a scanned block stack
reports each block rather than one.

Why it is off by default
------------------------
Every trace point is a host callback. Callbacks serialise the program
around them: the accelerator must stop and wait for the host, which
removes the pipelining that makes a compiled program fast in the first
place. With tracing on, a generation is expected to be several times
slower, and inside a scanned twenty-block stack a single trace point
becomes twenty callbacks per step.

That cost is worth paying when a number is wrong and worth nothing when
it is not, so this is a switch to turn on for a diagnosis rather than a
setting to leave enabled.

Why the switch is module-level
------------------------------
Tracing is cross-cutting: it concerns every numeric function regardless
of which component it belongs to. Threading a flag through the thirty
or so functions that would use it would obscure their signatures for a
facility used occasionally and deliberately. The switch is therefore
global, like a logging level, with an explicit API rather than a bare
variable.

Because the flag is read at trace time, a disabled trace point emits no
operation at all: the branch is resolved while the program is being
built, so there is nothing left in the compiled result. Disabled tracing
is genuinely free rather than merely cheap.

That same property has a consequence worth knowing: a program already
compiled with tracing off will not start reporting when tracing is
turned on, because it is not retraced. Both switches therefore clear
JAX's compilation caches, so the change takes effect on the next call.
The cost is a recompilation, which is the correct trade for a facility
that would otherwise appear silently broken.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# Statistics are computed in at least float32 before being reported. The
# same rule that applies to normalization and attention applies here:
# reducing bfloat16 in bfloat16 saturates, and a trace that reports an
# array of ones as having a mean near zero would send a reader hunting
# for a bug in the traced code rather than in the trace.
MINIMUM_STATISTIC_DTYPE = jnp.float32


@dataclass
class _TracingState:
    """
    Whether tracing is active, and which labels it applies to.

    A prefix filter matters more than it looks. Tracing everything in a
    twenty-block stack produces hundreds of lines per step, in which the
    one that matters is invisible. Narrowing to a prefix turns the
    facility from overwhelming into usable.
    """

    enabled: bool = False
    label_prefix: str = ""


_STATE = _TracingState()


def _clear_compiled_programs() -> None:
    """
    Discard compiled programs so the new tracing setting takes effect.

    Trace points are resolved when a function is traced, so a program
    compiled under one setting keeps that setting for its lifetime.
    Without this, enabling tracing after a first call would appear to do
    nothing at all.
    """
    jax.clear_caches()


def enable_model_tracing(label_prefix: str = "") -> None:
    """
    Turn on in-model tracing, optionally narrowed to a prefix.

    Parameters
    ----------
    label_prefix:
        Only trace points whose label starts with this are reported. An
        empty prefix reports everything, which is rarely what is wanted
        beyond a first look.

    Examples
    --------
    Narrow to one component, then to one operation within it:

        enable_model_tracing("vae")
        enable_model_tracing("vae.decoder.mid")
    """
    _STATE.enabled = True
    _STATE.label_prefix = label_prefix
    _clear_compiled_programs()


def disable_model_tracing() -> None:
    """Turn tracing off, returning the compiled programs to full speed."""
    _STATE.enabled = False
    _STATE.label_prefix = ""
    _clear_compiled_programs()


def is_tracing_enabled(label: str = "") -> bool:
    """
    Report whether a given label would be traced.

    Read at trace time, so a caller can skip building anything a
    disabled trace point would have needed.
    """
    return _STATE.enabled and label.startswith(_STATE.label_prefix)


def trace_tensor(label: str, tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Report a tensor's statistics from inside compiled code.

    Returns the tensor unchanged, so a trace point can be inserted into
    an expression without restructuring the surrounding code:

        activations = trace_tensor("block.attention", attended)

    Shape and dtype are printed from the trace-time metadata rather than
    computed at run time, since they are already known and printing them
    as values would send them through the callback needlessly.

    The statistics reported are deliberately few. A minimum, maximum,
    mean and standard deviation are enough to recognise the failures
    that matter, an activation that has collapsed to zero or exploded,
    and each additional statistic is another reduction over a tensor
    that may hold millions of elements.
    """
    if not is_tracing_enabled(label):
        return tensor

    statistic_dtype = jnp.promote_types(tensor.dtype, MINIMUM_STATISTIC_DTYPE)
    promoted = tensor.astype(statistic_dtype)

    jax.debug.print(
        "[trace] {label} shape={shape} dtype={dtype} "
        "min={minimum:.4f} max={maximum:.4f} mean={mean:.4f} std={deviation:.4f}",
        label=label,
        shape=str(tuple(tensor.shape)),
        dtype=str(tensor.dtype),
        minimum=jnp.min(promoted),
        maximum=jnp.max(promoted),
        mean=jnp.mean(promoted),
        deviation=jnp.std(promoted),
    )
    return tensor


def trace_finite(label: str, tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Report only whether a tensor contains non-finite values.

    Cheaper than a full trace, since it is one reduction rather than
    four, and it answers the single most useful question when output has
    gone wrong: which stage first produced something that is not a
    number. Use this across many points, then a full trace at the one
    that fires.
    """
    if not is_tracing_enabled(label):
        return tensor

    jax.debug.print(
        "[trace] {label} non_finite={count}",
        label=label,
        count=jnp.sum(~jnp.isfinite(tensor.astype(MINIMUM_STATISTIC_DTYPE))),
    )
    return tensor


def trace_parameters(label: str, parameters: dict) -> None:
    """
    Report a parameter group's contents at trace time.

    Weights do not change between executions, so unlike activations they
    need no run-time callback: this prints once, while the program is
    being built, which is exactly when the information is wanted and
    costs nothing in the compiled result.
    """
    if not is_tracing_enabled(label):
        return

    for name in sorted(parameters):
        value = parameters[name]
        print(f"[trace] {label}.{name} shape={tuple(value.shape)} dtype={value.dtype}")
