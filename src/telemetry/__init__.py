"""
Instrumentation for a generation run.

Answers three questions a log otherwise leaves to manual arithmetic:
where the time went, where the tensors live, and how much accelerator
memory is in use.

Two constraints shape everything here, and both are worth knowing
before extending it:

Timing must block. JAX dispatch is asynchronous, so a timer that does
not wait for its result measures queueing rather than work, and the two
differ by orders of magnitude.

Instrumentation cannot go inside compiled code. A log statement inside
a jit region runs once during tracing and never again, so everything
here sits at stage boundaries on the host.
"""

from .arrays import (
    ArrayDescription,
    TreeDescription,
    describe_array,
    describe_tree,
    format_bytes,
    summarise_values,
)
from .devices import DeviceMemory, log_memory_snapshot, log_platform, read_device_memory
from .stages import RunProfile, StageRecord, timed_stage
from .tracing import (
    disable_model_tracing,
    enable_model_tracing,
    is_tracing_enabled,
    trace_finite,
    trace_parameters,
    trace_tensor,
)

__all__ = [
    "ArrayDescription",
    "DeviceMemory",
    "RunProfile",
    "StageRecord",
    "TreeDescription",
    "describe_array",
    "disable_model_tracing",
    "enable_model_tracing",
    "is_tracing_enabled",
    "trace_finite",
    "trace_parameters",
    "trace_tensor",
    "describe_tree",
    "format_bytes",
    "log_memory_snapshot",
    "log_platform",
    "read_device_memory",
    "summarise_values",
    "timed_stage",
]
