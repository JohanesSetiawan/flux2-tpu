"""
Where each component's parameters live between uses.

The problem this solves is specific. At full precision the three
components together need roughly 14 GB of parameters, against 16 GB on
a single-chip accelerator, and the autoencoder's decode transiently
needs several more. They do not all fit at once.

They also do not all need to. Of the three, only the transformer runs
more than once per image: the text encoder runs once per prompt and the
decoder once per generation, while the transformer runs once per
sampling step. Keeping the transformer resident and moving the text
encoder in only when a prompt changes trades a host transfer for the
memory headroom that makes the whole thing fit, and pays that transfer
on the least frequent operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax

from ..config import MemoryResidencyStrategy


# Components that run once per generation or less are the ones worth
# evicting. The transformer is excluded deliberately: it runs once per
# sampling step, so evicting it would pay a transfer per step.
EVICTABLE_COMPONENTS = frozenset({"text_encoder"})


@dataclass(frozen=True)
class ComponentResidency:
    """
    Where one component's parameters are held between uses.

    `resident` means the parameters stay in accelerator memory for the
    session. Otherwise they live in host memory and are copied in for
    each use, then released.
    """

    component_name: str
    resident: bool


def plan_component_residency(
    strategy: MemoryResidencyStrategy,
    component_names: tuple[str, ...],
) -> tuple[ComponentResidency, ...]:
    """
    Decide which components stay in accelerator memory.

    Under FULLY_RESIDENT everything stays. Under SWAPPED only the
    components in EVICTABLE_COMPONENTS are moved out, which is a
    deliberately narrow set: evicting the transformer would cost a
    transfer on every sampling step, and evicting the decoder would save
    little since it is small.

    AUTO must be resolved before reaching here. It is rejected rather
    than guessed at, because the device count needed to resolve it is
    not available to this function and silently picking one would hide a
    caller's mistake.
    """
    if strategy is MemoryResidencyStrategy.AUTO:
        raise ValueError(
            "Residency strategy must be resolved before planning; call "
            "resolve_residency_strategy first"
        )

    fully_resident = strategy is MemoryResidencyStrategy.FULLY_RESIDENT

    return tuple(
        ComponentResidency(
            component_name=name,
            resident=fully_resident or name not in EVICTABLE_COMPONENTS,
        )
        for name in component_names
    )


def _host_device():
    """
    The CPU device parameters are parked on when evicted.

    Resolved at call time rather than at import, since the device list
    is not populated until the backend initialises.
    """
    return jax.devices("cpu")[0]


def evict_to_host(parameters: dict, logger: logging.Logger, component_name: str) -> dict:
    """
    Move a component's parameters to host memory.

    The returned tree replaces the original; the caller should drop its
    reference to the previous one, or the accelerator copy stays alive
    and no memory is actually freed. That is the most common way this
    optimisation silently does nothing.
    """
    logger.info("Evicting %s parameters to host memory", component_name)
    host = _host_device()
    return jax.tree_util.tree_map(lambda array: jax.device_put(array, host), parameters)


def move_to_accelerator(
    parameters: dict,
    logger: logging.Logger,
    component_name: str,
    sharding=None,
) -> dict:
    """
    Copy a component's parameters back into accelerator memory.

    The destination must be named explicitly. `jax.device_put(array)`
    with no destination is a no-op for an array already committed to a
    device, so an earlier version of this function silently left the
    evicted parameters on the host. Nothing failed at that point: the
    component ran on the host instead, and the mismatch only surfaced
    later, when its output met an accelerator-resident tensor inside a
    jit boundary.

    Parameters
    ----------
    sharding:
        Where to place the parameters. Defaults to the first accelerator
        device. Pass a mesh sharding to place them across a pod instead.
    """
    destination = sharding if sharding is not None else jax.devices()[0]
    logger.info(
        "Moving %s parameters to %s", component_name, destination
    )
    return jax.tree_util.tree_map(
        lambda array: jax.device_put(array, destination), parameters
    )
