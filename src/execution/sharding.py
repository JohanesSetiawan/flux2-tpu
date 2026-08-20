"""
Splitting parameters across devices.

Written so that a single device and a multi-device pod run the same
code path. The mesh is built from whatever devices are visible, and a
mesh of one is a valid mesh: sharding across it is a no-op rather than
a special case. Forking the implementation per platform would double
the surface that needs testing while guaranteeing the rarely-used path
rots.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# Name of the single mesh axis. One axis is enough for this model:
# parameters are either replicated or split along their widest
# dimension, and there is no second dimension worth splitting at this
# scale.
MESH_AXIS_NAME = "devices"


def build_device_mesh(logger: logging.Logger) -> Mesh:
    """
    Build a one dimensional mesh over every visible device.

    A mesh of one device is returned unremarkably, which is what lets
    single-chip and pod configurations share this code.
    """
    devices = jax.devices()
    logger.info("Building a mesh over %d device(s): %s", len(devices), devices[0].platform)
    return Mesh(devices, axis_names=(MESH_AXIS_NAME,))


def replicate_parameters(parameters: dict, mesh: Mesh) -> dict:
    """
    Place a full copy of every parameter on every device.

    Replication is the right default for anything small or used once,
    and for parameters whose shapes do not divide evenly across the
    mesh. It costs memory proportional to the device count but needs no
    collective operations at all.
    """
    sharding = NamedSharding(mesh, PartitionSpec())
    return jax.tree_util.tree_map(
        lambda array: jax.device_put(array, sharding), parameters
    )


# Which parameter groups of each component are large enough to be worth
# splitting across devices, rather than copied to every one.
#
# The distinction is not cosmetic and getting it wrong is expensive.
# Replication does not divide a component's memory across a pod; it
# multiplies it by the device count. An earlier version replicated the
# whole text encoder, 5.80 GiB, onto all eight chips of a v5e-8. That
# is 5.80 GiB resident per chip for a component used once per prompt,
# and 46 GiB of host transfer during load, and it exhausted the device
# mapping before a single image was generated.
#
# Splitting its layer stack instead brings that to 0.63 GiB per chip.
# The embedding table stays replicated: it is read by gather rather
# than matrix multiply, and splitting a lookup table across devices
# would make every lookup a collective.
SPLITTABLE_GROUPS_BY_COMPONENT = {
    "transformer": frozenset({"double_blocks", "single_blocks"}),
    "text_encoder": frozenset({"layers"}),
    # The autoencoder is 189 MiB. Replicating it costs little and avoids
    # collectives in a decoder that is already the heaviest stage.
    "vae": frozenset(),
}


def place_component(
    parameters: dict,
    component_name: str,
    mesh: Mesh,
    logger: logging.Logger,
) -> dict:
    """
    Place one component on the mesh, splitting the groups worth
    splitting and replicating the rest.

    Every component must be placed. Parameters that never touch the
    mesh stay wherever restore left them, which is the first device, so
    a component omitted here runs on one chip of a pod while the others
    idle. That happened to the autoencoder for an entire phase before it
    was noticed.

    Parameters
    ----------
    parameters:
        The component's parameter tree, keyed by group.
    component_name:
        Used to look up which groups are splittable. An unknown name
        replicates everything, which is the safe default: correct, and
        merely uses more memory than necessary.
    """
    splittable = SPLITTABLE_GROUPS_BY_COMPONENT.get(component_name, frozenset())

    if component_name not in SPLITTABLE_GROUPS_BY_COMPONENT:
        logger.info(
            "  %s has no placement policy; replicating every group",
            component_name,
        )

    placed = {}
    for group_name, group in parameters.items():
        if group_name in splittable:
            placed[group_name] = shard_stacked_blocks(group, mesh, logger)
        else:
            placed[group_name] = replicate_parameters(group, mesh)

    return placed


def shard_stacked_blocks(
    parameters: dict, mesh: Mesh, logger: logging.Logger
) -> dict:
    """
    Split each stacked block tensor along its widest non-block axis.

    Repeated-block parameters carry a leading axis over blocks, which
    must not be split: every device runs every block, so splitting there
    would break the scan. The axis worth splitting is the widest of the
    remaining ones, which for a projection is its output width.

    A tensor whose chosen axis does not divide evenly by the device
    count is replicated instead of being padded. Padding would work but
    introduces a shape that no longer matches the checkpoint, and the
    parameters this affects are small enough that replicating them costs
    little.
    """
    device_count = mesh.devices.size
    replicated = NamedSharding(mesh, PartitionSpec())

    def shard_one(array: jnp.ndarray) -> jnp.ndarray:
        if array.ndim < 2 or device_count == 1:
            return jax.device_put(array, replicated)

        # Axis zero indexes blocks; consider only the rest.
        candidate_axis = 1 + int(jnp.argmax(jnp.asarray(array.shape[1:])))
        if array.shape[candidate_axis] % device_count != 0:
            return jax.device_put(array, replicated)

        partition = [None] * array.ndim
        partition[candidate_axis] = MESH_AXIS_NAME
        return jax.device_put(array, NamedSharding(mesh, PartitionSpec(*partition)))

    sharded = jax.tree_util.tree_map(shard_one, parameters)
    logger.info("Sharded stacked block parameters across %d device(s)", device_count)
    return sharded
