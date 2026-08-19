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
