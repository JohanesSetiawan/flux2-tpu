"""
Restoring parameter pytrees from a downloaded bundle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from jax.sharding import Sharding, SingleDeviceSharding

from .hub import PARAMETERS_SUBDIRECTORY_NAME, validate_component_name


def restore_component(
    bundle_path: Path,
    component_name: str,
    logger: logging.Logger,
) -> dict:
    """
    Restore a single component's parameter pytree from an already
    downloaded bundle.

    Parameters
    ----------
    bundle_path:
        Local path returned by download_bundle.
    component_name:
        One of "text_encoder", "transformer", "vae".
    """
    validate_component_name(component_name)

    component_directory = bundle_path / PARAMETERS_SUBDIRECTORY_NAME / component_name
    logger.info("Restoring %s parameters from %s", component_name, component_directory)

    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(component_directory)

    logger.info("Restored %s parameters", component_name)
    return params


def component_metadata(bundle_path: Path, component_name: str, logger: logging.Logger) -> dict:
    """
    Read a component's structure without materialising its arrays.

    Returns a pytree of the same shape as the parameters, whose leaves
    carry shape and dtype rather than values. This makes structural
    checks cheap and, more importantly, possible at all in environments
    with less memory than the component's size: the text encoder alone
    is nearly six gigabytes, so verifying its layout by restoring it
    would fail on any machine that cannot hold it.

    Use this whenever the question is "is the checkpoint shaped the way
    the code expects"; use restore_component only when the values
    themselves are needed.
    """
    validate_component_name(component_name)

    component_directory = bundle_path / PARAMETERS_SUBDIRECTORY_NAME / component_name
    logger.info("Reading %s metadata from %s", component_name, component_directory)

    checkpointer = ocp.StandardCheckpointer()
    metadata = checkpointer.metadata(component_directory)

    # Orbax wraps the parameter tree in a step-level record; the tree
    # itself is what callers care about.
    return metadata.item_metadata if hasattr(metadata, "item_metadata") else metadata


def _as_sharding(destination) -> Sharding:
    """
    Accept either a device or a sharding, and return a sharding.

    Placement is naturally expressed as a device for a single chip and
    as a sharding for a mesh, and a caller should not have to know which
    form the restore path wants. ShapeDtypeStruct accepts only the
    latter, so a bare device is wrapped here rather than at every call
    site.
    """
    if isinstance(destination, Sharding):
        return destination
    return SingleDeviceSharding(destination)


def restore_component_with_sharding(
    bundle_path: Path,
    component_name: str,
    sharding,
    logger: logging.Logger,
) -> dict:
    """
    Restore a component directly onto its final devices.

    The obvious sequence, restore and then place, moves every byte
    twice: once from disk to wherever the default device is, and again
    to where it belongs. For the diffusion transformer that is a
    redundant 7.2 GB copy on a machine whose restore already dominates
    load time.

    Orbax can place arrays as it reads them if it is told the target
    layout up front, which is what this does: read the checkpoint's
    metadata for shapes and dtypes, attach the requested sharding, and
    restore against that.

    Falls back to the two-step path if the checkpoint's metadata cannot
    be read. That is deliberate rather than defensive noise: a fallback
    that loads correctly but slowly is much better than a load that
    fails, and the reason is logged so a silent regression to the slow
    path is still visible.
    """
    validate_component_name(component_name)
    component_directory = bundle_path / PARAMETERS_SUBDIRECTORY_NAME / component_name

    logger.info(
        "Restoring %s parameters from %s, placing directly on %s",
        component_name,
        component_directory,
        sharding,
    )

    checkpointer = ocp.StandardCheckpointer()
    sharding = _as_sharding(sharding)

    try:
        metadata = checkpointer.metadata(component_directory)
        tree = metadata.item_metadata if hasattr(metadata, "item_metadata") else metadata
        target = jax.tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=sharding
            ),
            tree,
        )
    except Exception as error:
        logger.info(
            "  could not read %s metadata (%s); restoring without placement, "
            "which copies every array twice",
            component_name,
            type(error).__name__,
        )
        return checkpointer.restore(component_directory)

    parameters = checkpointer.restore(component_directory, target)
    logger.info("Restored %s parameters", component_name)
    return parameters
