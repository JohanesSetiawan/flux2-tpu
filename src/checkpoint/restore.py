"""
Restoring parameter pytrees from a downloaded bundle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import orbax.checkpoint as ocp

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
