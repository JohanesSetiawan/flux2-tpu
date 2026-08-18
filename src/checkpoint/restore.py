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
