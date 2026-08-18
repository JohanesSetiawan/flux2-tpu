"""
Checkpoint download and restore.

This module is the only place in the inference package that talks to
the network or reads the checkpoint bundle from disk. It downloads the
bundle produced by the conversion pipeline (a manifest, three Orbax
parameter directories, and a tokenizer folder) from the Hub, and
restores individual components on request.

The bundle repository is private, so downloading it requires a Hugging
Face access token even for read access. Token resolution follows the
same layered approach used when the bundle was uploaded: a Kaggle
Secret first, then an environment variable, then an interactive prompt.
The token itself is never logged.
"""

from __future__ import annotations

import getpass
import logging
import os
from pathlib import Path

import orbax.checkpoint as ocp
from huggingface_hub import snapshot_download

from .config import CheckpointSourceConfig


VALID_COMPONENT_NAMES = frozenset({"text_encoder", "transformer", "vae"})
PARAMETERS_SUBDIRECTORY_NAME = "params"


def resolve_huggingface_token(logger: logging.Logger, token_environment_variable: str) -> str:
    """
    Resolve a Hugging Face access token from the first available
    source: a Kaggle Secret, then an environment variable, then an
    interactive prompt. The resolved token is never logged, only which
    source it came from.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(token_environment_variable)
        if token:
            logger.info("Resolved Hugging Face token from Kaggle Secrets")
            return token
    except Exception:
        pass

    token_from_environment = os.environ.get(token_environment_variable)
    if token_from_environment:
        logger.info(
            "Resolved Hugging Face token from the %s environment variable",
            token_environment_variable,
        )
        return token_from_environment

    logger.info("No Hugging Face token found in Kaggle Secrets or environment, prompting interactively")
    return getpass.getpass("Enter your Hugging Face access token (read access to the checkpoint repo): ")


def download_bundle(
    source_config: CheckpointSourceConfig,
    logger: logging.Logger,
    token: str | None = None,
) -> Path:
    """
    Download the full checkpoint bundle (manifest, three parameter
    directories, tokenizer files) from the Hub into the local cache
    directory.

    Returns
    -------
    Local filesystem path to the root of the downloaded bundle.
    """
    logger.info(
        "Downloading checkpoint bundle from %s (revision=%s)",
        source_config.huggingface_repo_id,
        source_config.huggingface_revision or "default",
    )

    local_path = Path(
        snapshot_download(
            repo_id=source_config.huggingface_repo_id,
            revision=source_config.huggingface_revision,
            local_dir=str(source_config.local_cache_directory),
            token=token,
        )
    )

    logger.info("Checkpoint bundle available at %s", local_path)
    return local_path


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
    if component_name not in VALID_COMPONENT_NAMES:
        raise ValueError(
            f"Unknown component '{component_name}'. Expected one of {sorted(VALID_COMPONENT_NAMES)}"
        )

    component_directory = bundle_path / PARAMETERS_SUBDIRECTORY_NAME / component_name
    logger.info("Restoring %s parameters from %s", component_name, component_directory)

    checkpointer = ocp.StandardCheckpointer()
    params = checkpointer.restore(component_directory)

    logger.info("Restored %s parameters", component_name)
    return params
