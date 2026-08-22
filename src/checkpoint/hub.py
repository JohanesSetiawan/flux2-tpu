"""
Downloading the weight bundle from the Hugging Face Hub.

This module and `restore` are the only places in the package that touch
the network or the filesystem. Both log every stage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

from ..config import CheckpointSourceConfig


VALID_COMPONENT_NAMES = frozenset({"text_encoder", "transformer", "vae"})
PARAMETERS_SUBDIRECTORY_NAME = "params"
TOKENIZER_SUBDIRECTORY_NAME = "tokenizer"
MANIFEST_FILE_NAME = "manifest.json"


def validate_component_name(component_name: str) -> None:
    """Raise a clear error if a component name is not one of the three."""
    if component_name not in VALID_COMPONENT_NAMES:
        raise ValueError(
            f"Unknown component '{component_name}'. "
            f"Expected one of {sorted(VALID_COMPONENT_NAMES)}"
        )


def resolve_huggingface_token(
    logger: logging.Logger, token_environment_variable: str
) -> str | None:
    """
    Resolve a Hugging Face access token, or return None if none is
    configured.

    The checkpoint bundle repository is public, so a token is not
    required to download it. This function therefore never prompts
    interactively and never raises when no token is found: it checks a
    Kaggle Secret, then an environment variable, and otherwise returns
    None so the caller proceeds anonymously.

    A token is still used when one happens to be available, because
    authenticated requests get a higher Hub rate limit than anonymous
    ones. That is a throughput benefit, not a requirement.

    The token value itself is never logged, only which source it came
    from.
    """
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret(token_environment_variable)
        if token:
            logger.info("Resolved Hugging Face token from Kaggle Secrets")
            return token
    except Exception:
        # Not running on Kaggle, or the secret is not configured. Both
        # are ordinary situations for a public repository.
        pass

    token_from_environment = os.environ.get(token_environment_variable)
    if token_from_environment:
        logger.info(
            "Resolved Hugging Face token from the %s environment variable",
            token_environment_variable,
        )
        return token_from_environment

    logger.info(
        "No Hugging Face token configured, proceeding anonymously "
        "(the checkpoint repository is public)"
    )
    return None


def bundle_is_complete(directory: Path) -> bool:
    """
    Report whether a directory already holds a usable bundle.

    Checks for the manifest, the tokenizer, and all three parameter
    components. A partial directory returns False so the caller fetches
    what is missing rather than failing later on an absent component.
    """
    if not (directory / MANIFEST_FILE_NAME).is_file():
        return False
    if not (directory / TOKENIZER_SUBDIRECTORY_NAME).is_dir():
        return False
    return all(
        (directory / PARAMETERS_SUBDIRECTORY_NAME / name).is_dir()
        for name in sorted(VALID_COMPONENT_NAMES)
    )


def _directory_is_writable(directory: Path) -> bool:
    """
    Report whether new files can be created under a directory.

    Tested by attempting a write rather than by inspecting permissions,
    because the case that matters is a read-only mount, where the
    permission bits can look perfectly ordinary.
    """
    probe = directory / ".write_probe"
    try:
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def component_download_patterns(component_names: Iterable[str]) -> list[str]:
    """
    Build Hub file patterns that fetch only the named components, plus
    the manifest and tokenizer files that are small and always needed.

    Downloading a subset matters because the full bundle is roughly
    13 GB, dominated by the transformer and text encoder. Work that
    only touches one component (developing or validating the VAE, for
    instance) should not pay for the other two.
    """
    patterns = [MANIFEST_FILE_NAME, f"{TOKENIZER_SUBDIRECTORY_NAME}/*"]
    for component_name in component_names:
        validate_component_name(component_name)
        patterns.append(f"{PARAMETERS_SUBDIRECTORY_NAME}/{component_name}/*")
    return patterns


def download_bundle(
    source_config: CheckpointSourceConfig,
    logger: logging.Logger,
    token: str | None = None,
    component_names: Iterable[str] | None = None,
) -> Path:
    """
    Download the checkpoint bundle from the Hub into the local cache
    directory.

    Parameters
    ----------
    source_config:
        Repository, revision and cache location.
    logger:
        Receives progress messages.
    token:
        Optional Hub access token. The repository is public, so None is
        valid and results in anonymous requests.
    component_names:
        When given, only these components' parameters are downloaded,
        along with the manifest and tokenizer. When None, the whole
        bundle is downloaded.

    Returns
    -------
    Local filesystem path to the root of the downloaded bundle.
    """
    directory = source_config.local_cache_directory

    # Skip the Hub entirely when everything is already present. This is
    # not merely an optimisation: the directory may be a read-only mount,
    # such as an attached dataset, and the download client writes its own
    # metadata alongside the files it fetches. Attempting a download
    # there fails on directory creation rather than on anything to do
    # with the download itself, which is a confusing way to learn that
    # the weights were sitting there all along.
    if bundle_is_complete(directory):
        logger.info("Using the bundle already present at %s; nothing to download", directory)
        return directory

    if directory.exists() and not _directory_is_writable(directory):
        raise PermissionError(
            f"{directory} is read-only and does not contain a complete bundle. "
            f"Expected {MANIFEST_FILE_NAME}, {TOKENIZER_SUBDIRECTORY_NAME}/, and "
            f"{PARAMETERS_SUBDIRECTORY_NAME}/ with "
            f"{sorted(VALID_COMPONENT_NAMES)}. Either attach a complete bundle or "
            f"point local_cache_directory somewhere writable so the missing parts "
            f"can be fetched."
        )

    allow_patterns = (
        None if component_names is None else component_download_patterns(component_names)
    )

    logger.info(
        "Downloading checkpoint bundle from %s (revision=%s, components=%s)",
        source_config.huggingface_repo_id,
        source_config.huggingface_revision or "default",
        "all" if component_names is None else sorted(component_names),
    )

    local_path = Path(
        snapshot_download(
            repo_id=source_config.huggingface_repo_id,
            revision=source_config.huggingface_revision,
            local_dir=str(source_config.local_cache_directory),
            token=token,
            allow_patterns=allow_patterns,
        )
    )

    logger.info("Checkpoint bundle available at %s", local_path)
    return local_path
