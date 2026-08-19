"""Persistent compilation cache."""

from __future__ import annotations

import logging
from pathlib import Path

import jax

from ..config import ExecutionConfig


def configure_compilation_cache(config: ExecutionConfig, logger: logging.Logger) -> bool:
    """
    Point XLA's persistent compilation cache at a directory, if one is
    configured.

    Returns whether the cache was enabled, so a caller can log or assert
    on it rather than guessing.

    This must run before the first compilation to have any effect;
    compiling first and enabling the cache afterwards silently does
    nothing. The pipeline calls it during setup for that reason.
    """
    directory = config.compilation_cache_directory
    if directory is None:
        logger.info("Persistent compilation cache disabled; no directory configured")
        return False

    directory.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        config.compilation_cache_minimum_seconds,
    )

    logger.info(
        "Persistent compilation cache enabled at %s, caching compilations over %.1fs",
        directory,
        config.compilation_cache_minimum_seconds,
    )
    return True
