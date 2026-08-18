"""
Logging configuration for the flux2_klein inference package.

Mirrors the converter's logging setup: every message goes to both the
console and a persistent text file, so a full run (from checkpoint
download through image generation) is traceable end to end without
relying on notebook cell output alone, which is lost when a session
ends.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_MESSAGE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_file_path: Path, logger_name: str) -> logging.Logger:
    """
    Create and return a logger that writes to both stdout and a text
    file. Calling this twice with the same logger_name does not
    duplicate handlers, so it is safe to call from a notebook cell that
    might be re-run.
    """
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_MESSAGE_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=str(log_file_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
