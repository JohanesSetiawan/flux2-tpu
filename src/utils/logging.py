"""
Logging configuration.

Every message goes to both the console and a persistent text file, so a
full run remains traceable after a notebook session ends and its cell
output is lost.

Logging belongs to orchestration, never to numeric code. A Python log
statement inside a jax.jit-compiled function executes once during
tracing and never again during the many actual invocations, so it
appears to report per-call behaviour while reporting a single trace.
That is worse than no logging at all.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_MESSAGE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_file_path: Path, logger_name: str) -> logging.Logger:
    """
    Create a logger that writes to both stdout and a text file.

    Calling this twice with the same logger_name does not duplicate
    handlers, so it is safe to call from a notebook cell that may be
    re-run.
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
