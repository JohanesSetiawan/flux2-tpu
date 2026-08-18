"""
Single entry point for the whole test suite.

Runs every suite in dependency order (configuration first, then the
primitives built on it) and writes a complete, timestamped log to a
text file as well as the console. Every suite added to the codebase
should be registered in TEST_SUITES below so that one command
exercises everything.

Exits with a non-zero status on the first failure, so it can be used
directly as a gate in a pre-commit hook or CI step.

Usage:
    python -m tests.run_all_tests
    python -m tests.run_all_tests --log-file path/to/log.txt
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from flux2_klein.logging_setup import configure_logging
from tests.test_checkpoint import run_checkpoint_tests
from tests.test_config import run_config_tests
from tests.test_layers import run_layer_tests
from tests.test_vae_blocks import run_vae_block_tests


DEFAULT_LOG_FILE_PATH = Path("test_run_log.txt")
TEST_RUNNER_LOGGER_NAME = "flux2_klein.tests"

# Ordered so that a failure appears in the most foundational layer
# first: a broken config makes every later suite's failure a
# consequence rather than a cause.
TEST_SUITES = (
    ("config", run_config_tests),
    ("checkpoint", run_checkpoint_tests),
    ("layers", run_layer_tests),
    ("vae_blocks", run_vae_block_tests),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full flux2_klein test suite.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE_PATH,
        help="Path of the text file the full run log is written to.",
    )
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, TEST_RUNNER_LOGGER_NAME)
    logger.info("Starting full test run, writing log to %s", arguments.log_file)

    for suite_name, suite_runner in TEST_SUITES:
        logger.info("--- Test suite: %s ---", suite_name)
        try:
            suite_runner(logger)
        except Exception:
            logger.error("Test suite '%s' FAILED", suite_name)
            logger.error("%s", traceback.format_exc())
            logger.error("Aborting: no further suites will run")
            return 1

    logger.info("Full test run completed, every suite passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
