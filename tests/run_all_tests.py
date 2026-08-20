"""
Single entry point for the whole unit test suite.

Runs every suite in dependency order (configuration first, then the
primitives built on it, then composites, then complete models) and
writes a timestamped log to a text file as well as the console. A
failure appears in the most foundational layer first, so the cause is
distinguishable from its consequences.

Every suite added to the codebase should be registered in TEST_SUITES
below, so that one command exercises everything. Integration tests are
deliberately excluded: they need network access and, for parity
testing, PyTorch. See tests/integration/README.md.

Exits non-zero on the first failure, so it can serve directly as a
pre-commit or CI gate.

Usage:
    python -m tests.run_all_tests
    python -m tests.run_all_tests --log-file path/to/log.txt

Regression suites compare against float64 oracles and therefore need
64-bit mode:
    JAX_ENABLE_X64=1 python -m tests.run_all_tests
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from src.utils import configure_logging
from tests.blocks.test_blocks import run_vae_block_tests
from tests.blocks.test_text_encoder_blocks import run_text_encoder_block_tests
from tests.checkpoint.test_hub import run_checkpoint_tests
from tests.config.test_runtime import run_config_tests
from tests.layers.test_masking import run_masking_tests
from tests.layers.test_primitives import run_layer_tests
from tests.layers.test_text_encoder_primitives import run_text_encoder_primitive_tests
from tests.layers.test_transformer_primitives import run_transformer_primitive_tests
from tests.models.test_text_encoder import run_text_encoder_tests
from tests.models.test_transformer import run_transformer_tests
from tests.execution.test_execution import run_execution_tests
from tests.interfaces.test_interfaces import run_interface_tests
from tests.sampling.test_sampling import run_sampling_tests
from tests.telemetry.test_telemetry import run_telemetry_tests
from tests.tokenization.test_fast_tokenizer import run_fast_tokenizer_tests
from tests.models.test_vae import run_vae_tests


DEFAULT_LOG_FILE_PATH = Path("test_run_log.txt")
TEST_RUNNER_LOGGER_NAME = "flux2_klein.tests"

TEST_SUITES = (
    ("config", run_config_tests),
    ("checkpoint", run_checkpoint_tests),
    ("layers", run_layer_tests),
    ("layers_text_encoder", run_text_encoder_primitive_tests),
    ("layers_masking", run_masking_tests),
    ("layers_transformer", run_transformer_primitive_tests),
    ("blocks", run_vae_block_tests),
    ("blocks_text_encoder", run_text_encoder_block_tests),
    ("models_vae", run_vae_tests),
    ("models_text_encoder", run_text_encoder_tests),
    ("models_transformer", run_transformer_tests),
    ("sampling", run_sampling_tests),
    ("execution", run_execution_tests),
    ("interfaces", run_interface_tests),
    ("telemetry", run_telemetry_tests),
    ("tokenization", run_fast_tokenizer_tests),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full unit test suite.")
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
