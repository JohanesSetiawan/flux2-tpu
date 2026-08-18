"""
Unit tests for checkpoint.py.

The bundle repository is private, and this sandbox has no Hugging Face
token for it, so download_bundle and restore_component are verified
here by mocking huggingface_hub.snapshot_download and
orbax.checkpoint.StandardCheckpointer rather than by a real network
call. What is verified is that this module calls those two libraries
with the correct arguments and raises clearly on an invalid component
name; an actual end-to-end download can only be verified by running
against the real, authenticated repository.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.checkpoint import (
    VALID_COMPONENT_NAMES,
    download_bundle,
    restore_component,
)
from src.config import CheckpointSourceConfig


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("checkpoint_tests")
    logger.addHandler(logging.NullHandler())
    return logger


def test_download_bundle_passes_repo_id_and_token_through() -> None:
    source_config = CheckpointSourceConfig(
        huggingface_repo_id="someone/some-repo",
        local_cache_directory=Path("/tmp/flux2_klein_test_cache"),
    )
    with patch("src.checkpoint.hub.snapshot_download") as mock_snapshot_download:
        mock_snapshot_download.return_value = "/tmp/flux2_klein_test_cache"

        result = download_bundle(source_config, _silent_logger(), token="fake_token_value")

        assert result == Path("/tmp/flux2_klein_test_cache")
        call_kwargs = mock_snapshot_download.call_args.kwargs
        assert call_kwargs["repo_id"] == "someone/some-repo"
        assert call_kwargs["token"] == "fake_token_value"


def test_restore_component_builds_correct_path_and_restores() -> None:
    with patch("src.checkpoint.restore.ocp.StandardCheckpointer") as mock_checkpointer_class:
        mock_checkpointer = MagicMock()
        mock_checkpointer.restore.return_value = {"fake": "params"}
        mock_checkpointer_class.return_value = mock_checkpointer

        result = restore_component(Path("/tmp/bundle"), "vae", _silent_logger())

        assert result == {"fake": "params"}
        restore_call_path = mock_checkpointer.restore.call_args.args[0]
        assert restore_call_path == Path("/tmp/bundle/params/vae")


def test_restore_component_rejects_unknown_component_name() -> None:
    try:
        restore_component(Path("/tmp/bundle"), "not_a_real_component", _silent_logger())
    except ValueError as error:
        assert "not_a_real_component" in str(error)
        return
    raise AssertionError("Expected ValueError for an unknown component name")


def test_valid_component_names_matches_the_three_bundle_components() -> None:
    assert VALID_COMPONENT_NAMES == frozenset({"text_encoder", "transformer", "vae"})


_CHECKPOINT_TESTS = [
    test_download_bundle_passes_repo_id_and_token_through,
    test_restore_component_builds_correct_path_and_restores,
    test_restore_component_rejects_unknown_component_name,
    test_valid_component_names_matches_the_three_bundle_components,
]


def run_checkpoint_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against checkpoint.py", len(_CHECKPOINT_TESTS))
    for test_function in _CHECKPOINT_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All checkpoint tests passed")
