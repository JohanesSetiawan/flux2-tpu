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
    bundle_is_complete,
    VALID_COMPONENT_NAMES,
    component_download_patterns,
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


def test_regression_download_without_token_still_proceeds() -> None:
    """
    The bundle repository is public, so a missing token must not block
    the download. It is passed through as None rather than triggering a
    prompt or an error.
    """
    source_config = CheckpointSourceConfig(local_cache_directory=Path("/tmp/flux2_klein_test_cache"))
    with patch("src.checkpoint.hub.snapshot_download") as mock_snapshot_download:
        mock_snapshot_download.return_value = "/tmp/flux2_klein_test_cache"

        download_bundle(source_config, _silent_logger(), token=None)

        assert mock_snapshot_download.call_args.kwargs["token"] is None


def test_regression_component_download_patterns_cover_only_requested_components() -> None:
    """
    Downloading a subset must still fetch the manifest and tokenizer,
    which are small and always needed, while excluding the components
    not asked for. The full bundle is roughly 13 GB, so work touching
    one component should not pay for the other two.
    """
    patterns = component_download_patterns(["vae"])

    assert any("vae" in pattern for pattern in patterns)
    assert any("tokenizer" in pattern for pattern in patterns)
    assert any("manifest" in pattern for pattern in patterns)
    assert not any("transformer" in pattern for pattern in patterns)
    assert not any("text_encoder" in pattern for pattern in patterns)


def test_regression_component_download_patterns_reject_unknown_component() -> None:
    try:
        component_download_patterns(["not_a_component"])
    except ValueError as error:
        assert "not_a_component" in str(error)
        return
    raise AssertionError("Expected ValueError for an unknown component name")


def _stage_complete_bundle(root: Path) -> Path:
    """Create a directory with the layout a usable bundle has."""
    for component in ("vae", "transformer", "text_encoder"):
        (root / "params" / component).mkdir(parents=True, exist_ok=True)
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "manifest.json").write_text("{}")
    return root


def test_regression_complete_bundle_is_recognised() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = _stage_complete_bundle(Path(directory))
        assert bundle_is_complete(root)


def test_regression_partial_bundle_is_not_treated_as_complete() -> None:
    """
    A directory missing any component must not be mistaken for a usable
    bundle, or the caller skips the download and fails later on an
    absent component instead.
    """
    import tempfile

    for missing in ("manifest.json", "tokenizer", "params/vae"):
        with tempfile.TemporaryDirectory() as directory:
            root = _stage_complete_bundle(Path(directory))
            target = root / missing
            if target.is_dir():
                target.rmdir()
            else:
                target.unlink()

            assert not bundle_is_complete(root), (
                f"a bundle missing {missing} was reported as complete"
            )


def test_regression_present_bundle_skips_the_download_entirely() -> None:
    """
    The fix for a real failure on Kaggle. Weights attached as a dataset
    sit on a read-only mount, and the download client writes its own
    metadata alongside whatever it fetches, so calling it there fails on
    directory creation rather than on anything to do with downloading.
    The weights were present the whole time.
    """
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        root = _stage_complete_bundle(Path(directory))
        source_config = CheckpointSourceConfig(local_cache_directory=root)

        with patch("src.checkpoint.hub.snapshot_download") as mock_snapshot_download:
            result = download_bundle(source_config, _silent_logger())

            mock_snapshot_download.assert_not_called()
            assert result == root


def test_regression_writability_probe_detects_a_read_only_mount() -> None:
    """
    Writability is tested by attempting a write rather than by reading
    permission bits, because the case that matters is a read-only mount,
    where the bits look ordinary. Note that chmod cannot stand in for
    this when running as root, which ignores them.
    """
    from src.checkpoint.hub import _directory_is_writable

    read_only_mounts = [Path("/mnt/skills/public"), Path("/proc/sys/kernel")]
    available = [path for path in read_only_mounts if path.is_dir()]

    for mount in available:
        assert not _directory_is_writable(mount), f"{mount} was reported writable"

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        assert _directory_is_writable(Path(directory))


_CHECKPOINT_TESTS = [
    test_download_bundle_passes_repo_id_and_token_through,
    test_restore_component_builds_correct_path_and_restores,
    test_restore_component_rejects_unknown_component_name,
    test_valid_component_names_matches_the_three_bundle_components,
    test_regression_download_without_token_still_proceeds,
    test_regression_component_download_patterns_cover_only_requested_components,
    test_regression_component_download_patterns_reject_unknown_component,
    test_regression_complete_bundle_is_recognised,
    test_regression_partial_bundle_is_not_treated_as_complete,
    test_regression_present_bundle_skips_the_download_entirely,
    test_regression_writability_probe_detects_a_read_only_mount,
]


def run_checkpoint_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against checkpoint.py", len(_CHECKPOINT_TESTS))
    for test_function in _CHECKPOINT_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All checkpoint tests passed")
