"""
Numerical parity of the JAX VAE decoder against the reference
PyTorch implementation.

This is the test that answers the question the unit suite cannot: not
"does the decoder run and produce sensible shapes", but "does it
produce the same numbers as the implementation it was ported from".

It is an integration test, kept out of the unit suite, because it needs
three things that suite deliberately avoids: network access, roughly
half a gigabyte of downloads, and PyTorch. PyTorch appears here only to
produce reference outputs; it is not a dependency of the flux2_klein
package and must not become one.

Method
------
Both implementations are given the identical latent, drawn from a
seeded generator and passed through as an array rather than
regenerated on each side, so the comparison isolates the algorithms
rather than the random draws.

Cases sweep square and non-square latents. The non-square case matters
disproportionately: a square latent cannot detect a transposition of
the height and width axes, since both are the same length. Any layout
error between the reference's NCHW and this implementation's NHWC would
survive a square-only test.

Agreement is reported as peak signal-to-noise ratio against the
reference. Exact bitwise agreement is not achievable and not the goal:
the two implementations schedule their float32 arithmetic differently,
so they differ at the level of float32 rounding. PARITY_MINIMUM_PSNR_DB
is set far above any level at which a difference could be visible, and
far below what a genuine algorithmic error would produce; a wrong
permutation, a missing residual, or a misordered normalization drops
this figure by tens of decibels, not fractions of one.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from flux2_klein.checkpoint import download_bundle, resolve_huggingface_token, restore_component
from flux2_klein.config import CheckpointSourceConfig, VaeDecoderConfig, VaeLayerConfig
from flux2_klein.logging_setup import configure_logging
from flux2_klein.vae import decode_latent


REFERENCE_SOURCE_REPOSITORY = "https://github.com/black-forest-labs/flux2"
REFERENCE_WEIGHTS_REPO_ID = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
REFERENCE_WEIGHTS_FILENAME = "split_files/vae/flux2-vae.safetensors"

# Latent shapes exercised, as (height, width) in latent space. The
# non-square case is the one that can detect an axis transposition.
PARITY_LATENT_SHAPES = ((16, 16), (12, 20))

PACKED_LATENT_CHANNELS = 128
PARITY_RANDOM_SEED = 20260818

# See the module docstring for why this threshold is where it is.
PARITY_MINIMUM_PSNR_DB = 100.0

DEFAULT_LOG_FILE_PATH = Path("vae_parity_log.txt")
PARITY_LOGGER_NAME = "flux2_klein.tests.integration.vae_parity"


class ReferenceImplementationUnavailableError(RuntimeError):
    """
    Raised when PyTorch or the reference source tree is not available.

    This is a distinct, clearly named failure so that "the reference
    implementation could not be loaded" is never mistaken for "the JAX
    implementation disagrees with the reference".
    """


def peak_signal_to_noise_ratio(actual: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute PSNR of `actual` against `reference`, using the reference's
    own value range rather than an assumed range, since the decoder's
    output is not clamped to a fixed interval.
    """
    mean_squared_error = float(np.mean((actual - reference) ** 2))
    if mean_squared_error == 0.0:
        return float("inf")
    data_range = float(reference.max() - reference.min())
    return 10.0 * float(np.log10((data_range ** 2) / mean_squared_error))


def load_reference_autoencoder(reference_source_path: Path, weights_path: Path):
    """
    Build the reference PyTorch autoencoder and load the original
    float32 weights into it.

    The weights are assigned rather than copied, and no dtype cast is
    applied, which is what makes the reference decode run in float32.
    That detail is the whole reason this project's decoder is float32
    too, so it is reproduced here rather than glossed over.
    """
    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as error:
        raise ReferenceImplementationUnavailableError(
            "PyTorch and safetensors are required to run the parity test. "
            "Install them with: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from error

    source_directory = reference_source_path / "src"
    if not source_directory.is_dir():
        raise ReferenceImplementationUnavailableError(
            f"Reference implementation not found at {source_directory}. "
            f"Clone it from {REFERENCE_SOURCE_REPOSITORY} and pass its path."
        )
    sys.path.insert(0, str(source_directory))

    from flux2.autoencoder import AutoEncoder, AutoEncoderParams

    state_dict = load_file(str(weights_path))
    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    autoencoder.load_state_dict(state_dict, strict=True, assign=True)
    return autoencoder.eval()


def reference_decode(autoencoder, latent_nchw: np.ndarray) -> np.ndarray:
    """Decode with the reference implementation, returning NHWC."""
    import torch

    with torch.no_grad():
        output_nchw = autoencoder.decode(torch.from_numpy(latent_nchw)).numpy()
    return np.transpose(output_nchw, (0, 2, 3, 1))


def run_vae_parity_test(
    reference_source_path: Path,
    logger: logging.Logger,
) -> None:
    """
    Compare both implementations across every latent shape in
    PARITY_LATENT_SHAPES, raising AssertionError if any falls below the
    parity threshold.
    """
    from huggingface_hub import hf_hub_download

    logger.info("Downloading reference weights from %s", REFERENCE_WEIGHTS_REPO_ID)
    reference_weights_path = Path(
        hf_hub_download(REFERENCE_WEIGHTS_REPO_ID, REFERENCE_WEIGHTS_FILENAME)
    )

    logger.info("Loading reference PyTorch autoencoder")
    autoencoder = load_reference_autoencoder(reference_source_path, reference_weights_path)

    source_config = CheckpointSourceConfig()
    token = resolve_huggingface_token(logger, source_config.huggingface_token_environment_variable)
    bundle_path = download_bundle(source_config, logger, token, component_names=["vae"])
    vae_parameters = restore_component(bundle_path, "vae", logger)

    config = VaeDecoderConfig(layer=VaeLayerConfig(attention_query_chunk_size=512))
    generator = np.random.default_rng(PARITY_RANDOM_SEED)

    for latent_height, latent_width in PARITY_LATENT_SHAPES:
        latent_nchw = generator.standard_normal(
            (1, PACKED_LATENT_CHANNELS, latent_height, latent_width)
        ).astype(np.float32)

        expected = reference_decode(autoencoder, latent_nchw)
        actual = np.asarray(
            decode_latent(np.transpose(latent_nchw, (0, 2, 3, 1)), vae_parameters, config)
        )

        assert actual.shape == expected.shape, (
            f"shape mismatch at latent {latent_height}x{latent_width}: "
            f"{actual.shape} against reference {expected.shape}"
        )

        psnr = peak_signal_to_noise_ratio(actual, expected)
        maximum_absolute_difference = float(np.max(np.abs(actual - expected)))

        logger.info(
            "latent %dx%d: PSNR %.2f dB, max abs diff %.3e",
            latent_height,
            latent_width,
            psnr,
            maximum_absolute_difference,
        )

        assert psnr >= PARITY_MINIMUM_PSNR_DB, (
            f"parity failed at latent {latent_height}x{latent_width}: "
            f"PSNR {psnr:.2f} dB is below the {PARITY_MINIMUM_PSNR_DB} dB threshold, "
            f"max abs diff {maximum_absolute_difference:.3e}"
        )

    logger.info("VAE parity test passed for every latent shape")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare the JAX VAE decoder against the reference PyTorch implementation."
    )
    parser.add_argument(
        "--reference-source-path",
        type=Path,
        required=True,
        help=f"Path to a checkout of {REFERENCE_SOURCE_REPOSITORY}",
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE_PATH)
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, PARITY_LOGGER_NAME)
    try:
        run_vae_parity_test(arguments.reference_source_path, logger)
    except ReferenceImplementationUnavailableError as error:
        logger.error("Cannot run parity test: %s", error)
        return 2
    except AssertionError as error:
        logger.error("PARITY FAILED: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
