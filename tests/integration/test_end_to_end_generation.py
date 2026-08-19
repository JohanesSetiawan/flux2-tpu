"""
End-to-end generation through the real sampler and the real decoder.

Every other test exercises one component. This one runs the full chain:
noise, four sampling steps through a transformer, unpacking, decoding,
and mapping to display range. It is the test that would catch a
mismatch between components that each pass their own tests, for
instance a latent packed in one order and unpacked in another, or a
decoder given a latent whose channel count it does not expect.

The autoencoder uses real checkpoint weights. The transformer is
synthetic and reduced in width, for a practical reason rather than a
principled one: the real transformer is 7.75 GB and this test must run
in environments with less memory than that. What is being verified here
is that the components connect and that shapes and value ranges survive
the whole chain, which a reduced transformer exercises exactly as well.
Numerical fidelity of the transformer itself is the separate concern of
test_transformer_parity.

Requires network access to download the autoencoder component.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.checkpoint import download_bundle, resolve_huggingface_token, restore_component
from src.config import (
    CheckpointSourceConfig,
    SamplingConfig,
    TransformerConfig,
    VaeDecoderConfig,
    VaeLayerConfig,
)
from src.models import decode_latent
from src.models.transformer import predict_velocity
from src.pipeline import LATENT_TOKEN_STRIDE, PACKED_LATENT_CHANNELS, to_display_range
from src.sampling import (
    compute_sigma_schedule,
    denoise_latent,
    pack_latent_to_tokens,
    unpack_tokens_to_latent,
)
from src.utils import configure_logging
from tests.models.test_transformer import make_transformer_parameters


# Small enough to run quickly, large enough that the decoder's
# upsampling levels and middle attention all execute.
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
CONDITIONING_TOKENS = 5
CONDITIONING_DIM = 48

GENERATION_SEED = 20260819

DEFAULT_LOG_FILE_PATH = Path("end_to_end_log.txt")
END_TO_END_LOGGER_NAME = "flux2_klein.tests.integration.end_to_end"


def build_reduced_transformer_config() -> TransformerConfig:
    """
    A reduced transformer that still accepts the real latent's channel
    count, since that is the interface the sampler and decoder agree on.
    """
    return TransformerConfig(
        in_channels=PACKED_LATENT_CHANNELS,
        context_dim=CONDITIONING_DIM,
        hidden_size=64,
        num_heads=2,
        num_double_blocks=2,
        num_single_blocks=2,
        positional_axes_dimensions=(8, 8, 8, 8),
        mlp_ratio=3.0,
    )


def run_end_to_end_test(logger: logging.Logger) -> None:
    """Generate one image and assert the chain held together."""
    source_config = CheckpointSourceConfig()
    token = resolve_huggingface_token(
        logger, source_config.huggingface_token_environment_variable
    )
    bundle_path = download_bundle(source_config, logger, token, component_names=["vae"])
    vae_parameters = restore_component(bundle_path, "vae", logger)

    transformer_config = build_reduced_transformer_config()
    generator = np.random.default_rng(GENERATION_SEED)
    transformer_parameters = make_transformer_parameters(generator, transformer_config)

    conditioning = jnp.asarray(
        generator.standard_normal((1, CONDITIONING_TOKENS, CONDITIONING_DIM)),
        dtype=jnp.float32,
    )

    key = jax.random.key(GENERATION_SEED)
    noise = jax.random.normal(
        key, (1, LATENT_HEIGHT, LATENT_WIDTH, PACKED_LATENT_CHANNELS)
    )
    latent_tokens = pack_latent_to_tokens(noise)

    image_tokens = LATENT_HEIGHT * LATENT_WIDTH
    sigma_schedule = compute_sigma_schedule(image_tokens, SamplingConfig())
    logger.info("schedule: %s", np.array2string(sigma_schedule, precision=4))

    def velocity_at(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        return predict_velocity(
            tokens,
            conditioning,
            timesteps,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            transformer_parameters,
            transformer_config,
        )

    denoised = denoise_latent(latent_tokens, sigma_schedule, velocity_at, logger)
    latent = unpack_tokens_to_latent(denoised, LATENT_HEIGHT, LATENT_WIDTH)

    vae_config = VaeDecoderConfig(layer=VaeLayerConfig(attention_query_chunk_size=512))
    decoded = decode_latent(latent, vae_parameters, vae_config)
    image = to_display_range(np.asarray(decoded[0]))

    expected_height = LATENT_HEIGHT * LATENT_TOKEN_STRIDE
    expected_width = LATENT_WIDTH * LATENT_TOKEN_STRIDE
    assert image.shape == (expected_height, expected_width, 3), (
        f"image shape {image.shape} does not match the latent scaled by "
        f"{LATENT_TOKEN_STRIDE}"
    )

    assert np.all(np.isfinite(image)), "generated image contains non-finite values"
    assert image.min() >= 0.0 and image.max() <= 1.0, (
        f"image values [{image.min()}, {image.max()}] escaped unit range"
    )

    # A constant image would satisfy every assertion above while
    # indicating the chain silently collapsed, so variation is checked
    # explicitly.
    assert float(image.std()) > 1e-3, (
        f"image has standard deviation {image.std()}, which suggests the pipeline "
        f"produced a constant rather than a decoded latent"
    )

    logger.info(
        "generated %dx%d image, range [%.4f, %.4f], std %.4f",
        expected_height,
        expected_width,
        image.min(),
        image.max(),
        image.std(),
    )
    logger.info("End-to-end generation test passed")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one full generation through the sampler and decoder."
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE_PATH)
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, END_TO_END_LOGGER_NAME)
    try:
        run_end_to_end_test(logger)
    except AssertionError as error:
        logger.error("END TO END FAILED: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
