"""
Numerical parity of the JAX diffusion transformer against the reference
BFL implementation.

This is the last and largest of the three parity tests, and the one
that exercises the most interacting pieces at once: multi-axis rotary
under the interleaved convention, per-head query and key normalization,
three shared modulation projections, joint attention over concatenated
text and image tokens, two structurally different block types, and a
final layer whose modulation has no gate.

The reference model is built at reduced width with both block types
present, which is what the reference's own debug configuration is for.
A full-size comparison would need the real checkpoint and far more
memory than a parity check warrants; what matters is that every code
path runs, and a reduced model exercises all of them.

Weights are ported into the converted checkpoint's layout, transposing
projections and stacking per-block tensors, so a layout
misunderstanding surfaces here rather than staying hidden.

Requires PyTorch and a checkout of the reference repository. Neither is
a dependency of the src package.
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.config import NumericPrecision, TransformerConfig
from src.models.transformer import predict_velocity
from src.utils import configure_logging


REFERENCE_SOURCE_REPOSITORY = "https://github.com/black-forest-labs/flux2"

REFERENCE_IN_CHANNELS = 16
REFERENCE_CONTEXT_DIM = 48
REFERENCE_HIDDEN_SIZE = 64
REFERENCE_NUM_HEADS = 2
REFERENCE_NUM_DOUBLE_BLOCKS = 2
REFERENCE_NUM_SINGLE_BLOCKS = 2
REFERENCE_AXES_DIMENSIONS = (8, 8, 8, 8)
REFERENCE_MLP_RATIO = 3.0

# Swept so that a bug appearing only at one aspect ratio or text length
# is caught at another. The non-square case matters for the same reason
# it does in the VAE test: it can detect a transposition of the row and
# column position axes, which a square grid cannot.
PARITY_LATENT_SHAPES = ((3, 4), (4, 4))
PARITY_TEXT_LENGTHS = (5, 1)

# Timesteps spanning the sampler's actual range, including the extremes
# where the sinusoidal embedding behaves least like the middle.
PARITY_TIMESTEPS = (1.0, 0.7672, 0.0)

PARITY_RANDOM_SEED = 20260819

# Both sides run in float64, but the reference builds its rotary tables
# in float32, so a residual at that scale is the reference's own
# precision rather than a disagreement. An algorithmic error would show
# up at the scale of the activations.
PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE = 1e-5

DOUBLE_BLOCK_PARAMETER_KEYS = (
    "img_attn.qkv.weight",
    "img_attn.proj.weight",
    "img_attn.norm.query_norm.scale",
    "img_attn.norm.key_norm.scale",
    "img_mlp.0.weight",
    "img_mlp.2.weight",
    "txt_attn.qkv.weight",
    "txt_attn.proj.weight",
    "txt_attn.norm.query_norm.scale",
    "txt_attn.norm.key_norm.scale",
    "txt_mlp.0.weight",
    "txt_mlp.2.weight",
)
SINGLE_BLOCK_PARAMETER_KEYS = (
    "linear1.weight",
    "linear2.weight",
    "norm.query_norm.scale",
    "norm.key_norm.scale",
)

DEFAULT_LOG_FILE_PATH = Path("transformer_parity_log.txt")
PARITY_LOGGER_NAME = "flux2_klein.tests.integration.transformer_parity"


class ReferenceImplementationUnavailableError(RuntimeError):
    """
    Raised when PyTorch or the reference source tree is unavailable, so
    that a missing dependency is never mistaken for a disagreement
    between implementations.
    """


def build_reference_model(reference_source_path: Path):
    """Construct the reduced reference transformer in float64."""
    try:
        import torch
    except ImportError as error:
        raise ReferenceImplementationUnavailableError(
            "PyTorch is required. Install with: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from error

    source_directory = reference_source_path / "src"
    if not source_directory.is_dir():
        raise ReferenceImplementationUnavailableError(
            f"Reference implementation not found at {source_directory}. Clone it from "
            f"{REFERENCE_SOURCE_REPOSITORY} and pass its path."
        )
    sys.path.insert(0, str(source_directory))

    from flux2.model import Flux2, Flux2Params

    parameters = Flux2Params(
        in_channels=REFERENCE_IN_CHANNELS,
        context_in_dim=REFERENCE_CONTEXT_DIM,
        hidden_size=REFERENCE_HIDDEN_SIZE,
        num_heads=REFERENCE_NUM_HEADS,
        depth=REFERENCE_NUM_DOUBLE_BLOCKS,
        depth_single_blocks=REFERENCE_NUM_SINGLE_BLOCKS,
        axes_dim=list(REFERENCE_AXES_DIMENSIONS),
        theta=int(TransformerConfig().rope_theta),
        mlp_ratio=REFERENCE_MLP_RATIO,
        use_guidance_embed=False,
    )
    torch.manual_seed(PARITY_RANDOM_SEED)
    return Flux2(parameters).eval().to(torch.float64)


def port_reference_weights(reference_model) -> dict:
    """
    Convert reference weights into the converted checkpoint's layout:
    projections transposed to (in_features, out_features), per-block
    tensors stacked along a leading axis, and global tensors flattened
    with dots replaced by underscores.
    """
    state_dict = reference_model.state_dict()

    def transposed_if_matrix(weight: np.ndarray) -> np.ndarray:
        return weight.T if weight.ndim == 2 else weight

    def stack_blocks(group_prefix: str, keys: tuple[str, ...], count: int) -> dict:
        stacked = {}
        for key in keys:
            per_block = [
                transposed_if_matrix(state_dict[f"{group_prefix}.{index}.{key}"].numpy())
                for index in range(count)
            ]
            stacked[key.replace(".", "_")] = jnp.asarray(np.stack(per_block, axis=0))
        return stacked

    global_parameters = {
        key.replace(".", "_"): jnp.asarray(transposed_if_matrix(value.numpy()))
        for key, value in state_dict.items()
        if not key.startswith(("double_blocks.", "single_blocks."))
    }

    return {
        "double_blocks": stack_blocks(
            "double_blocks", DOUBLE_BLOCK_PARAMETER_KEYS, REFERENCE_NUM_DOUBLE_BLOCKS
        ),
        "single_blocks": stack_blocks(
            "single_blocks", SINGLE_BLOCK_PARAMETER_KEYS, REFERENCE_NUM_SINGLE_BLOCKS
        ),
        "global": global_parameters,
    }


def build_reference_position_identifiers(
    text_length: int, latent_height: int, latent_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build position identifiers the way the reference command line does,
    by taking a cartesian product over the axes.

    Reproducing the reference's own construction, rather than reusing
    this implementation's build_position_identifiers, is deliberate: it
    means the test compares two independently derived layouts rather
    than feeding both sides the same possibly-wrong one.
    """
    image_identifiers = np.array(
        list(itertools.product([0], range(latent_height), range(latent_width), [0]))
    )[None]
    text_identifiers = np.array(
        list(itertools.product([0], [0], [0], range(text_length)))
    )[None]
    return image_identifiers, text_identifiers


def run_transformer_parity_test(reference_source_path: Path, logger: logging.Logger) -> None:
    """Compare both implementations across shapes, text lengths and timesteps."""
    import jax
    import torch

    jax.config.update("jax_enable_x64", True)

    reference_model = build_reference_model(reference_source_path)
    parameters = port_reference_weights(reference_model)

    config = TransformerConfig(
        in_channels=REFERENCE_IN_CHANNELS,
        context_dim=REFERENCE_CONTEXT_DIM,
        hidden_size=REFERENCE_HIDDEN_SIZE,
        num_heads=REFERENCE_NUM_HEADS,
        num_double_blocks=REFERENCE_NUM_DOUBLE_BLOCKS,
        num_single_blocks=REFERENCE_NUM_SINGLE_BLOCKS,
        positional_axes_dimensions=REFERENCE_AXES_DIMENSIONS,
        mlp_ratio=REFERENCE_MLP_RATIO,
        precision=NumericPrecision.HIGHEST,
    )

    generator = np.random.default_rng(PARITY_RANDOM_SEED)

    for (latent_height, latent_width), text_length, timestep in itertools.product(
        PARITY_LATENT_SHAPES, PARITY_TEXT_LENGTHS, PARITY_TIMESTEPS
    ):
        num_latent_tokens = latent_height * latent_width
        latent = generator.standard_normal((1, num_latent_tokens, REFERENCE_IN_CHANNELS))
        conditioning = generator.standard_normal((1, text_length, REFERENCE_CONTEXT_DIM))
        timesteps = np.array([timestep])

        image_identifiers, text_identifiers = build_reference_position_identifiers(
            text_length, latent_height, latent_width
        )

        with torch.no_grad():
            expected = reference_model(
                torch.from_numpy(latent),
                torch.from_numpy(image_identifiers).double(),
                torch.from_numpy(timesteps),
                torch.from_numpy(conditioning),
                torch.from_numpy(text_identifiers).double(),
                None,
            ).numpy()

        actual = np.asarray(
            predict_velocity(
                jnp.asarray(latent),
                jnp.asarray(conditioning),
                jnp.asarray(timesteps),
                latent_height,
                latent_width,
                parameters,
                config,
            )
        )

        assert actual.shape == expected.shape, (
            f"shape mismatch at latent {latent_height}x{latent_width}, text "
            f"{text_length}, timestep {timestep}: {actual.shape} against {expected.shape}"
        )

        maximum_difference = float(np.max(np.abs(actual - expected)))
        logger.info(
            "latent %dx%d, text %d, timestep %.4f: max abs diff %.3e",
            latent_height,
            latent_width,
            text_length,
            timestep,
            maximum_difference,
        )

        assert maximum_difference <= PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE, (
            f"parity failed at latent {latent_height}x{latent_width}, text "
            f"{text_length}, timestep {timestep}: max abs diff "
            f"{maximum_difference:.3e} exceeds {PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE}"
        )

    logger.info("Transformer parity test passed for every configuration")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare the JAX diffusion transformer against the reference."
    )
    parser.add_argument("--reference-source-path", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE_PATH)
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, PARITY_LOGGER_NAME)
    try:
        run_transformer_parity_test(arguments.reference_source_path, logger)
    except ReferenceImplementationUnavailableError as error:
        logger.error("Cannot run parity test: %s", error)
        return 2
    except AssertionError as error:
        logger.error("PARITY FAILED: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
