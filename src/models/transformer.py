"""
The FLUX.2 Klein-4B diffusion transformer.

Given a noisy latent, a timestep and text conditioning, it predicts the
velocity field the sampler integrates. It is not a denoiser in the
predict-the-clean-image sense; its output is a direction, which is why
the sampler adds it scaled by a step size rather than replacing the
latent with it.

Structure, from input to output:

  project the latent and the conditioning into hidden space
  derive modulation from the timestep, once for the whole forward pass
  run the double-stream blocks, image and text separately but attending
    jointly
  concatenate the two streams and run the single-stream blocks
  drop the text positions and project the image positions back out

The modulation projections are evaluated once here rather than inside
each block, which is possible because this model shares three
modulation modules across all blocks rather than carrying one per
block.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..blocks import double_stream_block, single_stream_block
from ..blocks.modulation import (
    TRIPLES_PER_DOUBLE_BLOCK,
    TRIPLES_PER_SINGLE_BLOCK,
    ModulationTriple,
    compute_modulation,
)
from ..checkpoint import require_parameter, select_parameter_group
from ..config import ExecutionConfig, TransformerConfig
from ..layers import (
    axial_rotation_table,
    build_position_identifiers,
    layer_normalization,
    timestep_embedding,
)


GLOBAL_GROUP = "global"
DOUBLE_BLOCK_GROUP = "double_blocks"
SINGLE_BLOCK_GROUP = "single_blocks"

LATENT_PROJECTION_KEY = "img_in_weight"
CONDITIONING_PROJECTION_KEY = "txt_in_weight"
TIMESTEP_INPUT_PROJECTION_KEY = "time_in_in_layer_weight"
TIMESTEP_OUTPUT_PROJECTION_KEY = "time_in_out_layer_weight"
FINAL_MODULATION_KEY = "final_layer_adaLN_modulation_1_weight"
FINAL_PROJECTION_KEY = "final_layer_linear_weight"

IMAGE_MODULATION_PREFIX = "double_stream_modulation_img"
TEXT_MODULATION_PREFIX = "double_stream_modulation_txt"
SINGLE_MODULATION_PREFIX = "single_stream_modulation"

# The final layer's modulation produces a shift and a scale but no
# gate, since its output leaves the residual stream rather than
# rejoining it.
FINAL_MODULATION_COMPONENTS = 2


class BlockCountMismatchError(Exception):
    """
    Raised when the checkpoint's block count disagrees with the
    configuration.

    Running the wrong number of blocks would produce a valid tensor from
    the wrong depth of network, so this fails loudly rather than
    proceeding.
    """


def _stacked_block_count(block_group: dict[str, np.ndarray]) -> int:
    """Read the block count from the leading axis of any stacked tensor."""
    return next(iter(block_group.values())).shape[0]


def _block_parameters_at(block_group: dict[str, np.ndarray], block_index: int) -> dict:
    """
    Slice one block's parameters out of the stacked arrays.

    The conversion stacked every block's tensors along a new leading
    axis so the stack can eventually be driven by a scan; until then,
    indexing that axis gives an ordinary per-block group.
    """
    return {key: value[block_index] for key, value in block_group.items()}


def compute_conditioning_vector(
    timesteps: jnp.ndarray,
    global_parameters: dict[str, np.ndarray],
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Turn timesteps into the vector every modulation projection reads.

    This is a sinusoidal encoding followed by a two-layer projection
    with the activation between them. Note the activation sits between
    the two layers, not before the first: the first layer receives the
    raw sinusoidal features.
    """
    from ..layers import sigmoid_linear_unit

    context = "compute_conditioning_vector"
    precision = config.precision.to_jax_precision()

    embedded = timestep_embedding(timesteps, config)
    hidden = jnp.matmul(
        embedded,
        require_parameter(global_parameters, TIMESTEP_INPUT_PROJECTION_KEY, context),
        precision=precision,
    )
    return jnp.matmul(
        sigmoid_linear_unit(hidden),
        require_parameter(global_parameters, TIMESTEP_OUTPUT_PROJECTION_KEY, context),
        precision=precision,
    )


def compute_all_modulation(
    conditioning_vector: jnp.ndarray,
    global_parameters: dict[str, np.ndarray],
    config: TransformerConfig,
) -> tuple[
    tuple[ModulationTriple, ModulationTriple],
    tuple[ModulationTriple, ModulationTriple],
    ModulationTriple,
]:
    """
    Evaluate all three modulation projections once.

    Every double block shares the returned image and text pairs, and
    every single block shares the returned triple. Computing them here
    rather than per block is not merely an optimisation: they depend on
    the timestep alone, so recomputing them inside each block would
    repeat identical work.
    """
    image_pair = compute_modulation(
        conditioning_vector,
        global_parameters,
        IMAGE_MODULATION_PREFIX,
        TRIPLES_PER_DOUBLE_BLOCK,
        config,
    )
    text_pair = compute_modulation(
        conditioning_vector,
        global_parameters,
        TEXT_MODULATION_PREFIX,
        TRIPLES_PER_DOUBLE_BLOCK,
        config,
    )
    single_triples = compute_modulation(
        conditioning_vector,
        global_parameters,
        SINGLE_MODULATION_PREFIX,
        TRIPLES_PER_SINGLE_BLOCK,
        config,
    )

    return (
        (image_pair[0], image_pair[1]),
        (text_pair[0], text_pair[1]),
        single_triples[0],
    )


def _apply_final_layer(
    activations: jnp.ndarray,
    conditioning_vector: jnp.ndarray,
    global_parameters: dict[str, np.ndarray],
    config: TransformerConfig,
) -> jnp.ndarray:
    """
    Normalize, modulate and project back to latent channels.

    This modulation is separate from the three shared ones and produces
    only a shift and a scale, with no gate, because its output leaves
    the residual stream rather than rejoining it.
    """
    from ..layers import sigmoid_linear_unit

    context = "final_layer"
    precision = config.precision.to_jax_precision()

    modulation_output = jnp.matmul(
        sigmoid_linear_unit(conditioning_vector),
        require_parameter(global_parameters, FINAL_MODULATION_KEY, context),
        precision=precision,
    )[:, None, :]

    shift, scale = jnp.split(modulation_output, FINAL_MODULATION_COMPONENTS, axis=-1)

    # Applied inline rather than through a ModulationTriple, because
    # this modulation genuinely has no gate. Constructing a triple with
    # a null gate would fake a shape the data does not have.
    normalized = layer_normalization(activations, config.layer_norm_epsilon)
    modulated = normalized * (1.0 + scale) + shift

    return jnp.matmul(
        modulated,
        require_parameter(global_parameters, FINAL_PROJECTION_KEY, context),
        precision=precision,
    )


def _run_double_blocks(
    image_activations: jnp.ndarray,
    text_activations: jnp.ndarray,
    block_group: dict[str, np.ndarray],
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
    image_modulation,
    text_modulation,
    config: TransformerConfig,
    execution: ExecutionConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run every double-stream block, either through a scan or an unrolled
    loop.

    The two paths compute identical values. They differ only in how the
    block body reaches the compiler: unrolled emits one copy per block,
    scanned emits one copy total. The regression suite asserts their
    agreement rather than assuming it.
    """
    if not execution.use_scan_over_blocks:
        for block_index in range(config.num_double_blocks):
            image_activations, text_activations = double_stream_block(
                image_activations,
                text_activations,
                _block_parameters_at(block_group, block_index),
                cosine_table,
                sine_table,
                image_modulation,
                text_modulation,
                config,
            )
        return image_activations, text_activations

    def run_one_block(carry, block_parameters):
        image, text = carry
        image, text = double_stream_block(
            image,
            text,
            block_parameters,
            cosine_table,
            sine_table,
            image_modulation,
            text_modulation,
            config,
        )
        # Nothing is collected per block, so the second element is None.
        return (image, text), None

    (image_activations, text_activations), _ = jax.lax.scan(
        run_one_block, (image_activations, text_activations), block_group
    )
    return image_activations, text_activations


def _run_single_blocks(
    activations: jnp.ndarray,
    block_group: dict[str, np.ndarray],
    cosine_table: jnp.ndarray,
    sine_table: jnp.ndarray,
    modulation: ModulationTriple,
    config: TransformerConfig,
    execution: ExecutionConfig,
) -> jnp.ndarray:
    """Run every single-stream block. See _run_double_blocks."""
    if not execution.use_scan_over_blocks:
        for block_index in range(config.num_single_blocks):
            activations = single_stream_block(
                activations,
                _block_parameters_at(block_group, block_index),
                cosine_table,
                sine_table,
                modulation,
                config,
            )
        return activations

    def run_one_block(carry, block_parameters):
        return (
            single_stream_block(
                carry, block_parameters, cosine_table, sine_table, modulation, config
            ),
            None,
        )

    activations, _ = jax.lax.scan(run_one_block, activations, block_group)
    return activations


def predict_velocity(
    latent_tokens: jnp.ndarray,
    conditioning: jnp.ndarray,
    timesteps: jnp.ndarray,
    latent_height: int,
    latent_width: int,
    parameters: dict,
    config: TransformerConfig,
    execution: ExecutionConfig | None = None,
) -> jnp.ndarray:
    """
    Predict the velocity field for one sampling step.

    Parameters
    ----------
    latent_tokens:
        Shape (batch, latent_height * latent_width, in_channels), the
        noisy latent flattened into tokens in row-major order.
    conditioning:
        Shape (batch, text_tokens, context_dim), from the text encoder.
    timesteps:
        Shape (batch,), the current noise level.
    latent_height, latent_width:
        Token grid dimensions, needed to build image position
        identifiers. Passed explicitly rather than inferred from the
        token count, since many height and width pairs share a product.
    parameters:
        The restored transformer component.
    config:
        Architecture and precision settings.
    execution:
        How to run the block stacks. Defaults to the standard settings,
        which use a scan.

    Returns
    -------
    Shape (batch, latent_height * latent_width, in_channels), matching
    the latent tokens.
    """
    execution = execution or ExecutionConfig()
    global_parameters = parameters[GLOBAL_GROUP]
    double_blocks = parameters[DOUBLE_BLOCK_GROUP]
    single_blocks = parameters[SINGLE_BLOCK_GROUP]

    for group, expected, name in (
        (double_blocks, config.num_double_blocks, "double"),
        (single_blocks, config.num_single_blocks, "single"),
    ):
        found = _stacked_block_count(group)
        if found != expected:
            raise BlockCountMismatchError(
                f"Checkpoint carries {found} {name}-stream blocks but the "
                f"configuration declares {expected}"
            )

    precision = config.precision.to_jax_precision()

    # One dtype governs the whole forward pass, taken from the latent
    # because that is what the caller is integrating and what the
    # velocity must be added to.
    #
    # Deriving it and casting at every entry point is not defensive
    # clutter. The image stream, the text stream and the modulation
    # vectors arrive from three different sources, and mixing their
    # dtypes promotes the residual stream partway through a block. Under
    # a scan that is fatal rather than merely slower: the carry's input
    # and output dtypes must match exactly, so a promotion anywhere
    # inside the block body stops the program from compiling at all.
    compute_dtype = latent_tokens.dtype

    image_activations = jnp.matmul(
        latent_tokens,
        require_parameter(global_parameters, LATENT_PROJECTION_KEY, "predict_velocity"),
        precision=precision,
    ).astype(compute_dtype)
    text_activations = jnp.matmul(
        conditioning.astype(compute_dtype),
        require_parameter(global_parameters, CONDITIONING_PROJECTION_KEY, "predict_velocity"),
        precision=precision,
    ).astype(compute_dtype)

    conditioning_vector = compute_conditioning_vector(
        timesteps.astype(compute_dtype), global_parameters, config
    ).astype(compute_dtype)
    image_modulation, text_modulation, single_modulation = compute_all_modulation(
        conditioning_vector, global_parameters, config
    )

    text_identifiers, image_identifiers = build_position_identifiers(
        text_activations.shape[1], latent_height, latent_width, config
    )
    position_identifiers = jnp.concatenate([text_identifiers, image_identifiers], axis=1)
    cosine_table, sine_table = axial_rotation_table(position_identifiers, config)

    image_activations, text_activations = _run_double_blocks(
        image_activations,
        text_activations,
        double_blocks,
        cosine_table,
        sine_table,
        image_modulation,
        text_modulation,
        config,
        execution,
    )

    num_text_tokens = text_activations.shape[1]
    activations = jnp.concatenate([text_activations, image_activations], axis=1)

    activations = _run_single_blocks(
        activations,
        single_blocks,
        cosine_table,
        sine_table,
        single_modulation,
        config,
        execution,
    )

    # Only the image positions carry the prediction; the text positions
    # were conditioning and are dropped here.
    image_activations = activations[:, num_text_tokens:]

    return _apply_final_layer(
        image_activations, conditioning_vector, global_parameters, config
    )
