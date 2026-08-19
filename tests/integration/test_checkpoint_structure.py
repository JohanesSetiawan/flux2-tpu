"""
Structural contract between the code and the real checkpoint bundle.

Every other test in this suite runs against synthetic parameters, which
proves the mathematics but says nothing about whether the real bundle
is shaped the way the code assumes. This test closes that gap: it
downloads the actual bundle and asserts the structure directly.

It exists because of a specific past failure. An early version of the
conversion pipeline assumed a tensor sat at the top level of the
autoencoder when it was in fact nested under the decoder, and nothing
caught it until a real download was attempted by hand. A structural
assumption held only in someone's head is one upstream reorganisation
away from being wrong, and the failure it produces (a missing key deep
inside a forward pass) is far less informative than an explicit
assertion here.

Structure is read from checkpoint metadata rather than by restoring
the parameters. Shapes and dtypes are all this test needs, and reading
them costs a few hundred megabytes instead of the roughly six
gigabytes the text encoder occupies, so the check runs on machines that
could not hold the component at all. An earlier version restored the
parameters in full and was killed by the memory manager partway
through.

This is an integration test: it needs network access and downloads
several gigabytes. It is not part of the unit suite.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.checkpoint import (
    component_metadata,
    download_bundle,
    resolve_huggingface_token,
    select_parameter_group,
)
from src.config import CheckpointSourceConfig, TextEncoderConfig, VaeDecoderConfig
from src.models.vae import (
    discover_residual_block_indices,
    discover_upsample_level_indices,
)
from src.utils import configure_logging


# Groups the VAE component must expose at its top level.
REQUIRED_VAE_GROUPS = ("decoder", "post_quant_conv", "latent_denormalize")

# Keys that must exist within those groups. post_quant_conv is listed
# explicitly because its location is precisely what was once assumed
# wrongly.
REQUIRED_POST_QUANT_CONV_KEYS = ("weight", "bias")
REQUIRED_LATENT_DENORMALIZE_KEYS = ("scale", "shift")

# Decoder keys addressed by name rather than discovered.
REQUIRED_DECODER_KEYS = (
    "conv_in_weight",
    "conv_in_bias",
    "conv_out_weight",
    "conv_out_bias",
    "norm_out_weight",
    "norm_out_bias",
)

# Prefixes whose groups must be non-empty.
REQUIRED_DECODER_GROUP_PREFIXES = ("mid_block_1", "mid_attn_1", "mid_block_2")

# Per-layer text encoder keys the transformer layer addresses.
REQUIRED_TEXT_ENCODER_LAYER_KEYS = (
    "input_layernorm_weight",
    "post_attention_layernorm_weight",
    "self_attn_q_proj_weight",
    "self_attn_k_proj_weight",
    "self_attn_v_proj_weight",
    "self_attn_o_proj_weight",
    "self_attn_q_norm_weight",
    "self_attn_k_norm_weight",
    "mlp_gate_proj_weight",
    "mlp_up_proj_weight",
    "mlp_down_proj_weight",
)

DEFAULT_LOG_FILE_PATH = Path("checkpoint_structure_log.txt")
STRUCTURE_LOGGER_NAME = "flux2_klein.tests.integration.checkpoint_structure"


def verify_vae_structure(vae_parameters: dict, logger: logging.Logger) -> None:
    """Assert the VAE component matches what src.models.vae addresses."""
    for group_name in REQUIRED_VAE_GROUPS:
        assert group_name in vae_parameters, (
            f"VAE component is missing the '{group_name}' group. Present groups: "
            f"{sorted(vae_parameters.keys())}"
        )

    for key in REQUIRED_POST_QUANT_CONV_KEYS:
        assert key in vae_parameters["post_quant_conv"], (
            f"post_quant_conv is missing '{key}'. This group's location is the one "
            f"that was previously assumed wrongly, so verify it against the "
            f"conversion pipeline before changing the code."
        )

    weight = vae_parameters["post_quant_conv"]["weight"]
    assert len(weight.shape) == 2, (
        f"post_quant_conv weight has {len(weight.shape)} dimensions, expected 2. The "
        f"conversion stores this 1x1 convolution as a plain matrix; a four "
        f"dimensional weight means the conversion changed."
    )

    for key in REQUIRED_LATENT_DENORMALIZE_KEYS:
        assert key in vae_parameters["latent_denormalize"], (
            f"latent_denormalize is missing '{key}'"
        )

    decoder = vae_parameters["decoder"]
    for key in REQUIRED_DECODER_KEYS:
        assert key in decoder, f"decoder is missing '{key}'"

    for prefix in REQUIRED_DECODER_GROUP_PREFIXES:
        group = select_parameter_group(decoder, prefix)
        assert group, f"decoder group '{prefix}' is empty"

    assert not any("encoder" in key for key in decoder), (
        "decoder contains encoder tensors; the conversion should have dropped them"
    )

    level_indices = discover_upsample_level_indices(decoder)
    logger.info("VAE upsampling levels, in execution order: %s", level_indices)
    for level_index in level_indices:
        block_indices = discover_residual_block_indices(decoder, level_index)
        logger.info("  level %d has residual blocks %s", level_index, block_indices)


def verify_text_encoder_structure(
    text_encoder_parameters: dict, config: TextEncoderConfig, logger: logging.Logger
) -> None:
    """Assert the text encoder component matches what src.models addresses."""
    assert "embed_tokens" in text_encoder_parameters, "text encoder is missing embed_tokens"
    assert "layers" in text_encoder_parameters, "text encoder is missing layers"

    embedding = text_encoder_parameters["embed_tokens"]["weight"]
    assert tuple(embedding.shape) == (config.vocab_size, config.hidden_size), (
        f"embedding table has shape {embedding.shape}, expected "
        f"({config.vocab_size}, {config.hidden_size}). Note this table keeps its "
        f"row layout while projections were transposed, so a transposed shape here "
        f"would indicate the conversion changed."
    )

    layers = text_encoder_parameters["layers"]
    for key in REQUIRED_TEXT_ENCODER_LAYER_KEYS:
        assert key in layers, (
            f"stacked layers are missing '{key}'. Present keys: {sorted(layers.keys())}"
        )

    layer_count = layers[REQUIRED_TEXT_ENCODER_LAYER_KEYS[0]].shape[0]
    assert layer_count >= config.num_layers_required, (
        f"checkpoint carries {layer_count} layers but the configured hidden state "
        f"selection requires {config.num_layers_required}"
    )
    logger.info("Text encoder carries %d stacked layers", layer_count)

    query_projection = layers["self_attn_q_proj_weight"]
    expected_query_width = config.num_attention_heads * config.head_dim
    assert tuple(query_projection.shape) == (layer_count, config.hidden_size, expected_query_width), (
        f"query projection has shape {query_projection.shape}, expected "
        f"({layer_count}, {config.hidden_size}, {expected_query_width}). This is the "
        f"shape that confirms head_dim is not hidden_size divided by head count."
    )

    key_projection = layers["self_attn_k_proj_weight"]
    expected_key_width = config.num_key_value_heads * config.head_dim
    assert tuple(key_projection.shape) == (layer_count, config.hidden_size, expected_key_width), (
        f"key projection has shape {key_projection.shape}, expected "
        f"({layer_count}, {config.hidden_size}, {expected_key_width}). A width equal "
        f"to the query projection's would mean this is not grouped-query attention."
    )


def run_checkpoint_structure_test(logger: logging.Logger) -> None:
    """Download the real bundle and verify both components' structure."""
    source_config = CheckpointSourceConfig()
    token = resolve_huggingface_token(
        logger, source_config.huggingface_token_environment_variable
    )
    bundle_path = download_bundle(
        source_config, logger, token, component_names=["vae", "text_encoder"]
    )

    logger.info("Verifying VAE component structure")
    verify_vae_structure(component_metadata(bundle_path, "vae", logger), logger)

    logger.info("Verifying text encoder component structure")
    verify_text_encoder_structure(
        component_metadata(bundle_path, "text_encoder", logger),
        TextEncoderConfig(),
        logger,
    )

    logger.info("Checkpoint structure matches every assumption the code makes")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify the real checkpoint bundle matches the code's structural assumptions."
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE_PATH)
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, STRUCTURE_LOGGER_NAME)
    try:
        run_checkpoint_structure_test(logger)
    except AssertionError as error:
        logger.error("STRUCTURE MISMATCH: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
