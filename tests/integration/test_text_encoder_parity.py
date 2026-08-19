"""
Numerical parity of the JAX text encoder against the reference
transformers Qwen3 implementation.

Like the VAE parity test, this answers the question the unit suite
cannot: whether the port produces the same numbers as the
implementation it was derived from. It is an integration test because
it needs PyTorch and transformers, neither of which is a dependency of
the src package.

Method
------
A reduced-width Qwen3 is constructed and its weights ported into the
converted checkpoint's layout: projections transposed to
(in_features, out_features), per-layer tensors stacked along a leading
axis. Both implementations then run the same token identifiers and the
same attention mask.

The reduced model deliberately preserves the two structural properties
that make the real one easy to get wrong: head_dim is not hidden_size
divided by head count, and there are four query heads per key/value
head.

Prompts are swept across padding levels, from no padding to almost
entirely padding. This is the point of the test. Under causal attention
with right-side padding it is tempting to think padding needs no mask,
but padded positions produce hidden states that become part of the
conditioning, and a missing key-side padding mask changes them. Heavily
padded cases expose that; an unpadded case cannot.

Layer depth selection also matters and is chosen deliberately. The
reference applies a final normalization after its last layer and
records the result as the deepest hidden state, so selecting a depth
equal to the layer count would compare against a normalized value. The
real configuration selects depth 27 from a 36-layer model, which is not
the last, so this test likewise selects depths strictly below its layer
count.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.config import NumericPrecision, TextEncoderConfig
from src.models.text_encoder import encode_prompt
from src.utils import configure_logging


# Reduced dimensions preserving the real model's structural ratios.
REFERENCE_HIDDEN_SIZE = 64
REFERENCE_INTERMEDIATE_SIZE = 128
REFERENCE_NUM_LAYERS = 5
REFERENCE_NUM_ATTENTION_HEADS = 8
REFERENCE_NUM_KEY_VALUE_HEADS = 2
REFERENCE_HEAD_DIM = 16
REFERENCE_VOCAB_SIZE = 200
REFERENCE_SEQUENCE_LENGTH = 12

# Strictly below REFERENCE_NUM_LAYERS, so none passes through the
# reference's final normalization. See the module docstring.
REFERENCE_OUTPUT_LAYERS = (1, 2, 4)

# Number of real tokens to test, from unpadded down to almost entirely
# padded.
PARITY_REAL_TOKEN_COUNTS = (12, 7, 3, 1)

PARITY_RANDOM_SEED = 20260819

# The reference builds its rotary tables in float32 even when the model
# itself runs in float64, so a residual at that scale is the
# reference's own precision rather than a disagreement. The threshold
# sits above it and far below any algorithmic error, which would show
# up at the scale of the activations themselves.
PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE = 1e-6

PER_LAYER_PARAMETER_KEYS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

DEFAULT_LOG_FILE_PATH = Path("text_encoder_parity_log.txt")
PARITY_LOGGER_NAME = "flux2_klein.tests.integration.text_encoder_parity"


class ReferenceImplementationUnavailableError(RuntimeError):
    """
    Raised when PyTorch or transformers is not installed.

    A distinct type so that "the reference could not be loaded" is never
    mistaken for "the implementations disagree"; those warrant very
    different responses.
    """


def build_reference_model():
    """Construct the reduced reference Qwen3 in float64."""
    try:
        import torch
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    except ImportError as error:
        raise ReferenceImplementationUnavailableError(
            "PyTorch and transformers are required for this test. Install with: "
            "pip install torch --index-url https://download.pytorch.org/whl/cpu "
            "and pip install transformers"
        ) from error

    configuration = Qwen3Config(
        hidden_size=REFERENCE_HIDDEN_SIZE,
        intermediate_size=REFERENCE_INTERMEDIATE_SIZE,
        num_hidden_layers=REFERENCE_NUM_LAYERS,
        num_attention_heads=REFERENCE_NUM_ATTENTION_HEADS,
        num_key_value_heads=REFERENCE_NUM_KEY_VALUE_HEADS,
        head_dim=REFERENCE_HEAD_DIM,
        rope_theta=TextEncoderConfig().rope_theta,
        rms_norm_eps=TextEncoderConfig().rms_norm_epsilon,
        vocab_size=REFERENCE_VOCAB_SIZE,
        attention_bias=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(PARITY_RANDOM_SEED)
    return Qwen3Model(configuration).eval().to(torch.float64)


def port_reference_weights(reference_model, num_layers: int) -> dict:
    """
    Convert reference weights into the layout the converted checkpoint
    uses: projections transposed, per-layer tensors stacked.

    This mirrors what the conversion pipeline does to the real
    checkpoint, so a layout misunderstanding here would surface as a
    parity failure rather than being hidden.
    """
    state_dict = reference_model.state_dict()

    stacked: dict[str, jnp.ndarray] = {}
    for key in PER_LAYER_PARAMETER_KEYS:
        per_layer = []
        for layer_index in range(num_layers):
            weight = state_dict[f"layers.{layer_index}.{key}"].numpy()
            # Two-dimensional projections are transposed; one
            # dimensional normalization scales are not.
            per_layer.append(weight.T if weight.ndim == 2 else weight)
        stacked[key.replace(".", "_")] = jnp.asarray(np.stack(per_layer, axis=0))

    return {
        "embed_tokens": {"weight": jnp.asarray(state_dict["embed_tokens.weight"].numpy())},
        "layers": stacked,
    }


def reference_conditioning(reference_model, token_ids, token_is_real, output_layers):
    """Run the reference and concatenate the selected hidden states."""
    import torch

    with torch.no_grad():
        outputs = reference_model(
            input_ids=torch.from_numpy(token_ids),
            attention_mask=torch.from_numpy(token_is_real),
            output_hidden_states=True,
        )
    selected = [outputs.hidden_states[depth] for depth in output_layers]
    return torch.cat(selected, dim=-1).numpy()


def run_text_encoder_parity_test(logger: logging.Logger) -> None:
    """
    Compare both implementations across every padding level in
    PARITY_REAL_TOKEN_COUNTS, raising AssertionError on any failure.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    reference_model = build_reference_model()

    config = TextEncoderConfig(
        hidden_size=REFERENCE_HIDDEN_SIZE,
        intermediate_size=REFERENCE_INTERMEDIATE_SIZE,
        num_attention_heads=REFERENCE_NUM_ATTENTION_HEADS,
        num_key_value_heads=REFERENCE_NUM_KEY_VALUE_HEADS,
        head_dim=REFERENCE_HEAD_DIM,
        vocab_size=REFERENCE_VOCAB_SIZE,
        hidden_states_output_layers=REFERENCE_OUTPUT_LAYERS,
        sequence_length=REFERENCE_SEQUENCE_LENGTH,
        precision=NumericPrecision.HIGHEST,
    )
    parameters = port_reference_weights(reference_model, config.num_layers_required)

    # Match the reference's float32 rotary tables rather than leaving
    # this implementation's float64 ones in place, so the comparison
    # measures the algorithms rather than a known precision difference.
    from src.layers.positional import rotary_frequency_table
    import src.models.text_encoder as text_encoder_module

    text_encoder_module.rotary_frequency_table = (
        lambda sequence_length, head_dim, theta: rotary_frequency_table(
            sequence_length, head_dim, theta, dtype=jnp.float64
        )
    )

    generator = np.random.default_rng(PARITY_RANDOM_SEED)

    for real_token_count in PARITY_REAL_TOKEN_COUNTS:
        token_ids = generator.integers(
            0, REFERENCE_VOCAB_SIZE, size=(2, REFERENCE_SEQUENCE_LENGTH)
        )
        token_is_real = np.zeros((2, REFERENCE_SEQUENCE_LENGTH), dtype=np.int64)
        token_is_real[:, :real_token_count] = 1

        expected = reference_conditioning(
            reference_model, token_ids, token_is_real, REFERENCE_OUTPUT_LAYERS
        )
        actual = np.asarray(
            encode_prompt(
                jnp.asarray(token_ids), jnp.asarray(token_is_real), parameters, config
            )
        )

        assert actual.shape == expected.shape, (
            f"shape mismatch at real_token_count={real_token_count}: "
            f"{actual.shape} against reference {expected.shape}"
        )

        maximum_difference = float(np.max(np.abs(actual - expected)))
        logger.info(
            "real tokens %2d of %d (padding %2d): max abs diff %.3e",
            real_token_count,
            REFERENCE_SEQUENCE_LENGTH,
            REFERENCE_SEQUENCE_LENGTH - real_token_count,
            maximum_difference,
        )

        assert maximum_difference <= PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE, (
            f"parity failed at real_token_count={real_token_count}: max abs diff "
            f"{maximum_difference:.3e} exceeds {PARITY_MAXIMUM_ABSOLUTE_DIFFERENCE}"
        )

    logger.info("Text encoder parity test passed at every padding level")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare the JAX text encoder against the reference implementation."
    )
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE_PATH)
    arguments = parser.parse_args()

    logger = configure_logging(arguments.log_file, PARITY_LOGGER_NAME)
    try:
        run_text_encoder_parity_test(logger)
    except ReferenceImplementationUnavailableError as error:
        logger.error("Cannot run parity test: %s", error)
        return 2
    except AssertionError as error:
        logger.error("PARITY FAILED: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
