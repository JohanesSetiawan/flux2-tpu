"""Configuration for the Qwen3 text encoder."""

from __future__ import annotations

from dataclasses import dataclass

from .precision import NumericPrecision


@dataclass(frozen=True)
class TextEncoderConfig:
    """
    Architectural parameters of the Qwen3-4B text encoder, as used for
    FLUX.2 Klein conditioning.

    Every value here is a property of the trained checkpoint taken from
    the upstream model configuration, not a free choice. Changing any of
    them produces different outputs from the same weights.

    Two details are easy to get wrong and are called out explicitly:

    head_dim is not derived from hidden_size. Thirty-two heads of
    dimension 128 total 4096, while hidden_size is 2560, so the query
    projection widens rather than partitions. Code that computes
    head_dim as hidden_size divided by head count would be silently
    wrong here.

    num_key_value_heads is smaller than num_attention_heads: this is
    grouped-query attention, where each key/value head is shared by
    several query heads.
    """

    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    rms_norm_epsilon: float = 1e-6
    vocab_size: int = 151936

    # The conditioning tensor fed to the diffusion transformer is built
    # by concatenating the hidden states emitted after these layers.
    # Producing the deepest of them requires exactly that many layers to
    # run, which is why the converted checkpoint keeps 27 of the
    # original 36 and drops the rest as provably dead weight.
    hidden_states_output_layers: tuple[int, ...] = (9, 18, 27)

    # Prompts are padded to exactly this length, and the padding is fed
    # to the diffusion transformer along with the real tokens. See
    # AGENTS.md: this is deliberate, and the transformer applies no mask.
    sequence_length: int = 512

    precision: NumericPrecision = NumericPrecision.HIGHEST

    @property
    def num_layers_required(self) -> int:
        """
        Number of transformer layers that must run to produce every
        referenced hidden state.

        Hidden states are indexed so that entry zero is the embedding
        output and entry k is the output after k layers, so reading
        entry k requires exactly k layers.
        """
        return max(self.hidden_states_output_layers)

    @property
    def query_heads_per_key_value_head(self) -> int:
        """
        How many query heads share each key/value head under
        grouped-query attention.
        """
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"Query head count {self.num_attention_heads} is not divisible by "
                f"key/value head count {self.num_key_value_heads}"
            )
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def conditioning_dimension(self) -> int:
        """
        Width of the conditioning tensor handed to the diffusion
        transformer, formed by concatenating the selected hidden states.
        """
        return self.hidden_size * len(self.hidden_states_output_layers)
