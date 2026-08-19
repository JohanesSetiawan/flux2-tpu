"""Configuration for the diffusion transformer."""

from __future__ import annotations

from dataclasses import dataclass

from .precision import NumericPrecision


@dataclass(frozen=True)
class TransformerConfig:
    """
    Architectural parameters of the FLUX.2 Klein-4B diffusion
    transformer.

    Unlike the text encoder, head_dim here is hidden_size divided by
    head count (3072 over 24 gives 128), and the positional axes sum to
    exactly that. Both relationships are asserted rather than assumed,
    since a checkpoint violating either would fail confusingly deep
    inside attention.

    Note that guidance is not embedded for this checkpoint. It is a
    distillation-guided model with no guidance_in tensor, and the
    guidance value accepted by the reference command line is ignored by
    the model itself.
    """

    in_channels: int = 128
    context_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    num_double_blocks: int = 5
    num_single_blocks: int = 20
    mlp_ratio: float = 3.0

    # Rotary position embedding is applied over four independent axes.
    # For text-to-image the first three carry no information for text
    # tokens and the last carries none for image tokens; see
    # build_position_identifiers for how they are populated.
    positional_axes_dimensions: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: float = 2000.0

    # Width of the sinusoidal timestep embedding before it is projected
    # to hidden size.
    timestep_embedding_dim: int = 256
    timestep_max_period: float = 10000.0
    timestep_scale_factor: float = 1000.0

    rms_norm_epsilon: float = 1e-6
    layer_norm_epsilon: float = 1e-6

    precision: NumericPrecision = NumericPrecision.HIGHEST

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Hidden size {self.hidden_size} is not divisible by head count "
                f"{self.num_heads}"
            )
        if sum(self.positional_axes_dimensions) != self.head_dim:
            raise ValueError(
                f"Positional axes {self.positional_axes_dimensions} sum to "
                f"{sum(self.positional_axes_dimensions)}, which does not match the "
                f"head dimension {self.head_dim}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def mlp_hidden_size(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @property
    def num_positional_axes(self) -> int:
        return len(self.positional_axes_dimensions)
