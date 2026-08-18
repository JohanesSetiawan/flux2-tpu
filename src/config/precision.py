"""Numerical precision selection."""

from __future__ import annotations

import enum


class NumericPrecision(enum.Enum):
    """
    Matrix-multiply and convolution precision, mapped onto JAX's three
    precision levels.

    On TPU, a float32 matmul or convolution is not a single native
    float32 operation; it is decomposed into multiple bfloat16 passes,
    trading accuracy against speed:

    DEFAULT: one pass. Fastest, lowest accuracy.
    HIGH: three passes. Intermediate.
    HIGHEST: six passes. Closest to true float32, slowest.

    The reference FLUX.2 implementation decodes in float32, so HIGHEST
    is the accuracy-matching default for the VAE. HIGH is offered
    because the decoder is a single feed-forward pass with no iterative
    error accumulation, so the reduced-pass variant may be visually
    indistinguishable at meaningfully lower cost.

    Note that this decomposition happens on TPU only. On CPU all three
    settings produce bit-identical results, so the choice between them
    cannot be evaluated in a CPU environment and must be measured on
    real hardware before either is treated as settled.
    """

    DEFAULT = "default"
    HIGH = "high"
    HIGHEST = "highest"

    def to_jax_precision(self) -> "jax.lax.Precision":
        """
        Translate to the corresponding jax.lax.Precision value.

        JAX is imported inside the function rather than at module scope
        so that configuration objects can be constructed and inspected
        without paying JAX's import cost, which matters for tooling that
        only reads configuration.
        """
        import jax

        return {
            NumericPrecision.DEFAULT: jax.lax.Precision.DEFAULT,
            NumericPrecision.HIGH: jax.lax.Precision.HIGH,
            NumericPrecision.HIGHEST: jax.lax.Precision.HIGHEST,
        }[self]
