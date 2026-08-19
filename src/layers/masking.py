"""
Attention masks.

The mask built here is the single most consequential piece of the text
encoder, and the easiest to get wrong without noticing. See the module
constants and `causal_padding_mask` for why.
"""

from __future__ import annotations

import jax.numpy as jnp


# Masked positions are given a large negative bias rather than negative
# infinity. True negative infinity produces NaN if an entire row is
# masked, because softmax then divides zero by zero; a large finite
# value degrades gracefully to a uniform distribution instead. The
# magnitude is far beyond any real score, so masked positions receive
# effectively zero weight.
MASKED_SCORE_BIAS = -1e9

# Masks are built in float32 and added to scores before the softmax.
MASK_DTYPE = jnp.float32


def causal_padding_mask(
    token_is_real: jnp.ndarray,
    sequence_length: int,
) -> jnp.ndarray:
    """
    Build an additive attention mask combining causal ordering with
    key-side padding suppression.

    Two conditions must both hold for a query at position i to attend to
    a key at position j: j must not be in the future (j <= i), and the
    key at j must be a real token rather than padding.

    Why the padding condition is not redundant
    ------------------------------------------
    With right-side padding and causal attention it is tempting to
    conclude that padding needs no mask, since real tokens always
    precede padding and causality already prevents attending forward.
    That reasoning holds for real queries but not for padded ones, and
    padded queries matter here.

    Prompts are padded to a fixed length, and every position of the
    resulting hidden state, padding included, is handed to the diffusion
    transformer as conditioning. So the hidden states at padded
    positions are part of the model's real input, not discarded
    leftovers. A padded query at position i would, under causality
    alone, attend to earlier padded keys as well as real ones, producing
    a different hidden state than the reference, which masks those keys
    out.

    An error here raises nothing and changes no shape. It produces a
    subtly different image. Validate against the reference with prompts
    of several lengths; short prompts, which have the most padding,
    expose it most clearly.

    Parameters
    ----------
    token_is_real:
        Shape (batch, sequence_length), non-zero where a token is real
        and zero where it is padding. This is the attention mask a
        tokenizer produces.
    sequence_length:
        Length the mask is built for. Passed explicitly rather than read
        from the array so the caller's intent is checked rather than
        assumed.

    Returns
    -------
    Additive mask of shape (batch, 1, sequence_length, sequence_length),
    zero where attention is permitted and MASKED_SCORE_BIAS where it is
    not. The singleton axis broadcasts across attention heads.
    """
    if token_is_real.shape[-1] != sequence_length:
        raise ValueError(
            f"Token mask has length {token_is_real.shape[-1]} but a mask for "
            f"length {sequence_length} was requested"
        )

    positions = jnp.arange(sequence_length)
    # is_not_future[i, j] is True when key j is at or before query i.
    is_not_future = positions[:, None] >= positions[None, :]

    # key_is_real[b, 1, 1, j] broadcasts the per-key validity across all
    # query positions.
    key_is_real = (token_is_real != 0)[:, None, None, :]

    attention_permitted = jnp.logical_and(is_not_future[None, None, :, :], key_is_real)

    return jnp.where(attention_permitted, 0.0, MASKED_SCORE_BIAS).astype(MASK_DTYPE)
