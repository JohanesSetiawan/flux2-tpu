"""
Tests for the diffusion transformer's primitives.

The single most important test here is
test_regression_axial_rotation_uses_interleaved_not_half_split_pairing.
This codebase now contains two rotary implementations under different
and incompatible pairing conventions: the text encoder pairs feature i
with i + head_dim/2, the diffusion transformer pairs 2i with 2i+1.
Applying one model's convention to the other's weights produces
plausible output that is wrong, and no shape check can distinguish
them, so each convention is pinned by a direct test.

The second load-bearing test is
test_regression_text_and_image_tokens_occupy_disjoint_position_axes,
which checks the property that makes a single unmasked attention over
both token types coherent.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.blocks import ModulationTriple, apply_modulated_normalization, compute_modulation
from src.blocks.modulation import COMPONENTS_PER_MODULATION, TRIPLES_PER_DOUBLE_BLOCK
from src.config import NumericPrecision, TransformerConfig
from src.layers import (
    apply_axial_rotation,
    axial_rotation_table,
    build_position_identifiers,
    timestep_embedding,
)


NUMERICAL_TOLERANCE = 1e-10

# Reduced width whose axes still sum to the head dimension, so the
# configuration's own validation is satisfied by a genuinely smaller
# model rather than by relaxing the check.
_TEST_CONFIG = TransformerConfig(
    hidden_size=128,
    num_heads=2,
    positional_axes_dimensions=(16, 16, 16, 16),
    precision=NumericPrecision.HIGHEST,
)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_smoke_axial_rotation_table_has_broadcastable_head_axis() -> None:
    rng = _random_generator(seed=0)
    identifiers = jnp.asarray(rng.integers(0, 20, size=(2, 9, 4)))

    cosine, sine = axial_rotation_table(identifiers, _TEST_CONFIG)

    assert cosine.shape == (2, 1, 9, _TEST_CONFIG.head_dim // 2)
    assert sine.shape == cosine.shape


def test_regression_axial_rotation_table_rejects_axis_count_mismatch() -> None:
    identifiers = jnp.zeros((1, 4, 3), dtype=jnp.int32)

    try:
        axial_rotation_table(identifiers, _TEST_CONFIG)
    except ValueError as error:
        assert "axes" in str(error)
        return
    raise AssertionError("Expected ValueError when the identifier axis count disagrees")


def test_regression_axial_rotation_uses_interleaved_not_half_split_pairing() -> None:
    """
    Pin the diffusion transformer's pairing convention.

    A unit value is placed at feature zero. Under the interleaved
    convention its partner is feature one, so the rotated component
    appears there. Under the half-split convention used by the text
    encoder it would appear at head_dim/2 instead. Checking which index
    receives the component identifies the convention unambiguously.

    The position is set on axis zero, not an arbitrary axis, because
    the head dimension is partitioned between axes: feature zero falls
    in axis zero's slice, so only a position along that axis rotates
    it. Setting a position on a different axis would leave feature zero
    untouched and make this test fail against correct code.
    """
    config = _TEST_CONFIG
    head_dim = config.head_dim
    identifiers = jnp.zeros((1, 2, 4), dtype=jnp.int32).at[0, 1, 0].set(3)

    vectors = np.zeros((1, 1, 2, head_dim))
    vectors[0, 0, 1, 0] = 1.0

    cosine, sine = axial_rotation_table(identifiers, config)
    rotated = np.asarray(
        apply_axial_rotation(jnp.asarray(vectors, dtype=jnp.float64), cosine, sine)
    )[0, 0, 1]

    assert abs(rotated[1]) > 1e-9, (
        "no rotated component appeared at feature 1, which means the implementation "
        "is not using the interleaved pairing the diffusion transformer requires"
    )
    assert abs(rotated[head_dim // 2]) < 1e-12, (
        "a rotated component appeared at feature head_dim/2, which indicates the "
        "half-split convention used by the text encoder rather than the interleaved one"
    )


def test_regression_axial_rotation_preserves_vector_norm() -> None:
    rng = _random_generator(seed=1)
    config = _TEST_CONFIG
    identifiers = jnp.asarray(rng.integers(0, 30, size=(1, 6, 4)))
    vectors = rng.standard_normal((1, 2, 6, config.head_dim))

    cosine, sine = axial_rotation_table(identifiers, config)
    rotated = np.asarray(
        apply_axial_rotation(jnp.asarray(vectors, dtype=jnp.float64), cosine, sine)
    )

    assert np.allclose(
        np.linalg.norm(vectors, axis=-1), np.linalg.norm(rotated, axis=-1), atol=1e-6
    ), "rotation changed vector norms, so the transform is not a rotation"


def test_regression_zero_position_leaves_vectors_unchanged() -> None:
    """
    A token at the origin of every axis has all-zero angles, so its
    rotation is the identity. This catches an off-by-one that would
    shift every token's encoding by one place.
    """
    rng = _random_generator(seed=2)
    config = _TEST_CONFIG
    identifiers = jnp.zeros((1, 3, 4), dtype=jnp.int32)
    vectors = rng.standard_normal((1, 2, 3, config.head_dim))

    cosine, sine = axial_rotation_table(identifiers, config)
    rotated = np.asarray(
        apply_axial_rotation(jnp.asarray(vectors, dtype=jnp.float64), cosine, sine)
    )

    assert np.allclose(rotated, vectors, atol=1e-12)


def test_regression_each_axis_rotates_an_independent_feature_slice() -> None:
    """
    Confirm the head dimension is partitioned between axes rather than
    shared.

    Moving a token along one axis must change only that axis's slice of
    the rotation table. If the slices overlapped, position along one
    axis would corrupt the encoding of another.
    """
    config = _TEST_CONFIG
    pairs_per_axis = config.positional_axes_dimensions[0] // 2

    at_origin = jnp.zeros((1, 1, 4), dtype=jnp.int32)
    moved_on_second_axis = at_origin.at[0, 0, 1].set(5)

    cosine_origin, _ = axial_rotation_table(at_origin, config)
    cosine_moved, _ = axial_rotation_table(moved_on_second_axis, config)

    difference = np.abs(np.asarray(cosine_moved - cosine_origin))[0, 0, 0]

    first_axis_slice = difference[:pairs_per_axis]
    second_axis_slice = difference[pairs_per_axis : 2 * pairs_per_axis]
    remaining_slices = difference[2 * pairs_per_axis :]

    assert np.all(first_axis_slice < 1e-12), "moving on axis 1 changed axis 0's slice"
    assert np.any(second_axis_slice > 1e-9), "moving on axis 1 did not change its own slice"
    assert np.all(remaining_slices < 1e-12), "moving on axis 1 changed a later axis's slice"


def test_regression_text_and_image_tokens_occupy_disjoint_position_axes() -> None:
    """
    Text and image tokens share one attention sequence with no mask
    separating them, so their position encodings must not collide. Text
    uses the final axis, images use the middle two, and neither writes
    to the other's.
    """
    text_length, height, width = 5, 3, 4
    text_identifiers, image_identifiers = build_position_identifiers(
        text_length, height, width, _TEST_CONFIG
    )

    text = np.asarray(text_identifiers)[0]
    image = np.asarray(image_identifiers)[0]

    assert text.shape == (text_length, 4)
    assert image.shape == (height * width, 4)

    assert np.array_equal(text[:, 3], np.arange(text_length)), (
        "text tokens should carry their sequence index on the final axis"
    )
    assert np.all(text[:, :3] == 0), "text tokens should be at the origin of the image axes"
    assert np.all(image[:, 3] == 0), "image tokens should be at the origin of the text axis"
    assert np.all(image[:, 0] == 0), "the first axis is unused in text-to-image mode"


def test_regression_image_positions_are_row_major() -> None:
    """
    Row-major ordering is what makes unpacking the latent a plain
    reshape rather than a scatter. A column-major layout would produce a
    transposed image while every shape stayed valid.
    """
    height, width = 3, 4
    _, image_identifiers = build_position_identifiers(1, height, width, _TEST_CONFIG)
    image = np.asarray(image_identifiers)[0]

    for index, (row, column) in enumerate(itertools.product(range(height), range(width))):
        assert image[index, 1] == row, f"token {index} should be at row {row}"
        assert image[index, 2] == column, f"token {index} should be at column {column}"


def test_smoke_timestep_embedding_has_expected_width() -> None:
    config = TransformerConfig()
    embedded = timestep_embedding(jnp.asarray(np.array([0.0, 0.5, 1.0])), config)

    assert embedded.shape == (3, config.timestep_embedding_dim)


def test_regression_timestep_embedding_distinguishes_nearby_timesteps() -> None:
    """
    The sampler's timesteps sit close together, so the embedding must
    separate them. This is what the scale factor exists for: without it
    every timestep would land in the flattest part of the lowest
    frequency and the embeddings would be nearly identical.
    """
    config = TransformerConfig()
    timesteps = jnp.asarray(np.array([0.7672, 0.7700]))

    embedded = np.asarray(timestep_embedding(timesteps, config))
    separation = float(np.max(np.abs(embedded[0] - embedded[1])))

    assert separation > 1e-3, (
        f"timesteps 0.0028 apart produced embeddings differing by only {separation}; "
        f"the scale factor may not be applied"
    )


def test_regression_modulation_splits_into_ordered_triples() -> None:
    """
    Modulation output is consumed as shift, scale, gate in that order,
    repeated per sub-layer. A different ordering would apply a gate
    where a shift belongs while keeping every shape valid, so the split
    is checked against a projection built to make each component
    identifiable.
    """
    config = TransformerConfig(hidden_size=4, num_heads=1, positional_axes_dimensions=(1, 1, 1, 1))
    hidden = config.hidden_size
    num_triples = TRIPLES_PER_DOUBLE_BLOCK
    width = num_triples * COMPONENTS_PER_MODULATION * hidden

    # A projection that maps a single active input feature to a
    # distinct constant per output block, so each component's identity
    # is recoverable from the result.
    weight = np.zeros((hidden, width))
    for block_index in range(num_triples * COMPONENTS_PER_MODULATION):
        weight[0, block_index * hidden : (block_index + 1) * hidden] = float(block_index + 1)

    conditioning = np.zeros((1, hidden))
    conditioning[0, 0] = 4.0  # silu(4) is close to 4, and strictly positive

    parameters = {"double_stream_modulation_img_lin_weight": jnp.asarray(weight, dtype=jnp.float64)}
    triples = compute_modulation(
        jnp.asarray(conditioning, dtype=jnp.float64),
        parameters,
        "double_stream_modulation_img",
        num_triples,
        config,
    )

    assert len(triples) == num_triples
    activated = 4.0 / (1.0 + np.exp(-4.0))

    for triple_index, triple in enumerate(triples):
        for component_index, component in enumerate(
            (triple.shift, triple.scale, triple.gate)
        ):
            block_index = triple_index * COMPONENTS_PER_MODULATION + component_index
            expected = activated * float(block_index + 1)
            assert np.allclose(np.asarray(component), expected, atol=1e-9), (
                f"triple {triple_index} component {component_index} carried the wrong "
                f"block of the projection; the shift, scale, gate ordering may be wrong"
            )


def test_regression_modulation_rejects_width_mismatch() -> None:
    config = TransformerConfig(hidden_size=4, num_heads=1, positional_axes_dimensions=(1, 1, 1, 1))
    parameters = {
        "single_stream_modulation_lin_weight": jnp.zeros((4, 8), dtype=jnp.float64)
    }

    try:
        compute_modulation(
            jnp.zeros((1, 4), dtype=jnp.float64),
            parameters,
            "single_stream_modulation",
            TRIPLES_PER_DOUBLE_BLOCK,
            config,
        )
    except ValueError as error:
        assert "width" in str(error)
        return
    raise AssertionError("Expected ValueError when the projection width disagrees")


def test_regression_zero_modulation_leaves_normalization_unchanged() -> None:
    """
    Scale is applied as one plus the modulation value, so a zero
    modulation is the identity. Applying the raw value instead would
    make a zero modulation annihilate the signal.
    """
    rng = _random_generator(seed=3)
    normalized = jnp.asarray(rng.standard_normal((1, 3, 8)), dtype=jnp.float64)
    zero_triple = ModulationTriple(
        shift=jnp.zeros((1, 1, 8), dtype=jnp.float64),
        scale=jnp.zeros((1, 1, 8), dtype=jnp.float64),
        gate=jnp.zeros((1, 1, 8), dtype=jnp.float64),
    )

    output = np.asarray(apply_modulated_normalization(normalized, zero_triple))

    assert np.allclose(output, np.asarray(normalized), atol=NUMERICAL_TOLERANCE)


_TRANSFORMER_PRIMITIVE_TESTS = [
    test_smoke_axial_rotation_table_has_broadcastable_head_axis,
    test_regression_axial_rotation_table_rejects_axis_count_mismatch,
    test_regression_axial_rotation_uses_interleaved_not_half_split_pairing,
    test_regression_axial_rotation_preserves_vector_norm,
    test_regression_zero_position_leaves_vectors_unchanged,
    test_regression_each_axis_rotates_an_independent_feature_slice,
    test_regression_text_and_image_tokens_occupy_disjoint_position_axes,
    test_regression_image_positions_are_row_major,
    test_smoke_timestep_embedding_has_expected_width,
    test_regression_timestep_embedding_distinguishes_nearby_timesteps,
    test_regression_modulation_splits_into_ordered_triples,
    test_regression_modulation_rejects_width_mismatch,
    test_regression_zero_modulation_leaves_normalization_unchanged,
]


def run_transformer_primitive_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests against the transformer primitives",
        len(_TRANSFORMER_PRIMITIVE_TESTS),
    )
    for test_function in _TRANSFORMER_PRIMITIVE_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All transformer primitive tests passed")
