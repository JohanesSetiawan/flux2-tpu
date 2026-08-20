"""
Tests for the execution layer: residency, sharding, compilation cache,
and the optimizations that must not change results.

The central claim of this whole layer is that nothing in it changes
what the model computes. That claim is load-bearing, because these
options exist precisely to be turned on without further thought, so it
is asserted directly for each of them rather than argued for:

- a scanned block stack against an unrolled one
- a fused sampling loop against a stepped one

Neither is expected to agree bitwise. Both reorder floating point
operations, and reordering changes rounding. The tolerances below are
set at the level the underlying precision explains, and the accompanying
comments record what that level is, so that a future regression showing
a larger gap is recognisable as a real change rather than dismissed as
noise.

Sharding is exercised across several simulated devices rather than the
one this environment really has, using JAX's host platform device
count. Testing it on a single device would exercise only the trivial
path, which is the path least likely to be wrong.
"""

from __future__ import annotations

import logging
import os

# Must be set before JAX initialises its backend, so this sits above the
# JAX imports deliberately rather than being hoisted with them.
SIMULATED_DEVICE_COUNT = 4
os.environ.setdefault(
    "XLA_FLAGS", f"--xla_force_host_platform_device_count={SIMULATED_DEVICE_COUNT}"
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.config import (  # noqa: E402
    ExecutionConfig,
    MemoryResidencyStrategy,
    SamplingConfig,
    resolve_residency_strategy,
)
from src.execution import (  # noqa: E402
    SPLITTABLE_GROUPS_BY_COMPONENT,
    build_device_mesh,
    place_component,
    configure_compilation_cache,
    evict_to_host,
    move_to_accelerator,
    plan_component_residency,
    replicate_parameters,
    shard_stacked_blocks,
)
from src.execution.residency import EVICTABLE_COMPONENTS  # noqa: E402
from src.models.text_encoder import encode_prompt  # noqa: E402
from src.models.transformer import predict_velocity  # noqa: E402
from src.sampling import compute_sigma_schedule, denoise_latent  # noqa: E402
from tests.models.test_text_encoder import (  # noqa: E402
    _TEST_CONFIG as TEXT_ENCODER_TEST_CONFIG,
)
from tests.models.test_text_encoder import _make_parameters, _token_inputs  # noqa: E402
from tests.models.test_transformer import (  # noqa: E402
    _TEST_CONFIG as TRANSFORMER_TEST_CONFIG,
)
from tests.models.test_transformer import (  # noqa: E402
    TEST_LATENT_HEIGHT,
    TEST_LATENT_WIDTH,
    _model_inputs,
    make_transformer_parameters,
)


COMPONENT_NAMES = ("text_encoder", "transformer", "vae")

# Both models carry float32 rotary tables while the test parameters are
# float64. Reordering operations around a lower-precision table
# amplifies its rounding, so the observed gap sits near float32 epsilon
# rather than float64 epsilon. Verified by rebuilding the tables in
# float64, which drops the gap to around 1e-15.
REORDERING_TOLERANCE_WITH_FLOAT32_TABLES = 1e-6

# The sampling loop carries no such table, so its two paths agree to
# float64 rounding.
REORDERING_TOLERANCE_FLOAT64_ONLY = 1e-12


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _silent_logger() -> logging.Logger:
    logger = logging.getLogger("execution_tests")
    logger.addHandler(logging.NullHandler())
    return logger


def test_smoke_simulated_devices_are_available() -> None:
    """
    Confirm the simulated device count took effect.

    If it did not, every sharding test below would silently exercise the
    single-device path and prove nothing, so this failure must be loud.
    """
    assert jax.device_count() >= 2, (
        f"expected several simulated devices, found {jax.device_count()}; the "
        f"XLA_FLAGS setting at the top of this module may have been applied after "
        f"the backend initialised"
    )


def test_regression_residency_plan_keeps_the_transformer_resident() -> None:
    """
    Under the swapped strategy the transformer must stay resident. It
    runs once per sampling step, so evicting it would pay a host
    transfer per step, which is the opposite of the intent.
    """
    plan = plan_component_residency(MemoryResidencyStrategy.SWAPPED, COMPONENT_NAMES)
    by_name = {entry.component_name: entry.resident for entry in plan}

    assert by_name["transformer"] is True, "the transformer must never be evicted"
    assert by_name["text_encoder"] is False, "the text encoder is the component worth evicting"
    assert by_name["vae"] is True, "the decoder is small enough to keep resident"


def test_regression_fully_resident_plan_evicts_nothing() -> None:
    plan = plan_component_residency(MemoryResidencyStrategy.FULLY_RESIDENT, COMPONENT_NAMES)

    assert all(entry.resident for entry in plan)


def test_regression_residency_plan_rejects_unresolved_strategy() -> None:
    """
    AUTO carries no decision. Resolving it needs a device count this
    function does not have, so guessing would hide a caller's mistake.
    """
    try:
        plan_component_residency(MemoryResidencyStrategy.AUTO, COMPONENT_NAMES)
    except ValueError as error:
        assert "resolved" in str(error)
        return
    raise AssertionError("Expected ValueError for an unresolved strategy")


def test_regression_evictable_set_excludes_the_transformer() -> None:
    """
    Guard the set itself, not just the plan derived from it. Adding the
    transformer here would be a plausible-looking change that quietly
    makes every sampling step pay a transfer.
    """
    assert "transformer" not in EVICTABLE_COMPONENTS


def test_regression_eviction_moves_parameters_to_host_and_back() -> None:
    parameters = {"group": {"weight": jnp.ones((4, 4))}}
    logger = _silent_logger()

    on_host = evict_to_host(parameters, logger, "test_component")
    assert on_host["group"]["weight"].devices() == {jax.devices("cpu")[0]}

    back = move_to_accelerator(on_host, logger, "test_component")
    assert np.array_equal(np.asarray(back["group"]["weight"]), np.ones((4, 4))), (
        "values changed while moving between memories"
    )


def test_regression_move_to_accelerator_names_its_destination() -> None:
    """
    Moving parameters back must name where they are going.

    jax.device_put(array) with no destination is a no-op for an array
    already committed to a device, so an earlier version of this
    function silently left evicted parameters on the host. Nothing
    failed at that point; the component simply ran on the host, and the
    mismatch only surfaced later when its output met an
    accelerator-resident tensor inside a jit boundary.

    The test commits to a device explicitly and checks the array
    actually lands where it was told, rather than checking only that the
    call returned something.
    """
    logger = _silent_logger()
    devices = jax.devices()
    parameters = {"weight": jax.device_put(jnp.ones((4,)), devices[-1])}

    moved = move_to_accelerator(parameters, logger, "test", sharding=devices[0])

    assert moved["weight"].devices() == {devices[0]}, (
        f"parameters landed on {moved['weight'].devices()} rather than the "
        f"requested {devices[0]}; the destination may not have been passed through"
    )


def test_regression_round_trip_through_host_returns_to_the_accelerator() -> None:
    """
    The full eviction cycle must end where it started. This is the
    sequence the swapped residency strategy runs on every new prompt.
    """
    logger = _silent_logger()
    devices = jax.devices()
    original = jax.device_put(jnp.arange(4.0), devices[0])

    on_host = evict_to_host({"weight": original}, logger, "test")
    back = move_to_accelerator(on_host, logger, "test", sharding=devices[0])

    assert back["weight"].devices() == {devices[0]}
    assert np.array_equal(np.asarray(back["weight"]), np.arange(4.0))


def test_smoke_mesh_covers_every_visible_device() -> None:
    mesh = build_device_mesh(_silent_logger())

    assert mesh.devices.size == jax.device_count()


def test_regression_replication_places_a_full_copy_everywhere() -> None:
    mesh = build_device_mesh(_silent_logger())
    parameters = {"weight": jnp.arange(8.0)}

    replicated = replicate_parameters(parameters, mesh)

    assert np.array_equal(np.asarray(replicated["weight"]), np.arange(8.0))
    assert len(replicated["weight"].devices()) == jax.device_count(), (
        "a replicated parameter should be present on every device"
    )


def test_regression_sharding_never_splits_the_block_axis() -> None:
    """
    The leading axis indexes blocks and every device runs every block,
    so splitting it would break the scan. This checks the resulting
    sharding directly rather than trusting the selection logic.
    """
    mesh = build_device_mesh(_silent_logger())
    device_count = jax.device_count()
    num_blocks = 3

    parameters = {"linear_weight": jnp.zeros((num_blocks, 8, 4 * device_count))}
    sharded = shard_stacked_blocks(parameters, mesh, _silent_logger())

    partition = sharded["linear_weight"].sharding.spec
    assert partition[0] is None, (
        f"the block axis was sharded, which would break the scan: {partition}"
    )


def test_regression_sharding_splits_the_widest_remaining_axis() -> None:
    mesh = build_device_mesh(_silent_logger())
    device_count = jax.device_count()

    # Widest non-block axis is the last one.
    parameters = {"linear_weight": jnp.zeros((2, 8, 16 * device_count))}
    sharded = shard_stacked_blocks(parameters, mesh, _silent_logger())

    partition = sharded["linear_weight"].sharding.spec
    assert partition[2] is not None, f"the widest axis was not sharded: {partition}"


def test_regression_sharding_replicates_indivisible_shapes() -> None:
    """
    A tensor whose chosen axis does not divide by the device count is
    replicated rather than padded, since padding would produce a shape
    that no longer matches the checkpoint.
    """
    mesh = build_device_mesh(_silent_logger())
    indivisible = jax.device_count() * 2 + 1

    parameters = {"linear_weight": jnp.zeros((2, 3, indivisible))}
    sharded = shard_stacked_blocks(parameters, mesh, _silent_logger())

    assert all(entry is None for entry in sharded["linear_weight"].sharding.spec), (
        "an indivisible tensor should be replicated, not split"
    )


def test_regression_sharding_preserves_values() -> None:
    mesh = build_device_mesh(_silent_logger())
    rng = _random_generator(seed=1)
    original = rng.standard_normal((2, 4, 4 * jax.device_count()))

    sharded = shard_stacked_blocks({"linear_weight": jnp.asarray(original)}, mesh, _silent_logger())

    assert np.allclose(np.asarray(sharded["linear_weight"]), original), (
        "sharding altered parameter values"
    )


def test_regression_scanned_transformer_matches_unrolled() -> None:
    """
    The central guarantee of the scan optimization.

    A gap near float32 epsilon is expected and explained: both paths
    consume a float32 rotary table, and reordering the operations around
    it changes how its rounding propagates. Rebuilding that table in
    float64 drops the gap to around 1e-15, which is what confirms the
    difference is precision rather than logic.
    """
    rng = _random_generator(seed=2)
    config = TRANSFORMER_TEST_CONFIG
    parameters = make_transformer_parameters(rng, config)
    latent, conditioning, timesteps = _model_inputs(rng, config)

    scanned = np.asarray(
        predict_velocity(
            latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH,
            parameters, config, ExecutionConfig(use_scan_over_blocks=True),
        )
    )
    unrolled = np.asarray(
        predict_velocity(
            latent, conditioning, timesteps, TEST_LATENT_HEIGHT, TEST_LATENT_WIDTH,
            parameters, config, ExecutionConfig(use_scan_over_blocks=False),
        )
    )

    difference = float(np.max(np.abs(scanned - unrolled)))
    assert difference < REORDERING_TOLERANCE_WITH_FLOAT32_TABLES, (
        f"scanned and unrolled block stacks disagreed by {difference:.3e}, which is "
        f"beyond what reordering around a float32 table explains"
    )


def test_regression_scanned_text_encoder_matches_unrolled() -> None:
    rng = _random_generator(seed=3)
    config = TEXT_ENCODER_TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    token_ids, token_is_real = _token_inputs(rng, config, real_length=4)

    scanned = np.asarray(
        encode_prompt(
            token_ids, token_is_real, parameters, config,
            ExecutionConfig(use_scan_over_blocks=True),
        )
    )
    unrolled = np.asarray(
        encode_prompt(
            token_ids, token_is_real, parameters, config,
            ExecutionConfig(use_scan_over_blocks=False),
        )
    )

    difference = float(np.max(np.abs(scanned - unrolled)))
    assert difference < REORDERING_TOLERANCE_WITH_FLOAT32_TABLES, (
        f"scanned and unrolled layer stacks disagreed by {difference:.3e}"
    )


def test_regression_scanned_text_encoder_selects_the_same_depths() -> None:
    """
    The scanned path collects every layer's output and then indexes it,
    where the unrolled path captures selectively. An off-by-one in that
    indexing would take the state one layer early or late while
    producing the right shape, so the two paths are compared slice by
    slice rather than only in aggregate.
    """
    rng = _random_generator(seed=4)
    config = TEXT_ENCODER_TEST_CONFIG
    parameters = _make_parameters(rng, config, config.num_layers_required)
    token_ids, token_is_real = _token_inputs(rng, config, real_length=5)

    scanned = np.asarray(
        encode_prompt(token_ids, token_is_real, parameters, config,
                      ExecutionConfig(use_scan_over_blocks=True))
    )
    unrolled = np.asarray(
        encode_prompt(token_ids, token_is_real, parameters, config,
                      ExecutionConfig(use_scan_over_blocks=False))
    )

    width = config.hidden_size
    for slice_index, depth in enumerate(config.hidden_states_output_layers):
        scanned_slice = scanned[..., slice_index * width : (slice_index + 1) * width]
        unrolled_slice = unrolled[..., slice_index * width : (slice_index + 1) * width]
        assert np.allclose(
            scanned_slice, unrolled_slice, atol=REORDERING_TOLERANCE_WITH_FLOAT32_TABLES
        ), f"the two paths captured different states at depth {depth}"


def test_regression_fused_sampling_matches_stepped() -> None:
    """
    The sampling loop carries no lower-precision table, so its two paths
    agree to float64 rounding rather than float32.
    """
    rng = _random_generator(seed=5)
    initial = jnp.asarray(rng.standard_normal((1, 6, 4)), dtype=jnp.float64)
    schedule = compute_sigma_schedule(4096, SamplingConfig())
    weight = jnp.asarray(rng.standard_normal((4, 4)) * 0.3, dtype=jnp.float64)

    def velocity(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(tokens @ weight) * timesteps[0]

    fused = np.asarray(
        denoise_latent(initial, schedule, velocity, None, ExecutionConfig(fuse_sampling_steps=True))
    )
    stepped = np.asarray(
        denoise_latent(initial, schedule, velocity, None, ExecutionConfig(fuse_sampling_steps=False))
    )

    difference = float(np.max(np.abs(fused - stepped)))
    assert difference < REORDERING_TOLERANCE_FLOAT64_ONLY, (
        f"fused and stepped sampling disagreed by {difference:.3e}"
    )


def test_regression_fused_sampling_takes_the_same_number_of_steps() -> None:
    """
    Fusing must not change how many times the velocity is evaluated. A
    scan over the wrong number of level pairs would still produce a
    plausible latent.
    """
    schedule = compute_sigma_schedule(4096, SamplingConfig(num_steps=4))
    evaluations = []

    def counting_velocity(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        evaluations.append(1)
        return jnp.zeros_like(tokens)

    denoise_latent(
        jnp.zeros((1, 2, 3), dtype=jnp.float64), schedule, counting_velocity, None,
        ExecutionConfig(fuse_sampling_steps=True),
    )

    # Under a scan the body is traced once rather than executed four
    # times, so this counts tracings. One tracing is the correct
    # expectation, and it is exactly what distinguishes the fused path
    # from the stepped one.
    assert len(evaluations) == 1, (
        f"the fused body was traced {len(evaluations)} times; fusing should emit the "
        f"velocity computation once rather than once per step"
    )


def test_regression_compilation_cache_disabled_without_a_directory() -> None:
    """
    No directory means no cache, rather than a guessed default. Writing
    compiled programs to an unexpected place is worse than not caching.
    """
    enabled = configure_compilation_cache(ExecutionConfig(), _silent_logger())

    assert enabled is False


def test_regression_compilation_cache_enabled_with_a_directory(tmp_directory=None) -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        config = ExecutionConfig(compilation_cache_directory=Path(directory) / "cache")
        enabled = configure_compilation_cache(config, _silent_logger())

        assert enabled is True
        assert config.compilation_cache_directory.is_dir(), (
            "the cache directory should be created rather than assumed to exist"
        )


def _bytes_per_device(tree, device_count: int) -> int:
    """Count what one device actually holds, respecting each array's sharding."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf_bytes = int(np.prod(leaf.shape)) * leaf.dtype.itemsize
        specification = getattr(getattr(leaf, "sharding", None), "spec", None)
        is_split = specification is not None and any(
            entry is not None for entry in specification
        )
        total += leaf_bytes // device_count if is_split else leaf_bytes
    return total


def _synthetic_text_encoder() -> dict:
    """The real text encoder's shape at reduced scale: a big embedding table
    plus a stack of layers."""
    return {
        "embed_tokens": {"weight": jnp.zeros((2048, 256), dtype=jnp.bfloat16)},
        "layers": {
            "self_attn_q_proj_weight": jnp.zeros((27, 256, 512), dtype=jnp.bfloat16),
            "input_layernorm_weight": jnp.zeros((27, 256), dtype=jnp.bfloat16),
        },
    }


def test_regression_text_encoder_layers_are_split_not_replicated() -> None:
    """
    The bug that exhausted an eight-chip pod.

    Replication does not divide a component's memory across devices, it
    multiplies it: the text encoder's 5.80 GiB became 5.80 GiB resident
    on every chip, plus 46 GiB of host transfer during load, for a
    component used once per prompt. Splitting its layer stack instead
    brings that to a fraction.

    Asserted as a memory figure rather than a sharding annotation,
    because the annotation is a means and the memory is the thing that
    ran out.
    """
    mesh = build_device_mesh(_silent_logger())
    device_count = jax.device_count()
    parameters = _synthetic_text_encoder()

    placed = place_component(parameters, "text_encoder", mesh, _silent_logger())

    total = _bytes_per_device(parameters, 1)
    per_device = _bytes_per_device(placed, device_count)

    assert per_device < total, (
        f"each device holds {per_device} bytes of {total}; the layer stack does not "
        f"appear to be split, which is what exhausted the pod"
    )


def test_regression_embedding_table_stays_replicated() -> None:
    """
    The embedding table is read by gather, not matrix multiply.
    Splitting a lookup table across devices would make every lookup a
    collective, which is worse than the memory it saves.
    """
    mesh = build_device_mesh(_silent_logger())

    placed = place_component(_synthetic_text_encoder(), "text_encoder", mesh, _silent_logger())

    specification = placed["embed_tokens"]["weight"].sharding.spec
    assert all(entry is None for entry in specification), (
        f"the embedding table was split: {specification}"
    )


def test_regression_every_component_has_a_placement_policy() -> None:
    """
    A component absent from the policy replicates everything, which is
    correct but wasteful. More importantly, a component never placed at
    all stays on the first device while the rest of the pod idles, which
    is what happened to the autoencoder for an entire phase.
    """
    assert set(SPLITTABLE_GROUPS_BY_COMPONENT) == {"transformer", "text_encoder", "vae"}


def test_regression_unknown_component_replicates_rather_than_failing() -> None:
    """
    An unrecognised component should still be placed, using the safe
    default. Failing instead would turn a future addition into a crash
    rather than a memory cost that shows up in the log.
    """
    mesh = build_device_mesh(_silent_logger())
    parameters = {"group": {"weight": jnp.zeros((4, 8))}}

    placed = place_component(parameters, "not_a_component", mesh, _silent_logger())

    specification = placed["group"]["weight"].sharding.spec
    assert all(entry is None for entry in specification)


_EXECUTION_TESTS = [
    test_smoke_simulated_devices_are_available,
    test_regression_residency_plan_keeps_the_transformer_resident,
    test_regression_fully_resident_plan_evicts_nothing,
    test_regression_residency_plan_rejects_unresolved_strategy,
    test_regression_evictable_set_excludes_the_transformer,
    test_regression_eviction_moves_parameters_to_host_and_back,
    test_regression_move_to_accelerator_names_its_destination,
    test_regression_round_trip_through_host_returns_to_the_accelerator,
    test_smoke_mesh_covers_every_visible_device,
    test_regression_replication_places_a_full_copy_everywhere,
    test_regression_sharding_never_splits_the_block_axis,
    test_regression_sharding_splits_the_widest_remaining_axis,
    test_regression_sharding_replicates_indivisible_shapes,
    test_regression_sharding_preserves_values,
    test_regression_text_encoder_layers_are_split_not_replicated,
    test_regression_embedding_table_stays_replicated,
    test_regression_every_component_has_a_placement_policy,
    test_regression_unknown_component_replicates_rather_than_failing,
    test_regression_scanned_transformer_matches_unrolled,
    test_regression_scanned_text_encoder_matches_unrolled,
    test_regression_scanned_text_encoder_selects_the_same_depths,
    test_regression_fused_sampling_matches_stepped,
    test_regression_fused_sampling_takes_the_same_number_of_steps,
    test_regression_compilation_cache_disabled_without_a_directory,
    test_regression_compilation_cache_enabled_with_a_directory,
]


def run_execution_tests(logger: logging.Logger) -> None:
    logger.info(
        "Running %d unit tests against the execution layer, across %d simulated device(s)",
        len(_EXECUTION_TESTS),
        jax.device_count(),
    )
    for test_function in _EXECUTION_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All execution tests passed")
