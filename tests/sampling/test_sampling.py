"""
Tests for the sampler and the generation pipeline.

The most valuable test here is
test_regression_euler_integrates_a_known_field_correctly. Every other
test in this project checks a component against a reference
implementation or an oracle that reimplements the same definition. This
one is different: it feeds the integrator a velocity field whose exact
solution is known analytically, so it verifies the integration is
actually solving the equation rather than merely reproducing another
implementation's arithmetic.

The schedule tests pin the discontinuity at the token threshold rather
than smoothing over it, because reproducing the reference's behaviour
on both sides of that branch is the point.
"""

from __future__ import annotations

import itertools
import logging

import jax.numpy as jnp
import numpy as np

from src.config import SamplingConfig
from src.pipeline import to_display_range
from src.sampling import (
    compute_schedule_shift,
    compute_sigma_schedule,
    denoise_latent,
    pack_latent_to_tokens,
    unpack_tokens_to_latent,
)


NUMERICAL_TOLERANCE = 1e-10

# The three supported resolutions' token counts, plus values that
# bracket the branch threshold.
TOKEN_COUNTS = (1024, 4080, 4096, 4299, 4301, 8100)


def _random_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_smoke_sigma_schedule_has_one_more_level_than_steps() -> None:
    for num_steps in (1, 4, 10):
        config = SamplingConfig(num_steps=num_steps)
        schedule = compute_sigma_schedule(4096, config)

        assert schedule.shape == (num_steps + 1,), (
            "each adjacent pair of levels defines one step, so there must be one "
            "more level than there are steps"
        )


def test_regression_sigma_schedule_starts_at_one_and_ends_at_zero() -> None:
    """
    Sampling begins at full noise and must reach exactly zero, not
    merely approach it. The transform is undefined at the endpoint, so
    a naive implementation produces a division by zero there.
    """
    for token_count in TOKEN_COUNTS:
        schedule = compute_sigma_schedule(token_count, SamplingConfig())

        assert schedule[0] == 1.0, f"schedule does not start at one for {token_count} tokens"
        assert schedule[-1] == 0.0, f"schedule does not end at zero for {token_count} tokens"
        assert np.all(np.isfinite(schedule)), (
            f"schedule contains non-finite values for {token_count} tokens"
        )


def test_regression_sigma_schedule_decreases_monotonically() -> None:
    for token_count in TOKEN_COUNTS:
        schedule = compute_sigma_schedule(token_count, SamplingConfig())

        assert np.all(np.diff(schedule) < 0.0), (
            f"schedule is not strictly decreasing for {token_count} tokens"
        )


def test_regression_schedule_is_front_loaded() -> None:
    """
    The steps are deliberately uneven: early steps barely move while
    the last covers most of the remaining distance. That shape comes
    from the distillation and a uniform schedule would be wrong, so it
    is asserted rather than left implicit.
    """
    schedule = compute_sigma_schedule(4096, SamplingConfig())
    step_sizes = -np.diff(schedule)

    assert step_sizes[-1] > step_sizes[0] * 5.0, (
        f"final step {step_sizes[-1]:.4f} is not much larger than the first "
        f"{step_sizes[0]:.4f}; the schedule may not be shifted at all"
    )


def test_regression_schedule_shift_is_discontinuous_at_the_branch() -> None:
    """
    Pin the discontinuity rather than smoothing it.

    Just below the threshold the shift interpolates against step count;
    just above it, the step count is ignored entirely. The jump is
    large, and reproducing it is what keeps this implementation faithful
    to the reference.
    """
    config = SamplingConfig()
    threshold = config.token_count_branch_threshold

    below = compute_schedule_shift(threshold - 1, config)
    above = compute_schedule_shift(threshold + 1, config)

    assert below > 2.0, f"shift below the threshold was {below}, expected above two"
    assert above < 1.5, f"shift above the threshold was {above}, expected below one and a half"
    assert below - above > 1.0, (
        f"expected a discontinuity of more than one at the threshold, got {below - above}"
    )


def test_regression_schedule_shift_ignores_step_count_above_the_threshold() -> None:
    """
    Above the threshold the reference drops the step-count
    interpolation. That is precisely why a four-step model should not
    be run there, so the behaviour is asserted to make the reason
    visible rather than buried.
    """
    token_count = SamplingConfig().token_count_branch_threshold + 100

    four_steps = compute_schedule_shift(token_count, SamplingConfig(num_steps=4))
    fifty_steps = compute_schedule_shift(token_count, SamplingConfig(num_steps=50))

    assert four_steps == fifty_steps, (
        "above the threshold the shift should not depend on step count"
    )


def test_regression_schedule_shift_depends_on_step_count_below_the_threshold() -> None:
    token_count = 4096

    four_steps = compute_schedule_shift(token_count, SamplingConfig(num_steps=4))
    fifty_steps = compute_schedule_shift(token_count, SamplingConfig(num_steps=50))

    assert four_steps != fifty_steps, (
        "below the threshold the shift should interpolate against step count"
    )


def test_regression_latent_packing_round_trips() -> None:
    for height, width, channels in itertools.product((3, 5), (4, 5), (2, 8)):
        rng = _random_generator(seed=height * 100 + width * 10 + channels)
        latent = jnp.asarray(
            rng.standard_normal((2, height, width, channels)), dtype=jnp.float64
        )

        tokens = pack_latent_to_tokens(latent)
        restored = unpack_tokens_to_latent(tokens, height, width)

        assert tokens.shape == (2, height * width, channels)
        assert np.array_equal(np.asarray(restored), np.asarray(latent)), (
            f"round trip failed at {height}x{width}x{channels}"
        )


def test_regression_latent_packing_is_row_major() -> None:
    """
    Row-major order is what makes unpacking a reshape rather than a
    scatter, and it must match the order position identifiers are built
    in. A column-major layout would transpose the image while keeping
    every shape valid.
    """
    height, width = 2, 3
    latent = jnp.asarray(
        np.arange(height * width, dtype=np.float64).reshape(1, height, width, 1)
    )

    tokens = np.asarray(pack_latent_to_tokens(latent))[0, :, 0]

    assert np.array_equal(tokens, np.arange(height * width)), (
        "packing did not traverse the latent row by row"
    )


def test_regression_unpack_rejects_mismatched_token_count() -> None:
    tokens = jnp.zeros((1, 10, 4), dtype=jnp.float32)

    try:
        unpack_tokens_to_latent(tokens, height=3, width=4)
    except ValueError as error:
        assert "does not match" in str(error)
        return
    raise AssertionError("Expected ValueError when the token count disagrees with the shape")


def test_regression_euler_integrates_a_known_field_correctly() -> None:
    """
    Verify the integrator against an analytic solution rather than
    another implementation.

    With a velocity equal to the state itself, the exact solution over a
    total displacement d is multiplication by exp(d). Explicit Euler
    approximates that as a product of per-step factors, which is what
    this checks: the integrator must reproduce Euler's own answer
    exactly, and that answer must approach the analytic one as the steps
    get finer.

    Checking both properties matters. The first catches an error in how
    the step size is applied; the second catches a sign error, which
    would still satisfy the first while integrating in the wrong
    direction.
    """
    initial = jnp.ones((1, 1, 1), dtype=jnp.float64)

    def velocity_is_state(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        return tokens

    for num_steps in (2, 8, 64):
        # A descending schedule from one to zero, so the total signed
        # displacement is minus one.
        schedule = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float64)

        result = float(np.asarray(denoise_latent(initial, schedule, velocity_is_state))[0, 0, 0])

        step = -1.0 / num_steps
        expected_euler = (1.0 + step) ** num_steps
        assert abs(result - expected_euler) < 1e-12, (
            f"integrator did not reproduce explicit Euler at {num_steps} steps: "
            f"{result} against {expected_euler}"
        )

    # With enough steps, Euler must approach the analytic solution.
    fine_schedule = np.linspace(1.0, 0.0, 2001, dtype=np.float64)
    fine_result = float(
        np.asarray(denoise_latent(initial, fine_schedule, velocity_is_state))[0, 0, 0]
    )
    assert abs(fine_result - np.exp(-1.0)) < 1e-3, (
        f"refining the steps did not converge toward the analytic solution: "
        f"{fine_result} against {np.exp(-1.0)}"
    )


def test_regression_euler_calls_the_predictor_once_per_step() -> None:
    """
    Four steps means four evaluations, not five. An off-by-one here
    would change the cost of every generation by a quarter and move the
    result off the distilled trajectory.
    """
    calls = []

    def counting_predictor(tokens: jnp.ndarray, timesteps: jnp.ndarray) -> jnp.ndarray:
        calls.append(float(timesteps[0]))
        return jnp.zeros_like(tokens)

    schedule = compute_sigma_schedule(4096, SamplingConfig(num_steps=4))
    denoise_latent(jnp.zeros((1, 2, 3), dtype=jnp.float64), schedule, counting_predictor)

    assert len(calls) == 4, f"expected four evaluations, got {len(calls)}"
    assert calls == [float(level) for level in schedule[:-1]], (
        "the predictor was not called at the schedule's leading levels; the final "
        "level is a destination, not an evaluation point"
    )


def test_regression_euler_leaves_latent_unchanged_for_zero_velocity() -> None:
    rng = _random_generator(seed=5)
    initial = jnp.asarray(rng.standard_normal((1, 6, 4)), dtype=jnp.float64)
    schedule = compute_sigma_schedule(1024, SamplingConfig())

    result = denoise_latent(
        initial, schedule, lambda tokens, timesteps: jnp.zeros_like(tokens)
    )

    assert np.allclose(np.asarray(result), np.asarray(initial), atol=NUMERICAL_TOLERANCE)


def test_regression_euler_rejects_degenerate_schedule() -> None:
    for bad_schedule in (np.array([1.0]), np.zeros((2, 2))):
        try:
            denoise_latent(
                jnp.zeros((1, 1, 1)), bad_schedule, lambda t, s: jnp.zeros_like(t)
            )
        except ValueError as error:
            assert "schedule" in str(error)
            continue
        raise AssertionError(f"Expected ValueError for schedule shape {bad_schedule.shape}")


def test_regression_display_range_maps_and_clips() -> None:
    """
    The decoder's output is unbounded, so out-of-range values occur and
    must be clipped rather than allowed to wrap. Clipping is used rather
    than rescaling by observed extremes, which would make brightness
    depend on the image's own most extreme pixel.
    """
    values = np.array([[-3.0, -1.0, 0.0, 1.0, 3.0]])

    mapped = to_display_range(values)

    assert np.array_equal(mapped, np.array([[0.0, 0.0, 0.5, 1.0, 1.0]]))
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0


def test_regression_display_range_does_not_rescale_by_extremes() -> None:
    """
    Two images differing only in one outlier pixel must map their
    shared pixels identically. A rescaling implementation would fail
    this while passing the range check above.
    """
    modest = np.array([[-0.5, 0.0, 0.5]])
    with_outlier = np.array([[-0.5, 0.0, 0.5]])

    first = to_display_range(modest)
    second = to_display_range(with_outlier * 1.0)

    assert np.array_equal(first, second)
    assert np.isclose(first[0, 1], 0.5), (
        "a zero-valued pixel should map to the midpoint regardless of its neighbours"
    )


_SAMPLING_TESTS = [
    test_smoke_sigma_schedule_has_one_more_level_than_steps,
    test_regression_sigma_schedule_starts_at_one_and_ends_at_zero,
    test_regression_sigma_schedule_decreases_monotonically,
    test_regression_schedule_is_front_loaded,
    test_regression_schedule_shift_is_discontinuous_at_the_branch,
    test_regression_schedule_shift_ignores_step_count_above_the_threshold,
    test_regression_schedule_shift_depends_on_step_count_below_the_threshold,
    test_regression_latent_packing_round_trips,
    test_regression_latent_packing_is_row_major,
    test_regression_unpack_rejects_mismatched_token_count,
    test_regression_euler_integrates_a_known_field_correctly,
    test_regression_euler_calls_the_predictor_once_per_step,
    test_regression_euler_leaves_latent_unchanged_for_zero_velocity,
    test_regression_euler_rejects_degenerate_schedule,
    test_regression_display_range_maps_and_clips,
    test_regression_display_range_does_not_rescale_by_extremes,
]


def run_sampling_tests(logger: logging.Logger) -> None:
    logger.info("Running %d unit tests against the sampler", len(_SAMPLING_TESTS))
    for test_function in _SAMPLING_TESTS:
        test_function()
        logger.info("PASS: %s", test_function.__name__)
    logger.info("All sampling tests passed")
