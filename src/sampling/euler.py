"""
Explicit Euler integration of the velocity field.

The sampler starts from noise and integrates toward a clean latent. At
each step the transformer predicts a velocity, which is a direction
rather than a destination, and the latent moves along it by the
difference between the current and next noise levels.

Why the integrator is not upgradeable
-------------------------------------
Substituting a higher-order solver here, Heun or a DPM variant, is a
natural instinct and would be wrong. This checkpoint was distilled
against explicit Euler at four specific noise levels: the weights
encode that trajectory. A higher-order method corrects toward the true
solution of the underlying differential equation, which is not the
trajectory the model was trained to follow, so at equal evaluation
count it can produce worse output rather than better.

The reference enforces this by rejecting attempts to change the step
count at all. Speed on this model comes from compiling and scheduling
the same arithmetic better, never from changing what arithmetic runs.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..config import ExecutionConfig


# The reference accumulates the latent in bfloat16, the dtype the
# transformer's weights carry. Accumulating in float32 would be
# marginally more accurate but is a deviation from the reference, and
# across only four steps the difference is far below anything
# measurable in the decoded image. Matching the reference wins.
VelocityPredictor = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def denoise_latent(
    initial_latent_tokens: jnp.ndarray,
    sigma_schedule: np.ndarray,
    predict_velocity: VelocityPredictor,
    logger=None,
    execution: ExecutionConfig | None = None,
) -> jnp.ndarray:
    """
    Integrate the velocity field from the first noise level to the last.

    Parameters
    ----------
    initial_latent_tokens:
        Shape (batch, num_tokens, channels), pure noise scaled to the
        first noise level.
    sigma_schedule:
        The noise levels, of length num_steps + 1. Each adjacent pair
        defines one step.
    predict_velocity:
        Called as predict_velocity(latent_tokens, timesteps) and
        returning a velocity of the same shape as its first argument.
        Taking a callable rather than the model itself keeps this
        module independent of the transformer: the sampler does not
        need to know what produces the velocity, and tests can supply a
        known field with an analytically checkable solution.
    execution:
        Whether to fuse the steps into one compiled program. Fusing is
        the default and gives up per-step logging, since a Python log
        statement inside a compiled region runs once at trace time.
    logger:
        Optional. Receives one line per step when the steps are not
        fused, and a single summary line when they are. Sampling is the outermost
        loop of a generation, so progress here is the most useful thing
        to log; the numeric functions it calls deliberately log nothing.

    Returns
    -------
    The final latent tokens, same shape as the input.
    """
    if sigma_schedule.ndim != 1 or sigma_schedule.shape[0] < 2:
        raise ValueError(
            f"Sigma schedule must be a one dimensional array of at least two levels, "
            f"got shape {sigma_schedule.shape}"
        )

    execution = execution or ExecutionConfig()
    latent_tokens = initial_latent_tokens
    batch = latent_tokens.shape[0]

    if execution.fuse_sampling_steps:
        return _denoise_fused(latent_tokens, sigma_schedule, predict_velocity, logger)

    for step_index in range(sigma_schedule.shape[0] - 1):
        current_sigma = float(sigma_schedule[step_index])
        next_sigma = float(sigma_schedule[step_index + 1])

        timesteps = jnp.full((batch,), current_sigma, dtype=latent_tokens.dtype)
        velocity = predict_velocity(latent_tokens, timesteps)

        # The step is the signed difference between noise levels, which
        # is negative because the schedule descends. Writing it as
        # next minus current rather than negating a positive step keeps
        # the direction implicit in the schedule rather than duplicated
        # here.
        latent_tokens = latent_tokens + (next_sigma - current_sigma) * velocity

        if logger is not None:
            logger.info(
                "sampling step %d of %d: sigma %.4f to %.4f",
                step_index + 1,
                sigma_schedule.shape[0] - 1,
                current_sigma,
                next_sigma,
            )

    return latent_tokens


def _denoise_fused(
    initial_latent_tokens: jnp.ndarray,
    sigma_schedule: np.ndarray,
    predict_velocity: VelocityPredictor,
    logger=None,
) -> jnp.ndarray:
    """
    Run every step inside one compiled program.

    The steps become a scan over adjacent pairs of noise levels, so the
    compiler sees the whole trajectory at once: it can keep the latent
    in place between steps rather than round-tripping it, and emits the
    velocity computation once rather than once per step.

    The levels are passed as traced values rather than baked in as
    constants, which is what lets one compiled program serve any
    schedule of the same length. Baking them in would recompile whenever
    a resolution changed the schedule.

    Per-step logging is not available here, for the same reason logging
    is absent from every numeric function in this package: inside a
    compiled region it would report tracing rather than execution.
    """
    current_levels = jnp.asarray(sigma_schedule[:-1], dtype=initial_latent_tokens.dtype)
    next_levels = jnp.asarray(sigma_schedule[1:], dtype=initial_latent_tokens.dtype)
    batch = initial_latent_tokens.shape[0]

    if logger is not None:
        logger.info(
            "Running %d sampling steps as one fused program", current_levels.shape[0]
        )

    def take_one_step(latent_tokens, levels):
        current_sigma, next_sigma = levels
        timesteps = jnp.full((batch,), current_sigma, dtype=latent_tokens.dtype)
        velocity = predict_velocity(latent_tokens, timesteps)
        return latent_tokens + (next_sigma - current_sigma) * velocity, None

    final_tokens, _ = jax.lax.scan(
        take_one_step, initial_latent_tokens, (current_levels, next_levels)
    )
    return final_tokens
