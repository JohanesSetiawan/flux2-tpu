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

import jax.numpy as jnp
import numpy as np


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
    logger:
        Optional. Receives one line per step. Sampling is the outermost
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

    latent_tokens = initial_latent_tokens
    batch = latent_tokens.shape[0]

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
