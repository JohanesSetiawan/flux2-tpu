"""Configuration for the sampling schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingConfig:
    """
    Parameters of the rectified-flow sampling schedule.

    The step count is not a tunable quality-versus-speed dial. This
    checkpoint was distilled against one specific four-point Euler
    trajectory, and the reference implementation rejects attempts to
    change either the step count or the guidance value. Raising the step
    count does not improve quality and lowering it does not merely
    degrade gracefully: both move the sampler off the trajectory the
    weights were trained for.

    The empirical coefficients below fit a curve relating image token
    count to schedule shape. They are reproduced from the reference
    rather than derived, and there is no interpretation of them beyond
    that.
    """

    num_steps: int = 4

    # Above this token count the reference switches to a formula that
    # ignores the step count entirely. See the note on the branch in
    # compute_schedule_shift for why that matters for a four-step model.
    token_count_branch_threshold: int = 4300

    low_step_slope: float = 8.73809524e-05
    low_step_intercept: float = 1.89833333
    high_step_slope: float = 0.00016927
    high_step_intercept: float = 0.45666666

    # The two reference points the interpolation runs between, in steps.
    interpolation_low_steps: float = 10.0
    interpolation_high_steps: float = 200.0

    # Exponent in the signal-to-noise shift. The reference passes one
    # and offers no other value.
    snr_shift_exponent: float = 1.0
