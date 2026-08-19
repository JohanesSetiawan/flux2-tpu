"""
The noise schedule.

Two functions: one computes a shift parameter from the image size, the
other turns that into the sequence of noise levels the sampler steps
through.
"""

from __future__ import annotations

import numpy as np

from ..config import SamplingConfig


def compute_schedule_shift(image_token_count: int, config: SamplingConfig) -> float:
    """
    Compute the shift parameter that determines the schedule's shape.

    Larger images need more of the schedule spent at high noise, and
    this parameter encodes that relationship. The reference fits it
    empirically as a function of token count, interpolated against step
    count.

    A discontinuity worth understanding
    -----------------------------------
    Above config.token_count_branch_threshold the reference abandons the
    step-count interpolation and uses the high-step formula alone. The
    jump is not small: at 4299 tokens the shift is about 2.31, and at
    4301 it drops to about 1.18, producing a materially different
    sequence of noise levels.

    For a model distilled against four steps this matters, because the
    formula above the threshold is the one derived for two hundred. That
    is why the supported resolutions all sit below the threshold, and
    why exceeding it is a quality decision rather than merely a memory
    one.

    This function reproduces the reference's behaviour on both sides of
    the branch rather than smoothing it, since the goal is to match the
    reference rather than to improve on it.
    """
    high_step_value = (
        config.high_step_slope * image_token_count + config.high_step_intercept
    )

    if image_token_count > config.token_count_branch_threshold:
        return float(high_step_value)

    low_step_value = config.low_step_slope * image_token_count + config.low_step_intercept

    step_span = config.interpolation_high_steps - config.interpolation_low_steps
    slope = (high_step_value - low_step_value) / step_span
    intercept = high_step_value - config.interpolation_high_steps * slope

    return float(slope * config.num_steps + intercept)


def compute_sigma_schedule(image_token_count: int, config: SamplingConfig) -> np.ndarray:
    """
    Build the sequence of noise levels the sampler steps through.

    Returns config.num_steps + 1 values, from one down to zero. Each
    adjacent pair defines one step: the sampler moves from the first to
    the second.

    The values are not evenly spaced. A uniform sequence is generated
    first and then bent by the shift parameter, which concentrates the
    steps at high noise. For this checkpoint the effect is pronounced:
    the first three steps barely move while the last covers most of the
    remaining distance. That shape is a property of the distillation,
    not an error.

    Computed in float64 on the host and returned as a plain array. It
    depends only on the token count, so a caller should compute it once
    per resolution rather than per generation.
    """
    shift = compute_schedule_shift(image_token_count, config)

    uniform = np.linspace(1.0, 0.0, config.num_steps + 1, dtype=np.float64)

    # The transform is undefined at zero, where the reciprocal diverges.
    # The limit there is zero, which is also the value the schedule must
    # end on, so it is substituted directly rather than computed.
    shifted = np.empty_like(uniform)
    interior = uniform > 0.0
    shifted[~interior] = 0.0

    exponential_shift = np.exp(shift)
    reciprocal_term = (1.0 / uniform[interior] - 1.0) ** config.snr_shift_exponent
    shifted[interior] = exponential_shift / (exponential_shift + reciprocal_term)

    return shifted
