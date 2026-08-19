"""
The rectified-flow sampler.

Turns a noise sample into a latent by integrating the velocity field
the transformer predicts. The integration is explicit Euler at a fixed
number of points; see `euler` for why neither the integrator nor the
step count is adjustable.
"""

from .euler import denoise_latent
from .latent import pack_latent_to_tokens, unpack_tokens_to_latent
from .schedule import compute_schedule_shift, compute_sigma_schedule

__all__ = [
    "compute_schedule_shift",
    "compute_sigma_schedule",
    "denoise_latent",
    "pack_latent_to_tokens",
    "unpack_tokens_to_latent",
]
