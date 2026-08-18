"""
Complete networks assembled from blocks.

Models sit at the top of the dependency order: they may import from
every other package, and nothing imports from them.
"""

from .vae import decode_latent

__all__ = ["decode_latent"]
