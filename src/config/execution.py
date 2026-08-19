"""Configuration for how work is compiled and where weights live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionConfig:
    """
    Settings that change how the model is executed without changing what
    it computes.

    Every option here is required to be output-preserving. The test
    suite asserts that directly rather than trusting it: a scanned block
    stack and an unrolled one must agree exactly, and so must a fused
    sampling loop and a stepped one. An option that changed results
    would belong in a model configuration, not here.
    """

    # Drive repeated blocks with jax.lax.scan rather than a Python loop.
    #
    # Both produce identical results. The difference is compilation: a
    # Python loop emits the block body once per block, so the diffusion
    # transformer's twenty-five blocks become twenty-five copies in the
    # compiled program. A scan emits it once. On TPU that is the
    # difference between a compile measured in minutes and one measured
    # in seconds, and it also shrinks the memory the compiler needs to
    # plan the program.
    #
    # The unrolled path is kept rather than deleted because scan makes
    # runtime errors considerably harder to localise: a failure inside a
    # scanned body reports the scan, not the block. Turning this off is
    # the fastest way to find out which block is at fault.
    use_scan_over_blocks: bool = True

    # Compile the whole sampling loop into one program rather than
    # dispatching each step separately.
    #
    # Fusing removes per-step dispatch overhead and lets the compiler
    # keep the latent in place across steps instead of round-tripping
    # it. The cost is that per-step progress logging becomes impossible,
    # since a Python log statement inside a compiled region runs once at
    # trace time and never again.
    fuse_sampling_steps: bool = True

    # Directory for XLA's persistent compilation cache. When set,
    # compiled programs survive process restarts, which matters most in
    # hosted notebooks where sessions end frequently and every restart
    # would otherwise pay full compilation again.
    #
    # None disables the cache rather than choosing a default location,
    # because writing to an unexpected directory is worse than not
    # caching.
    compilation_cache_directory: Path | None = None

    # Minimum compilation time before a program is worth caching. Very
    # short compilations cost more to store and reload than to repeat.
    compilation_cache_minimum_seconds: float = 1.0
