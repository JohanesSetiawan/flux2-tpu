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

    # Record a per-stage timing profile, tensor placements and device
    # memory for every run.
    #
    # On by default. The cost is a few log lines and, more importantly,
    # the loss of overlap between stages: accurate timing requires
    # waiting for each stage's result, which prevents JAX from running
    # the next one early. Across a handful of coarse stages that
    # overlap is small, and a profile that cannot be trusted is worth
    # less than the overlap it costs.
    enable_telemetry: bool = True

    # Restore parameters directly onto their target devices instead of
    # restoring first and placing afterwards.
    #
    # The theory is sound: the two-step path moves every byte twice,
    # which for the diffusion transformer is a redundant 7.2 GB copy.
    # The evidence is not. Measured on CPU it came out slower, roughly
    # 3.4 seconds against 1.4, presumably because placement there is
    # nearly free and the metadata read is not.
    #
    # It is therefore off by default rather than shipped as an
    # improvement on the strength of an argument. Whether it helps
    # depends on how expensive placement actually is, which differs
    # between a single chip, where the second copy may be elided
    # entirely, and a pod, where restoring to one device and then
    # replicating to eight is a real cost. Turn it on and compare the
    # restore stage in the profile.
    restore_directly_onto_devices: bool = False

    # Import the tokenizer library on a background thread while the
    # checkpoint downloads.
    #
    # The import costs around 55 seconds on Colab, against a download of
    # around 99. Run in sequence that is 154 seconds; overlapped, the
    # import disappears into the download entirely. It is pure waiting
    # on both sides, so there is nothing to contend over.
    preload_tokenizer_during_download: bool = True

    # Also read back and summarise each stage's output values.
    #
    # Off by default because it forces a copy from the accelerator to
    # the host for tensors that would otherwise stay put, which is
    # genuinely expensive at full resolution. Turn it on when a stage is
    # suspected of producing something numerically wrong, since it is
    # the fastest way to find which stage first emits a non-finite
    # value.
    enable_value_summaries: bool = False
