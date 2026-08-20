"""Reporting what the accelerator is and how much of it is in use."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax

from .arrays import format_bytes


# Keys JAX exposes in a device's memory statistics. Not every backend
# provides them, and CPU provides none at all, so each is read
# defensively rather than assumed.
BYTES_IN_USE_KEY = "bytes_in_use"
PEAK_BYTES_IN_USE_KEY = "peak_bytes_in_use"
BYTES_LIMIT_KEY = "bytes_limit"


@dataclass(frozen=True)
class DeviceMemory:
    """A snapshot of one device's memory, where the backend reports it."""

    device_label: str
    bytes_in_use: int | None
    peak_bytes_in_use: int | None
    bytes_limit: int | None

    @property
    def available(self) -> bool:
        return self.bytes_in_use is not None

    def format(self) -> str:
        if not self.available:
            return f"{self.device_label}: memory statistics not reported by this backend"

        parts = [f"{self.device_label}: {format_bytes(self.bytes_in_use)} in use"]
        if self.bytes_limit:
            share = 100.0 * self.bytes_in_use / self.bytes_limit
            parts.append(f"of {format_bytes(self.bytes_limit)} ({share:.1f}%)")
        if self.peak_bytes_in_use:
            parts.append(f"peak {format_bytes(self.peak_bytes_in_use)}")
        return " ".join(parts)


def read_device_memory(device) -> DeviceMemory:
    """
    Read one device's memory statistics, tolerating backends that do not
    report them.

    CPU reports nothing at all, so this must degrade to a stated absence
    rather than an error or, worse, a zero that would read as an empty
    device.
    """
    statistics = None
    try:
        statistics = device.memory_stats()
    except Exception:
        # Some backends raise rather than returning None. Either way the
        # answer is the same: no statistics available.
        statistics = None

    if not statistics:
        return DeviceMemory(str(device), None, None, None)

    return DeviceMemory(
        device_label=str(device),
        bytes_in_use=statistics.get(BYTES_IN_USE_KEY),
        peak_bytes_in_use=statistics.get(PEAK_BYTES_IN_USE_KEY),
        bytes_limit=statistics.get(BYTES_LIMIT_KEY),
    )


def log_platform(logger: logging.Logger) -> None:
    """Record what JAX is running on, once, at startup."""
    devices = jax.devices()
    first = devices[0]

    logger.info("JAX %s on %s", jax.__version__, first.platform)
    logger.info(
        "  %d device(s): %s, kind %s",
        len(devices),
        first.platform,
        getattr(first, "device_kind", "unknown"),
    )
    logger.info("  x64 mode: %s", jax.config.jax_enable_x64)

    for device in devices:
        memory = read_device_memory(device)
        if memory.available:
            logger.info("  %s", memory.format())


def log_memory_snapshot(logger: logging.Logger, label: str) -> None:
    """
    Record accelerator memory at a point in time.

    Only the first device is reported for a multi-device run. On a pod
    every chip holds a near-identical share, so reporting all of them
    would multiply the log without adding information; a genuine
    imbalance shows up in the sharding descriptions instead.
    """
    memory = read_device_memory(jax.devices()[0])
    if memory.available:
        logger.info("memory after %s: %s", label, memory.format())
