"""
Describing arrays and parameter trees for the run log.

What is worth recording about a tensor is not its values but its
shape, its dtype, how many bytes it occupies, and where it lives. The
last of those has already caused two real bugs in this project: an
array left on the host when it was meant to be on the accelerator, and
a component that never reached the device mesh at all. Both were
invisible until much later, and both would have been obvious in a log
that stated placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


BYTES_PER_MEBIBYTE = 1024 ** 2
BYTES_PER_GIBIBYTE = 1024 ** 3


@dataclass(frozen=True)
class ArrayDescription:
    """What the log records about a single array."""

    shape: tuple[int, ...]
    dtype: str
    bytes_used: int
    devices: str
    sharding: str

    def format(self) -> str:
        return (
            f"shape={self.shape} dtype={self.dtype} "
            f"size={format_bytes(self.bytes_used)} "
            f"on={self.devices} sharding={self.sharding}"
        )


@dataclass(frozen=True)
class TreeDescription:
    """What the log records about a whole parameter tree."""

    leaf_count: int
    total_bytes: int
    dtypes: dict[str, int]
    devices: str
    sharded_leaf_count: int

    def format(self) -> str:
        dtype_summary = ", ".join(
            f"{name} x{count}" for name, count in sorted(self.dtypes.items())
        )
        return (
            f"{self.leaf_count} arrays, {format_bytes(self.total_bytes)} total, "
            f"dtypes [{dtype_summary}], on {self.devices}, "
            f"{self.sharded_leaf_count} of {self.leaf_count} split across devices"
        )


def format_bytes(count: int) -> str:
    """Render a byte count at a readable scale."""
    if count >= BYTES_PER_GIBIBYTE:
        return f"{count / BYTES_PER_GIBIBYTE:.2f} GiB"
    if count >= BYTES_PER_MEBIBYTE:
        return f"{count / BYTES_PER_MEBIBYTE:.1f} MiB"
    return f"{count} B"


def _describe_devices(array) -> str:
    """
    Name the devices an array occupies, compactly.

    A replicated array reports every device, which is noise once there
    are eight of them, so anything beyond a couple is summarised by
    count and platform instead.
    """
    try:
        devices = array.devices()
    except AttributeError:
        return "unknown"

    if len(devices) == 1:
        return str(next(iter(devices)))
    platform = next(iter(devices)).platform
    return f"{len(devices)}x {platform}"


def _describe_sharding(array) -> str:
    """
    Summarise how an array is split, or say that it is not.

    Reported as the partition specification rather than the full
    sharding object, since the specification is the part that answers
    the question actually being asked: which axis, if any, is divided.
    """
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        return "none"

    # A single-device sharding carries no partition specification. It is
    # reported by name rather than falling through to the class name,
    # because "sitting on one device" is a distinct and important state:
    # it is what an unplaced parameter looks like, and mistaking it for
    # a split one is how the decoder came to run on a single chip of an
    # eight-chip pod unnoticed.
    specification = getattr(sharding, "spec", None)
    if specification is None:
        return "single device"
    if all(entry is None for entry in specification):
        return "replicated"
    return str(specification)


def describe_array(array) -> ArrayDescription:
    """Describe one array without reading its values."""
    return ArrayDescription(
        shape=tuple(array.shape),
        dtype=str(array.dtype),
        bytes_used=int(np.prod(array.shape)) * array.dtype.itemsize,
        devices=_describe_devices(array),
        sharding=_describe_sharding(array),
    )


def describe_tree(tree) -> TreeDescription:
    """
    Summarise a parameter tree.

    Reports the mix of dtypes rather than a single dtype, because a tree
    that is mostly one dtype with a few stragglers is exactly the
    situation that broke the scanned block stack, and a summary claiming
    one dtype would have hidden it.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return TreeDescription(0, 0, {}, "none", 0)

    dtype_counts: dict[str, int] = {}
    total_bytes = 0
    # Counts leaves genuinely split across devices, not merely placed on
    # one. See _describe_sharding for why that distinction matters.
    sharded_leaves = 0

    for leaf in leaves:
        name = str(leaf.dtype)
        dtype_counts[name] = dtype_counts.get(name, 0) + 1
        total_bytes += int(np.prod(leaf.shape)) * leaf.dtype.itemsize
        if _describe_sharding(leaf) not in ("none", "replicated", "single device"):
            sharded_leaves += 1

    return TreeDescription(
        leaf_count=len(leaves),
        total_bytes=total_bytes,
        dtypes=dtype_counts,
        devices=_describe_devices(leaves[0]),
        sharded_leaf_count=sharded_leaves,
    )


def summarise_values(array, sample_limit: int = 100_000) -> str:
    """
    Report an array's value range, for spotting a stage that has gone
    numerically wrong.

    Reading values forces the computation and copies to the host, so
    this is expensive and only called at high verbosity. Large arrays
    are sampled rather than read whole, since a range estimate does not
    need every element and copying a gigabyte to describe it would cost
    more than the stage being described.

    Non-finite values are counted rather than summarised, because a
    single infinity makes a minimum or maximum meaningless while being
    exactly what a reader most needs to know.
    """
    values = np.asarray(jax.device_get(array)).ravel()
    if values.size > sample_limit:
        stride = values.size // sample_limit
        values = values[::stride]

    # Promote before computing statistics. Summing even a modest number
    # of bfloat16 values in bfloat16 saturates: an array of ten thousand
    # ones reports a mean near zero, which would send a reader hunting
    # for a bug in the stage being described rather than in the
    # description. The same promotion rule applies here as everywhere
    # else in this codebase.
    values = values.astype(np.float32)

    finite = np.isfinite(values)
    non_finite_count = int((~finite).sum())
    if non_finite_count:
        return f"NON-FINITE: {non_finite_count} of {values.size} sampled values"

    return (
        f"min={values.min():.4f} max={values.max():.4f} "
        f"mean={values.mean():.4f} std={values.std():.4f}"
    )
