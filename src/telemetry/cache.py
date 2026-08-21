"""
Reporting whether the compilation cache was used.

XLA's persistent cache turns minutes of compilation into seconds, but
it is silent about whether it worked. A cache that misses behaves
exactly like no cache at all: nothing fails, nothing warns, the run is
simply slow again. On a platform where a session costs several minutes
to reach, that is worth reporting explicitly rather than inferring from
a stopwatch.

What is in the cache is compiled machine code, not metadata. One decode
program is around 16 MB compressed and 78 MB expanded, and it carries
the paths of the source files it was compiled from. That is why a cache
entry is tied to the exact program that produced it: change the code so
the graph changes, and the key changes with it.

Three things decide whether an entry is found:

  the program        the traced graph, which includes parameter
                     sharding, so a placement change invalidates entries
  the runtime        jaxlib's version
  the hardware       chip generation and topology

A miss on any of them is harmless. XLA compiles as it would have anyway
and stores the result. The only cost is time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .arrays import format_bytes


# Entries are written as one file per compiled program, named after the
# function and a hash of everything the key depends on.
CACHE_ENTRY_SUFFIX = "-cache"


@dataclass(frozen=True)
class CacheContents:
    """A snapshot of what a cache directory holds."""

    entry_names: frozenset[str]
    total_bytes: int

    @property
    def entry_count(self) -> int:
        return len(self.entry_names)

    def format(self) -> str:
        if not self.entry_names:
            return "empty"
        return f"{self.entry_count} entries, {format_bytes(self.total_bytes)}"


def read_cache_contents(directory: Path | None) -> CacheContents:
    """
    List what a cache directory currently holds.

    A missing directory reports as empty rather than raising: no cache
    configured and an empty cache lead to the same behaviour, and a
    caller comparing before against after should not have to special
    case the first run.
    """
    if directory is None or not directory.is_dir():
        return CacheContents(entry_names=frozenset(), total_bytes=0)

    entries = [path for path in directory.iterdir() if path.name.endswith(CACHE_ENTRY_SUFFIX)]
    return CacheContents(
        entry_names=frozenset(path.name for path in entries),
        total_bytes=sum(path.stat().st_size for path in entries),
    )


def report_cache_effect(
    before: CacheContents,
    after: CacheContents,
    logger: logging.Logger,
) -> None:
    """
    Say what the cache did, by comparing before against after.

    Entries added during a stage are programs that had to be compiled,
    which is a miss. Entries that were already present and no new ones
    appearing means every program was found, which is a hit.

    This is inferred from the directory rather than read from XLA, which
    exposes no hit counter. The inference is sound for the question
    being asked: a program that compiled left a new file behind, and one
    that did not, did not.
    """
    added = after.entry_names - before.entry_names

    if not before.entry_names and not added:
        logger.info("Compilation cache: nothing cached and nothing compiled")
        return

    if not before.entry_names:
        logger.info(
            "Compilation cache was empty and now holds %s. The next run reuses this "
            "and should start far faster.",
            after.format(),
        )
        return

    if not added:
        logger.info(
            "Compilation cache hit for every program: %s reused, nothing compiled",
            before.format(),
        )
        return

    logger.info(
        "Compilation cache partially reused: %d of %d programs were already cached, "
        "%d had to be compiled and were added",
        before.entry_count,
        after.entry_count,
        len(added),
    )
    for name in sorted(added):
        # Trimmed to the function name; the hash is long and identifies
        # the exact graph rather than anything a reader can act on.
        logger.info("    compiled: %s", name.split("-")[0])
    logger.info(
        "  A miss usually means the code, the JAX version, or the accelerator "
        "changed since these entries were written. It costs time, nothing else."
    )
