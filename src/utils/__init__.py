"""
Cross-cutting utilities with no model or configuration knowledge.

Nothing here may import from any other package in src, which keeps
these modules usable from every layer without creating a cycle.
"""

from .logging import configure_logging

__all__ = ["configure_logging"]
