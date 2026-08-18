"""
Parameter access helpers.

The converted checkpoint stores each component's tensors in a flat
dictionary whose keys are the original PyTorch module paths with dots
replaced by underscores, for example `mid_block_1_conv1_weight` or
`up_3_block_0_nin_shortcut_bias`. Flattening happened during conversion
because it makes the checkpoint trivially inspectable and keeps the
Orbax pytree shallow.

Model code, by contrast, is written per block: a residual block wants
`conv1_weight`, not `up_3_block_0_conv1_weight`. This module bridges the
two, and is the only place that knows about the flat naming convention.
Keeping that knowledge here means a change to the conversion's naming
scheme is a change to one function rather than to every block
implementation.
"""

from __future__ import annotations

import numpy as np


PARAMETER_PATH_SEPARATOR = "_"


class MissingParameterError(KeyError):
    """
    Raised when a block is asked to run without a tensor it requires.

    This is a distinct exception type rather than a bare KeyError so
    that a missing weight, which almost always means the checkpoint and
    the code disagree about structure, is not silently caught by an
    unrelated `except KeyError` somewhere up the stack.
    """


def select_parameter_group(parameters: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    """
    Return every parameter whose key begins with `prefix`, with the
    prefix and its trailing separator removed from each key.

    Parameters
    ----------
    parameters:
        A flat parameter dictionary, typically one component of the
        restored checkpoint.
    prefix:
        Module path prefix without a trailing separator, for example
        "mid_block_1" or "up_3_block_0".

    Returns
    -------
    A new dictionary containing only the matching entries, re-keyed
    relative to the prefix. Returns an empty dictionary if nothing
    matches; callers that require a non-empty result should use
    require_parameter to fail with a useful message instead of checking
    emptiness here.

    Examples
    --------
    A flat dictionary containing "mid_block_1_conv1_weight" yields
    "conv1_weight" when selected with prefix "mid_block_1".
    """
    prefix_with_separator = prefix + PARAMETER_PATH_SEPARATOR
    return {
        key[len(prefix_with_separator):]: value
        for key, value in parameters.items()
        if key.startswith(prefix_with_separator)
    }


def require_parameter(parameters: dict[str, np.ndarray], key: str, context: str) -> np.ndarray:
    """
    Look up a required parameter, raising a diagnostic error if absent.

    Parameters
    ----------
    parameters:
        The parameter group to look in.
    key:
        The required key.
    context:
        Human-readable description of what was being built, included in
        the error message so a failure identifies which block is
        mismatched rather than only which key was missing.
    """
    if key not in parameters:
        available = sorted(parameters.keys())
        raise MissingParameterError(
            f"{context} requires parameter '{key}', which is not present. "
            f"Available parameters in this group: {available}"
        )
    return parameters[key]


def has_parameter_group(parameters: dict[str, np.ndarray], prefix: str) -> bool:
    """
    Report whether any parameter exists under a prefix.

    Used for genuinely optional sub-modules, most notably a residual
    block's projection shortcut, which is present only when the block
    changes channel count.
    """
    prefix_with_separator = prefix + PARAMETER_PATH_SEPARATOR
    return any(key.startswith(prefix_with_separator) for key in parameters)
