"""
JAX-native inference for FLUX.2 Klein-4B.

The package is organised by architectural role rather than by model
component, so that a reader looking for "how is normalization done"
finds one place, not four:

    config      configuration dataclasses and enumerations
    utils       cross-cutting concerns with no model knowledge
    checkpoint  loading weights from the Hub and addressing them
    layers      individual mathematical primitives
    blocks      composites assembled from primitives
    models      complete networks assembled from blocks

Dependencies run strictly downward through that list. A layer may
import from config and utils but never from blocks; a block may import
from layers but never from models. This keeps the numeric core free of
IO and makes each level independently testable.
"""
