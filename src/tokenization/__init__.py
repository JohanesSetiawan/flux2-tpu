"""
Turning prompt text into padded token identifiers.

This is the one part of the pipeline that is not numerical, and the one
place where an error is invisible to every numerical test: a prompt
tokenized differently from the reference produces a valid conditioning
tensor of exactly the right shape, carrying different content.
"""

from .prompt import (
    TokenizedPrompt,
    load_tokenizer,
    tokenize_prompts,
)

__all__ = ["TokenizedPrompt", "load_tokenizer", "tokenize_prompts"]
