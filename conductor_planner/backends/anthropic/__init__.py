"""Anthropic provider.

This is the default for all three model roles (planner, grounder, monitor).
Nothing here needs a GPU.
"""
from .backend import AnthropicBackend, MODELS

__all__ = ["AnthropicBackend", "MODELS"]
