"""OpenAI provider.

Also covers any OpenAI-compatible endpoint -- a local vLLM or llama.cpp server,
Azure, OpenRouter -- by pointing `base_url` at it. That keeps one class for the
hosted path and the local path, which matters because the local path is how a
16 GB card would ever serve a grounder.
"""
from .backend import MODELS, OpenAIBackend, OpenAICompatBackend

__all__ = ["OpenAIBackend", "OpenAICompatBackend", "MODELS"]
