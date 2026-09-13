"""Backend factory.

`build_backend(cfg)` takes a dict straight out of a config's `models` block and
returns a `VLMBackend`. Adding a provider means adding one folder and one entry
in the table below -- never a change anywhere else in the package.

Three model roles, all defaulting to the Anthropic API:

  planner   long-horizon composition and strict JSON
  grounder  phrase -> pixel, for `observe`
  monitor   yes/no questions about a scene, and goal verification

None of them needs a GPU. If you later want the grounder local, point the
`openai_compat` kind at a vLLM server -- `docs/STRETCH_SETUP.md` covers what
fits in 16 GB.
"""
from __future__ import annotations

from typing import Any

from .base import BackendError, ChatMessage, VLMBackend, extract_json
from .keys import MissingKeyError, status as key_status
from .scripted import ScriptedBackend

__all__ = [
    "VLMBackend", "ChatMessage", "BackendError", "extract_json",
    "ScriptedBackend", "build_backend", "key_status", "MissingKeyError",
]

_KINDS = {
    "anthropic",        # backends/anthropic/  -- hosted Claude, the default
    "openai",           # backends/openai/     -- hosted OpenAI
    "openai_compat",    # backends/openai/     -- any OpenAI-shaped endpoint
    "hf_local",         # in-process transformers, needs a GPU
    "scripted",         # deterministic fake, for tests
}


def build_backend(cfg: dict[str, Any]) -> VLMBackend:
    kind = cfg.get("kind", "anthropic")
    params = {k: v for k, v in cfg.items() if k != "kind"}

    if kind == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend(**params)
    if kind == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend(**params)
    if kind == "openai_compat":
        from .openai import OpenAICompatBackend
        params.setdefault("require_key", False)
        return OpenAICompatBackend(**params)
    if kind == "hf_local":
        from .hf_local import HFLocalBackend
        return HFLocalBackend(**params)
    if kind == "scripted":
        return ScriptedBackend(**params)

    raise BackendError(
        f"unknown backend kind {kind!r}; expected one of {sorted(_KINDS)}")
