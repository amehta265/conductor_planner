"""A deterministic fake backend.

Not a toy: this is how you test the conductor, the memory abstraction and the
failure-recovery paths without burning GPU time or robot time, and how the
test suite stays hermetic. Give it a list of responses, or a callable that
inspects the conversation and decides.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from .base import BackendError, ChatMessage, VLMBackend

Responder = Callable[[list[ChatMessage]], str | dict[str, Any]]


class ScriptedBackend(VLMBackend):
    def __init__(self, responses: list[str | dict[str, Any]] | Responder,
                 name: str = "scripted"):
        self.name = name
        self.supports_guided_json = False
        self._responder: Responder | None = None
        self._queue: list[str | dict[str, Any]] = []
        if callable(responses):
            self._responder = responses
        else:
            self._queue = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 1024,
             temperature: float = 0.0, json_schema: dict[str, Any] | None = None) -> str:
        self.calls.append(list(messages))
        if self._responder is not None:
            out = self._responder(messages)
        elif self._queue:
            out = self._queue.pop(0)
        else:
            raise BackendError("scripted backend ran out of responses")
        return out if isinstance(out, str) else json.dumps(out)
