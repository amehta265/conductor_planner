"""OpenAI Chat Completions backend.

Plain HTTP, no SDK. Two entry points over the same implementation:

  `OpenAIBackend`        talks to api.openai.com and resolves a key through
                         `backends.keys` like the Anthropic side does.

  `OpenAICompatBackend`  talks to any OpenAI-shaped endpoint you name. Used for
                         a local vLLM server, where the "key" is a placeholder.
                         This is the path to use if you ever want the grounder
                         running on the 5070 Ti instead of paying per call.

Structured output goes through `response_format: json_schema` where the server
supports it, and `guided_json` as well when the server is vLLM. Servers ignore
keys they do not recognise, so sending both is safe and saves a branch.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..base import BackendError, ChatMessage, VLMBackend, encode_image, mime_for
from ..keys import MissingKeyError, resolve

MODELS = {
    "planner": "gpt-5",
    "grounder": "gpt-5",
    "monitor": "gpt-5-mini",
}


class OpenAICompatBackend(VLMBackend):
    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        provider: str = "openai",
        require_key: bool = True,
        timeout: float = 180.0,
        guided_json: bool = True,
        max_retries: int = 3,
        extra_body: dict[str, Any] | None = None,
    ):
        try:
            key = resolve(provider, api_key, api_key_file, required=require_key)
        except MissingKeyError as e:
            raise BackendError(str(e)) from None
        # a local vLLM server accepts any bearer token
        self.api_key = key or "EMPTY"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_guided_json = guided_json
        self.extra_body = extra_body or {}
        self.name = f"{provider}:{model}"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}

    # ------------------------------------------------------------------
    def _content(self, m: ChatMessage) -> Any:
        if not m.images:
            return m.text
        parts: list[dict[str, Any]] = []
        for img in m.images:
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_for(img)};base64,{encode_image(img)}"},
            })
        parts.append({"type": "text", "text": m.text})
        return parts

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": self._content(m)} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if temperature:
            payload["temperature"] = temperature
        payload.update(self.extra_body)
        if json_schema is not None and self.supports_guided_json:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "primitive_call", "schema": json_schema,
                                "strict": False},
            }
            payload["guided_json"] = json_schema        # vLLM; ignored elsewhere

        body = self._send(payload)
        u = body.get("usage") or {}
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        self.usage["cached_tokens"] += (
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise BackendError(f"{self.name} unexpected response: {str(body)[:400]}") from None

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        last = ""
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last = f"HTTP {e.code}: {detail}"
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 401:
                    raise BackendError(
                        f"{self.name}: the API key was rejected (401). Check "
                        "keys/openai.key or OPENAI_API_KEY.") from None
                raise BackendError(f"{self.name} {last}") from None
            except urllib.error.URLError as e:
                last = (f"unreachable at {self.base_url}: {e.reason}. "
                        "If this is a local server, is it running?")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise BackendError(f"{self.name} {last}") from None
        raise BackendError(f"{self.name} {last}")

    def cost_report(self) -> str:
        u = self.usage
        pct = 100.0 * u["cached_tokens"] / u["prompt_tokens"] if u["prompt_tokens"] else 0.0
        return (f"{self.name}: {u['prompt_tokens']} input tokens "
                f"({pct:.0f}% cached), {u['completion_tokens']} output")


class OpenAIBackend(OpenAICompatBackend):
    """api.openai.com, with a key resolved from keys/openai.key or the env."""

    def __init__(self, model: str = MODELS["planner"], **kw: Any):
        kw.setdefault("base_url", "https://api.openai.com/v1")
        kw.setdefault("require_key", True)
        super().__init__(model=model, **kw)
