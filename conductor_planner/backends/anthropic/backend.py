"""Anthropic Messages API backend.

Plain HTTP -- no SDK dependency, so the robot needs nothing but the standard
library to run the planner.

Two things here earn their complexity:

**Forced tool call for structured output.** Asking a model to "reply with only
JSON" works most of the time; declaring a tool and forcing it works every time,
because the schema is enforced server-side. When `json_schema` is passed we
force a single `emit` tool call and return its input. The planner's repair loop
then almost never fires, which matters when every retry is a paid round trip.

**Prompt caching.** The system prompt is roughly 14k characters of primitive
vocabulary and memory catalogue, and it is byte-identical on every turn of an
episode. Marking it `cache_control: ephemeral` means you pay full price for it
once and a small fraction thereafter. On a 30-step episode that is most of the
bill. This is also why `prompts.py` keeps volatile memory OUT of the system
prompt and puts it in the per-turn message -- see the note in `planner.py`.
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

# Starting points. Any current Claude model id works; these are roles, not a
# hard list, and model ids change -- check the docs rather than trusting this.
MODELS = {
    "planner": "claude-sonnet-4-5",
    "grounder": "claude-sonnet-4-5",
    "monitor": "claude-haiku-4-5",
}


class AnthropicBackend(VLMBackend):
    def __init__(
        self,
        model: str = MODELS["planner"],
        api_key: str | None = None,
        api_key_file: str | Path | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 180.0,
        version: str = "2023-06-01",
        cache_system_prompt: bool = True,
        max_retries: int = 3,
    ):
        try:
            self.api_key = resolve("anthropic", api_key, api_key_file)
        except MissingKeyError as e:
            raise BackendError(str(e)) from None
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.version = version
        self.cache_system_prompt = cache_system_prompt
        self.max_retries = max_retries
        self.name = f"anthropic:{model}"
        # structured output comes from a forced tool call, not from decoding
        # constraints, so the generic guided-json path stays off
        self.supports_guided_json = False
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        system_blocks = self._system_blocks(messages)
        turns = self._turns(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system_blocks:
            payload["system"] = system_blocks
        if json_schema is not None:
            payload["tools"] = [{
                "name": "emit",
                "description": "Emit the structured result. Always use this tool.",
                "input_schema": json_schema,
            }]
            payload["tool_choice"] = {"type": "tool", "name": "emit"}

        body = self._send(payload)
        u = body.get("usage", {})
        for k in self.usage:
            self.usage[k] += u.get(k, 0)

        for block in body.get("content", []):
            if block.get("type") == "tool_use":
                return json.dumps(block["input"])
        text = "".join(b.get("text", "") for b in body.get("content", [])
                       if b.get("type") == "text")
        if not text:
            raise BackendError(
                f"{self.name} returned no usable content "
                f"(stop_reason={body.get('stop_reason')})")
        return text

    # ------------------------------------------------------------------
    def _system_blocks(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        text = "\n\n".join(m.text for m in messages if m.role == "system")
        if not text:
            return []
        block: dict[str, Any] = {"type": "text", "text": text}
        if self.cache_system_prompt:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _turns(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            content: list[dict[str, Any]] = []
            for img in (m.images or []):
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime_for(img),
                               "data": encode_image(img)},
                })
            content.append({"type": "text", "text": m.text})
            role = "assistant" if m.role == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"].extend(content)   # the API rejects repeats
            else:
                turns.append({"role": role, "content": content})
        if turns and turns[0]["role"] != "user":
            turns.insert(0, {"role": "user", "content": [{"type": "text", "text": "Begin."}]})
        return turns

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with backoff on the retryable statuses.

        529 (overloaded) and 429 (rate limit) are transient and common enough
        that failing an episode over one would be a bad trade -- an episode
        holds a physical robot in a half-finished state.
        """
        last = ""
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/messages",
                data=json.dumps(payload).encode(),
                headers={"content-type": "application/json",
                         "x-api-key": self.api_key,
                         "anthropic-version": self.version},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last = f"HTTP {e.code}: {detail}"
                if e.code in (429, 500, 502, 503, 529) and attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                if e.code == 401:
                    raise BackendError(
                        f"{self.name}: the API key was rejected (401). Check "
                        "keys/anthropic.key or ANTHROPIC_API_KEY.") from None
                raise BackendError(f"{self.name} {last}") from None
            except urllib.error.URLError as e:
                last = f"unreachable: {e.reason}"
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise BackendError(f"{self.name} {last}") from None
        raise BackendError(f"{self.name} {last}")

    # ------------------------------------------------------------------
    def cost_report(self) -> str:
        u = self.usage
        cached = u["cache_read_input_tokens"]
        fresh = u["input_tokens"]
        total_in = cached + fresh + u["cache_creation_input_tokens"]
        pct = 100.0 * cached / total_in if total_in else 0.0
        return (f"{self.name}: {total_in} input tokens "
                f"({pct:.0f}% served from cache), {u['output_tokens']} output")
