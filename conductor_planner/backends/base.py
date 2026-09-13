"""Backend interface.

Two *roles*, deliberately separated, because they want different models:

  planner   long-horizon composition, strict JSON, instruction following.
            Frontier models are still ~20 points ahead of open weights here
            (RoboBench, 2026). A 7-8B embodied model works but needs guided
            decoding and a repair loop.

  grounder  phrase -> pixel -> 3D. This is where the household-specific open
            models (RoboBrain 2.5, MiMo-Embodied) genuinely win: they were
            post-trained on pointing and affordance data that frontier models
            never saw. Small, local, fast, no per-call cost.

Nothing above this file knows which model is behind either role.
"""
from __future__ import annotations

import abc
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ChatMessage:
    role: str                      # "system" | "user" | "assistant"
    text: str
    images: list[str] | None = None    # file paths, attached in order


class BackendError(RuntimeError):
    pass


class VLMBackend(abc.ABC):
    """Minimal surface: text in, text out, images optional."""

    name: str = "abstract"
    supports_guided_json: bool = False

    @abc.abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        ...

    # -- convenience ------------------------------------------------------
    def chat_json(
        self,
        messages: list[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Ask for JSON, and if the model produces junk, show it the junk.

        A 7B model will emit prose around its JSON perhaps one call in five.
        Rather than crash the episode, we extract, and failing that we hand
        the parse error back as a new turn. This loop is the single biggest
        reliability difference between a demo that runs and one that doesn't
        when the planner is a small open model.
        """
        convo = list(messages)
        last_err = ""
        for attempt in range(retries + 1):
            raw = self.chat(
                convo,
                max_tokens=max_tokens,
                temperature=temperature if attempt == 0 else max(temperature, 0.2),
                json_schema=json_schema if self.supports_guided_json else None,
            )
            try:
                return extract_json(raw)
            except ValueError as e:
                last_err = f"{e}. You returned:\n{raw[:800]}"
                convo = convo + [
                    ChatMessage("assistant", raw),
                    ChatMessage(
                        "user",
                        "That was not valid JSON for the required schema. "
                        f"Error: {e}. Reply with ONE JSON object and nothing else -- "
                        "no markdown fence, no commentary.",
                    ),
                ]
        raise BackendError(f"backend {self.name} failed to produce JSON after "
                           f"{retries + 1} attempts: {last_err}")


# --------------------------------------------------------------------------
# JSON extraction: tolerant of fences, prose, and trailing text
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("empty response")
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    except json.JSONDecodeError:
        pass
    # brace matching, first balanced object
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"malformed JSON object: {e}") from None
    raise ValueError("unterminated JSON object")


def encode_image(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise BackendError(f"image not found: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def mime_for(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(ext, "image/png")
