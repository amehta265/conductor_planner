"""In-process transformers backend. Optional, and rarely the right choice.

The planner, grounder and monitor all default to a hosted API, so nothing in
this package requires a GPU. This module exists for one case: running the
GROUNDER locally to avoid paying per `observe` call.

**Hard constraint on this project: a single RTX 5070 Ti, 16 GB VRAM.**
That rules out most of what the literature recommends. What actually fits:

    RoboBrain2.5-4B      bf16   ~9 GB    fits, with room for a display
    Qwen3-VL-4B          bf16   ~9 GB    fits
    RoboBrain2.5-8B-NV   bf16   ~17 GB   DOES NOT FIT -- needs AWQ/FP8 (~9 GB)
    MiMo-Embodied-7B     bf16   ~16 GB   does not fit alongside anything else

Vision models spike well above their weight footprint on the image tower, so
budget 2-3 GB of headroom over the numbers above and cap resolution with
`max_pixels`. The default here is deliberately low for that reason.

The 5070 Ti is Blackwell (compute capability 12.0). Older PyTorch wheels have
no sm_120 kernels and fail at load with "no kernel image is available for
execution on the device". You need a build against CUDA 12.8 or newer.

Serving through vLLM (`kind: openai_compat`) is faster and frees the weights
between runs; prefer it over this module unless you are actively debugging the
model call itself.
"""
from __future__ import annotations

from typing import Any

from .base import BackendError, ChatMessage, VLMBackend


class HFLocalBackend(VLMBackend):
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_pixels: int | None = 768 * 28 * 28,   # keep the vision tower inside 16 GB
        trust_remote_code: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise BackendError(
                "HFLocalBackend needs `pip install transformers accelerate torch`"
            ) from e
        self.model_id = model_id
        self.name = f"hf:{model_id}"
        self.supports_guided_json = False
        kw: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if max_pixels:
            kw["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_id, **kw)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self._torch = torch

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        conv = []
        for m in messages:
            content: list[dict[str, Any]] = []
            for img in (m.images or []):
                content.append({"type": "image", "url": img})
            content.append({"type": "text", "text": m.text})
            conv.append({"role": m.role, "content": content})
        inputs = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        return self.processor.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
