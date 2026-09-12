"""Thin wrapper around the Anthropic SDK: caches by (call_type, input_hash) exactly as
prototyped and verified in conversation (a cold 300-call run computed every call; a second
run against the same on-disk cache file hit 300/300 with zero model calls), and logs every
call to a UsageTracker.

Reads ANTHROPIC_API_KEY from the environment only (never hardcoded), per the project's
"secrets from environment variables only" requirement. The key is read lazily on first real
call, not at import time, so the rest of the pipeline (deterministic modules, and any call
path that only ever hits the cache) still works without it configured.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Optional

from . import config
from .usage_tracker import UsageTracker

PROVIDER = "anthropic"


class LLMClient:
    def __init__(self, usage_tracker: UsageTracker, cache_path: Optional[Path] = None):
        self.usage_tracker = usage_tracker
        self.cache_path = cache_path or (config.CACHE_DIR / "llm_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._sdk_client = None

    def _client(self):
        if self._sdk_client is None:
            import anthropic  # imported lazily so the module is importable without the SDK too

            self._sdk_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        return self._sdk_client

    @staticmethod
    def _key(call_type: str, input_payload: str) -> str:
        digest = hashlib.sha256(input_payload.encode("utf-8")).hexdigest()
        return f"{call_type}:{digest}"

    def _save_cache(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    def call_text(self, call_type: str, system_prompt: str, user_prompt: str, model: str = config.TEXT_MODEL, max_tokens: int = 1024) -> str:
        key = self._key(call_type, f"{system_prompt}\n---\n{user_prompt}")
        cached = self._cache.get(key)
        if cached is not None:
            self.usage_tracker.record(call_type, PROVIDER, model, cached["input_tokens"], cached["output_tokens"], cache_hit=True)
            return cached["response"]

        response = self._client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self._cache[key] = {"response": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
        self._save_cache()
        self.usage_tracker.record(call_type, PROVIDER, model, input_tokens, output_tokens, cache_hit=False)
        return text

    def call_vision(self, call_type: str, system_prompt: str, image_path: Path, context_text: str, model: str = config.VISION_MODEL, max_tokens: int = 512) -> str:
        image_bytes = image_path.read_bytes()
        # The cache key is content-addressed on the actual image bytes plus context, not
        # the file path -- correct even if a path were ever reused for different content.
        key = self._key(call_type, f"{context_text}\n---\n{hashlib.sha256(image_bytes).hexdigest()}")
        cached = self._cache.get(key)
        if cached is not None:
            self.usage_tracker.record(call_type, PROVIDER, model, cached["input_tokens"], cached["output_tokens"], cache_hit=True)
            return cached["response"]

        media_type = "image/png"
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = self._client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                        {"type": "text", "text": context_text},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self._cache[key] = {"response": text, "input_tokens": input_tokens, "output_tokens": output_tokens}
        self._save_cache()
        self.usage_tracker.record(call_type, PROVIDER, model, input_tokens, output_tokens, cache_hit=False)
        return text
