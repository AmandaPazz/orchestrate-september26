"""Thin wrapper around two possible LLM providers: Anthropic (primary, when
ANTHROPIC_API_KEY is set) and Gemini (fallback, or primary when only GEMINI_API_KEY is
set -- the situation this was built for: no Anthropic credit, running entirely on Gemini).

Provider selection happens per call, at call time, from environment variables only (never
hardcoded): try Anthropic first if ANTHROPIC_API_KEY is present; on any failure (missing
key, auth error, network error, rate limit), fall through to Gemini if GEMINI_API_KEY is
present; if neither is configured, or both fail, raise the last error. Every caller
(vision_llm.py, messages_llm.py, explain_llm.py) is provider-agnostic -- they only see
`call_text`/`call_vision`, never which provider actually answered.

Caches by (call_type, input_hash) regardless of which provider serves the call -- a cached
Gemini answer is reused even if Anthropic credit becomes available later, and vice versa;
the whole point of the cache is "was this exact input already answered," not "by whom."
The cache entry itself records which provider/model actually served it, so a cache HIT
still logs accurate provider/model attribution to usage_tracker.py, not just whichever
provider the current call happened to try first.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Optional

from . import config
from .usage_tracker import UsageTracker


class AllProvidersFailedError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, usage_tracker: UsageTracker, cache_path: Optional[Path] = None):
        self.usage_tracker = usage_tracker
        self.cache_path = cache_path or (config.CACHE_DIR / "llm_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._anthropic_client = None
        self._gemini_client = None

    # -- provider clients, constructed lazily so the module is importable and cache-only
    #    calls work without either SDK/key configured -------------------------------- #

    def _anthropic(self):
        if self._anthropic_client is None:
            import anthropic

            self._anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._anthropic_client

    def _gemini(self):
        if self._gemini_client is None:
            from google import genai

            self._gemini_client = genai.Client()  # reads GEMINI_API_KEY
        return self._gemini_client

    @staticmethod
    def _provider_order() -> list[str]:
        import os

        order = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            order.append("anthropic")
        if os.environ.get("GEMINI_API_KEY"):
            order.append("gemini")
        return order

    # -- cache ------------------------------------------------------------------------ #

    @staticmethod
    def _key(call_type: str, input_payload: str) -> str:
        digest = hashlib.sha256(input_payload.encode("utf-8")).hexdigest()
        return f"{call_type}:{digest}"

    def _save_cache(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    def _record_from_cache_entry(self, call_type: str, entry: dict) -> str:
        self.usage_tracker.record(
            call_type, entry["provider"], entry["model"], entry["input_tokens"], entry["output_tokens"], cache_hit=True
        )
        return entry["response"]

    def _store_and_record(self, key: str, call_type: str, provider: str, model: str, text: str, input_tokens: int, output_tokens: int) -> str:
        self._cache[key] = {
            "response": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider": provider,
            "model": model,
        }
        self._save_cache()
        self.usage_tracker.record(call_type, provider, model, input_tokens, output_tokens, cache_hit=False)
        return text

    # -- per-provider text calls -------------------------------------------------------- #

    def _call_text_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, int, int]:
        response = self._anthropic().messages.create(
            model=config.ANTHROPIC_TEXT_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens

    def _call_text_gemini(self, system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, int, int]:
        from google.genai import types

        response = self._gemini().models.generate_content(
            model=config.GEMINI_TEXT_MODEL,
            contents=[types.Part.from_text(text=user_prompt)],
            config=types.GenerateContentConfig(system_instruction=system_prompt, max_output_tokens=max_tokens),
        )
        text = response.text or ""
        usage = response.usage_metadata
        return text, usage.prompt_token_count or 0, usage.candidates_token_count or 0

    # -- per-provider vision calls ------------------------------------------------------ #

    def _call_vision_anthropic(self, system_prompt: str, image_bytes: bytes, context_text: str, max_tokens: int) -> tuple[str, int, int]:
        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        response = self._anthropic().messages.create(
            model=config.ANTHROPIC_VISION_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                        {"type": "text", "text": context_text},
                    ],
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage.input_tokens, response.usage.output_tokens

    def _call_vision_gemini(self, system_prompt: str, image_bytes: bytes, context_text: str, max_tokens: int) -> tuple[str, int, int]:
        from google.genai import types

        response = self._gemini().models.generate_content(
            model=config.GEMINI_VISION_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=context_text),
            ],
            config=types.GenerateContentConfig(system_instruction=system_prompt, max_output_tokens=max_tokens),
        )
        text = response.text or ""
        usage = response.usage_metadata
        return text, usage.prompt_token_count or 0, usage.candidates_token_count or 0

    # -- public API, provider-agnostic to every caller ---------------------------------- #

    def call_text(self, call_type: str, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        key = self._key(call_type, f"{system_prompt}\n---\n{user_prompt}")
        cached = self._cache.get(key)
        if cached is not None:
            return self._record_from_cache_entry(call_type, cached)

        providers = self._provider_order()
        if not providers:
            raise AllProvidersFailedError("Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY is set.")

        last_error = None
        for provider in providers:
            try:
                if provider == "anthropic":
                    text, in_tok, out_tok = self._call_text_anthropic(system_prompt, user_prompt, max_tokens)
                    model = config.ANTHROPIC_TEXT_MODEL
                else:
                    text, in_tok, out_tok = self._call_text_gemini(system_prompt, user_prompt, max_tokens)
                    model = config.GEMINI_TEXT_MODEL
                return self._store_and_record(key, call_type, provider, model, text, in_tok, out_tok)
            except Exception as e:
                last_error = e
                continue
        raise AllProvidersFailedError(f"All providers failed for call_type={call_type}: {last_error!r}") from last_error

    def call_vision(self, call_type: str, system_prompt: str, image_path: Path, context_text: str, max_tokens: int = 512) -> str:
        image_bytes = image_path.read_bytes()
        # Content-addressed on the actual image bytes, not the path -- correct even if a
        # path were ever reused for different content.
        key = self._key(call_type, f"{context_text}\n---\n{hashlib.sha256(image_bytes).hexdigest()}")
        cached = self._cache.get(key)
        if cached is not None:
            return self._record_from_cache_entry(call_type, cached)

        providers = self._provider_order()
        if not providers:
            raise AllProvidersFailedError("Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY is set.")

        last_error = None
        for provider in providers:
            try:
                if provider == "anthropic":
                    text, in_tok, out_tok = self._call_vision_anthropic(system_prompt, image_bytes, context_text, max_tokens)
                    model = config.ANTHROPIC_VISION_MODEL
                else:
                    text, in_tok, out_tok = self._call_vision_gemini(system_prompt, image_bytes, context_text, max_tokens)
                    model = config.GEMINI_VISION_MODEL
                return self._store_and_record(key, call_type, provider, model, text, in_tok, out_tok)
            except Exception as e:
                last_error = e
                continue
        raise AllProvidersFailedError(f"All providers failed for call_type={call_type}: {last_error!r}") from last_error
