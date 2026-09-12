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


# Gemini 3.6 Flash spends tokens on internal "thinking" before producing visible output --
# verified directly: a call with max_output_tokens=1024 consumed 981 of them on hidden
# reasoning, leaving 39 for the actual answer and truncating it mid-JSON
# (finish_reason=MAX_TOKENS). This reserve is added on top of the caller's requested
# max_tokens for every Gemini call so "how much output text I want" stays decoupled from
# "how much the model needs to think first" -- Anthropic calls need no such reserve.
_GEMINI_THINKING_RESERVE = 3072


def _looks_like_quota_exhaustion(error: Exception) -> bool:
    text = str(error)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "rate_limit" in text.lower()


class LLMClient:
    def __init__(self, usage_tracker: UsageTracker, cache_path: Optional[Path] = None, exhausted_path: Optional[Path] = None):
        self.usage_tracker = usage_tracker
        self.cache_path = cache_path or (config.CACHE_DIR / "llm_cache.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)
        self._anthropic_client = None
        self._gemini_client = None

        # Circuit breaker: (provider, model) pairs known exhausted TODAY. Once a
        # quota-exhaustion error is seen for a given model, every later call skips it
        # immediately rather than re-attempting (and waiting out) a doomed request --
        # verified real case: a 250-request run kept re-trying the same exhausted model
        # 240+ times, each a wasted network round-trip, instead of failing fast.
        # Persisted to disk (keyed by UTC date, cheap proxy for the Pacific-midnight daily
        # reset -- worst case this retries once right at the reset boundary, which is
        # harmless) so a FRESH process invocation doesn't repeat the one wasted call a
        # purely in-memory breaker would take before rediscovering the same exhaustion --
        # verified real gap: without this, every new `python3 main.py` run re-tried a
        # known-dead model once before falling through.
        self.exhausted_path = exhausted_path or (config.CACHE_DIR / "llm_exhausted.json")
        self._exhausted: set[tuple[str, str]] = self._load_exhausted()

        # Sliding-window pacing for Gemini's confirmed 15 RPM cap (checked against the
        # user's real AI Studio dashboard) -- a per-minute limit resets on its own, so this
        # only needs to live in memory for this process's calls, unlike the daily
        # exhaustion breaker above. Only counts real network attempts, never cache hits.
        self._gemini_call_times: list[float] = []

    def _throttle_gemini(self) -> None:
        import time

        now = time.monotonic()
        self._gemini_call_times = [t for t in self._gemini_call_times if now - t < 60.0]
        if len(self._gemini_call_times) >= config.GEMINI_RPM_LIMIT:
            sleep_for = 60.0 - (now - self._gemini_call_times[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._gemini_call_times = [t for t in self._gemini_call_times if now - t < 60.0]
        self._gemini_call_times.append(now)

    def _load_exhausted(self) -> set[tuple[str, str]]:
        import datetime

        if not self.exhausted_path.exists():
            return set()
        with open(self.exhausted_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        return {(e["provider"], e["model"]) for e in data if e.get("date") == today}

    def _save_exhausted(self) -> None:
        import datetime

        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        data = [{"provider": p, "model": m, "date": today} for p, m in self._exhausted]
        with open(self.exhausted_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _mark_exhausted(self, provider: str, model: str) -> None:
        self._exhausted.add((provider, model))
        self._save_exhausted()

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

    def _call_text_gemini(self, model: str, system_prompt: str, user_prompt: str, max_tokens: int) -> tuple[str, int, int]:
        from google.genai import types

        self._throttle_gemini()
        response = self._gemini().models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=user_prompt)],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, max_output_tokens=max_tokens + _GEMINI_THINKING_RESERVE
            ),
        )
        return self._extract_gemini_text_and_usage(response)

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

    def _call_vision_gemini(self, model: str, system_prompt: str, image_bytes: bytes, context_text: str, max_tokens: int) -> tuple[str, int, int]:
        from google.genai import types

        self._throttle_gemini()
        response = self._gemini().models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                types.Part.from_text(text=context_text),
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt, max_output_tokens=max_tokens + _GEMINI_THINKING_RESERVE
            ),
        )
        return self._extract_gemini_text_and_usage(response)

    @staticmethod
    def _extract_gemini_text_and_usage(response) -> tuple[str, int, int]:
        finish_reason = response.candidates[0].finish_reason if response.candidates else None
        usage = response.usage_metadata
        if str(finish_reason) not in ("FinishReason.STOP", "STOP") and not (response.text or "").strip():
            # MAX_TOKENS with empty/truncated text (e.g. the whole budget went to hidden
            # "thinking" tokens) -- surface this as a clear failure so the caller's
            # existing graceful-fallback path handles it, rather than returning a
            # truncated JSON fragment that would silently parse as "no facts."
            raise RuntimeError(
                f"Gemini response incomplete: finish_reason={finish_reason}, "
                f"thoughts_tokens={getattr(usage, 'thoughts_token_count', None)}, "
                f"output_tokens={getattr(usage, 'candidates_token_count', None)}"
            )
        return response.text or "", usage.prompt_token_count or 0, usage.candidates_token_count or 0

    def _candidates(self, model_chain_key: str) -> list[tuple[str, str]]:
        """Returns [(provider, model), ...] to try in order: Anthropic (single model) first
        if configured, then every model in Gemini's chain (config.GEMINI_TEXT_MODEL_CHAIN or
        GEMINI_VISION_MODEL_CHAIN) if configured -- skipping any (provider, model) already
        known exhausted this run."""
        candidates: list[tuple[str, str]] = []
        providers = self._provider_order()
        if "anthropic" in providers:
            model = config.ANTHROPIC_TEXT_MODEL if model_chain_key == "text" else config.ANTHROPIC_VISION_MODEL
            candidates.append(("anthropic", model))
        if "gemini" in providers:
            chain = config.GEMINI_TEXT_MODEL_CHAIN if model_chain_key == "text" else config.GEMINI_VISION_MODEL_CHAIN
            candidates.extend(("gemini", model) for model in chain)
        return [c for c in candidates if c not in self._exhausted]

    # -- public API, provider-agnostic to every caller ---------------------------------- #

    def call_text(self, call_type: str, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        key = self._key(call_type, f"{system_prompt}\n---\n{user_prompt}")
        cached = self._cache.get(key)
        if cached is not None:
            return self._record_from_cache_entry(call_type, cached)

        candidates = self._candidates("text")
        if not candidates:
            raise AllProvidersFailedError("No available (provider, model) left -- none configured, or all exhausted this run.")

        last_error = None
        for provider, model in candidates:
            try:
                if provider == "anthropic":
                    text, in_tok, out_tok = self._call_text_anthropic(system_prompt, user_prompt, max_tokens)
                else:
                    text, in_tok, out_tok = self._call_text_gemini(model, system_prompt, user_prompt, max_tokens)
                return self._store_and_record(key, call_type, provider, model, text, in_tok, out_tok)
            except Exception as e:
                last_error = e
                if _looks_like_quota_exhaustion(e):
                    self._mark_exhausted(provider, model)
                continue
        raise AllProvidersFailedError(f"All providers/models failed for call_type={call_type}: {last_error!r}") from last_error

    def call_vision(self, call_type: str, system_prompt: str, image_path: Path, context_text: str, max_tokens: int = 512) -> str:
        image_bytes = image_path.read_bytes()
        # Content-addressed on the actual image bytes, not the path -- correct even if a
        # path were ever reused for different content.
        key = self._key(call_type, f"{context_text}\n---\n{hashlib.sha256(image_bytes).hexdigest()}")
        cached = self._cache.get(key)
        if cached is not None:
            return self._record_from_cache_entry(call_type, cached)

        candidates = self._candidates("vision")
        if not candidates:
            raise AllProvidersFailedError("No available (provider, model) left -- none configured, or all exhausted this run.")

        last_error = None
        for provider, model in candidates:
            try:
                if provider == "anthropic":
                    text, in_tok, out_tok = self._call_vision_anthropic(system_prompt, image_bytes, context_text, max_tokens)
                else:
                    text, in_tok, out_tok = self._call_vision_gemini(model, system_prompt, image_bytes, context_text, max_tokens)
                return self._store_and_record(key, call_type, provider, model, text, in_tok, out_tok)
            except Exception as e:
                last_error = e
                if _looks_like_quota_exhaustion(e):
                    self._mark_exhausted(provider, model)
                continue
        raise AllProvidersFailedError(f"All providers/models failed for call_type={call_type}: {last_error!r}") from last_error
