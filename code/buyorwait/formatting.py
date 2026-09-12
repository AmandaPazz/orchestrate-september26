"""Shared numeric formatting for anything that ends up in output.csv or in text handed to
an LLM prompt. Its own tiny module so candidates.py, spending_changes.py, and explain_llm.py
can all use it without a circular import between them.
"""
from __future__ import annotations


def strip_json_fence(raw: str) -> str:
    """Strips a leading/trailing markdown code fence (```json ... ``` or ``` ... ```)
    around a model's JSON response. Verified real case: gemini-3.6-flash wraps its JSON
    array in ```json fences despite the prompt saying "respond with ONLY a JSON array (no
    prose)" -- json.loads() would otherwise fail on every such response regardless of
    provider, so every JSON-parsing call site should run its raw response through this
    first. A response with no fence passes through unchanged.
    """
    s = raw.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    if s.startswith("json"):
        s = s[4:]
    end = s.rfind("```")
    if end != -1:
        s = s[:end]
    return s.strip()


def format_amount(amount: float) -> str:
    """Plain decimal formatting, never scientific notation. `f"{x:g}"` switches to
    exponential above ~999,999 (e.g. `f"{46018000.0:g}"` -> "4.6018e+07"), which is a real
    bug for this dataset -- IDR requested_amounts in the tens of millions are common.
    """
    s = f"{amount:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
