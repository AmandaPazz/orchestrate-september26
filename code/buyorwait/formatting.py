"""Shared numeric formatting for anything that ends up in output.csv or in text handed to
an LLM prompt. Its own tiny module so candidates.py, spending_changes.py, and explain_llm.py
can all use it without a circular import between them.
"""
from __future__ import annotations


def format_amount(amount: float) -> str:
    """Plain decimal formatting, never scientific notation. `f"{x:g}"` switches to
    exponential above ~999,999 (e.g. `f"{46018000.0:g}"` -> "4.6018e+07"), which is a real
    bug for this dataset -- IDR requested_amounts in the tens of millions are common.
    """
    s = f"{amount:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
