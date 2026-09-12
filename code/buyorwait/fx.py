"""Fixed, dated currency conversion. Verified against the shipped dataset: every
foreign-currency financial_events row has an exact (settlement_date, from, to) match in
exchange_rates.csv (0/140 missing). The missing-rate branch below is a defensive fallback
only, never expected to fire on this dataset.
"""
from __future__ import annotations

from datetime import date

RunLog = list[str]


def convert(
    amount: float,
    from_currency: str,
    to_currency: str,
    rate_date: date,
    rates: dict[tuple[date, str, str], float],
    run_log: RunLog | None = None,
    context: str = "",
) -> float | None:
    """Convert `amount` from `from_currency` to `to_currency` using the exact-date rate.

    Returns None (never guesses) if `from_currency == to_currency` is false and no exact
    (rate_date, from_currency, to_currency) row exists — callers must treat None as
    "unresolvable," per the same rule as an unresolvable blank amount / low-confidence
    extraction (events.py applies that fallback).
    """
    if from_currency == to_currency:
        return amount

    key = (rate_date, from_currency, to_currency)
    rate = rates.get(key)
    if rate is not None:
        return amount * rate

    if run_log is not None:
        run_log.append(
            f"MISSING_FX_RATE date={rate_date} from={from_currency} to={to_currency} "
            f"context={context}"
        )
    return None
