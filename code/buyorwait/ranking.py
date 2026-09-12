"""The spec's 6-level tie-break, applied generically over any number of safe, eligible
candidates (real data only ever produces up to 2 simultaneously eligible-and-safe
candidates — see tests/test_ranking.py for synthetic coverage of levels 4-6, which no real
request exercises).

Ranking order (best first):
1. Completes the full request by desired_completion_date.
2. Requires no spending changes (fewer changes ranks better).
3. Minimizes total amount paid.
4. Starts earliest.
5. Uses fewest payments.
6. Lowest payment_option_id (numeric-aware — "payment_option_9" must sort before
   "payment_option_10", not lexicographically after).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class PlanCandidate:
    method: str  # full_payment | partial_payment | installments | wait
    completes_by_deadline: bool
    spending_changes: tuple[str, ...]  # formatted "stop:<id>" / "reduce_to:<id>:<amt>" strings; () if none
    total_amount_paid: float
    start_date: date
    number_of_payments: int
    payment_option_id: Optional[str] = None  # None for full_payment/partial_payment/wait
    payload: object = field(default=None, compare=False)  # whatever candidates.py needs to build the output row


_ID_NUMBER_RE = re.compile(r"(\d+)")


def _payment_option_sort_key(payment_option_id: Optional[str]) -> tuple[int, float, str]:
    """Lower is better. A candidate with no payment_option_id (full_payment,
    partial_payment, wait) never had one to be "lowest" among, so it sorts after any
    candidate that does — this tie-break only meaningfully discriminates between
    installment options anyway."""
    if payment_option_id is None:
        return (1, float("inf"), "")
    match = _ID_NUMBER_RE.search(payment_option_id)
    numeric = float(match.group(1)) if match else float("inf")
    return (0, numeric, payment_option_id)


def rank_candidates(candidates: list[PlanCandidate]) -> list[PlanCandidate]:
    """Returns `candidates` sorted best-first per the spec's 6-level tie-break. Works for
    any N >= 1; callers pass in only candidates that are already known to be individually
    eligible and safe — this function only breaks ties among them, it does not filter.
    """
    return sorted(
        candidates,
        key=lambda c: (
            0 if c.completes_by_deadline else 1,
            len(c.spending_changes),
            c.total_amount_paid,
            c.start_date,
            c.number_of_payments,
            _payment_option_sort_key(c.payment_option_id),
        ),
    )


def best_candidate(candidates: list[PlanCandidate]) -> Optional[PlanCandidate]:
    ranked = rank_candidates(candidates)
    return ranked[0] if ranked else None
