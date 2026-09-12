"""Search for the smallest, cheapest combination of up to 3 spending changes (stop /
reduce) that makes the full request safe to pay on `request_date`.

Verified against real data (request_112/user_112, see conversation walkthrough): a
technically-eligible flexible event (category in the user's willing-list, event flagged
stoppable/reducible) is not automatically a solution — the search has to actually re-run
the 90-day forecast per candidate combination and check whether it closes the gap, because
a small subscription stop can be far too small to matter even when it's "allowed."

Only 1-change cases occur in the 25 sample_requests.csv rows; the 2-change and 3-change
paths, and the "prefer cheapest among multiple solutions" rule, are covered by synthetic
tests in tests/test_spending_changes.py since no real request needs more than one change.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from datetime import date
from typing import Optional

from . import forecast
from .forecast import RecurringSeries
from .io_loader import Event, Profile


@dataclass(frozen=True)
class ChangeOption:
    action: str  # "stop" | "reduce"
    template_event_id: str
    category: str
    original_amount: float
    new_amount: float  # 0 for stop, minimum_allowed_amount for reduce
    impact: float  # original_amount - new_amount; per-occurrence amount given back to the user

    def as_output_string(self) -> str:
        if self.action == "stop":
            return f"stop:{self.template_event_id}"
        return f"reduce_to:{self.template_event_id}:{self.new_amount:g}"


def enumerate_change_options(series_list: list[RecurringSeries], profile: Profile) -> list[ChangeOption]:
    """One or two ChangeOptions per flexible series: a `reducible_or_stoppable` series
    contributes both a stop option and a reduce option (mutually exclusive with each
    other at combination time, never both chosen for the same template_event_id)."""
    options: list[ChangeOption] = []
    for s in series_list:
        can_stop = s.flexibility in ("stoppable", "reducible_or_stoppable") and s.category in profile.expense_categories_user_is_willing_to_stop
        can_reduce = (
            s.flexibility in ("reducible", "reducible_or_stoppable")
            and s.category in profile.expense_categories_user_is_willing_to_reduce
            and s.minimum_allowed_amount is not None
            and s.minimum_allowed_amount < s.projected_amount
        )
        if can_stop:
            options.append(
                ChangeOption("stop", s.template_event_id, s.category, s.projected_amount, 0.0, s.projected_amount)
            )
        if can_reduce:
            options.append(
                ChangeOption(
                    "reduce",
                    s.template_event_id,
                    s.category,
                    s.projected_amount,
                    s.minimum_allowed_amount,
                    s.projected_amount - s.minimum_allowed_amount,
                )
            )
    return options


def _apply_changes(series_list: list[RecurringSeries], combo: tuple[ChangeOption, ...]) -> list[RecurringSeries]:
    by_id = {c.template_event_id: c for c in combo}
    result = []
    for s in series_list:
        change = by_id.get(s.template_event_id)
        if change is None:
            result.append(s)
        elif change.action == "stop":
            continue  # drop the series entirely -- no future occurrences projected
        else:
            reduced = copy.copy(s)
            reduced.projected_amount = change.new_amount
            result.append(reduced)
    return result


def _valid_combo(combo: tuple[ChangeOption, ...]) -> bool:
    """Stopping and reducing the same event are mutually exclusive -- a combo must never
    contain two options that share a template_event_id (the flat option list can offer
    both for a reducible_or_stoppable series, so itertools.combinations alone would
    otherwise happily pick both)."""
    ids = [c.template_event_id for c in combo]
    return len(ids) == len(set(ids))


def find_best_combo(
    profile: Profile,
    resolved_events: list[Event],
    series_list: list[RecurringSeries],
    request_date: date,
    requested_amount: float,
    max_changes: int = 3,
) -> Optional[tuple[ChangeOption, ...]]:
    """Returns the smallest, then cheapest (lowest total per-occurrence impact) 1-to-3
    change combination that makes `requested_amount` safe to pay in full on
    `request_date`, or None if no combination up to `max_changes` succeeds.

    "Smallest first": every combination of size 1 is tried before any of size 2, and size
    2 before size 3 -- a 1-change solution is always preferred over a 2-change solution
    even if the 2-change one happens to be iterated first or has a lower nominal impact,
    per the spec's "requires no/fewer spending changes" preference. Within one size, the
    combination with the lowest total impact (least disruption to the user's spending, not
    most) wins among those that succeed.
    """
    options = enumerate_change_options(series_list, profile)

    for size in range(1, max_changes + 1):
        successes: list[tuple[ChangeOption, ...]] = []
        for combo in itertools.combinations(options, size):
            if not _valid_combo(combo):
                continue
            modified_series = _apply_changes(series_list, combo)
            result = forecast.run_forecast(
                profile.current_available_balance,
                profile.minimum_balance_to_keep,
                request_date,
                requested_amount,
                resolved_events,
                modified_series,
            )
            if result.amount_safe_to_pay >= requested_amount - 1e-6:
                successes.append(combo)
        if successes:
            return min(successes, key=lambda c: sum(o.impact for o in c))
    return None
