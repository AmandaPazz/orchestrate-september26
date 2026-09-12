"""Per-user event resolution: status filtering, currency conversion, blank-amount and
low-confidence handling, and message-derived amendments.

Investigated against the real dataset before writing this: of the 58 `linked_event_id`
pairs, 25 have both legs surviving plain status filtering. 19 of those 25 are genuinely
independent cash movements (a settled expense linked to its later settled refund/employer
reimbursement, or a settled investment purchase linked to its later settled investment
sale) and must both count. But 6 of the 25 are real duplicates: a `settled` debit linked to
a `pending` debit of the identical amount and currency (e.g. event_12708 "Original card
charge" settled 134.75 EUR -> event_12709 "Possible duplicate card charge" pending 134.75
EUR). Those 6 are the literal target of the spec's "ignore duplicate records" rule and are
detected structurally below (linked, both debits, same amount+currency, source settled /
destination pending) rather than by matching the English description text, so the same
pattern still gets caught if the hidden 250-row dataset phrases it differently or in
another language. Every other linked pair (settled->cancelled/failed/unrealized, or a
pending *credit* destination) is already excluded by plain status filtering with no
special-casing needed. A full-file scan additionally found zero exact-duplicate rows
(same user/description/amount/currency/date/direction/status) elsewhere in the file.

Recurrence detection is NOT done here — it's a forecasting concern (needs the 90-day
horizon) and lives in forecast.py, which consumes the resolved events this module
produces.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable, Optional

from . import fx
from .io_loader import Event, Image, Profile

# status -> included in the numeric cash-flow forecast at all
_ALWAYS_INCLUDED_STATUSES = {"settled", "scheduled"}
_NEVER_INCLUDED_STATUSES = {"cancelled", "failed", "unrealized"}
# "pending": debits are reserved (included), credits are excluded (spec: don't count
# pending credits, bonuses, commissions, refunds, lottery proceeds until they settle).


@dataclass(frozen=True)
class MessageFact:
    """One atomic fact extracted from a message by messages_llm.py."""

    message_id: str
    fact_type: str  # amend_amount | amend_date | cancel | confirm | no_actionable_info
    target_event_id: Optional[str]
    recurring_scope: Optional[str]  # e.g. "salary" for a user-level recurring-income amendment
    new_amount: Optional[float]
    new_date: Optional[date]
    effective_date: Optional[date]
    confidence: str  # "high" | "low"
    sent_at: str  # ISO timestamp string, used to rank "newer record from the same source"


AmountResolver = Callable[[Event, Image], tuple[Optional[float], str]]
"""(event, image) -> (amount_in_event_currency_or_None, confidence). Injected so events.py
is testable without a live vision LLM call; vision_llm.py (group d) supplies the real one."""


def _classify_inclusion(event: Event) -> tuple[bool, Optional[str]]:
    """Returns (include_in_numeric_forecast, excluded_reason)."""
    if event.status in _NEVER_INCLUDED_STATUSES:
        return False, f"status_{event.status}"
    if event.status == "pending" and event.direction == "credit":
        return False, "pending_credit"
    if event.status in _ALWAYS_INCLUDED_STATUSES or event.status == "pending":
        return True, None
    return False, f"unknown_status_{event.status}"


def resolve_user_events(
    user_id: str,
    raw_events: list[Event],
    profile: Profile,
    images_by_event: dict[str, Image],
    rates: dict[tuple[date, str, str], float],
    run_log: list[str],
    resolve_amount: Optional[AmountResolver] = None,
    message_facts: Optional[list[MessageFact]] = None,
) -> list[Event]:
    """Returns every event for this user (not just included ones), each annotated with
    `home_currency_amount`, `amount_confidence`, and `excluded_reason`. Callers filter on
    `excluded_reason is None` for cash-flow purposes; excluded events are still returned so
    forecast.py can use flexible/recurring templates and spending_changes.py can reference
    their `event_id` even when a particular historical instance itself isn't in the ledger.
    """
    message_facts = message_facts or []
    facts_by_event: dict[str, list[MessageFact]] = {}
    for fact in message_facts:
        if fact.target_event_id:
            facts_by_event.setdefault(fact.target_event_id, []).append(fact)

    events_by_id = {e.event_id: e for e in raw_events}

    resolved: list[Event] = []
    for event in raw_events:
        # 0. Structural duplicate detection: a pending debit linked to a settled debit of
        # the identical amount+currency is the same charge counted twice (see module
        # docstring). Drop the pending duplicate leg; the settled original still counts.
        if event.status == "pending" and event.direction == "debit" and event.linked_event_id:
            source = events_by_id.get(event.linked_event_id)
            if (
                source is not None
                and source.status == "settled"
                and source.direction == "debit"
                and source.amount == event.amount
                and source.currency == event.currency
            ):
                event.excluded_reason = "duplicate_pending_charge"
                resolved.append(event)
                run_log.append(
                    f"DUPLICATE_PENDING_CHARGE event_id={event.event_id} "
                    f"original_event_id={source.event_id} user={user_id}"
                )
                continue

        # 1. Resolve blank amounts via the linked image.
        if event.amount is None:
            image = images_by_event.get(event.event_id)
            if image is not None and resolve_amount is not None:
                amount, confidence = resolve_amount(event, image)
                event.amount = amount
                event.amount_confidence = confidence
            if event.amount is None:
                event.excluded_reason = "unresolvable_blank_amount"
                resolved.append(event)
                run_log.append(
                    f"UNRESOLVED_BLANK_AMOUNT event_id={event.event_id} user={user_id} "
                    f"direction={event.direction}"
                )
                continue

        # 2. Apply message-derived amendments/cancellations for this specific event.
        for fact in sorted(facts_by_event.get(event.event_id, []), key=lambda f: f.sent_at):
            if fact.fact_type == "cancel":
                event.status = "cancelled"
            elif fact.fact_type == "amend_amount" and fact.new_amount is not None:
                event.amount = fact.new_amount
                event.amount_confidence = fact.confidence
            elif fact.fact_type == "amend_date" and fact.new_date is not None:
                event.settlement_date = fact.new_date
            # "confirm" and "no_actionable_info" don't change the event.

        # 2b. Low-confidence extraction (from either the image resolver or a message
        # amendment) is treated exactly like an unresolvable blank amount, per the locked
        # design decision: excluded from the numeric forecast either way; a low-confidence
        # *debit* additionally signals a one-tier affordability downgrade (applied by
        # candidates.py via `summarize_forecast_risk` below), a low-confidence *credit* is
        # simply dropped since crediting unverified income is the risky direction.
        if event.amount_confidence == "low":
            event.excluded_reason = "low_confidence_extraction"
            resolved.append(event)
            run_log.append(
                f"LOW_CONFIDENCE_EXTRACTION event_id={event.event_id} user={user_id} "
                f"direction={event.direction}"
            )
            continue

        # 3. Status-based inclusion.
        included, excluded_reason = _classify_inclusion(event)
        event.excluded_reason = excluded_reason

        # 4. Currency conversion (only meaningful for events still in play).
        settle = event.settlement_date or event.event_date
        converted = fx.convert(
            event.amount,
            event.currency,
            profile.home_currency,
            settle,
            rates,
            run_log=run_log,
            context=f"event_id={event.event_id}",
        )
        if converted is None:
            event.excluded_reason = event.excluded_reason or "unresolvable_fx_rate"
            run_log.append(f"UNRESOLVED_FX event_id={event.event_id} user={user_id}")
        else:
            event.home_currency_amount = converted

        resolved.append(event)

    return resolved


_RISK_DOWNGRADE_REASONS = {
    "unresolvable_blank_amount",
    "low_confidence_extraction",
    "unresolvable_fx_rate",
}


def summarize_forecast_risk(events: list[Event]) -> list[Event]:
    """Excluded debit events whose exclusion should demote affordability_status one tier
    (unresolvable amount, low-confidence extraction, or unresolvable FX rate). Excluded
    credits are just dropped silently — no downgrade, per the locked design decision.
    """
    return [
        e
        for e in events
        if e.direction == "debit" and e.excluded_reason in _RISK_DOWNGRADE_REASONS
    ]
