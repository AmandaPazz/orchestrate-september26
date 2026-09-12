"""Deterministic pre-write verification, checking every invariant established over the
course of building this pipeline (not just the spec-mandated ones) so a future regression
can't silently reintroduce something already fixed once. See the conversation checklist
this module was written against for the full enumerated list; each check below is tagged
with its checklist item number in a comment.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

from . import config
from .io_loader import Event, PaymentOption, Profile, Request

_PLAN_ENTRY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}:-?\d+(\.\d+)?$")
_SCIENTIFIC_NOTATION_RE = re.compile(r"[eE][+-]\d")
_STOP_RE = re.compile(r"^stop:(\S+)$")
_REDUCE_RE = re.compile(r"^reduce_to:(\S+):(-?\d+(?:\.\d+)?)$")


def _parse_plan(payment_plan: str) -> Optional[list[tuple[date, float]]]:
    if payment_plan == "none":
        return []
    entries = []
    for chunk in payment_plan.split("|"):
        if not _PLAN_ENTRY_RE.match(chunk):
            return None
        d_str, amt_str = chunk.split(":", 1)
        try:
            entries.append((date.fromisoformat(d_str), float(amt_str)))
        except ValueError:
            return None
    return entries


def verify_row(
    row: dict,
    request: Request,
    profile: Profile,
    payment_options: list[PaymentOption],
    events_by_id: dict[str, Event],
) -> list[str]:
    """Returns a list of violation descriptions for one output.csv row; empty means clean."""
    v: list[str] = []
    rid = row["request_id"]

    # --- raw field presence / enum membership -------------------------------------- #
    for field_name in config.OUTPUT_COLUMNS:
        if field_name not in row:
            v.append(f"{rid}: missing field {field_name}")
    if row.get("affordability_status") not in config.AFFORDABILITY_STATUSES:
        v.append(f"{rid}: invalid affordability_status {row.get('affordability_status')!r}")
    if row.get("recommended_payment_method") not in config.PAYMENT_METHODS:
        v.append(f"{rid}: invalid recommended_payment_method {row.get('recommended_payment_method')!r}")

    # --- checklist #39: no scientific notation anywhere ------------------------------ #
    for field_name in ("amount_safe_to_pay", "payment_plan", "spending_changes_needed"):
        if _SCIENTIFIC_NOTATION_RE.search(str(row.get(field_name, ""))):
            v.append(f"{rid}: scientific notation in {field_name}: {row.get(field_name)!r}")

    # --- checklist #3: 0 <= amount_safe_to_pay <= requested_amount ------------------- #
    try:
        amount_safe_to_pay = float(row["amount_safe_to_pay"])
    except (TypeError, ValueError):
        v.append(f"{rid}: amount_safe_to_pay is not numeric: {row.get('amount_safe_to_pay')!r}")
        amount_safe_to_pay = None
    if amount_safe_to_pay is not None and not (-1e-6 <= amount_safe_to_pay <= request.requested_amount + 1e-6):
        v.append(f"{rid}: amount_safe_to_pay {amount_safe_to_pay} out of [0, {request.requested_amount}]")

    status = row.get("affordability_status")
    method = row.get("recommended_payment_method")

    # --- checklist #6: affordable_now => earliest_date == request_date --------------- #
    earliest = row.get("earliest_date_for_full_payment", "")
    if status == "affordable_now" and earliest != request.request_date.isoformat():
        v.append(f"{rid}: affordable_now but earliest_date_for_full_payment={earliest!r} != request_date")

    # --- checklist #16: not_affordable/not_recommended => plan none ------------------ #
    if status == "not_affordable" or method == "not_recommended":
        if status != "not_affordable" or method != "not_recommended":
            v.append(f"{rid}: not_affordable and not_recommended must go together (status={status}, method={method})")
        if row.get("payment_plan") != "none":
            v.append(f"{rid}: not_affordable/not_recommended but payment_plan={row.get('payment_plan')!r} != none")

    # --- checklist #16 (companion): affordable_now with no changes ------------------- #
    if status == "affordable_now" and row.get("spending_changes_needed", "none") != "none" and method != "full_payment":
        v.append(f"{rid}: affordable_now with method {method!r}, expected full_payment")
    # checklist #35: full_payment WITH a spending change must be affordable_with_plan, not affordable_now
    if method == "full_payment" and row.get("spending_changes_needed", "none") != "none" and status == "affordable_now":
        v.append(f"{rid}: full_payment with a spending change must be affordable_with_plan, not affordable_now")

    # --- checklist #8: payment_plan format + chronological order --------------------- #
    plan_entries = _parse_plan(row.get("payment_plan", ""))
    if plan_entries is None:
        v.append(f"{rid}: payment_plan malformed: {row.get('payment_plan')!r}")
        plan_entries = []
    else:
        dates_only = [d for d, _ in plan_entries]
        if dates_only != sorted(dates_only):
            v.append(f"{rid}: payment_plan not chronological: {row.get('payment_plan')!r}")

    # --- checklist #9: partial_payment shape ------------------------------------------ #
    if method == "partial_payment":
        if status != "affordable_with_plan":
            v.append(f"{rid}: partial_payment must be affordable_with_plan, got {status}")
        if len(plan_entries) != 2:
            v.append(f"{rid}: partial_payment must have exactly 2 payments, got {len(plan_entries)}")
        else:
            (d1, a1), (d2, a2) = plan_entries
            if d1 != request.request_date:
                v.append(f"{rid}: partial_payment first payment date {d1} != request_date {request.request_date}")
            if abs(a1 - amount_safe_to_pay) > 0.01:
                v.append(f"{rid}: partial_payment first amount {a1} != amount_safe_to_pay {amount_safe_to_pay}")
            if abs((a1 + a2) - request.requested_amount) > 0.01:
                v.append(f"{rid}: partial_payment total {a1 + a2} != requested_amount {request.requested_amount}")
            if d2 > request.desired_completion_date:
                v.append(f"{rid}: partial_payment second payment {d2} after desired_completion_date {request.desired_completion_date}")
        if amount_safe_to_pay is not None and not (0 < amount_safe_to_pay < request.requested_amount - 1e-9):
            v.append(f"{rid}: partial_payment requires 0 < amount_safe_to_pay < requested_amount")
        if not request.allows_partial_payment:
            v.append(f"{rid}: partial_payment recommended but request.allows_partial_payment is false")
        if "partial_payment" not in profile.payment_methods_user_will_consider:
            v.append(f"{rid}: partial_payment recommended but not in payment_methods_user_will_consider")

    # --- checklist #10: installments must exactly match a real option ---------------- #
    if method == "installments":
        matched = False
        for opt in payment_options:
            if opt.payment_method != "installments":
                continue
            expected = []
            d = opt.first_payment_date
            for _ in range(opt.number_of_payments):
                expected.append((d, opt.payment_amount))
                d = d + timedelta(days=opt.payment_frequency_days or 30)
            if len(expected) == len(plan_entries) and all(
                e[0] == a[0] and abs(e[1] - a[1]) < 0.01 for e, a in zip(expected, plan_entries)
            ):
                matched = True
                break
        if not matched:
            v.append(f"{rid}: installments payment_plan {row.get('payment_plan')!r} doesn't match any real payment_option")
        if "installments" not in profile.payment_methods_user_will_consider:
            v.append(f"{rid}: installments recommended but not in payment_methods_user_will_consider")
        if plan_entries and plan_entries[-1][0] > request.desired_completion_date:
            v.append(f"{rid}: installments last payment {plan_entries[-1][0]} after desired_completion_date")

    # --- checklist #34 (wait/full_payment eligibility) -------------------------------- #
    if method in ("full_payment", "wait") and "full_payment" not in profile.payment_methods_user_will_consider:
        v.append(f"{rid}: {method} recommended but full_payment not in payment_methods_user_will_consider")
    if method == "wait" and status != "affordable_later":
        v.append(f"{rid}: wait must be affordable_later, got {status}")
    if method == "wait" and plan_entries:
        if plan_entries[0][0] > request.desired_completion_date:
            v.append(f"{rid}: wait payment date {plan_entries[0][0]} after desired_completion_date")

    # --- checklist #11/12/13/14: spending_changes_needed -------------------------------- #
    changes_str = row.get("spending_changes_needed", "none")
    if changes_str != "none":
        entries = changes_str.split("|")
        if len(entries) > 3:
            v.append(f"{rid}: more than 3 spending changes: {changes_str!r}")
        stopped_ids, reduced_ids = set(), set()
        for entry in entries:
            m_stop, m_reduce = _STOP_RE.match(entry), _REDUCE_RE.match(entry)
            if m_stop:
                event_id = m_stop.group(1)
                stopped_ids.add(event_id)
                ev = events_by_id.get(event_id)
                if ev is None:
                    v.append(f"{rid}: stop targets unknown event_id {event_id}")
                elif ev.flexibility not in ("stoppable", "reducible_or_stoppable"):
                    v.append(f"{rid}: stop targets event {event_id} with flexibility={ev.flexibility}")
                elif ev.category not in profile.expense_categories_user_is_willing_to_stop:
                    v.append(f"{rid}: stop targets category {ev.category} not in user's willing-to-stop list")
            elif m_reduce:
                event_id, new_amount_str = m_reduce.group(1), m_reduce.group(2)
                reduced_ids.add(event_id)
                ev = events_by_id.get(event_id)
                if ev is None:
                    v.append(f"{rid}: reduce_to targets unknown event_id {event_id}")
                else:
                    if ev.flexibility not in ("reducible", "reducible_or_stoppable"):
                        v.append(f"{rid}: reduce_to targets event {event_id} with flexibility={ev.flexibility}")
                    if ev.category not in profile.expense_categories_user_is_willing_to_reduce:
                        v.append(f"{rid}: reduce_to targets category {ev.category} not in user's willing-to-reduce list")
                    if ev.minimum_allowed_amount is not None and abs(float(new_amount_str) - ev.minimum_allowed_amount) > 0.01:
                        v.append(
                            f"{rid}: reduce_to:{event_id} amount {new_amount_str} != minimum_allowed_amount {ev.minimum_allowed_amount}"
                        )
            else:
                v.append(f"{rid}: malformed spending_changes_needed entry {entry!r}")
        overlap = stopped_ids & reduced_ids
        if overlap:
            v.append(f"{rid}: same event both stopped and reduced: {overlap}")

    # --- decision_explanation non-empty --------------------------------------------- #
    if not row.get("decision_explanation", "").strip():
        v.append(f"{rid}: decision_explanation is empty")

    return v


def verify_output(rows: list[dict], requests: list[Request]) -> list[str]:
    """Set-level checks: exactly one row per request_id, matching requests.csv exactly."""
    v: list[str] = []
    request_ids = [r.request_id for r in requests]
    row_ids = [r["request_id"] for r in rows]

    if len(row_ids) != len(set(row_ids)):
        seen = set()
        dupes = {rid for rid in row_ids if rid in seen or seen.add(rid)}
        v.append(f"duplicate request_id rows in output: {dupes}")

    missing = set(request_ids) - set(row_ids)
    if missing:
        v.append(f"missing rows for request_ids: {missing}")

    extra = set(row_ids) - set(request_ids)
    if extra:
        v.append(f"output rows for unknown request_ids: {extra}")

    return v
