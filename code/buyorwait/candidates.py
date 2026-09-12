"""Enumerate every eligible, safe payment candidate for a request, and pick the winner via
ranking.py's 6-level tie-break. Verified end-to-end against real requests (request_56: a
genuine 2-way tie between partial_payment and an installments option, broken at level 3 by
the installment option's financing fee; request_112: a request with no safe candidate at
all until a spending change is applied) -- see conversation walkthroughs.

Eligibility rules (independent of safety):
- full_payment / wait: require "full_payment" in payment_methods_user_will_consider.
  wait additionally requires earliest_date_for_full_payment to exist, be later than
  request_date, and be on or before desired_completion_date -- a later-but-past-deadline
  date is still reported in the output field but is not treated as a safe recommendation
  (locked design decision from the forecast.py conversation).
- partial_payment: requires allows_partial_payment on the request, "partial_payment" in
  payment_methods_user_will_consider, 0 < amount_safe_to_pay < requested_amount, and
  earliest_date_for_full_payment on or before desired_completion_date.
- installments: requires "installments" in payment_methods_user_will_consider, the
  option's span (in months) within max_installment_months (None means the user won't
  consider installments at all), and the option's last payment on or before
  desired_completion_date. Safety is checked by simulating the option's exact schedule
  against the ledger, not by amount_safe_to_pay (which only covers a single lump sum).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from . import config, events, forecast, ranking, spending_changes
from .forecast import ForecastResult, RecurringSeries
from .formatting import format_amount
from .io_loader import Event, PaymentOption, Profile, Request


@dataclass
class Decision:
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str  # formatted "date:amount|date:amount" or "none"
    earliest_date_for_full_payment: Optional[date]
    spending_changes_needed: str  # formatted or "none"
    winning_candidate: Optional[ranking.PlanCandidate]  # for explain_llm.py grounding


def _months_span(number_of_payments: int, frequency_days: Optional[int]) -> float:
    if number_of_payments <= 1:
        return 0.0
    return (number_of_payments - 1) * (frequency_days or 30) / 30.0


def _installment_schedule(option: PaymentOption) -> list[tuple[date, float]]:
    schedule = []
    d = option.first_payment_date
    for _ in range(option.number_of_payments):
        schedule.append((d, option.payment_amount))
        d = d + timedelta(days=option.payment_frequency_days or 30)
    return schedule


def _simulate_schedule_safe(
    profile: Profile,
    resolved_events: list[Event],
    series: list[RecurringSeries],
    request_date: date,
    schedule: list[tuple[date, float]],
    horizon_days: int = 90,
) -> bool:
    horizon_end = request_date + timedelta(days=horizon_days)
    known = forecast._known_committed_entries(resolved_events, request_date, horizon_end)
    ledger = list(known)
    for s in series:
        projected = forecast._project_series(s, horizon_end)
        ledger.extend(forecast._dedupe_against_known(projected, known))
    for d, amount in schedule:
        ledger.append(forecast.LedgerEntry(d, -amount, "payment", "candidate"))
    ledger.sort(key=lambda le: le.entry_date)

    balance = profile.current_available_balance
    worst = balance
    last_date = request_date
    for entry in ledger:
        balance += entry.signed_amount
        worst = min(worst, balance)
        last_date = entry.entry_date
    return worst >= profile.minimum_balance_to_keep - 1e-9


def _format_plan(schedule: list[tuple[date, float]]) -> str:
    if not schedule:
        return "none"
    return "|".join(f"{d.isoformat()}:{format_amount(amount)}" for d, amount in schedule)


def _build_candidates(
    profile: Profile,
    request: Request,
    resolved_events: list[Event],
    series: list[RecurringSeries],
    payment_options: list[PaymentOption],
    baseline: ForecastResult,
    spending_change_strings: tuple[str, ...] = (),
) -> list[ranking.PlanCandidate]:
    methods_ok = set(profile.payment_methods_user_will_consider)
    out: list[ranking.PlanCandidate] = []

    full_safe = baseline.amount_safe_to_pay >= request.requested_amount - 1e-6
    if "full_payment" in methods_ok and full_safe:
        schedule = [(request.request_date, request.requested_amount)]
        out.append(
            ranking.PlanCandidate(
                method="full_payment",
                completes_by_deadline=request.request_date <= request.desired_completion_date,
                spending_changes=spending_change_strings,
                total_amount_paid=request.requested_amount,
                start_date=request.request_date,
                number_of_payments=1,
                payload=schedule,
            )
        )

    if (
        "full_payment" in methods_ok
        and baseline.earliest_date_for_full_payment
        and baseline.earliest_date_for_full_payment != request.request_date
        and baseline.earliest_date_for_full_payment <= request.desired_completion_date
    ):
        schedule = [(baseline.earliest_date_for_full_payment, request.requested_amount)]
        out.append(
            ranking.PlanCandidate(
                method="wait",
                completes_by_deadline=True,
                spending_changes=spending_change_strings,
                total_amount_paid=request.requested_amount,
                start_date=baseline.earliest_date_for_full_payment,
                number_of_payments=1,
                payload=schedule,
            )
        )

    if (
        request.allows_partial_payment
        and "partial_payment" in methods_ok
        and 0 < baseline.amount_safe_to_pay < request.requested_amount - 1e-9
        and baseline.earliest_date_for_full_payment
        and baseline.earliest_date_for_full_payment <= request.desired_completion_date
    ):
        remainder = request.requested_amount - baseline.amount_safe_to_pay
        schedule = [
            (request.request_date, baseline.amount_safe_to_pay),
            (baseline.earliest_date_for_full_payment, remainder),
        ]
        out.append(
            ranking.PlanCandidate(
                method="partial_payment",
                completes_by_deadline=True,
                spending_changes=spending_change_strings,
                total_amount_paid=request.requested_amount,
                start_date=request.request_date,
                number_of_payments=2,
                payload=schedule,
            )
        )

    if "installments" in methods_ok and profile.max_installment_months is not None:
        for option in payment_options:
            if option.payment_method != "installments":
                continue
            if _months_span(option.number_of_payments, option.payment_frequency_days) > profile.max_installment_months + 0.5:
                continue
            schedule = _installment_schedule(option)
            if schedule[-1][0] > request.desired_completion_date:
                continue
            if not _simulate_schedule_safe(profile, resolved_events, series, request.request_date, schedule):
                continue
            out.append(
                ranking.PlanCandidate(
                    method="installments",
                    completes_by_deadline=True,
                    spending_changes=spending_change_strings,
                    total_amount_paid=option.total_payable_amount,
                    start_date=schedule[0][0],
                    number_of_payments=option.number_of_payments,
                    payment_option_id=option.payment_option_id,
                    payload=schedule,
                )
            )

    return out


def _apply_risk_downgrade(decision: Decision, resolved_events: list[Event], run_log: Optional[list[str]] = None) -> Decision:
    """Locked design decision (see events.py): an unresolvable-or-low-confidence DEBIT
    demotes affordability_status one tier (unless already not_affordable); a credit is
    just dropped, no downgrade -- already handled by events.py excluding it from the
    numeric forecast, nothing further needed here. Dormant on the current dataset's
    deterministic dry run (0 blank-amount events lack a resolvable image) but live once
    real LLM calls can return confidence="low".
    """
    at_risk = events.summarize_forecast_risk(resolved_events)
    if not at_risk:
        return decision

    new_status = config.AFFORDABILITY_DOWNGRADE[decision.affordability_status]
    if run_log is not None:
        run_log.append(
            f"AFFORDABILITY_DOWNGRADE_APPLIED from={decision.affordability_status} to={new_status} "
            f"at_risk_events={[e.event_id for e in at_risk]}"
        )
    if new_status == decision.affordability_status:
        return decision

    if new_status == "not_affordable":
        # Canonical not_affordable shape (invariant: not_affordable/not_recommended =>
        # payment_plan == "none") -- the amount itself is untouched, it's still the
        # baseline value; only the recommendation is withdrawn.
        return Decision(
            amount_safe_to_pay=decision.amount_safe_to_pay,
            affordability_status="not_affordable",
            recommended_payment_method="not_recommended",
            payment_plan="none",
            earliest_date_for_full_payment=decision.earliest_date_for_full_payment,
            spending_changes_needed="none",
            winning_candidate=None,
        )

    # affordable_now -> affordable_with_plan, or affordable_later unchanged (already the
    # bottom rung before not_affordable): keep the existing method/plan, just the status
    # reflects the added uncertainty. Rare, dormant path -- flagged rather than perfected.
    return Decision(
        amount_safe_to_pay=decision.amount_safe_to_pay,
        affordability_status=new_status,
        recommended_payment_method=decision.recommended_payment_method,
        payment_plan=decision.payment_plan,
        earliest_date_for_full_payment=decision.earliest_date_for_full_payment,
        spending_changes_needed=decision.spending_changes_needed,
        winning_candidate=decision.winning_candidate,
    )


def decide(
    profile: Profile,
    request: Request,
    resolved_events: list[Event],
    series: list[RecurringSeries],
    payment_options: list[PaymentOption],
    run_log: Optional[list[str]] = None,
) -> Decision:
    baseline = forecast.run_forecast(
        profile.current_available_balance,
        profile.minimum_balance_to_keep,
        request.request_date,
        request.requested_amount,
        resolved_events,
        series,
    )

    candidates = _build_candidates(profile, request, resolved_events, series, payment_options, baseline)
    winner = ranking.best_candidate(candidates)

    if winner is not None:
        decision = _decision_from_candidate(winner, baseline, spending_changes_str="none")
        return _apply_risk_downgrade(decision, resolved_events, run_log)

    # Nothing safe without a change -- search for the smallest spending-change
    # combination that makes full_payment safe (only meaningful if the user accepts it).
    if "full_payment" in profile.payment_methods_user_will_consider:
        combo = spending_changes.find_best_combo(
            profile, resolved_events, series, request.request_date, request.requested_amount
        )
        if combo is not None:
            # find_best_combo already verified this combo makes the full request safe on
            # request_date (it reruns the forecast internally). amount_safe_to_pay and
            # earliest_date_for_full_payment in the OUTPUT are still the ORIGINAL, pre-change
            # baseline -- confirmed against real samples (request_06/request_11: both
            # recommend full_payment via a spending change, yet report the smaller
            # pre-change amount_safe_to_pay, not the requested amount actually paid).
            change_strings = tuple(c.as_output_string() for c in combo)
            schedule = [(request.request_date, request.requested_amount)]
            candidate = ranking.PlanCandidate(
                method="full_payment",
                completes_by_deadline=True,
                spending_changes=change_strings,
                total_amount_paid=request.requested_amount,
                start_date=request.request_date,
                number_of_payments=1,
                payload=schedule,
            )
            decision = _decision_from_candidate(candidate, baseline, spending_changes_str="|".join(change_strings))
            return _apply_risk_downgrade(decision, resolved_events, run_log)

    # Nothing safe at all: not_affordable / not_recommended. (Already the floor -- the
    # risk downgrade is a no-op here since AFFORDABILITY_DOWNGRADE maps not_affordable to
    # itself -- but applied anyway for a uniform code path and so the log entry still
    # fires if an at-risk debit was part of why nothing was affordable.)
    decision = Decision(
        amount_safe_to_pay=max(0.0, min(baseline.amount_safe_to_pay, request.requested_amount)),
        affordability_status="not_affordable",
        recommended_payment_method="not_recommended",
        payment_plan="none",
        earliest_date_for_full_payment=baseline.earliest_date_for_full_payment,
        spending_changes_needed="none",
        winning_candidate=None,
    )
    return _apply_risk_downgrade(decision, resolved_events, run_log)


def _decision_from_candidate(candidate: ranking.PlanCandidate, baseline: ForecastResult, spending_changes_str: str) -> Decision:
    """`amount_safe_to_pay` is always the BASELINE forecast value (before any optional
    spending change), clamped to [0, requested_amount] -- confirmed against real samples
    (request_06/request_11: both recommend full_payment via a spending change, yet
    amount_safe_to_pay is reported as the smaller pre-change figure, not the requested
    amount actually paid). It never depends on which candidate wins.
    """
    if candidate.method == "wait":
        status = "affordable_later"
    elif candidate.method == "full_payment" and not candidate.spending_changes:
        status = "affordable_now"
    else:  # full_payment-via-spending-change, partial_payment, installments
        status = "affordable_with_plan"

    amount_safe_to_pay = baseline.amount_safe_to_pay

    return Decision(
        amount_safe_to_pay=amount_safe_to_pay,
        affordability_status=status,
        recommended_payment_method=candidate.method,
        payment_plan=_format_plan(candidate.payload),
        earliest_date_for_full_payment=baseline.earliest_date_for_full_payment,
        spending_changes_needed=spending_changes_str if spending_changes_str != "" else "none",
        winning_candidate=candidate,
    )
