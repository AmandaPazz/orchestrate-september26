"""Entry point: python3 code/main.py

Reads dataset/*.csv, produces one prediction per row of dataset/requests.csv, writes
output.csv to the repository root, verifies every row against the full invariant
checklist, and writes evaluation/usage_report.md from the same run.

Runs LLM-dependent work in four priority passes rather than per-request, so a quota cutoff
partway through always sacrifices the lowest-value calls first, never a random subset:
1. Vision extraction (highest priority -- a blank-amount event has no structural fallback
   at all if unresolved; it's simply excluded, per events.py's tested fallback).
2. Message extraction (structural recurrence detection is a decent, tested fallback if
   this is unavailable -- see forecast.py).
3. Deterministic forecasting/decisions (no LLM calls; consumes whatever passes 1-2
   resolved, gracefully degrading for anything they didn't).
4. Decision explanations (lowest priority -- explain_llm.py has a safe, grounded
   deterministic fallback template if this fails or the quota is gone).
Verified real case this matters for: a 250-request run hit a Gemini free-tier daily quota
wall partway through: per-request processing would have spent the whole budget on
explain_llm calls for the first ~10 requests, leaving zero budget for the vision/message
calls of the other 240 -- exactly backwards from their actual value.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from buyorwait import (
    candidates,
    config,
    events,
    explain_llm,
    forecast,
    io_loader,
    messages_llm,
    verify,
    vision_llm,
)
from buyorwait.formatting import format_amount
from buyorwait.llm_client import LLMClient
from buyorwait.usage_tracker import UsageTracker


def main() -> int:
    run_log: list[str] = []

    print("Loading dataset...")
    profiles = io_loader.load_profiles()
    events_by_user = io_loader.load_events()
    images_by_event = io_loader.load_images()
    rates = io_loader.load_exchange_rates()
    requests = io_loader.load_requests()
    payment_options_by_request = io_loader.load_payment_options()
    messages_by_user = io_loader.load_messages()

    all_events_by_id = {e.event_id: e for evs in events_by_user.values() for e in evs}

    usage_tracker = UsageTracker()
    client = LLMClient(usage_tracker)
    resolve_amount = vision_llm.make_resolver(client, run_log=run_log)
    call_llm = messages_llm.make_call_llm(client, run_log=run_log)

    seen: set[str] = set()
    relevant_users = [
        r.user_id for r in requests if profiles.get(r.user_id) is not None and not (r.user_id in seen or seen.add(r.user_id))
    ]

    # --- Pass 1/4: vision extraction -- highest priority. ---------------------------- #
    print("Pass 1/4: vision extraction...")
    vision_events = [
        (uid, e)
        for uid in relevant_users
        for e in events_by_user.get(uid, [])
        if e.amount is None and e.event_id in images_by_event
    ]
    for uid, e in vision_events:
        resolve_amount(e, images_by_event[e.event_id])  # populates the LLMClient cache
    print(f"  {len(vision_events)} blank-amount events processed.")

    # --- Pass 2/4: message extraction -- next priority. ------------------------------- #
    print("Pass 2/4: message extraction...")
    facts_by_user: dict[str, tuple[list, list]] = {}
    for uid in relevant_users:
        user_messages = messages_by_user.get(uid, [])
        event_scoped_messages = [m for m in user_messages if m.related_event_id]
        event_scoped_facts = messages_llm.extract_message_facts(event_scoped_messages, all_events_by_id, call_llm)
        salary_facts = messages_llm.extract_salary_facts(user_messages, call_llm)
        facts_by_user[uid] = (event_scoped_facts, salary_facts)
    print(f"  {len(relevant_users)} users processed.")

    # --- Pass 3/4: deterministic forecasting and decisions -- no LLM calls; consumes ---
    # whatever passes 1-2 resolved (from cache) and gracefully degrades for the rest. --- #
    print("Pass 3/4: deterministic forecasting and decisions...")
    decisions = []
    for request in requests:
        profile = profiles.get(request.user_id)
        if profile is None:
            run_log.append(f"NO_PROFILE user_id={request.user_id} request_id={request.request_id}")
            continue

        event_scoped_facts, salary_facts = facts_by_user.get(request.user_id, ([], []))
        resolved = events.resolve_user_events(
            request.user_id,
            list(events_by_user.get(request.user_id, [])),
            profile,
            images_by_event,
            rates,
            run_log,
            resolve_amount=resolve_amount,
            message_facts=event_scoped_facts,
        )
        eligible = [e for e in resolved if e.excluded_reason is None]
        series = forecast.detect_recurring_series(eligible)
        series = forecast.apply_salary_message_facts(series, salary_facts, run_log=run_log)

        payment_options = payment_options_by_request.get(request.request_id, [])
        decision = candidates.decide(profile, request, resolved, series, payment_options, run_log=run_log)
        decisions.append((request, profile, decision, resolved, payment_options))
    print(f"  {len(decisions)} decisions computed.")

    # --- Pass 4/4: decision explanations -- lowest priority, safe fallback exists. ----- #
    print("Pass 4/4: decision explanations...")
    rows: list[dict] = []
    all_violations: list[str] = []
    for i, (request, profile, decision, resolved, payment_options) in enumerate(decisions, 1):
        explanation = explain_llm.generate_explanation(client, request, profile, decision)

        row = {
            "request_id": request.request_id,
            "amount_safe_to_pay": format_amount(decision.amount_safe_to_pay),
            "affordability_status": decision.affordability_status,
            "recommended_payment_method": decision.recommended_payment_method,
            "payment_plan": decision.payment_plan,
            "earliest_date_for_full_payment": decision.earliest_date_for_full_payment.isoformat() if decision.earliest_date_for_full_payment else "",
            "spending_changes_needed": decision.spending_changes_needed,
            "decision_explanation": explanation,
        }
        rows.append(row)

        events_by_id_for_user = {e.event_id: e for e in resolved}
        all_violations.extend(verify.verify_row(row, request, profile, payment_options, events_by_id_for_user))

        if i % 50 == 0:
            print(f"  {i}/{len(decisions)} done")

    all_violations.extend(verify.verify_output(rows, requests))

    print(f"\nWriting {config.OUTPUT_CSV}...")
    with open(config.OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    usage_tracker.write_report(num_requests=len(rows))
    print(f"Wrote {config.EVALUATION_DIR / 'usage_report.md'}")

    log_path = config.REPO_ROOT / "code" / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(run_log))
    print(f"Wrote {log_path} ({len(run_log)} entries)")

    status_counts = Counter(r["affordability_status"] for r in rows)
    method_counts = Counter(r["recommended_payment_method"] for r in rows)
    print(f"\nRows written: {len(rows)}")
    print(f"affordability_status distribution: {dict(status_counts)}")
    print(f"recommended_payment_method distribution: {dict(method_counts)}")

    if all_violations:
        print(f"\n{len(all_violations)} VERIFICATION VIOLATIONS:")
        for v in all_violations[:50]:
            print(f"  {v}")
        if len(all_violations) > 50:
            print(f"  ... and {len(all_violations) - 50} more")
        return 1

    print("\nverify.py: 0 violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
