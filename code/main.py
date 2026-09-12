"""Entry point: python3 code/main.py

Reads dataset/*.csv, produces one prediction per row of dataset/requests.csv, writes
output.csv to the repository root, verifies every row against the full invariant
checklist, and writes evaluation/usage_report.md from the same run.
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

    rows: list[dict] = []
    all_violations: list[str] = []
    requests_by_id = {r.request_id: r for r in requests}

    print(f"Processing {len(requests)} requests...")
    for i, request in enumerate(requests, 1):
        profile = profiles.get(request.user_id)
        if profile is None:
            run_log.append(f"NO_PROFILE user_id={request.user_id} request_id={request.request_id}")
            continue

        user_messages = messages_by_user.get(request.user_id, [])

        # Target-event-scoped facts (validated real use case: 39 messages dataset-wide
        # carry related_event_id). General recurring-scope facts beyond salary are not
        # extracted -- no validated real need for them in this dataset (see conversation:
        # every non-salary recurring-pattern ambiguity found was the salary case itself).
        event_scoped_messages = [m for m in user_messages if m.related_event_id]
        event_scoped_facts = messages_llm.extract_message_facts(event_scoped_messages, all_events_by_id, call_llm)

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

        salary_facts = messages_llm.extract_salary_facts(user_messages, call_llm)
        series = forecast.apply_salary_message_facts(series, salary_facts, run_log=run_log)

        payment_options = payment_options_by_request.get(request.request_id, [])
        decision = candidates.decide(profile, request, resolved, series, payment_options, run_log=run_log)
        explanation = explain_llm.generate_explanation(client, request, profile, decision)

        row = {
            "request_id": request.request_id,
            "amount_safe_to_pay": _format_amount_for_csv(decision.amount_safe_to_pay),
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
            print(f"  {i}/{len(requests)} done")

    all_violations.extend(verify.verify_output(rows, requests))

    print(f"\nWriting {config.OUTPUT_CSV}...")
    with open(config.OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = usage_tracker.write_report(num_requests=len(rows))
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


def _format_amount_for_csv(amount: float) -> str:
    from buyorwait.formatting import format_amount

    return format_amount(amount)


if __name__ == "__main__":
    sys.exit(main())
