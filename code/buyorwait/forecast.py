"""90-day cash-flow forecast: recurrence detection, ledger simulation, and the two
headline numbers (amount_safe_to_pay, earliest_date_for_full_payment) computed before any
optional spending change. candidates.py (group c) re-runs this with spending changes
applied to evaluate affordable_with_plan options.

Design decisions locked after investigating the real dataset (see the walkthrough in
conversation for the worked example on user_46 / request_46):

- Recurring series are grouped by (user, category, direction) only — never description,
  which rotates through several templates per occurrence even for a genuinely weekly
  series (verified: user_46's groceries cycles through 4+ description templates on an
  exact 7-day cadence).
- Recurrence threshold: >=2 settled occurrences when the gap matches a common cadence
  (7, 14, or 28-31 days, within tolerance); >=3 occurrences otherwise. Validated
  dataset-wide: of 17 two-occurrence (user,category,direction) groups, 16 are real
  monthly-cadence recurring salary ("First-job payroll", flat amount, 30-31 day gap) and
  1 is a false positive (22-day gap, mismatched amounts/descriptions, correctly excluded).
- Conservative projection for variable-amount series: max of the trailing min(3, n)
  occurrences for a debit series (assume the worst realistic future expense), min of the
  trailing min(3, n) for a credit series (assume the worst realistic future income) — not
  a single "conservative = max" rule, since applying max to income would be reckless.
- A projected-forward event landing exactly on `request_date` is treated as "about to
  happen today" and included in the forecast (the safer/more conservative choice).
- 34 users' salary recurrence is genuinely undecidable from financial_events.csv alone
  (seasonal contracts, leave returns, first-job records with only 1-2 payslips so far) —
  and every one of them has an employer message in messages.csv that resolves it
  explicitly. High-confidence salary MessageFacts (from messages_llm.py) override
  structural detection for the salary series (explicit stop, or explicit resume/confirm
  at a stated amount+date); low-confidence ones are ignored, falling back to whatever the
  structural detector alone would say.
"""
from __future__ import annotations

import calendar
import dataclasses
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

from . import config
from .events import MessageFact
from .io_loader import Event


_COMMON_CADENCES_DAYS = (7, 14, 28, 29, 30, 31)

# financial_events.csv descriptions are dataset-generated, fixed-vocabulary English (unlike
# messages.csv, which is free-form and multilingual) -- a plain substring check here is
# safe and appropriate. Verified real case: user_05's last salary occurrence is literally
# described "Final employer payroll" (7 occurrences dataset-wide with this exact text);
# without this check the series projects a job that has already ended, overstating future
# income. Scanned every description in financial_events.csv for related termination
# language (final/last/closing/closed/ended/terminated/cancelled) -- this is the only
# recurring-series-relevant hit; "Cancelled ... authorization" rows are already
# status=cancelled and excluded by status filtering regardless.
_TERMINAL_DESCRIPTION_MARKERS = ("final",)


def _cadence_tolerance(avg_gap: float) -> int:
    return max(3, round(0.20 * avg_gap))


def _matches_common_cadence(avg_gap: float, tolerance: int) -> bool:
    return any(abs(avg_gap - c) <= tolerance for c in _COMMON_CADENCES_DAYS)


@dataclass
class RecurringSeries:
    category: str
    direction: str  # "credit" | "debit"
    interval_days: int  # nominal interval used for projection cadence classification
    is_monthly: bool  # True -> project by calendar month (same day-of-month); False -> fixed day interval
    projected_amount: float
    last_known_date: date
    template_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[float]
    most_recent_amount: float = 0.0  # the literal last settled/scheduled occurrence's own amount
    # (distinct from projected_amount, which is the conservative max/min of a trailing
    # window) -- used as the baseline for the message-fact magnitude sanity check.
    stop_after_date: Optional[date] = None  # set by a high-confidence "cancel" salary fact


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def _gap_consistency(evs_sorted: list[Event]) -> Optional[tuple[float, bool]]:
    """Median-based, majority-vote consistency check over the most recent gaps. Returns
    (median_gap, is_monthly) if consistent, else None.

    Uses the median (not the mean) as the tolerance anchor, and allows one date-outlier
    among the last 4 gaps rather than requiring all of them to pass — a single delayed
    settlement (verified real case: user_07's payroll settled 8 days late once) shouldn't
    break an otherwise clean monthly series. A group with only 1-2 gaps has no slack for
    an outlier; a real pattern break still fails.

    The excused outlier's OWN amount must also be within 30% of the group's median amount.
    A date-outlier alone isn't enough to excuse — verified real case: user_04's "Quarterly
    performance bonus" (10.5M) landed close enough in date terms to slip past the gap check
    among five 38.19M payrolls, and because it was also the most recent occurrence, it got
    picked as the projection anchor, cutting projected future salary by ~72%. Requiring the
    amount to match too correctly rejects this while still excusing user_07's same-amount,
    merely-late settlement.
    """
    dates = [e.settlement_date or e.event_date for e in evs_sorted]
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    if not gaps:
        return None
    recent_gaps = gaps[-4:]
    recent_events = evs_sorted[-5:]  # the events bounding recent_gaps (one more than gaps)
    median_gap = _median(recent_gaps)
    tolerance = _cadence_tolerance(median_gap)
    median_amount = _median([e.home_currency_amount for e in evs_sorted])

    outlier_count = 0
    for i, g in enumerate(recent_gaps):
        if abs(g - median_gap) <= tolerance:
            continue
        # gap i sits between recent_events[i] and recent_events[i+1]; the later event is
        # the one whose date broke the pattern, so its amount is what must still match.
        outlier_event = recent_events[i + 1]
        amount_ok = (
            median_amount > 0
            and abs(outlier_event.home_currency_amount - median_amount) / median_amount <= 0.30
        )
        if not amount_ok:
            return None
        outlier_count += 1

    allowed_outliers = 1 if len(recent_gaps) >= 3 else 0
    if outlier_count > allowed_outliers:
        return None
    is_monthly = 25 <= median_gap <= 35
    return median_gap, is_monthly


def _cluster_by_day_of_month(evs_sorted: list[Event], tolerance_days: int = 3) -> list[list[Event]]:
    """Fallback recovery for a group whose gaps aren't consistent as one series: bins
    events by day-of-month (greedy, sorted). Verified real cases this resolves: a base
    salary + a separate variable commission under the same category (day 15 vs day 24),
    a primary + secondary household income (day 15 vs day 20), and a one-time
    bonus/arrears credit mixed into an otherwise clean monthly payroll (forms its own
    too-small cluster and is correctly dropped rather than breaking the whole series).

    tolerance_days=3, not 5: at 5, a one-time same-category credit just 5 days off the
    real pattern (verified real case: user_03's "Promotion arrears payment" landed 5 days
    after the regular day-15 payroll) got pulled into the real cluster, corrupting its
    anchor date/amount. 3 days cleanly separates that case and user_13's day-15/day-20
    two-household-income case while still tolerating ordinary weekend/holiday drift within
    one real series.
    """
    by_day = sorted(evs_sorted, key=lambda e: (e.settlement_date or e.event_date).day)
    clusters: list[list[Event]] = []
    for e in by_day:
        day = (e.settlement_date or e.event_date).day
        if clusters and abs(day - (clusters[-1][-1].settlement_date or clusters[-1][-1].event_date).day) <= tolerance_days:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return [sorted(c, key=lambda e: e.settlement_date or e.event_date) for c in clusters]


def _build_series_from_group(
    category: str, direction: str, evs_sorted: list[Event]
) -> Optional[RecurringSeries]:
    if len(evs_sorted) < 2:
        return None
    consistency = _gap_consistency(evs_sorted)
    if consistency is None:
        return None
    median_gap, is_monthly = consistency

    tolerance = _cadence_tolerance(median_gap)
    min_occurrences = 2 if _matches_common_cadence(median_gap, tolerance) else 3
    if len(evs_sorted) < min_occurrences:
        return None

    last = evs_sorted[-1]
    window = evs_sorted[-min(3, len(evs_sorted)):]
    amounts = [e.home_currency_amount for e in window]
    # Conservative projection: max-of-trailing-window for a debit series, min-of-trailing-
    # window for a credit series. Tried swapping in median-of-trailing-3 (compared against
    # all 25 sample_requests.csv rows): no exact-match regressions and no safety violations,
    # but only 9/25 rows got closer to ground truth against 8/25 that got meaningfully
    # worse (several near-exact matches moved materially further off) -- not a clean win,
    # so kept max/min. The residual ~5-15% conservative gap on some rows is a known,
    # spec-aligned trade-off (the spec asks for essential variable spending to be forecast
    # conservatively), not a bug.
    projected_amount = min(amounts) if direction == "credit" else max(amounts)
    last_date = last.settlement_date or last.event_date

    is_terminal = any(marker in last.description.lower() for marker in _TERMINAL_DESCRIPTION_MARKERS)

    return RecurringSeries(
        category=category,
        direction=direction,
        interval_days=round(median_gap),
        is_monthly=is_monthly,
        projected_amount=projected_amount,
        last_known_date=last_date,
        template_event_id=last.event_id,
        flexibility=last.flexibility,
        minimum_allowed_amount=last.minimum_allowed_amount,
        most_recent_amount=last.home_currency_amount,
        stop_after_date=last_date if is_terminal else None,
    )


def detect_recurring_series(
    home_currency_events: list[Event],
) -> list[RecurringSeries]:
    """`home_currency_events` must already have `home_currency_amount` populated (i.e.
    events.py has run). Considers `settled` and `scheduled` events (a `scheduled` row is
    the dataset's own "next confirmed" signal — verified real case: a user with only 1
    settled salary event plus 1 scheduled one has no other way to establish a pattern).
    Groups by (category, direction); when the whole group's gaps aren't consistent as one
    series, falls back to day-of-month sub-clustering to recover genuinely distinct
    interleaved series (base salary + commission, primary + secondary income, gig-work
    payment days) rather than losing the category's income/expense entirely.
    """
    eligible = [e for e in home_currency_events if e.status in ("settled", "scheduled")]
    groups: dict[tuple[str, str], list[Event]] = {}
    for e in eligible:
        groups.setdefault((e.category, e.direction), []).append(e)

    series: list[RecurringSeries] = []
    for (category, direction), evs in groups.items():
        evs_sorted = sorted(evs, key=lambda e: e.settlement_date or e.event_date)

        built = _build_series_from_group(category, direction, evs_sorted)
        if built is not None:
            series.append(built)
            continue

        for cluster in _cluster_by_day_of_month(evs_sorted):
            built = _build_series_from_group(category, direction, cluster)
            if built is not None:
                series.append(built)
    return series


def _add_month(d: date, months: int = 1) -> date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


_MAGNITUDE_DEVIATION_THRESHOLD = 0.50


def _sent_date(message_iso_timestamp: str) -> Optional[date]:
    try:
        return datetime.fromisoformat(message_iso_timestamp.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return None


def _magnitude_sanity_check(
    salary_facts: list[MessageFact],
    reference_amount: Optional[float],
    template_event_id: Optional[str],
    run_log: Optional[list[str]],
) -> list[MessageFact]:
    """Deterministic guard against trusting the model's self-reported confidence alone for
    an amount change: a "confirm" fact whose new_amount deviates more than 50% from the
    series' most recent known settled amount gets downgraded to confidence="low"
    regardless of what the model claimed — UNLESS the fact also carries an explicit
    effective_date on or after the message's own sent_at (a plausible forward-looking
    change: a raise, a new contract amount), which is the same shape every real
    amendment message in this dataset uses (verified: "Regular salary of X resumes on
    <date>" / "Your first salary will be X, confirmed credit date <date>" are always sent
    before the date they describe). No reference amount (no existing structural series)
    means there's nothing to sanity-check against, so those facts pass through unchanged.

    Same category of defense as the structural duplicate-charge check in events.py: don't
    rely on a single signal (there, linked_event_id; here, the model's self-reported
    confidence) when a second, independent, deterministic signal is available.
    """
    if reference_amount is None or reference_amount <= 0:
        return salary_facts

    result = []
    for fact in salary_facts:
        if fact.fact_type != "confirm" or fact.new_amount is None:
            result.append(fact)
            continue

        deviation = abs(fact.new_amount - reference_amount) / reference_amount
        if deviation <= _MAGNITUDE_DEVIATION_THRESHOLD:
            result.append(fact)
            continue

        sent = _sent_date(fact.sent_at)
        # Strictly AFTER, not on-or-after: every real amendment message in this dataset is
        # advance notice of a change that takes effect later (12-19 days out in the two
        # verified examples), not an immediate/same-day/backdated change -- an "effective
        # today" claim on a wildly different amount is itself the anomalous pattern, not a
        # plausible raise announcement.
        plausible_forward_change = (
            fact.effective_date is not None and sent is not None and fact.effective_date > sent
        )
        if plausible_forward_change:
            result.append(fact)
            continue

        if fact.confidence == "high" and run_log is not None:
            run_log.append(
                f"MAGNITUDE_SANITY_OVERRIDE series={template_event_id} "
                f"model_claimed_confidence={fact.confidence} "
                f"reference_amount={reference_amount:g} claimed_new_amount={fact.new_amount:g} "
                f"deviation={deviation:.1%} message_id={fact.message_id}"
            )
        result.append(dataclasses.replace(fact, confidence="low"))
    return result


def _find_household_consolidation(
    salary_facts: list[MessageFact],
) -> Optional[MessageFact]:
    """Detects the "one household income record ended, remaining confirmed salary is X"
    shape: a paired cancel (no amount) + confirm (with amount) from the SAME message.
    Verified real pattern: 7 of 10 users dataset-wide with two structurally distinct
    salary series (found via day-of-month clustering: "Primary household salary" + a
    separate "Second household income") have exactly this message shape, and the
    confirmed amount consistently doesn't match either individual stream's historical
    value (it's a restated/consolidated figure, not a simple continuation) -- so it would
    otherwise always fail the plain per-series magnitude check despite being a clearly
    legitimate, consistently-templated dataset pattern, not an anomaly.

    Returns the confirm fact if this shape is found in `salary_facts` (already grouped to
    one user), else None. Caller is responsible for the further gate: only actually use
    this if the user has >=2 REAL structural salary series (see apply_salary_message_facts)
    -- an attacker can write anything in a message, but can't fabricate a second series in
    financial_events.csv, so requiring structural corroboration keeps this from being just
    a bigger loophole in the magnitude check for a single-stream user.
    """
    by_message: dict[str, list[MessageFact]] = {}
    for f in salary_facts:
        by_message.setdefault(f.message_id, []).append(f)
    for group in by_message.values():
        has_bare_cancel = any(f.fact_type == "cancel" and f.new_amount is None for f in group)
        confirms = [f for f in group if f.fact_type == "confirm" and f.new_amount is not None]
        if has_bare_cancel and confirms:
            return confirms[0]
    return None


def apply_salary_message_facts(
    series_list: list[RecurringSeries],
    salary_facts: list[MessageFact],
    run_log: Optional[list[str]] = None,
) -> list[RecurringSeries]:
    """High-confidence salary facts override the structural salary series: a "cancel"
    stops projection after its effective_date; a "confirm" resets the projected amount
    (and, if the structural detector found nothing at all, synthesizes the series from
    the message alone — covers the pure first-job case where only 1 settled payslip
    exists and the rest of the picture comes entirely from the message). Low-confidence
    facts are ignored; the structural result (or absence of one) stands as the safer
    fallback.

    Before trusting confidence at all, a deterministic magnitude sanity check
    (_magnitude_sanity_check) downgrades any "confirm" fact whose claimed amount is a
    wild, unexplained jump from the series' real recent history — see that function's
    docstring. This runs regardless of what the model self-reported.

    Household-consolidation exception: when the user has >=2 structurally-detected salary
    series AND the message pairs a bare "cancel" with a "confirm" (see
    _find_household_consolidation), the magnitude check is skipped for that confirm and
    the whole multi-stream setup is replaced by one new consolidated series at the
    confirmed amount — gated on real structural corroboration, not just the message's own
    say-so, so a single-stream user's fabricated "another income ended, salary is now
    $999,999,999" claim still can't buy its way past the magnitude check this way.
    """
    salary_series_indices = [i for i, s in enumerate(series_list) if s.category == "salary"]
    result = list(series_list)

    if len(salary_series_indices) >= 2:
        consolidation_fact = _find_household_consolidation(salary_facts)
        if consolidation_fact is not None and consolidation_fact.confidence == "high":
            if run_log is not None:
                run_log.append(
                    f"HOUSEHOLD_CONSOLIDATION_APPLIED series_count={len(salary_series_indices)} "
                    f"new_amount={consolidation_fact.new_amount:g} message_id={consolidation_fact.message_id}"
                )
            anchor_date = (
                (consolidation_fact.new_date - timedelta(days=30))
                if consolidation_fact.new_date
                else max(series_list[i].last_known_date for i in salary_series_indices)
            )
            result = [s for i, s in enumerate(result) if i not in salary_series_indices]
            result.append(
                RecurringSeries(
                    category="salary",
                    direction="credit",
                    interval_days=30,
                    is_monthly=True,
                    projected_amount=consolidation_fact.new_amount,
                    last_known_date=anchor_date,
                    template_event_id=f"message:{consolidation_fact.message_id}",
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                )
            )
            return result

    # Single-series path (the common case; unchanged behavior for every previously
    # validated real case, since every one of those users has exactly one salary series).
    salary_idx = salary_series_indices[0] if salary_series_indices else None
    reference_amount = series_list[salary_idx].most_recent_amount if salary_idx is not None else None
    reference_id = series_list[salary_idx].template_event_id if salary_idx is not None else None
    salary_facts = _magnitude_sanity_check(salary_facts, reference_amount, reference_id, run_log)

    high_confidence = [f for f in salary_facts if f.confidence == "high"]
    if not high_confidence:
        return result

    high_confidence.sort(key=lambda f: f.effective_date or date.min)
    latest = high_confidence[-1]

    salary_idx = next((i for i, s in enumerate(result) if s.category == "salary"), None)

    if latest.fact_type == "cancel":
        if salary_idx is not None:
            result[salary_idx].stop_after_date = latest.effective_date
        return result

    if latest.fact_type == "confirm" and latest.new_amount is not None:
        if salary_idx is not None:
            result[salary_idx].projected_amount = latest.new_amount
            if latest.new_date:
                result[salary_idx].last_known_date = latest.new_date - timedelta(days=30)
        else:
            # No structural series at all (e.g. exactly one settled "first" payslip) —
            # the message alone is a confirmed, dated, amounted fact, not an invented one.
            result.append(
                RecurringSeries(
                    category="salary",
                    direction="credit",
                    interval_days=30,
                    is_monthly=True,
                    projected_amount=latest.new_amount,
                    last_known_date=(latest.new_date - timedelta(days=30)) if latest.new_date else date.min,
                    template_event_id=f"message:{latest.message_id}",
                    flexibility="fixed",
                    minimum_allowed_amount=None,
                )
            )
    return result


@dataclass
class LedgerEntry:
    entry_date: date
    signed_amount: float  # positive for credit, negative for debit
    category: str
    source_event_id: str  # real event_id, or "message:<id>", or "series:<category>:<date>"


def _project_series(series: RecurringSeries, horizon_end: date) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    sign = 1 if series.direction == "credit" else -1
    d = _add_month(series.last_known_date, 1) if series.is_monthly else series.last_known_date + timedelta(days=series.interval_days)
    while d <= horizon_end:
        if series.stop_after_date is not None and d > series.stop_after_date:
            break
        entries.append(
            LedgerEntry(
                entry_date=d,
                signed_amount=sign * series.projected_amount,
                category=series.category,
                source_event_id=f"series:{series.category}:{d.isoformat()}",
            )
        )
        d = _add_month(d, 1) if series.is_monthly else d + timedelta(days=series.interval_days)
    return entries


def _known_committed_entries(
    resolved_events: list[Event], request_date: date, horizon_end: date
) -> list[LedgerEntry]:
    """Direct (non-projected) future cash flow: scheduled events, and pending debits,
    whose settlement_date falls in [request_date, horizon_end]."""
    entries: list[LedgerEntry] = []
    for e in resolved_events:
        if e.excluded_reason is not None:
            continue
        settle = e.settlement_date or e.event_date
        if settle is None or not (request_date <= settle <= horizon_end):
            continue
        if e.status not in ("scheduled", "pending"):
            continue
        sign = 1 if e.direction == "credit" else -1
        entries.append(
            LedgerEntry(
                entry_date=settle,
                signed_amount=sign * (e.home_currency_amount or 0.0),
                category=e.category,
                source_event_id=e.event_id,
            )
        )
    return entries


def _dedupe_against_known(
    projected: list[LedgerEntry], known: list[LedgerEntry], tolerance_days: int = 3
) -> list[LedgerEntry]:
    """Drops a projected occurrence that coincides (same category, within
    tolerance_days) with a real known-committed event — the real record wins over the
    projection (e.g. a "scheduled" salary row beats a projected one for the same month)."""
    result = []
    for p in projected:
        collides = any(
            k.category == p.category and abs((k.entry_date - p.entry_date).days) <= tolerance_days
            for k in known
        )
        if not collides:
            result.append(p)
    return result


@dataclass
class ForecastResult:
    amount_safe_to_pay: float
    earliest_date_for_full_payment: Optional[date]
    ledger: list[LedgerEntry]
    points: list[tuple[date, float]]  # (date, balance_after) chronological
    suffix_min: list[float]  # aligned with points


def run_forecast(
    starting_balance: float,
    minimum_balance_to_keep: float,
    request_date: date,
    requested_amount: float,
    resolved_events: list[Event],
    recurring_series: list[RecurringSeries],
    horizon_days: int = config.FORECAST_HORIZON_DAYS,
) -> ForecastResult:
    horizon_end = request_date + timedelta(days=horizon_days)

    known = _known_committed_entries(resolved_events, request_date, horizon_end)
    ledger: list[LedgerEntry] = list(known)
    for series in recurring_series:
        projected = _project_series(series, horizon_end)
        ledger.extend(_dedupe_against_known(projected, known))

    ledger.sort(key=lambda le: le.entry_date)

    points: list[tuple[date, float]] = [(request_date, starting_balance)]
    bal = starting_balance
    for entry in ledger:
        bal += entry.signed_amount
        if points[-1][0] == entry.entry_date:
            points[-1] = (entry.entry_date, bal)
        else:
            points.append((entry.entry_date, bal))

    n = len(points)
    suffix_min = [0.0] * n
    suffix_min[-1] = points[-1][1]
    for i in range(n - 2, -1, -1):
        suffix_min[i] = min(points[i][1], suffix_min[i + 1])

    amount_safe_to_pay = max(0.0, min(requested_amount, suffix_min[0] - minimum_balance_to_keep))

    target = requested_amount + minimum_balance_to_keep
    earliest_date_for_full_payment = None
    for i, (d, _b) in enumerate(points):
        if suffix_min[i] >= target:
            earliest_date_for_full_payment = d
            break

    return ForecastResult(
        amount_safe_to_pay=amount_safe_to_pay,
        earliest_date_for_full_payment=earliest_date_for_full_payment,
        ledger=ledger,
        points=points,
        suffix_min=suffix_min,
    )
