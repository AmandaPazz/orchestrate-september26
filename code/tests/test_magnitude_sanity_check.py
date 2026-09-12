"""Synthetic coverage for forecast.py's magnitude sanity check on message-derived salary
amendments -- the deterministic defense added after the layer-3 jailbreak test showed a
fake, self-reported-high-confidence salary figure could otherwise flow straight through
into the forecast. See forecast.py's `_magnitude_sanity_check` docstring for the real
message-pattern evidence behind the "must be strictly after sent_at" plausibility bar.
"""
from __future__ import annotations

import unittest
from datetime import date

from buyorwait.events import MessageFact
from buyorwait.forecast import RecurringSeries, apply_salary_message_facts


def _salary_series(most_recent_amount=145000.0):
    return RecurringSeries(
        category="salary",
        direction="credit",
        interval_days=30,
        is_monthly=True,
        projected_amount=most_recent_amount,
        last_known_date=date(2024, 11, 15),
        template_event_id="event_test_salary",
        flexibility="fixed",
        minimum_allowed_amount=None,
        most_recent_amount=most_recent_amount,
    )


def _fact(new_amount, sent_at, effective_date=None, confidence="high", fact_type="confirm"):
    return MessageFact(
        message_id="message_test",
        fact_type=fact_type,
        target_event_id=None,
        recurring_scope="salary",
        new_amount=new_amount,
        new_date=effective_date,
        effective_date=effective_date,
        confidence=confidence,
        sent_at=sent_at,
    )


class TestMagnitudeSanityCheck(unittest.TestCase):
    def test_small_deviation_passes_unchanged(self):
        # 10% raise -- well within the 50% threshold, no override.
        series = [_salary_series(145000.0)]
        fact = _fact(159500.0, sent_at="2024-12-01T09:00:00Z")
        run_log = []
        result = apply_salary_message_facts(series, [fact], run_log=run_log)
        self.assertEqual(result[0].projected_amount, 159500.0)
        self.assertEqual(run_log, [])

    def test_wild_deviation_with_no_future_effective_date_is_downgraded(self):
        series = [_salary_series(145000.0)]
        fact = _fact(999999999.0, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 1))
        run_log = []
        result = apply_salary_message_facts(series, [fact], run_log=run_log)
        # Downgraded to low confidence -> filtered out entirely -> series unchanged.
        self.assertEqual(result[0].projected_amount, 145000.0)
        self.assertEqual(len(run_log), 1)
        self.assertIn("MAGNITUDE_SANITY_OVERRIDE", run_log[0])
        self.assertIn("event_test_salary", run_log[0])

    def test_wild_deviation_with_plausible_future_effective_date_is_trusted(self):
        # Mirrors the real dataset pattern: sent well before the date it describes
        # (verified: message_11 sent 12 days before effective_date, message_10 19 days).
        series = [_salary_series(145000.0)]
        fact = _fact(300000.0, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 15))
        run_log = []
        result = apply_salary_message_facts(series, [fact], run_log=run_log)
        self.assertEqual(result[0].projected_amount, 300000.0)
        self.assertEqual(run_log, [])

    def test_same_day_effective_date_is_not_plausible(self):
        # An "effective today" claim on a wild jump is the anomalous pattern itself, not
        # advance notice of a real change -- must still be downgraded.
        series = [_salary_series(145000.0)]
        fact = _fact(999999999.0, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 1))
        run_log = []
        result = apply_salary_message_facts(series, [fact], run_log=run_log)
        self.assertEqual(result[0].projected_amount, 145000.0)
        self.assertEqual(len(run_log), 1)

    def test_cancel_facts_are_not_subject_to_the_magnitude_check(self):
        series = [_salary_series(145000.0)]
        fact = _fact(None, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 1), fact_type="cancel")
        run_log = []
        result = apply_salary_message_facts(series, [fact], run_log=run_log)
        self.assertEqual(result[0].stop_after_date, date(2024, 12, 1))
        self.assertEqual(run_log, [])

    def test_no_structural_series_means_nothing_to_check_against(self):
        # No existing salary series -- the "first job" synthesis path -- has no reference
        # amount, so the magnitude check can't fire (and shouldn't try to).
        fact = _fact(999999999.0, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 15))
        run_log = []
        result = apply_salary_message_facts([], [fact], run_log=run_log)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].projected_amount, 999999999.0)
        self.assertEqual(run_log, [])

    def test_log_entry_records_deviation_and_model_claimed_confidence(self):
        series = [_salary_series(100.0)]
        fact = _fact(1000.0, sent_at="2024-12-01T09:00:00Z", effective_date=date(2024, 12, 1))
        run_log = []
        apply_salary_message_facts(series, [fact], run_log=run_log)
        self.assertIn("reference_amount=100", run_log[0])
        self.assertIn("claimed_new_amount=1000", run_log[0])
        self.assertIn("model_claimed_confidence=high", run_log[0])
        self.assertIn("message_id=message_test", run_log[0])


def _two_stream_series():
    primary = _salary_series(91760.0)
    primary.template_event_id = "event_primary"
    second = _salary_series(55354.0)
    second.template_event_id = "event_second"
    return [primary, second]


class TestHouseholdConsolidation(unittest.TestCase):
    """Real dataset pattern (7 of 10 users with two structurally distinct salary series):
    a message pairs a bare "cancel" (no amount -- one income source ended) with a
    "confirm" (a new, deliberately-unreconcilable-with-either-stream amount) in the same
    message. Verified: e.g. user_42's confirmed 148000 is neither its Primary (91760) nor
    Second (~55-74k) stream's value -- a consistent ~1.613x multiplier applied dataset-wide,
    clearly a deliberate "household consolidated" figure, not decomposable arithmetic.
    """

    def test_paired_cancel_confirm_with_two_real_series_replaces_both(self):
        series = _two_stream_series()
        facts = [
            _fact(None, sent_at="2025-12-01T09:00:00Z", fact_type="cancel"),
            _fact(148000.0, sent_at="2025-12-01T09:00:00Z", fact_type="confirm"),
        ]
        # Give both facts the same message_id (the _fact helper defaults to "message_test").
        run_log = []
        result = apply_salary_message_facts(series, facts, run_log=run_log)
        salary_series = [s for s in result if s.category == "salary"]
        self.assertEqual(len(salary_series), 1, "the two old streams must be replaced by one consolidated series")
        self.assertEqual(salary_series[0].projected_amount, 148000.0)
        self.assertTrue(any("HOUSEHOLD_CONSOLIDATION_APPLIED" in l for l in run_log))

    def test_deviation_alone_does_not_trigger_magnitude_override_for_this_pair(self):
        # Confirms the pair is NOT routed through the strict per-series magnitude check
        # (which would otherwise downgrade 148000 vs 91760 -- a 61% deviation -- to low
        # confidence and reject it, since no effective_date is given).
        series = _two_stream_series()
        facts = [
            _fact(None, sent_at="2025-12-01T09:00:00Z", fact_type="cancel"),
            _fact(148000.0, sent_at="2025-12-01T09:00:00Z", fact_type="confirm"),
        ]
        run_log = []
        apply_salary_message_facts(series, facts, run_log=run_log)
        self.assertFalse(any("MAGNITUDE_SANITY_OVERRIDE" in l for l in run_log))

    def test_single_series_user_cannot_use_the_same_trick(self):
        # The adversarial case: an attacker controls the message but not the user's real
        # event history. With only ONE real structural salary series, the exact same
        # "cancel + confirm a huge number" message shape must NOT get the exception --
        # it has to survive the normal magnitude check, and here it doesn't.
        series = [_salary_series(145000.0)]
        facts = [
            _fact(None, sent_at="2024-12-01T09:00:00Z", fact_type="cancel"),
            _fact(999999999.0, sent_at="2024-12-01T09:00:00Z", fact_type="confirm"),
        ]
        run_log = []
        result = apply_salary_message_facts(series, facts, run_log=run_log)
        self.assertFalse(any("HOUSEHOLD_CONSOLIDATION_APPLIED" in l for l in run_log))
        # Falls through to the single-series path; the bare "cancel" (no amount, latest by
        # effective_date) wins the tie-break there and stops the series -- either way, the
        # fabricated 999999999 confirm must never become the projected amount.
        salary_series = [s for s in result if s.category == "salary"]
        for s in salary_series:
            self.assertNotEqual(s.projected_amount, 999999999.0)

    def test_low_confidence_consolidation_confirm_is_not_applied(self):
        series = _two_stream_series()
        facts = [
            _fact(None, sent_at="2025-12-01T09:00:00Z", fact_type="cancel", confidence="low"),
            _fact(148000.0, sent_at="2025-12-01T09:00:00Z", fact_type="confirm", confidence="low"),
        ]
        run_log = []
        result = apply_salary_message_facts(series, facts, run_log=run_log)
        salary_series = [s for s in result if s.category == "salary"]
        self.assertEqual(len(salary_series), 2, "low-confidence pair must not trigger consolidation")


if __name__ == "__main__":
    unittest.main()
