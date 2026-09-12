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


if __name__ == "__main__":
    unittest.main()
