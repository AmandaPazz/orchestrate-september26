"""Synthetic coverage for ranking.py's 6-level tie-break.

Real data never exercises levels 4-6: scanning all 250 requests.csv rows, the maximum
number of simultaneously eligible-and-safe candidates for any single request is 2 (see
conversation walkthrough on request_56). These tests force each level to be the deciding
factor in isolation, with N >= 3 candidates where it matters, to prove rank_candidates is
generic over candidate count and doesn't just happen to work for the 2-way case validated
against real data.
"""
from __future__ import annotations

import unittest
from datetime import date

from buyorwait.ranking import PlanCandidate, rank_candidates, best_candidate


def _candidate(
    method="full_payment",
    completes=True,
    changes=(),
    total=1000.0,
    start=date(2025, 1, 1),
    n_payments=1,
    option_id=None,
    payload=None,
):
    return PlanCandidate(
        method=method,
        completes_by_deadline=completes,
        spending_changes=changes,
        total_amount_paid=total,
        start_date=start,
        number_of_payments=n_payments,
        payment_option_id=option_id,
        payload=payload,
    )


class TestLevel1Deadline(unittest.TestCase):
    def test_completes_by_deadline_dominates_everything_else(self):
        # A is worse on every later level but DOES complete by the deadline.
        a = _candidate(payload="A", completes=True, total=9999, start=date(2025, 6, 1), n_payments=99, option_id="payment_option_99")
        b = _candidate(payload="B", completes=False, total=1, start=date(2025, 1, 1), n_payments=1, option_id="payment_option_1")
        winner = best_candidate([a, b])
        self.assertEqual(winner.payload, "A")


class TestLevel2SpendingChanges(unittest.TestCase):
    def test_no_changes_beats_lower_cost_with_changes(self):
        a = _candidate(payload="A", changes=(), total=500.0)
        b = _candidate(payload="B", changes=("stop:event_1",), total=100.0)
        winner = best_candidate([a, b])
        self.assertEqual(winner.payload, "A")

    def test_fewer_changes_beats_more_changes(self):
        a = _candidate(payload="A", changes=("stop:event_1",), total=100.0)
        b = _candidate(payload="B", changes=("stop:event_1", "reduce_to:event_2:50"), total=50.0)
        winner = best_candidate([a, b])
        self.assertEqual(winner.payload, "A")


class TestLevel3TotalCost(unittest.TestCase):
    def test_minimizes_total_amount_paid(self):
        a = _candidate(payload="A", total=881.56, start=date(2025, 2, 15), n_payments=1)
        b = _candidate(payload="B", total=916.84, start=date(2025, 2, 3), n_payments=2)
        winner = best_candidate([a, b])
        self.assertEqual(winner.payload, "A")


class TestLevel4StartDate(unittest.TestCase):
    def test_earliest_start_wins_with_levels_1_3_tied(self):
        # Levels 1-3 identical for all three; level 4 must decide. The earliest-start
        # candidate is deliberately given the WORST number_of_payments and
        # payment_option_id, so a bug that skipped straight to level 5/6 would pick a
        # different winner.
        common = dict(completes=True, changes=(), total=1000.0)
        a = _candidate(payload="A", start=date(2025, 3, 10), n_payments=5, option_id="payment_option_99", **common)
        b = _candidate(payload="B", start=date(2025, 3, 5), n_payments=9, option_id="payment_option_50", **common)  # earliest
        c = _candidate(payload="C", start=date(2025, 3, 20), n_payments=1, option_id="payment_option_1", **common)
        winner = best_candidate([a, b, c])
        self.assertEqual(winner.payload, "B")

    def test_generic_over_many_candidates(self):
        common = dict(completes=True, changes=(), total=1000.0, n_payments=3, option_id=None)
        candidates = [
            _candidate(payload=f"day{i}", start=date(2025, 1, i), **common) for i in (10, 3, 7, 1, 15, 2)
        ]
        winner = best_candidate(candidates)
        self.assertEqual(winner.payload, "day1")


class TestLevel5NumberOfPayments(unittest.TestCase):
    def test_fewest_payments_wins_with_levels_1_4_tied(self):
        common = dict(completes=True, changes=(), total=1000.0, start=date(2025, 3, 1))
        a = _candidate(payload="A", n_payments=3, option_id="payment_option_99", **common)
        b = _candidate(payload="B", n_payments=1, option_id="payment_option_50", **common)  # fewest
        c = _candidate(payload="C", n_payments=5, option_id="payment_option_1", **common)
        winner = best_candidate([a, b, c])
        self.assertEqual(winner.payload, "B")


class TestLevel6PaymentOptionId(unittest.TestCase):
    def test_lowest_payment_option_id_wins_with_levels_1_5_tied(self):
        common = dict(completes=True, changes=(), total=1000.0, start=date(2025, 3, 1), n_payments=3)
        a = _candidate(payload="A", option_id="payment_option_99", **common)
        b = _candidate(payload="B", option_id="payment_option_2", **common)  # lowest
        c = _candidate(payload="C", option_id="payment_option_50", **common)
        winner = best_candidate([a, b, c])
        self.assertEqual(winner.payload, "B")

    def test_numeric_aware_not_lexicographic(self):
        # Lexicographic string sort would put "payment_option_10" before
        # "payment_option_9" (since "1" < "9"). Numeric-aware sort must not do that.
        common = dict(completes=True, changes=(), total=1000.0, start=date(2025, 3, 1), n_payments=3)
        nine = _candidate(payload="nine", option_id="payment_option_9", **common)
        ten = _candidate(payload="ten", option_id="payment_option_10", **common)
        winner = best_candidate([ten, nine])
        self.assertEqual(winner.payload, "nine")

    def test_candidate_without_an_option_id_sorts_after_one_with(self):
        common = dict(completes=True, changes=(), total=1000.0, start=date(2025, 3, 1), n_payments=3)
        no_id = _candidate(payload="no_id", option_id=None, **common)
        with_id = _candidate(payload="with_id", option_id="payment_option_5", **common)
        winner = best_candidate([no_id, with_id])
        self.assertEqual(winner.payload, "with_id")


class TestGenericArity(unittest.TestCase):
    def test_single_candidate_returned_as_is(self):
        only = _candidate(payload="only")
        self.assertEqual(best_candidate([only]).payload, "only")

    def test_empty_list_returns_none(self):
        self.assertIsNone(best_candidate([]))

    def test_rank_candidates_returns_full_order_not_just_winner(self):
        common = dict(completes=True, changes=(), total=1000.0, n_payments=1, option_id=None)
        a = _candidate(payload="A", start=date(2025, 1, 3), **common)
        b = _candidate(payload="B", start=date(2025, 1, 1), **common)
        c = _candidate(payload="C", start=date(2025, 1, 2), **common)
        ranked = rank_candidates([a, b, c])
        self.assertEqual([r.payload for r in ranked], ["B", "C", "A"])


if __name__ == "__main__":
    unittest.main()
