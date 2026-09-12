"""Synthetic coverage for spending_changes.py.

request_112 (real data, see conversation walkthrough) only validates the 1-change path.
These tests construct fake profiles/series to force the 2-change and 3-change paths, the
stop/reduce mutual-exclusion rule, and the "prefer cheapest among multiple solutions" rule
-- none of which any real request in requests.csv exercises (the search only ever needs a
single change in the real dataset).
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

from buyorwait.forecast import RecurringSeries
from buyorwait.io_loader import Profile
from buyorwait.spending_changes import ChangeOption, _valid_combo, find_best_combo

REQUEST_DATE = date(2025, 1, 1)


def _profile(willing_to_stop=(), willing_to_reduce=(), balance=1000.0, min_balance=500.0):
    return Profile(
        user_id="user_test",
        home_currency="USD",
        current_available_balance=balance,
        minimum_balance_to_keep=min_balance,
        financial_priorities=(),
        expense_categories_to_protect=(),
        expense_categories_user_is_willing_to_reduce=willing_to_reduce,
        expense_categories_user_is_willing_to_stop=willing_to_stop,
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )


def _single_occurrence_series(category, amount, event_id, flexibility="stoppable", min_allowed=None):
    """A flexible debit series with exactly one projected occurrence landing precisely on
    REQUEST_DATE (interval=200 days, last known 200 days before REQUEST_DATE, so the next
    projected occurrence is REQUEST_DATE and the one after falls outside the 90-day
    horizon) -- makes the arithmetic exact and easy to hand-verify."""
    return RecurringSeries(
        category=category,
        direction="debit",
        interval_days=200,
        is_monthly=False,
        projected_amount=amount,
        last_known_date=REQUEST_DATE - timedelta(days=200),
        template_event_id=event_id,
        flexibility=flexibility,
        minimum_allowed_amount=min_allowed,
    )


class TestValidCombo(unittest.TestCase):
    def test_rejects_same_event_stop_and_reduce(self):
        stop = ChangeOption("stop", "event_1", "cat", 300, 0, 300)
        reduce = ChangeOption("reduce", "event_1", "cat", 300, 100, 200)
        self.assertFalse(_valid_combo((stop, reduce)))

    def test_accepts_distinct_events(self):
        a = ChangeOption("stop", "event_1", "cat", 300, 0, 300)
        b = ChangeOption("stop", "event_2", "cat", 50, 0, 50)
        self.assertTrue(_valid_combo((a, b)))


class TestSingleChange(unittest.TestCase):
    def test_one_change_closes_the_gap(self):
        # Mirrors request_112 in shape: one flexible debit, stopping it is enough.
        profile = _profile(willing_to_stop=("cat_a",))
        series = [_single_occurrence_series("cat_a", 300.0, "event_a")]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=200.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 1)
        self.assertEqual(combo[0].template_event_id, "event_a")


class TestTwoChangesRequired(unittest.TestCase):
    def test_no_single_change_suffices_but_a_pair_does(self):
        # balance=1000, min=500, headroom=500. Two independent flexible debits, A=100,
        # B=450 (sum=550). requested=401 (one dollar above what stopping B alone gives).
        # baseline (both active): balance=1000-550=450, safe=0.
        # stop A alone (B still active): balance=1000-450=550, safe=50  -> insufficient.
        # stop B alone (A still active): balance=1000-100=900, safe=400 -> still short by 1.
        # stop A+B (both):               balance=1000,         safe=500 -> succeeds.
        profile = _profile(willing_to_stop=("cat_a", "cat_b"))
        series = [
            _single_occurrence_series("cat_a", 100.0, "event_a"),
            _single_occurrence_series("cat_b", 450.0, "event_b"),
        ]
        requested = 401.0
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=requested)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 2)
        ids = {c.template_event_id for c in combo}
        self.assertEqual(ids, {"event_a", "event_b"})


class TestThreeChangesRequired(unittest.TestCase):
    def test_no_pair_suffices_but_the_triple_does(self):
        # balance=1000, min=500, headroom=500. Three equal flexible debits of 200 each
        # (sum=600). requested=480.
        # baseline: balance=400, safe=0.
        # any single stopped (impact 200): remaining debit 400, balance=600, safe=100 -> insufficient.
        # any pair stopped (impact 400): remaining debit 200, balance=800, safe=300 -> insufficient.
        # all three stopped (impact 600): balance=1000, safe=500 >= 480 -> succeeds.
        profile = _profile(willing_to_stop=("cat_a", "cat_b", "cat_c"))
        series = [
            _single_occurrence_series("cat_a", 200.0, "event_a"),
            _single_occurrence_series("cat_b", 200.0, "event_b"),
            _single_occurrence_series("cat_c", 200.0, "event_c"),
        ]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=480.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 3)
        ids = {c.template_event_id for c in combo}
        self.assertEqual(ids, {"event_a", "event_b", "event_c"})

    def test_max_changes_cap_respected(self):
        # Same setup, but cap the search at 2 changes -- must correctly report no solution
        # exists within that cap, not silently return a size-3 combo.
        profile = _profile(willing_to_stop=("cat_a", "cat_b", "cat_c"))
        series = [
            _single_occurrence_series("cat_a", 200.0, "event_a"),
            _single_occurrence_series("cat_b", 200.0, "event_b"),
            _single_occurrence_series("cat_c", 200.0, "event_c"),
        ]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=480.0, max_changes=2)
        self.assertIsNone(combo)


class TestPrefersCheapest(unittest.TestCase):
    def test_picks_lower_impact_single_solution_not_first_found(self):
        # Two independent flexible debits, EACH individually enough to close the gap when
        # the other stays active: D1=250 (cheaper), D2=280 (pricier). Both succeed alone;
        # the cheaper one (D1, less disruption to the user) must win.
        # Deliberately list D2 before D1 in the series list -- if the search just returned
        # the first successful combo instead of minimizing impact, it would wrongly pick D2.
        profile = _profile(willing_to_stop=("cat_1", "cat_2"))
        series = [
            _single_occurrence_series("cat_2", 280.0, "event_d2"),
            _single_occurrence_series("cat_1", 250.0, "event_d1"),
        ]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=200.0)
        self.assertIsNotNone(combo)
        self.assertEqual(len(combo), 1)
        self.assertEqual(combo[0].template_event_id, "event_d1")


class TestMutualExclusionIntegration(unittest.TestCase):
    def test_never_returns_both_stop_and_reduce_for_the_same_event(self):
        # One reducible_or_stoppable event E (stop impact 300, reduce impact 200) plus one
        # plain stoppable event F (impact 50). Neither alone suffices; only stopping BOTH
        # distinct events (E and F) does -- the search must not "solve" this by pairing
        # stop-E with reduce-E for the same event.
        profile = _profile(willing_to_stop=("cat_e", "cat_f"), willing_to_reduce=("cat_e",))
        series = [
            _single_occurrence_series("cat_e", 300.0, "event_e", flexibility="reducible_or_stoppable", min_allowed=100.0),
            _single_occurrence_series("cat_f", 50.0, "event_f"),
        ]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=470.0)
        self.assertIsNotNone(combo)
        ids = [c.template_event_id for c in combo]
        self.assertEqual(len(ids), len(set(ids)), "same event must never appear twice in one combo")
        self.assertEqual(set(ids), {"event_e", "event_f"})


class TestNoSolution(unittest.TestCase):
    def test_returns_none_when_even_all_changes_are_not_enough(self):
        profile = _profile(willing_to_stop=("cat_a",))
        series = [_single_occurrence_series("cat_a", 50.0, "event_a")]
        combo = find_best_combo(profile, [], series, REQUEST_DATE, requested_amount=10000.0)
        self.assertIsNone(combo)


if __name__ == "__main__":
    unittest.main()
