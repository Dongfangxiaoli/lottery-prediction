"""Contract tests for future-only jackpot evidence statistics."""
import unittest

from scipy.stats import binomtest

from evidence_statistics import (
    BONFERRONI_ALPHA,
    exact_jackpot_probability,
    estimate_game_horizon,
    estimate_required_horizon,
    evaluate_preregistered_horizon,
    power_for_horizon,
)


class EvidenceStatisticsTests(unittest.TestCase):
    def test_exact_probability_reuses_all_seven_legal_game_modes(self):
        cases = [
            ("ssq", None, 1), ("dlt", None, 1), ("pls", "直选", 1),
            ("pls", "组选3", 3), ("pls", "组选6", 6), ("plw", None, 1), ("qxc", None, 1),
        ]
        for game, play, numerator in cases:
            with self.subTest(game=game, play=play):
                result = exact_jackpot_probability(game, 1, play)
                self.assertEqual(result["favorable_outcomes_per_ticket"], numerator)
                self.assertAlmostEqual(result["p0"], numerator / result["outcome_count"])
                five_tickets = exact_jackpot_probability(game, 5, play)
                self.assertAlmostEqual(five_tickets["p0"], 5 * numerator / result["outcome_count"])
        self.assertAlmostEqual(exact_jackpot_probability("pls", 10, "组选6")["p0"], 0.06)

    def test_reports_scipy_one_sided_pvalue_and_two_sided_exact_ci(self):
        result = evaluate_preregistered_horizon("pls", 1, successes=2, trials=10, planned_horizon=10,
                                                play="直选", evidence_provenance="independent_archive")
        expected_pvalue = binomtest(2, n=10, p=.001, alternative="greater").pvalue
        expected_ci = binomtest(2, n=10, alternative="two-sided").proportion_ci(method="exact")
        self.assertAlmostEqual(result["pvalue_greater"], expected_pvalue)
        self.assertAlmostEqual(result["ci95_clopper_pearson"][0], expected_ci.low)
        self.assertAlmostEqual(result["ci95_clopper_pearson"][1], expected_ci.high)
        self.assertEqual(result["bonferroni_alpha"], BONFERRONI_ALPHA)

    def test_early_high_hit_result_is_descriptive_only(self):
        result = evaluate_preregistered_horizon("pls", 1, successes=1, trials=1, planned_horizon=2,
                                                play="直选", evidence_provenance="independent_archive")
        self.assertLess(result["pvalue_greater"], BONFERRONI_ALPHA)
        self.assertFalse(result["statistical_signal"])
        self.assertFalse(result["verified_advantage"])
        self.assertEqual(result["status"], "DESCRIPTIVE_PENDING_FROZEN_HORIZON")

    def test_historical_exploration_cannot_be_verified(self):
        result = evaluate_preregistered_horizon("pls", 1, successes=10, trials=10, planned_horizon=10,
                                                play="直选", historical_exploration=True,
                                                evidence_provenance="independent_archive")
        self.assertTrue(result["statistical_signal"])
        self.assertFalse(result["verified_advantage"])
        self.assertEqual(result["status"], "EXPLORATORY_OR_UNREGISTERED_NOT_VERIFIED")

    def test_local_only_record_remains_conservative(self):
        result = evaluate_preregistered_horizon("pls", 1, successes=10, trials=10, planned_horizon=10,
                                                play="直选")
        self.assertTrue(result["statistical_signal"])
        self.assertFalse(result["verified_advantage"])
        self.assertEqual(result["status"], "STATISTICAL_SIGNAL_LOCAL_ONLY_NOT_INDEPENDENTLY_VERIFIED")

    def test_self_reported_archive_is_only_eligible_for_independent_verification(self):
        result = evaluate_preregistered_horizon("pls", 1, successes=10, trials=10, planned_horizon=10,
                                                play="直选", evidence_provenance="independent_archive")
        self.assertTrue(result["statistical_signal"])
        self.assertTrue(result["registered_statistical_signal"])
        self.assertTrue(result["eligible_for_independent_verification"])
        self.assertFalse(result["verified_advantage"])
        self.assertEqual(result["status"], "REGISTERED_STATISTICAL_SIGNAL_INDEPENDENT_VERIFICATION_PENDING")

    def test_invalid_counts_and_too_many_tickets_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_preregistered_horizon("pls", 1, successes=2, trials=1, planned_horizon=1, play="直选")
        with self.assertRaises(ValueError):
            evaluate_preregistered_horizon("pls", 1, successes=0, trials=2, planned_horizon=1, play="直选")
        with self.assertRaises(ValueError):
            exact_jackpot_probability("pls", 121, "组选6")

    def test_exact_integer_power_plan_has_minimal_returned_horizon(self):
        plan = estimate_required_horizon(.01, max_horizon=100_000)
        self.assertGreaterEqual(plan["power"], .8)
        self.assertGreater(plan["trials"], 1)
        prior = power_for_horizon(plan["trials"] - 1, .01)
        self.assertLess(prior["power"], .8)
        self.assertEqual(plan["search_method"], "exact_integer_by_critical_region")
        wrapped = estimate_game_horizon("pls", 10, "直选", max_horizon=100_000)
        self.assertAlmostEqual(wrapped["baseline"]["p0"], .01)

    def test_power_plan_is_bounded_and_rejects_impossible_doubled_probability(self):
        with self.assertRaisesRegex(ValueError, "达不到"):
            estimate_required_horizon(1e-12, max_horizon=1_000)
        with self.assertRaises(ValueError):
            estimate_required_horizon(.6)

    def test_exact_search_does_not_wrongly_reject_after_a_later_power_drop(self):
        # For p0=.03, n=486 has a higher critical count and lower power than
        # n=483.  A one-point upper-bound precheck would miss the valid n=483.
        self.assertFalse(any(power_for_horizon(n, .03)["power"] >= .8 for n in range(1, 483)))
        plan = estimate_required_horizon(.03, max_horizon=486)
        self.assertEqual(plan["trials"], 483)
        self.assertGreaterEqual(plan["power"], .8)
        self.assertLess(power_for_horizon(486, .03)["power"], .8)

    def test_alpha_and_relative_lift_reject_nonfinite_or_boundary_values(self):
        for alpha in (1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    power_for_horizon(10, .01, alpha=alpha)
        for lift in (float("nan"), float("inf")):
            with self.subTest(lift=lift):
                with self.assertRaises(ValueError):
                    estimate_required_horizon(.01, relative_lift=lift)


if __name__ == "__main__":
    unittest.main()
