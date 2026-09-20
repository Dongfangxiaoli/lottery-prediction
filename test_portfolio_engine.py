"""Independent contract tests for the pure variable-count portfolio engine."""
import unittest
from itertools import product

import numpy as np

from portfolio_engine import _dlt_prize, _qxc_prize, generate_portfolio


def _probs(game):
    if game == "ssq":
        return {**{f"red{i}": np.ones(33) for i in range(1, 7)}, "blue": np.ones(16)}
    if game == "dlt":
        return {**{f"front{i}": np.ones(35) for i in range(1, 6)},
                **{f"back{i}": np.ones(12) for i in range(1, 3)}}
    classes = {"pls": (10, 10, 10), "plw": (10,) * 5, "qxc": (10,) * 6 + (15,)}[game]
    return {f"digit{i + 1}": np.ones(size) for i, size in enumerate(classes)}


class PortfolioEngineTests(unittest.TestCase):
    def test_uniform_legality_and_uniqueness_all_seven_modes(self):
        modes = [("ssq", None), ("dlt", None), ("pls", "直选"), ("pls", "组选3"),
                 ("pls", "组选6"), ("plw", None), ("qxc", None)]
        for game, play in modes:
            with self.subTest(game=game, play=play):
                result = generate_portfolio(game, play, 7, "uniform", seed=23, n_eval=31)
                self.assertEqual(len(result["tickets"]), 7)
                self.assertEqual(len({str(item) for item in result["tickets"]}), 7)
                self.assertAlmostEqual(result["comparison_rows"][0]["paired_delta_vs_uniform"], 0.0)

    def test_exact_jackpot_probability_and_group_ticket_ceilings(self):
        direct = generate_portfolio("pls", "直选", 10, n_eval=3)
        group3 = generate_portfolio("pls", "组选3", 10, n_eval=3)
        group6 = generate_portfolio("pls", "组选6", 10, n_eval=3)
        self.assertAlmostEqual(direct["exact_jackpot_probability"], .01)
        self.assertAlmostEqual(group3["exact_jackpot_probability"], .03)
        self.assertAlmostEqual(group6["exact_jackpot_probability"], .06)
        with self.assertRaisesRegex(ValueError, "90"):
            generate_portfolio("pls", "组选3", 91)
        with self.assertRaisesRegex(ValueError, "120"):
            generate_portfolio("pls", "组选6", 121)
        with self.assertRaisesRegex(ValueError, "500"):
            generate_portfolio("ssq", n_tickets=501)

    def test_entire_group_space_does_not_cover_all_draw_shapes(self):
        group3 = generate_portfolio("pls", "组选3", 90)
        group6 = generate_portfolio("pls", "组选6", 120)
        self.assertEqual(group3["exact_jackpot_probability"], .27)
        self.assertEqual(group6["exact_jackpot_probability"], .72)
        self.assertEqual(len(group3["tickets"]), 90)
        self.assertEqual(len(group6["tickets"]), 120)

    def test_rules_at_prize_boundaries_and_invalid_targets(self):
        ssq = generate_portfolio("ssq", n_tickets=1, target_prize=6, n_eval=2)
        dlt = generate_portfolio("dlt", n_tickets=1, target_prize=7, n_eval=2)
        qxc = generate_portfolio("qxc", n_tickets=1, target_prize=6, n_eval=2)
        self.assertGreater(ssq["exact_single_ticket_target_probability"], 0)
        self.assertGreater(dlt["exact_single_ticket_target_probability"], 0)
        self.assertGreater(qxc["exact_single_ticket_target_probability"], 0)
        for game, play in (("pls", "直选"), ("pls", "组选3"), ("plw", None)):
            with self.subTest(game=game):
                with self.assertRaises(ValueError):
                    generate_portfolio(game, play, target_prize=2)
        self.assertEqual(_dlt_prize(3, 1), 6)
        self.assertEqual(_dlt_prize(0, 2), 7)
        self.assertEqual(_qxc_prize((True, True, False, False, False, False, True)), 6)  # 2+1
        self.assertEqual(_qxc_prize((True, True, True, False, False, False, True)), 5)  # 3+1
        self.assertEqual(_qxc_prize((True, True, True, True, False, False, False)), 5)  # 4+0
        from backtest import _prize_level_digit
        for bits in product((False, True), repeat=7):
            self.assertEqual(_qxc_prize(bits), _prize_level_digit(list(bits), "qxc"))

    def test_coverage_is_deterministic_and_has_no_guarantee_language(self):
        first = generate_portfolio("qxc", n_tickets=8, strategy="coverage", target_prize=6, seed=99, n_eval=61)
        second = generate_portfolio("qxc", n_tickets=8, strategy="coverage", target_prize=6, seed=99, n_eval=61)
        self.assertEqual(first["tickets"], second["tickets"])
        self.assertEqual(first["comparison_rows"], second["comparison_rows"])
        self.assertIn("不构成保证", first["report"])
        head = generate_portfolio("dlt", n_tickets=4, strategy="coverage", target_prize=1, seed=7, n_eval=3)
        self.assertIn("不进行无意义的贪心", head["provenance"])

    def test_invalid_inputs_and_model_requires_selector_or_probs(self):
        with self.assertRaises(ValueError): generate_portfolio("ssq", n_tickets=True)
        with self.assertRaises(ValueError): generate_portfolio("ssq", strategy="other")
        with self.assertRaises(ValueError): generate_portfolio("ssq", n_eval=0)
        with self.assertRaises(ValueError): generate_portfolio("ssq", strategy="model")

    def test_model_mode_is_deterministic_when_given_uncalibrated_scores(self):
        first = generate_portfolio("pls", "直选", 8, "model", probs=_probs("pls"), seed=4, n_eval=13)
        second = generate_portfolio("pls", "直选", 8, "model", probs=_probs("pls"), seed=4, n_eval=13)
        self.assertEqual(first["tickets"], second["tickets"])
        self.assertIn("未校准", first["report"])

    def test_seed_streams_keep_baseline_fixed_when_eval_size_changes(self):
        small = generate_portfolio("ssq", n_tickets=4, strategy="coverage", target_prize=6, seed=81, n_eval=11)
        large = generate_portfolio("ssq", n_tickets=4, strategy="coverage", target_prize=6, seed=81, n_eval=29)
        self.assertEqual(small["tickets"], large["tickets"])
        self.assertEqual(small["baseline_tickets"], large["baseline_tickets"])

    def test_zero_sample_difference_has_valid_nonzero_uncertainty(self):
        from portfolio_engine import _paired_ci
        delta, interval = _paired_ci(np.zeros(100), np.zeros(100))
        self.assertEqual(delta, 0)
        self.assertLess(interval[0], -0.01)
        self.assertGreater(interval[1], 0.01)
        exact = generate_portfolio("dlt", n_tickets=10, n_eval=100)
        self.assertEqual(exact["evaluation_scope"], "exact_uniform_jackpot")
        self.assertEqual(exact["actual_eval_draw_count"], 0)
        self.assertIsNone(exact["comparison"][0]["best_prize_counts"])


if __name__ == "__main__":
    unittest.main()
