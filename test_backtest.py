import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import (
    _prize_level_dlt,
    _random_baseline_hits,
    _validate_backtest_inputs,
    backtest_digit,
    format_backtest_summary,
)


class BacktestInvariantTest(unittest.TestCase):
    def test_combined_lottery_summary_counts_winning_periods_not_tickets(self):
        summary = {
            "game": "ssq", "n_test": 5, "n_groups": 4, "holdout_size": 5,
            "sampling_mode": "merged", "n_ensemble": 1,
            "baseline": _random_baseline_hits("ssq"),
            "unbiased_mean_red_hits": 0.0, "unbiased_blue_hit_rate": 0.0,
            "red_ci95": (0.0, 0.0), "blue_ci95": (0.0, 0.0),
            "unbiased_red_distribution": {}, "mean_best_red_hits": 0.0,
            "winning_period_count": 2,
        }
        text = format_backtest_summary(summary)
        self.assertIn("中奖期数（至少一注中奖）: 2/5", text)
        self.assertNotIn("中奖注数（任意等级）", text)

    def test_pl3_group_rates_and_current_dlt_levels_are_explicit(self):
        self.assertEqual(_random_baseline_hits("pls", "组选3")["exact_hit_rate"], .003)
        self.assertEqual(_random_baseline_hits("pls", "组选6")["exact_hit_rate"], .006)
        self.assertEqual(_prize_level_dlt(4, 0), 5)
        self.assertEqual(_prize_level_dlt(3, 1), 6)
        self.assertEqual(_prize_level_dlt(0, 2), 7)

    def test_backtest_inputs_require_a_real_frozen_window(self):
        df = pd.DataFrame({"issue": range(10)})
        info = {"seq_len": 3, "holdout_size": 4}
        for n_test, n_groups in ((0, 1), (1, 0), (5, 1), (2, True), (1.5, 1)):
            with self.assertRaises(ValueError):
                _validate_backtest_inputs(df, info, n_test, n_groups)
        with self.assertRaises(ValueError):
            _validate_backtest_inputs(df, {"seq_len": 3, "holdout_size": 8}, 1, 1)
        with self.assertRaises(ValueError):
            _validate_backtest_inputs(df, {"seq_len": 9, "holdout_size": 1}, 2, 1)

    def test_digit_backtest_window_stops_before_target_issue(self):
        df = pd.DataFrame({
            "issue": [f"i{i}" for i in range(10)],
            "digit1": [i % 10 for i in range(10)],
            "digit2": [(i + 1) % 10 for i in range(10)],
            "digit3": [(i + 2) % 10 for i in range(10)],
        })
        info = {
            "seq_len": 3, "input_size": 2, "hidden_size": 1, "num_layers": 1,
            "dropout": 0.0, "holdout_size": 4,
        }
        windows = []

        def fake_probabilities(_model, window, _seq_len):
            windows.append(window.copy())
            return {f"digit{i}": np.full(10, .1) for i in range(1, 4)}

        with patch("backtest.build_features_digit", return_value=(np.arange(20).reshape(10, 2), None, None)), \
             patch("backtest._load_model", return_value=object()), \
             patch("backtest._get_probabilities", side_effect=fake_probabilities), \
             patch("backtest._weighted_sample_digits", return_value=[{"digits": [0, 1, 2]}]):
            result = backtest_digit(df, info, "pls", n_test=2, n_groups=1)

        self.assertEqual(result["n_test"], 2)
        self.assertEqual([len(window) for window in windows], [8, 9])
        self.assertEqual([int(window[-1, 0]) for window in windows], [14, 16])


if __name__ == "__main__":
    unittest.main()
