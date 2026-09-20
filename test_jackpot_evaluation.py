import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import jackpot_evaluation as evaluation


def _df(n=8):
    return pd.DataFrame({"issue": [str(i) for i in range(n)],
                         **{f"digit{i}": [(r + i) % 10 for r in range(n)] for i in range(1, 4)}})


def _info():
    return {"holdout_size": 3, "seq_len": 2, "input_size": 2, "hidden_size": 1,
            "num_layers": 1, "dropout": 0.0, "n_ensemble": 1}


def _probs():
    return {f"digit{i}": np.full(10, .1) for i in range(1, 4)}


class TopFiveEvaluationTests(unittest.TestCase):
    def _run(self, play="直选"):
        seen = []
        def get_probs(_model, features, seq_len):
            seen.append(len(features)); self.assertEqual(seq_len, 2); return _probs()
        selector = lambda *_args: [{"digits": [i, i, i]} for i in range(5)] if play == "直选" else [
            {"digits": [0, 0, i]} for i in range(1, 6)]
        with patch.object(evaluation, "build_features_digit", return_value=(np.arange(16).reshape(8, 2), {}, None)), \
             patch.object(evaluation, "_load_model", return_value=object()), \
             patch.object(evaluation, "_get_probabilities", side_effect=get_probs), \
             patch.object(evaluation, "_weighted_sample_digits", return_value=selector()), \
             patch.object(evaluation, "_sample_pl3_play", return_value=selector()), \
             patch("jackpot_selection.select_top_five", side_effect=selector):
            result = evaluation.backtest_top_five(_df(), _info(), "pls", play, n_test=3)
        self.assertEqual(seen, [5, 6, 7]) # target indexes; no target feature enters a window
        for strategy in result["strategies"].values():
            self.assertTrue(all(row["每期注数"] == 5 for row in strategy["rows"]))
        return result

    def test_windows_five_tickets_and_deterministic_uniform_baseline(self):
        one = self._run()
        two = self._run()
        self.assertEqual(one["uniform_random"]["rows"], two["uniform_random"]["rows"])
        self.assertEqual(one["evaluation_scope"], "exploratory_existing_holdout")
        self.assertIn("未来新开奖", evaluation.format_top_five_backtest(one))

    def test_pls_grouped_matches_multiset_not_position(self):
        self.assertTrue(evaluation._is_jackpot({"digits": [1, 1, 2]}, np.array([2, 1, 1]), "pls", "组选3"))
        self.assertFalse(evaluation._is_jackpot({"digits": [1, 1, 2]}, np.array([2, 1, 1]), "pls", "直选"))
        self.assertEqual(evaluation._random_period_probability("pls", "组选3"), .015)
        self.assertEqual(evaluation._random_period_probability("pls", "组选6"), .03)

    def test_wilson_and_unique_guard(self):
        low, high = evaluation._wilson_interval(0, 50)
        self.assertGreater(low, -1e-12); self.assertLess(high, .1)
        with self.assertRaisesRegex(RuntimeError, "重复整票"):
            evaluation._require_five_unique([{"digits": [1, 1, 2]}] * 5, "pls", "组选3", "x")
        self.assertAlmostEqual(evaluation._random_period_probability("ssq", ""), 5 / (1_107_568 * 16))


if __name__ == "__main__":
    unittest.main()
