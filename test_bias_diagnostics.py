import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bias_diagnostics import analyze_game_bias


def ssq_frame(n, seed=8):
    rng = np.random.default_rng(seed)
    reds = np.array([rng.choice(np.arange(1, 34), size=6, replace=False) for _ in range(n)])
    frame = pd.DataFrame(reds, columns=[f"red{i}" for i in range(1, 7)])
    frame["blue"] = rng.integers(1, 17, size=n)
    return frame


class BiasDiagnosticsTests(unittest.TestCase):
    def test_joe_corrected_formula_for_set_margin(self):
        frame = ssq_frame(100)
        row = next(x for x in analyze_game_bias(frame, "ssq")
                   if x["component"] == "red" and "边际" in x["label"])
        counts = np.bincount(frame[[f"red{i}" for i in range(1, 7)]].to_numpy().ravel(), minlength=34)[1:]
        expected = 100 * 6 / 33
        manual = (33 - 1) / (33 - 6) * np.sum((counts - expected) ** 2 / expected)
        self.assertAlmostEqual(row["statistic"], manual)
        self.assertEqual(row["scope"], "exploratory_diagnostic_not_prediction")

    def test_uniform_and_obvious_marginal_bias(self):
        uniform = next(x for x in analyze_game_bias(ssq_frame(500), "ssq")
                       if x["component"] == "red" and "边际" in x["label"])
        self.assertGreater(uniform["p_value"], 0.001)
        biased = pd.DataFrame({f"red{i}": [i] * 200 for i in range(1, 7)})
        biased["blue"] = 1
        result = next(x for x in analyze_game_bias(biased, "ssq")
                      if x["component"] == "red" and "边际" in x["label"])
        self.assertLess(result["p_value"], 1e-20)

    def test_lag_permutation_is_deterministic_and_input_is_unchanged(self):
        frame = ssq_frame(30)
        original = frame.copy(deep=True)
        legacy_state = np.random.get_state()
        first = analyze_game_bias(frame, "ssq")
        second = analyze_game_bias(frame, "ssq")
        assert_frame_equal(frame, original)
        self.assertEqual(first, second)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(legacy_state, np.random.get_state())))
        unsorted = frame.copy(deep=True)
        unsorted[["red1", "red6"]] = unsorted[["red6", "red1"]].to_numpy()
        self.assertEqual(first, analyze_game_bias(unsorted, "ssq"))
        lag = next(x for x in first if x["component"] == "red" and "跨期" in x["label"])
        self.assertGreaterEqual(lag["p_value"], 0.001)
        self.assertLessEqual(lag["p_value"], 1.0)

    def test_short_invalid_and_position_specific_inputs(self):
        short = analyze_game_bias(ssq_frame(4), "ssq")
        self.assertTrue(all(x["p_value"] is None for x in short if "跨期" in x["label"]))
        malformed = ssq_frame(20)
        malformed.loc[0, "red2"] = malformed.loc[0, "red1"]
        with self.assertRaises(ValueError):
            analyze_game_bias(malformed, "ssq")
        qxc = pd.DataFrame({f"digit{i}": np.arange(25) % (15 if i == 7 else 10) for i in range(1, 8)})
        result = analyze_game_bias(qxc, "qxc")
        seventh = next(x for x in result if x["component"] == "digit7" and "边际" in x["label"])
        self.assertIsNone(seventh["p_value"])  # 25 / 15 < 5: must not use an asymptotic p-value.


if __name__ == "__main__":
    unittest.main()
