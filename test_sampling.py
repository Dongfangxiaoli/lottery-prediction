"""采样边界与记录保存回归测试。"""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import predictor
from predictor import (
    _sample_pl3_play,
    _weighted_sample_digits,
    format_numbers_copy,
    format_numbers_table,
)


def _uniform_pl3_probs():
    return {f"digit{i}": np.full(10, 0.1) for i in range(1, 4)}


class SamplingValidationTests(unittest.TestCase):
    def test_pl3_grouped_top_p_shortage_is_clear_error(self):
        with self.assertRaisesRegex(ValueError, r"有效唯一候选.*不足请求"):
            _sample_pl3_play(
                _uniform_pl3_probs(), "组选6", n_groups=2, temperature=1.0,
                top_p=0.1, rng=np.random.default_rng(1))

    def test_pl3_grouped_sampling_returns_requested_unique_groups(self):
        numbers = _sample_pl3_play(
            _uniform_pl3_probs(), "组选6", n_groups=20, temperature=1.0,
            top_p=1.0, rng=np.random.default_rng(2))
        self.assertEqual(len(numbers), 20)
        self.assertEqual(len({tuple(item["digits"]) for item in numbers}), 20)

    def test_pl3_direct_shortage_is_clear_error(self):
        concentrated = {f"digit{i}": np.eye(10)[0] for i in range(1, 4)}
        with self.assertRaisesRegex(RuntimeError, r"仅生成 1/2 注"):
            _weighted_sample_digits(
                concentrated, "pls", n_groups=2, temperature=1.0,
                top_p=0.5, rng=np.random.default_rng(3))

    def test_sampling_options_and_probability_values_are_validated(self):
        probs = _uniform_pl3_probs()
        for kwargs in (
            {"temperature": 0},
            {"top_p": 0},
            {"top_p": float("nan")},
            {"freq_alpha": 1.1},
            {"n_groups": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                _weighted_sample_digits(probs, "pls", **kwargs)

        bad = _uniform_pl3_probs()
        bad["digit1"][0] = -0.1
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            _weighted_sample_digits(bad, "pls", n_groups=1)

        bad = _uniform_pl3_probs()
        bad["digit1"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "有限数值"):
            _weighted_sample_digits(bad, "pls", n_groups=1)

    def test_pl3_coverage_counts_unique_effective_tickets(self):
        numbers = [
            {"digits": [1, 2, 3], "prob": 0.1},
            {"digits": [3, 2, 1], "prob": 0.1},
            {"digits": [1, 1, 2], "prob": 0.1},
        ]
        text = format_numbers_table("pls", numbers, "组选6")
        self.assertIn("1 注有效唯一号码理论覆盖率: 0.60%", text)

    def test_copy_can_include_game_and_play_without_changing_default(self):
        numbers = [{"digits": [0, 1, 2]}]
        self.assertEqual(format_numbers_copy("pls", numbers, "组选6"), "012")
        self.assertEqual(
            format_numbers_copy("pls", numbers, "组选6", include_play=True),
            "排列3（组选6）\n012")


class PredictionSaveTests(unittest.TestCase):
    def test_save_uses_collision_suffix_without_overwriting(self):
        original_dir = predictor.RESULTS_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor.RESULTS_DIR = temp_dir
            try:
                with patch.object(predictor, "datetime") as mocked_datetime:
                    mocked_datetime.now.return_value.strftime.return_value = "20260906_120000_123456"
                    first = predictor._save_prediction("pls", [{"digits": [0, 1, 2]}])
                    second = predictor._save_prediction("pls", [{"digits": [3, 4, 5]}])
            finally:
                predictor.RESULTS_DIR = original_dir
            self.assertNotEqual(first, second)
            self.assertTrue(os.path.exists(first))
            self.assertTrue(os.path.exists(second))
            self.assertRegex(os.path.basename(first), r"^pls_\d{8}_\d{6}_\d{6}(?:_\d{3})?\.json$")
            self.assertTrue(second.endswith("pls_20260906_120000_123456_001.json"))


if __name__ == "__main__":
    unittest.main()
