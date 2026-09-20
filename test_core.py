import json
import unittest

import numpy as np
import pandas as pd

from backtest import (
    format_backtest_summary,
    _prize_level_dlt,
    _prize_level_ssq,
    _random_baseline_hits,
    _ssq_special_prize_flags,
)
from data_fetcher_digit import parse as parse_digit
from data_fetcher_dlt import _normalize_history as normalize_dlt
from data_fetcher_ssq import _normalize_history as normalize_ssq
from feature_engineering import _scale_features, build_features_digit
from game_config import DIGIT_GAME_CONFIGS, GAME_PLAY_DEFAULTS, digit_columns, normalize_play
from lstm_model import build_digit_model
from predictor import (
    _weighted_sample_dlt_merged,
    _weighted_sample_ssq,
    _weighted_sample_ssq_merged,
    _weighted_sample_digits,
    _sample_pl3_play,
)


class CoreInvariantTest(unittest.TestCase):
    def test_core_invariants(self):
        duplicated = pd.DataFrame({
            "issue": ["26002", "2026001", "26001"],
            "date": ["2026-01-04", "2026-01-02", "2026-01-02"],
            "value": [2, 1, 99],
        })
        for normalize in (normalize_ssq, normalize_dlt):
            clean = normalize(duplicated)
            self.assertEqual(clean["issue"].tolist(), ["2026001", "26002"])
            self.assertTrue(clean["date"].is_monotonic_increasing)

        scaled, scaler = _scale_features(
            np.array([[0.0], [1.0], [100.0]], dtype=np.float32), fit_end=2)
        self.assertEqual(scaler.n_samples_seen_, 2)
        self.assertGreater(float(scaled[-1, 0]), 1.0)

        ssq_probs = {f"red{i}": np.full(33, 1 / 33) for i in range(1, 7)}
        ssq_probs["blue"] = np.full(16, 1 / 16)
        ssq = _weighted_sample_ssq_merged(
            ssq_probs, n_groups=20, freq_alpha=0, top_p=1,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(len(ssq), 20)
        self.assertTrue(all(len(set(row["red"])) == 6 for row in ssq))
        per_position = _weighted_sample_ssq(
            ssq_probs, 3, rng=np.random.default_rng(7))
        self.assertEqual(
            per_position,
            _weighted_sample_ssq(ssq_probs, 3, rng=np.random.default_rng(7)),
        )
        json.dumps(per_position)

        dlt_probs = {f"front{i}": np.full(35, 1 / 35) for i in range(1, 6)}
        dlt_probs.update({f"back{i}": np.full(12, 1 / 12) for i in range(1, 3)})
        dlt = _weighted_sample_dlt_merged(
            dlt_probs, n_groups=20, freq_alpha=0, top_p=1,
            rng=np.random.default_rng(42),
        )
        self.assertEqual(len(dlt), 20)
        self.assertTrue(all(len(set(row["front"])) == 5 and len(set(row["back"])) == 2 for row in dlt))

        self.assertAlmostEqual(_random_baseline_hits("ssq")["mean_red_hits"], 6 * 6 / 33)
        self.assertAlmostEqual(_random_baseline_hits("dlt")["mean_back_hits"], 2 * 2 / 12)
        self.assertEqual(_prize_level_dlt(3, 2), 5)
        self.assertEqual(_prize_level_dlt(4, 0), 5)
        self.assertEqual(_prize_level_dlt(3, 1), 6)
        self.assertEqual(_prize_level_dlt(0, 2), 7)
        self.assertEqual(_prize_level_ssq(3, False, special_prize=True), 7)
        self.assertEqual(_prize_level_ssq(3, True, special_prize=True), 5)
        self.assertEqual(
            _ssq_special_prize_flags(pd.DataFrame({"date": ["2026-02-01"]})),
            [False],
        )

    def test_digit_game_invariants(self):
        html = """
        <table id="tablelist">
          <tr class="t_tr1"><td>26002</td><td>0 1 2</td><td>3</td><td>50000</td><td>2026-01-01</td></tr>
          <tr class="t_tr1"><td>26001</td><td>3 0 4</td><td>7</td><td>50001</td><td>2026-01-01</td></tr>
        </table>
        """
        parsed = parse_digit(html, "pls")
        self.assertEqual(parsed["issue"].tolist(), ["26001", "26002"])
        self.assertEqual(parsed["digit2"].tolist(), [0, 1])  # 0 和同日不同期均须保留。
        from prediction_review import _find_draw_after
        next_draw = _find_draw_after(
            "pls", {"source_date": "2026-01-01", "source_issue": "26001"}, parsed)
        self.assertEqual(next_draw["issue"], "26002")

        for game in ("pls", "plw", "qxc"):
            cfg = DIGIT_GAME_CONFIGS[game]
            columns = list(digit_columns(game))
            values = [[0] * len(columns), [limit - 1 for limit in cfg["classes"]], [1] * len(columns)]
            frame = pd.DataFrame({"issue": ["26001", "26002", "26003"], "date": ["2026-01-01"] * 3})
            for index, column in enumerate(columns):
                frame[column] = [row[index] for row in values]
            features, labels, _ = build_features_digit(frame, game)
            self.assertEqual(features.shape[0], len(frame))
            self.assertEqual(labels["digits"].shape, (len(frame), len(columns)))
            self.assertEqual(int(labels["digits"][0, 0]), 0)

            model = build_digit_model(game, features.shape[1], hidden_size=8, num_layers=1,
                                      dropout=0, use_attention=False)
            output = model(__import__("torch").zeros((1, 2, features.shape[1])))
            self.assertEqual([output[f"digit{i + 1}"].shape[-1] for i in range(len(columns))], list(cfg["classes"]))

            probabilities = {f"digit{i + 1}": np.eye(size)[0] for i, size in enumerate(cfg["classes"])}
            sampled = _weighted_sample_digits(probabilities, game, n_groups=1,
                                              rng=np.random.default_rng(0))
            self.assertEqual(sampled[0]["digits"], [0] * len(columns))
            self.assertTrue(all(0 <= value < limit for value, limit in
                                zip(sampled[0]["digits"], cfg["classes"])))

        # 由数字彩回测模块提供；覆盖 7星彩全部奖级及“不兼中”的最高奖级映射。
        from backtest import _prize_level_digit
        self.assertEqual(_prize_level_digit([True] * 7, "qxc"), 1)
        self.assertEqual(_prize_level_digit([True] * 6 + [False], "qxc"), 2)
        self.assertEqual(_prize_level_digit([True] * 5 + [False, True], "qxc"), 3)
        self.assertEqual(_prize_level_digit([True] * 5 + [False, False], "qxc"), 4)
        self.assertEqual(_prize_level_digit([True] * 4 + [False] * 3, "qxc"), 5)
        self.assertEqual(_prize_level_digit([True] * 3 + [False] * 4, "qxc"), 6)
        self.assertEqual(_prize_level_digit([False] * 6 + [True], "qxc"), 6)
        self.assertEqual(_prize_level_digit([True, False, False, False, False, False, True], "qxc"), 6)
        self.assertEqual(_prize_level_digit([True, True, False, False, False, False, False], "qxc"), 0)

        pls_probs = {f"digit{i}": np.full(10, 0.1) for i in range(1, 4)}
        for play, distinct in (("组选3", 2), ("组选6", 3)):
            samples = _sample_pl3_play(
                pls_probs, play, 20, 1.0, rng=np.random.default_rng(42))
            groups = [tuple(row["digits"]) for row in samples]
            self.assertEqual(len(groups), 20)
            self.assertEqual(len(set(groups)), 20)
            self.assertTrue(all(tuple(sorted(group)) == group for group in groups))
            self.assertTrue(all(len(set(group)) == distinct for group in groups))

        self.assertEqual(GAME_PLAY_DEFAULTS["pls"], "组选6")
        self.assertEqual(normalize_play("pls"), "直选")  # 旧记录兼容
        self.assertAlmostEqual(_random_baseline_hits("pls", "直选")["exact_hit_rate"], .001)
        self.assertAlmostEqual(_random_baseline_hits("pls", "组选3")["exact_hit_rate"], .003)
        self.assertAlmostEqual(_random_baseline_hits("pls", "组选6")["exact_hit_rate"], .006)
        grouped_summary = {
            "game": "pls", "play": "组选6", "n_test": 2, "n_groups": 5,
            "holdout_size": 50, "sampling_mode": "per_position", "n_ensemble": 1,
            "baseline": _random_baseline_hits("pls", "组选6"),
            "exact_hit_rate": 0.0, "prize_count": 0,
        }
        self.assertIn("中奖注数（任意等级）: 0/10", format_backtest_summary(grouped_summary))


if __name__ == "__main__":
    unittest.main()
