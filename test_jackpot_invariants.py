"""双色球/大乐透整票采样数量与唯一性不变量测试。"""
import unittest
from unittest.mock import patch

import numpy as np

import predictor


def _one_ticket_probs(game):
    if game == "ssq":
        probs = {f"red{i}": np.eye(33)[i - 1] for i in range(1, 7)}
        probs["blue"] = np.eye(16)[0]
        return probs
    probs = {f"front{i}": np.eye(35)[i - 1] for i in range(1, 6)}
    probs.update({f"back{i}": np.eye(12)[i - 1] for i in range(1, 3)})
    return probs


def _uniform_probs(game):
    if game == "ssq":
        return {**{f"red{i}": np.full(33, 1 / 33) for i in range(1, 7)},
                "blue": np.full(16, 1 / 16)}
    return {**{f"front{i}": np.full(35, 1 / 35) for i in range(1, 6)},
            **{f"back{i}": np.full(12, 1 / 12) for i in range(1, 3)}}


class JackpotSamplingInvariantTests(unittest.TestCase):
    def test_probability_mass_supporting_one_ticket_reports_shortage(self):
        for game, ordinary, merged in (
            ("ssq", predictor._weighted_sample_ssq, predictor._weighted_sample_ssq_merged),
            ("dlt", predictor._weighted_sample_dlt, predictor._weighted_sample_dlt_merged),
        ):
            with self.subTest(game=game):
                probs = _one_ticket_probs(game)
                for sampler in (ordinary, merged):
                    with self.assertRaisesRegex(RuntimeError, r"仅生成 .*3 注"):
                        sampler(probs, n_groups=3, rng=np.random.default_rng(42))

    def test_uniform_sampling_returns_requested_unique_legal_tickets(self):
        for game, ordinary, merged in (
            ("ssq", predictor._weighted_sample_ssq, predictor._weighted_sample_ssq_merged),
            ("dlt", predictor._weighted_sample_dlt, predictor._weighted_sample_dlt_merged),
        ):
            for sampler in (ordinary, merged):
                for requested in (5, 20):
                    with self.subTest(game=game, sampler=sampler.__name__, requested=requested):
                        numbers = sampler(_uniform_probs(game), requested,
                                          rng=np.random.default_rng(42))
                        self.assertEqual(len(numbers), requested)
                        self.assertEqual(
                            len({_key(item, game) for item in numbers}), requested)
                        for item in numbers:
                            if game == "ssq":
                                self.assertEqual(len(item["red"]), 6)
                                self.assertEqual(len(set(item["red"])), 6)
                                self.assertTrue(all(1 <= n <= 33 for n in item["red"]))
                                self.assertTrue(1 <= item["blue"] <= 16)
                            else:
                                self.assertEqual(len(item["front"]), 5)
                                self.assertEqual(len(item["front"]), len(set(item["front"])))
                                self.assertEqual(len(item["back"]), 2)
                                self.assertEqual(len(item["back"]), len(set(item["back"])))
                                self.assertTrue(all(1 <= n <= 35 for n in item["front"]))
                                self.assertTrue(all(1 <= n <= 12 for n in item["back"]))

    def test_annealing_deduplicates_across_temperatures(self):
        a = {"red": [1, 2, 3, 4, 5, 6], "blue": 1}
        b = {"red": [1, 2, 3, 4, 5, 7], "blue": 1}
        c = {"red": [1, 2, 3, 4, 5, 8], "blue": 1}

        def fake_sampler(_probs, n_groups, temperature):
            return [a, b] if temperature == 1 else [b, c]

        with patch.object(predictor, "_weighted_sample_ssq", side_effect=fake_sampler):
            result = predictor._sample_with_annealing({}, 3, [1, 2], "ssq")
        self.assertEqual([predictor._make_key(item, "ssq") for item in result],
                         [predictor._make_key(a, "ssq"), predictor._make_key(b, "ssq"),
                          predictor._make_key(c, "ssq")])

    def test_annealing_does_not_require_twice_the_final_ticket_count(self):
        samples = [{"red": [1, 2, 3, 4, 5, 6], "blue": blue} for blue in (1, 2)]

        def limited_sampler(_probs, n_groups, temperature):
            self.assertEqual(n_groups, 2)
            return predictor._require_unique_samples(samples, n_groups, "ssq", "测试采样")

        with patch.object(predictor, "_weighted_sample_ssq", side_effect=limited_sampler):
            result = predictor._sample_with_annealing({}, 2, [1, 2], "ssq")
        self.assertEqual(len(result), 2)

    def test_guard_rejects_duplicate_complete_tickets(self):
        sample = {"front": [1, 2, 3, 4, 5], "back": [1, 2]}
        with self.assertRaisesRegex(RuntimeError, "重复整票"):
            predictor._require_unique_samples([sample, sample], 2, "dlt", "测试采样")

    def test_annealing_shortage_is_error(self):
        sample = {"red": [1, 2, 3, 4, 5, 6], "blue": 1}
        with patch.object(predictor, "_weighted_sample_ssq", return_value=[sample]):
            with self.assertRaisesRegex(RuntimeError, r"仅生成 .*3 注"):
                predictor._sample_with_annealing({}, 3, [1, 2], "ssq")


def _key(item, game):
    return predictor._make_key(item, game)


if __name__ == "__main__":
    unittest.main()
