"""Regression tests for deterministic, model-head-only top-five selection."""
from itertools import combinations, product
import unittest

import numpy as np

from jackpot_selection import _top_k_independent, _top_k_sorted, select_top_five


def _uniform_ball_probs(game):
    if game == "ssq":
        return {**{f"red{i}": np.ones(33) for i in range(1, 7)}, "blue": np.ones(16)}
    return {**{f"front{i}": np.ones(35) for i in range(1, 6)},
            **{f"back{i}": np.ones(12) for i in range(1, 3)}}


def _pls_probs():
    return {f"digit{i}": np.arange(1, 11, dtype=float) for i in range(1, 4)}


class TopKHelperTests(unittest.TestCase):
    def test_sorted_dp_matches_bruteforce_for_random_spaces_larger_than_top_five(self):
        for seed, domain_size, positions in ((17, 6, 3), (29, 8, 4), (43, 8, 4)):
            with self.subTest(seed=seed, domain_size=domain_size, positions=positions):
                rng = np.random.default_rng(seed)
                heads = [rng.random(domain_size) + 0.01 for _ in range(positions)]
                heads = [head / head.sum() for head in heads]
                expected = sorted(
                    [(sum(float(np.log(head[i])) for head, i in zip(heads, path)), path)
                     for path in combinations(range(domain_size), positions)],
                    key=lambda item: (-item[0], item[1]),
                )[:5]
                self.assertGreater(len(list(combinations(range(domain_size), positions))), 5)
                actual = _top_k_sorted(heads)
                self.assertEqual([path for _, path in actual], [path for _, path in expected])
                np.testing.assert_allclose([score for score, _ in actual],
                                           [score for score, _ in expected])

    def test_sorted_dp_ties_use_lexicographic_top_five(self):
        heads = [np.ones(6) / 6 for _ in range(3)]
        expected_paths = list(combinations(range(6), 3))[:5]
        actual = _top_k_sorted(heads)
        self.assertEqual([path for _, path in actual], expected_paths)
        self.assertEqual(len(actual), 5)

    def test_independent_beam_matches_bruteforce(self):
        heads = [np.array([.7, .2, .1]), np.array([.2, .5, .3]), np.array([.3, .1, .6])]
        expected = sorted(
            [(sum(np.log(head[i] / head.sum()) for head, i in zip(heads, path)), path)
             for path in product(range(3), repeat=3)], key=lambda item: (-item[0], item[1]))[:5]
        actual = _top_k_independent([head / head.sum() for head in heads])
        self.assertEqual([path for _, path in actual], [path for _, path in expected])
        np.testing.assert_allclose([score for score, _ in actual], [score for score, _ in expected])


class SelectionTests(unittest.TestCase):
    def test_actual_ball_dimensions_return_five_unique_legal_tickets(self):
        for game in ("双色球", "dlt"):
            with self.subTest(game=game):
                tickets = select_top_five(game, _uniform_ball_probs("ssq" if game == "双色球" else "dlt"))
                self.assertEqual(len(tickets), 5)
                keys = set()
                for ticket in tickets:
                    if game == "双色球":
                        self.assertEqual(ticket["red"], sorted(ticket["red"]))
                        self.assertEqual(len(ticket["red"]), 6)
                        self.assertEqual(len(set(ticket["red"])), 6)
                        self.assertTrue(1 <= ticket["blue"] <= 16)
                        keys.add(tuple(ticket["red"] + [ticket["blue"]]))
                    else:
                        self.assertEqual(ticket["front"], sorted(ticket["front"]))
                        self.assertEqual(ticket["back"], sorted(ticket["back"]))
                        self.assertEqual(len(set(ticket["front"])), 5)
                        self.assertEqual(len(set(ticket["back"])), 2)
                        keys.add(tuple(ticket["front"] + ticket["back"]))
                    self.assertLessEqual(ticket["joint_log_score"], 0.0)
                    self.assertGreater(ticket["prob"], 0.0)
                self.assertEqual(len(keys), 5)

    def test_digit_direct_is_deterministic_greedy_top_five(self):
        tickets = select_top_five("pls", _pls_probs(), "直选")
        self.assertEqual([ticket["digits"] for ticket in tickets],
                         [[9, 9, 9], [8, 9, 9], [9, 8, 9], [9, 9, 8], [8, 8, 9]])
        self.assertEqual(tickets, select_top_five("排列3", _pls_probs(), "直选"))

    def test_pl3_group_scores_sum_valid_permutations(self):
        probs = {"digit1": np.array([.85, .10, .05] + [0.] * 7),
                 "digit2": np.array([.10, .85, .05] + [0.] * 7),
                 "digit3": np.array([.75, .20, .05] + [0.] * 7)}
        grouped = select_top_five("pls", probs, "组选3")
        self.assertEqual(grouped[0]["digits"], [0, 0, 1])
        expected = np.log(.85 * .10 * .20 + .85 * .85 * .75 + .10 * .10 * .75)
        self.assertAlmostEqual(grouped[0]["joint_log_score"], expected)

    def test_pl3_group6_matches_full_directed_ticket_enumeration(self):
        rng = np.random.default_rng(20260906)
        heads = [rng.random(10) + 0.01 for _ in range(3)]
        heads = [head / head.sum() for head in heads]
        probs = {f"digit{i + 1}": head for i, head in enumerate(heads)}
        grouped = select_top_five("pls", probs, "组选6")

        masses = {}
        for ordered in product(range(10), repeat=3):
            if len(set(ordered)) != 3:
                continue
            canonical = tuple(sorted(ordered))
            masses[canonical] = masses.get(canonical, 0.0) + float(
                np.prod([head[digit] for head, digit in zip(heads, ordered)])
            )
        expected = sorted(
            [(float(np.log(mass)), canonical) for canonical, mass in masses.items()],
            key=lambda item: (-item[0], item[1]),
        )[:5]
        self.assertEqual([ticket["digits"] for ticket in grouped],
                         [list(path) for _, path in expected])
        np.testing.assert_allclose([ticket["joint_log_score"] for ticket in grouped],
                                   [score for score, _ in expected])

    def test_plw_and_qxc_return_five_full_dimension_legal_tickets(self):
        for game, classes in (("plw", (10, 10, 10, 10, 10)),
                              ("qxc", (10, 10, 10, 10, 10, 10, 15))):
            with self.subTest(game=game):
                probs = {f"digit{i + 1}": np.ones(size) for i, size in enumerate(classes)}
                tickets = select_top_five(game, probs)
                self.assertEqual(len(tickets), 5)
                self.assertEqual(len({tuple(ticket["digits"]) for ticket in tickets}), 5)
                for ticket in tickets:
                    self.assertEqual(len(ticket["digits"]), len(classes))
                    self.assertTrue(all(0 <= value < size
                                        for value, size in zip(ticket["digits"], classes)))

    def test_ties_are_deterministic(self):
        first = select_top_five("ssq", _uniform_ball_probs("ssq"))
        second = select_top_five("ssq", _uniform_ball_probs("ssq"))
        self.assertEqual(first, second)
        self.assertEqual([ticket["red"] for ticket in first], [[1, 2, 3, 4, 5, 6]] * 5)
        self.assertEqual([ticket["blue"] for ticket in first], [1, 2, 3, 4, 5])

    def test_invalid_probabilities_and_insufficient_candidates_are_clear(self):
        bad = _pls_probs(); bad["digit1"] = np.array([np.nan] * 10)
        with self.assertRaisesRegex(ValueError, "有限数值"):
            select_top_five("pls", bad)
        bad = _pls_probs(); bad["digit2"] = np.array([-1.] + [1.] * 9)
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            select_top_five("pls", bad)
        bad = _pls_probs(); bad["digit3"] = np.ones(9)
        with self.assertRaisesRegex(ValueError, "维度应为 10"):
            select_top_five("pls", bad)
        bad = _pls_probs(); bad["digit3"] = np.zeros(10)
        with self.assertRaisesRegex(ValueError, "总和必须大于 0"):
            select_top_five("pls", bad)
        concentrated = {f"digit{i}": np.eye(10)[0] for i in range(1, 4)}
        with self.assertRaisesRegex(ValueError, "不足固定的 5 注"):
            select_top_five("pls", concentrated, "直选")


if __name__ == "__main__":
    unittest.main()
