"""Focused regression coverage for variable-size jackpot selection."""
import unittest
from itertools import combinations, product
from math import log

import numpy as np

from jackpot_selection import select_top_five, select_top_k, _top_k_sorted, _top_k_independent


def _heads(sizes):
    return {f"digit{i}": np.ones(size) for i, size in enumerate(sizes, 1)}


class VariableTopKTests(unittest.TestCase):
    def test_variable_k_helpers_match_complete_small_space(self):
        heads = [np.array([.1, .4, .3, .2]), np.array([.2, .1, .4, .3])]
        for helper, paths in ((_top_k_sorted, list(combinations(range(4), 2))),
                              (_top_k_independent, list(product(range(4), repeat=2)))):
            exhaustive = sorted([(sum(log(heads[i][v]) for i, v in enumerate(path)), path)
                                 for path in paths], key=lambda item: (-item[0], item[1]))
            for k in (1, 3, 6):
                actual = helper(heads, k)
                self.assertEqual([item[1] for item in actual], [item[1] for item in exhaustive[:k]])
                np.testing.assert_allclose([item[0] for item in actual], [item[0] for item in exhaustive[:k]])

    def test_all_games_support_k_one_and_ten(self):
        cases = (
            ("ssq", {**{f"red{i}": np.ones(33) for i in range(1, 7)}, "blue": np.ones(16)}, None),
            ("dlt", {**{f"front{i}": np.ones(35) for i in range(1, 6)}, **{f"back{i}": np.ones(12) for i in range(1, 3)}}, None),
            ("pls", _heads((10, 10, 10)), "直选"),
            ("plw", _heads((10, 10, 10, 10, 10)), None),
            ("qxc", _heads((10, 10, 10, 10, 10, 10, 15)), None),
        )
        for game, probs, play in cases:
            for k in (1, 10):
                with self.subTest(game=game, k=k):
                    tickets = select_top_k(game, probs, k, play)
                    self.assertEqual(len(tickets), k)
                    self.assertEqual(len({repr(ticket) for ticket in tickets}), k)
                    self.assertTrue(all(ticket["selection_strategy"] == "joint_topk" for ticket in tickets))

    def test_grouped_support_limits_are_reported(self):
        probs = _heads((10, 10, 10))
        with self.assertRaisesRegex(ValueError, "不足请求的 91 注"):
            select_top_k("pls", probs, 91, "组选3")
        with self.assertRaisesRegex(ValueError, "不足请求的 121 注"):
            select_top_k("pls", probs, 121, "组选6")

    def test_k_is_strict_and_legacy_wrapper_keeps_strategy(self):
        probs = _heads((10, 10, 10))
        for invalid in (True, 0, 501, 1.0, 1.5, float("nan")):
            with self.subTest(k=invalid):
                with self.assertRaises(ValueError):
                    select_top_k("pls", probs, invalid)
        self.assertTrue(all(t["selection_strategy"] == "joint_top5"
                            for t in select_top_five("pls", probs)))


if __name__ == "__main__":
    unittest.main()
