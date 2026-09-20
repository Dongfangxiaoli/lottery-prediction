import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

import prediction_review as review


class PredictionReviewTest(unittest.TestCase):
    def test_damaged_and_wrong_play_records_are_not_mixed(self):
        direct = {"game": "pls", "timestamp": "20260906_120000",
                  "numbers": [{"digits": [0, 1, 2]}]}
        grouped = {**direct, "play": "组选6", "timestamp": "20260906_120001"}
        invalid = {**grouped, "numbers": [{"digits": [1, 1, 2]}]}
        with tempfile.TemporaryDirectory() as temp:
            files = {"pls_0.json": json.dumps(direct), "pls_1.json": json.dumps(grouped),
                     "pls_2.json": json.dumps(invalid), "pls_3.json": "{broken",
                     "pls_4.json": "[]", "unrelated.txt": "keep"}
            for name, text in files.items():
                Path(temp, name).write_text(text, encoding="utf-8")
            before = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
            with patch.object(review, "RESULTS_DIR", temp):
                records, skipped = review._read_prediction_records("pls", "组选6")
                self.assertEqual(records, [grouped])
                self.assertEqual(skipped, 3)
                self.assertEqual(review._load_latest_prediction("pls", "直选"), direct)
                text = review.load_prediction_history("排列3", "组选6")
                self.assertIn("已跳过 3", text)
                self.assertIn("排列3（组选6）", text)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(temp).iterdir()})

    def test_ticket_validation_and_group_multiset(self):
        self.assertEqual(review._prediction_digits({"digits": "012"}), [0, 1, 2])
        for digits in ([1, 1, 2], [0, 1, 14], [0, 1, 2.5], [True, 1, 2]):
            self.assertFalse(review._valid_ticket("pls", {"digits": digits}, "组选6"))
        pred = {"numbers": [{"digits": [1, 1, 2]}]}
        draw = {"issue": "26200", "date": "2026-09-06", "digits": [2, 1, 1]}
        _, table = review._review_numeric(pred, draw, "pls", "组选3")
        self.assertEqual(table.iloc[0]["整组命中"], "✓")
        self.assertNotIn("位置命中", table.columns)

    def test_history_rates_are_weighted_by_ticket_count(self):
        winning = {"digits": [1, 2, 3]}
        losing = {"digits": [4, 5, 6]}
        records = [{"timestamp": "one", "source_date": "2026-09-01", "numbers": [winning]},
                   {"timestamp": "nine", "source_date": "2026-09-01", "numbers": [losing] * 9}]
        frame = pd.DataFrame({"issue": ["26201"], "date": ["2026-09-02"],
                              "digit1": [1], "digit2": [2], "digit3": [3]})
        with patch.object(review, "_read_prediction_records", return_value=(records, 0)), \
                patch.object(review.os.path, "exists", return_value=True), \
                patch.object(review.pd, "read_csv", return_value=frame):
            text, _, fig = review.review_history_stats("pls", "直选")
        self.assertIn("共 10 注，匹配 1 个开奖期", text)
        self.assertIn("实际命中率 10.00%", text)
        self.assertIn("实际平均命中 0.300", text)
        self.assertNotIn("nan", text.lower())
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
