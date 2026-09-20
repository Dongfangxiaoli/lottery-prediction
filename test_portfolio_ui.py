"""New portfolio page contracts; all export checks use temporary directories."""
import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import gradio_app
import portfolio_ui


class PortfolioUiTests(unittest.TestCase):
    def test_play_specific_limits_and_awards(self):
        self.assertEqual(portfolio_ui.portfolio_options("排列3", "组选3")[0], 90)
        self.assertEqual(portfolio_ui.portfolio_options("排列3", "组选6")[0], 120)
        self.assertEqual(len(portfolio_ui.portfolio_options("排列5", "直选")[1]), 1)
        self.assertEqual(len(portfolio_ui.portfolio_options("大乐透", None)[1]), 7)

    def test_browser_numeric_values_are_validated_not_rounded(self):
        self.assertEqual(portfolio_ui._integer_input(10.0, "注数"), 10)
        for value in (10.5, None, float("nan"), True):
            with self.assertRaises(ValueError):
                portfolio_ui._integer_input(value, "注数")

    def test_variable_generation_needs_no_training_or_prediction_save(self):
        with patch("predictor._save_prediction") as save:
            values = portfolio_ui.make_portfolio("排列5", "直选", 12, "均匀随机去重", 1, 42, {})
        self.assertEqual(len(values[2]), 12)
        self.assertEqual(len(values[4]["tickets"]), 12)
        self.assertIn("排列5", values[1])
        self.assertIn("未绑定实际待开奖期", values[0])
        save.assert_not_called()

    def test_model_requires_session_and_preserves_cutoff(self):
        with self.assertRaisesRegex(ValueError, "需要先"):
            portfolio_ui.make_portfolio("排列3", "直选", 10, "模型联合排序（实验）", 1, 42, {})
        probs = {f"digit{i}": np.ones(10) for i in range(1, 4)}
        with patch.object(portfolio_ui, "_session_probabilities", return_value=(
                probs, {"source_issue": "26001", "source_date": "2026-01-01"})):
            values = portfolio_ui.make_portfolio("排列3", "直选", 10, "模型联合排序（实验）", 1, 42, {})
        self.assertEqual(values[4]["source_issue"], "26001")
        self.assertIn("不是实际中奖概率", values[0])

    def test_exclusive_json_export_and_empty_state_guard(self):
        values = portfolio_ui.make_portfolio("排列3", "组选6", 10, "均匀随机去重", 1, 42, {})
        with tempfile.TemporaryDirectory() as directory, patch.object(portfolio_ui, "datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 20, 12, 0, 0)
            first = portfolio_ui.export_portfolio(values[4], directory)
            original = Path(first).read_bytes()
            second = portfolio_ui.export_portfolio(values[4], directory)
            third = portfolio_ui.export_portfolio(values[4], directory)
            self.assertEqual(len({first, second, third}), 3)
            self.assertEqual(Path(first).read_bytes(), original)
            self.assertEqual(Path(second).read_bytes(), original)
            self.assertEqual(Path(third).read_bytes(), original)
            result = json.loads(Path(first).read_text(encoding="utf-8"))
            self.assertEqual(len(result["tickets"]), 10)
        with self.assertRaises(ValueError):
            portfolio_ui.export_portfolio(None)

    def test_page_callbacks_clear_results_and_restrict_targets(self):
        app = gradio_app.build_ui()
        callbacks = {event.fn.__name__: event for event in app.fns.values() if event.fn}
        for name, interactive in (("lock_portfolio", False), ("unlock_portfolio", True)):
            event = callbacks[name]
            updates = event.fn()
            self.assertEqual(len(updates), len(event.outputs))
            self.assertTrue(all(value["interactive"] == interactive for value in updates))
        for name, args in (("refresh_portfolio_game", ("排列3",)),
                           ("refresh_portfolio_play", ("排列3", "组选3")),
                           ("clear_portfolio", ())):
            event = callbacks[name]
            values = event.fn(*args)
            self.assertEqual(len(values), len(event.outputs))
            self.assertIsNone(values[-2])  # No stale exportable State.
            self.assertIsNone(values[-1])
        values = callbacks["refresh_portfolio_play"].fn("排列3", "组选3")
        self.assertEqual(values[0]["maximum"], 90)
        self.assertEqual(values[1]["choices"], [("最高奖", 1)])
        error = callbacks["run_portfolio"].fn("双色球", "单式投注", 0, "均匀随机去重", 1, 42)
        self.assertIn("无法生成", error[0])
        self.assertIsNone(error[-2])


if __name__ == "__main__":
    unittest.main()
