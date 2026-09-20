"""最高奖固定5注的推理保存、文案与UI入口集成验证。"""
import unittest
from unittest.mock import patch

import numpy as np

import gradio_app
import predictor


class TopFiveIntegrationTests(unittest.TestCase):
    def test_top_five_uses_joint_ranking_and_persists_strategy(self):
        probs = {f"digit{i}": np.ones(10) / 10 for i in range(1, 6)}
        with patch.object(predictor, "_load_model"), \
             patch.object(predictor, "_get_probabilities", return_value=probs), \
             patch.object(predictor, "_save_prediction") as save, \
             patch.object(predictor, "_weighted_sample_digits") as weighted:
            numbers, _ = predictor.predict_numbers(
                "plw", np.ones((40, 5)), n_groups=5, selection_strategy="joint_top5",
                source_issue="26001", source_date="2026-01-01")
        self.assertEqual(len(numbers), 5)
        weighted.assert_not_called()
        self.assertEqual(save.call_args.kwargs["selection_strategy"], "joint_top5")
        text = predictor.format_numbers_table("plw", numbers, "直选")
        self.assertIn("模型联合log分", text)
        self.assertIn("不是实际中奖概率", text)
        self.assertIn("0.005000000%", text)
        self.assertNotIn("倾向分:", text)

    def test_no_more_than_five_and_no_unknown_strategy(self):
        for kwargs in ({"selection_strategy": "joint_top5", "n_groups": 6},
                       {"selection_strategy": "unknown", "n_groups": 5}):
            with patch.object(predictor, "_load_model") as loader:
                with self.assertRaises(ValueError):
                    predictor.predict_numbers("ssq", np.ones((40, 4)), **kwargs)
                loader.assert_not_called()

    def test_ui_entry_is_fixed_five_without_sampling_bias_controls(self):
        with patch.object(gradio_app, "predict", return_value=("ok", "", None, None)) as predict:
            gradio_app.generate_top_five("排列3", "组选6")
        self.assertEqual(predict.call_args.args[2], 5)
        self.assertEqual(predict.call_args.kwargs["selection_strategy"], "joint_top5")
        self.assertEqual(predict.call_args.args[6:10], (0.0, 1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
