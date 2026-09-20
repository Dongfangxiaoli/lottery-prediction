"""Gradio 默认值与彩种切换回调的轻量回归测试。"""
import unittest

import gradio_app


class UiDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = gradio_app.build_ui()

    @classmethod
    def _event(cls, name):
        return next(event for event in cls.app.fns.values()
                    if getattr(event.fn, "__name__", "") == name)

    def _call_and_postprocess(self, name, *args):
        event = self._event(name)
        values = event.fn(*args)
        self.assertEqual(len(values), len(event.outputs), name)
        for component, value in zip(event.outputs, values):
            if isinstance(value, dict):
                if "value" in value:
                    component.postprocess(value["value"])
            else:
                component.postprocess(value)
        return values

    def test_game_descriptions_are_bound_to_game_and_play(self):
        self.assertIn("5 位数字", gradio_app.game_play_description("排列5", "直选"))
        self.assertIn("1/100,000", gradio_app.game_play_description("排列5", "直选"))
        self.assertIn("3 位数字", gradio_app.game_play_description("排列3", "直选"))
        self.assertIn("整注全部命中", gradio_app.game_play_description("7星彩", "单式投注"))

    def test_digit_and_ball_game_switches_clear_and_toggle_controls(self):
        digit = self._call_and_postprocess("clear_incompatible_and_results", "排列3")
        for value in digit[:6]:
            self.assertEqual(value["value"], 0.0)
            self.assertFalse(value["visible"])
        self.assertEqual(digit[6]["value"], "per_position")
        self.assertFalse(digit[6]["visible"])
        self.assertFalse(digit[11]["visible"])  # 集合覆盖核心池
        self.assertIn("上一彩种的结果已清空", digit[19])

        ball = self._call_and_postprocess("clear_incompatible_and_results", "双色球")
        self.assertTrue(ball[0]["visible"])
        self.assertEqual(ball[6]["value"], "merged")
        self.assertTrue(ball[11]["visible"])
        self.assertIsNone(ball[17]["value"])  # 覆盖号码表格必须清空，而非写入提示文本

    def test_restore_defaults_and_play_change_clear_results(self):
        defaults = self._call_and_postprocess("restore_defaults", "排列5")
        self.assertEqual(defaults[0]["value"], "直选")
        self.assertEqual(defaults[2:7], (30, 30, 0.001, 64, 128))
        self.assertEqual(defaults[13:18], ("merged", 0.0, 1.0, 0.0, 0.0))

        cleared = self._call_and_postprocess("clear_results", "排列3", "组选6")
        self.assertIn("排列3（组选6）", cleared[0])
        self.assertEqual(cleared[6], "")
        self.assertIsNone(cleared[1])

    def test_first_tab_entry_only_refreshes_visibility(self):
        for game, visible in (("排列5", False), ("大乐透", True)):
            values = self._call_and_postprocess("refresh_applicability", game)
            self.assertEqual(len(values), 19)
            self.assertTrue(all(item["visible"] == visible for item in values[:18]))
            self.assertEqual(values[-1]["visible"], not visible)
            self.assertTrue(all("value" not in item for item in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
