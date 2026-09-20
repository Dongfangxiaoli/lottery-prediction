"""Offline data and real checkpoint restoration, always in temporary directories."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import gradio_app
import local_history
import predictor
import train as training_backend
from portfolio_ui import _session_probabilities


class LocalSessionUiTests(unittest.TestCase):
    def setUp(self):
        self.previous = dict(gradio_app._state)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.models = self.root / "models"
        self.data.mkdir(); self.models.mkdir()
        # Keep this test self-contained: the public source copy intentionally has
        # no historical data.  The fixed seed makes a valid synthetic PL3 history
        # reproducible while remaining clearly test-only data.
        rng = np.random.default_rng(20260920)
        self.frame = pd.DataFrame({
            "issue": [str(30000 + index) for index in range(120)],
            "date": pd.date_range("2025-01-01", periods=120, freq="D").strftime("%Y-%m-%d"),
            "digit1": rng.integers(0, 10, size=120),
            "digit2": rng.integers(0, 10, size=120),
            "digit3": rng.integers(0, 10, size=120),
        })
        self.frame.to_csv(self.data / "pls_history.csv", index=False)
        self.patches = [patch.object(local_history, "DATA_DIR", self.data),
                        patch.object(gradio_app, "SCALER_DIR", str(self.models)),
                        patch.object(training_backend, "MODELS_DIR", str(self.models)),
                        patch.object(predictor, "MODELS_DIR", str(self.models)),
                        patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected network"))]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        gradio_app._state.clear(); gradio_app._state.update(self.previous)
        import matplotlib.pyplot as plt
        plt.close("all")
        self.tmp.cleanup()

    def test_local_read_is_offline_and_preserves_files(self):
        path = self.data / "pls_history.csv"
        before = path.read_bytes()
        output = gradio_app.load_local_data("排列3", lambda *a, **k: None)
        self.assertTrue(output[0].startswith("✅"), output[0])
        self.assertIn("未联网", output[0])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.models.iterdir()), [])
        self.assertEqual(len(gradio_app._state["pls_df"]), 120)

    def test_bad_local_file_does_not_replace_current_state(self):
        gradio_app._state["pls_train_info"] = {"marker": "previous"}
        gradio_app._state["pls_features"] = np.ones((2, 3))
        before = gradio_app._state["pls_features"]
        bad = self.frame.copy(); bad.loc[0, "digit1"] = 99
        bad.to_csv(self.data / "pls_history.csv", index=False)
        output = gradio_app.load_local_data("排列3", lambda *a, **k: None)
        self.assertIn("超出", output[0])
        self.assertIs(gradio_app._state["pls_features"], before)
        self.assertEqual(gradio_app._state["pls_train_info"], {"marker": "previous"})

    def test_invalid_online_result_is_transactional(self):
        before = dict(gradio_app._state)
        with patch.object(gradio_app, "fetch_digit", return_value=pd.DataFrame()):
            result = gradio_app.fetch_data("排列3", lambda *a, **k: None)
        self.assertIn("未替换", result[0])
        self.assertTrue(all(gradio_app._state[key] is value for key, value in before.items()))

    def test_actual_train_restart_restore_and_drift_rejection(self):
        torch.set_num_threads(2)
        loaded = gradio_app.load_local_data("排列3", lambda *a, **k: None)
        self.assertTrue(loaded[0].startswith("✅"), loaded[0])
        trained = gradio_app.train("排列3", 10, 1, 0.001, 16, 32,
                                  use_attention=False, progress=lambda *a, **k: None)
        self.assertIn("训练档案已保存", trained[0], trained[0])
        expected, _ = _session_probabilities("pls", gradio_app._state)
        weight_hash = hashlib.sha256((self.models / "pls_lstm.pth").read_bytes()).hexdigest()
        moved = self.root / "moved models"
        shutil.copytree(self.models, moved)
        gradio_app._state.update({f"pls_{suffix}": None for suffix in
                                 ("df", "features", "labels", "scaler", "train_info")})
        with patch.object(gradio_app, "SCALER_DIR", str(moved)), patch.object(predictor, "MODELS_DIR", str(moved)), \
                patch.object(gradio_app, "train_model", side_effect=AssertionError("Restore must not retrain")):
            output = gradio_app.restore_saved_session("排列3", lambda *a, **k: None)
            self.assertIn("已恢复", output[3], output)
            restored, _ = _session_probabilities("pls", gradio_app._state)
            for head in expected:
                np.testing.assert_array_equal(restored[head], expected[head])
            self.assertEqual(hashlib.sha256((moved / "pls_lstm.pth").read_bytes()).hexdigest(), weight_hash)
            current_info = gradio_app._state["pls_train_info"]
            changed = self.frame.copy(); changed.loc[0, "digit1"] = (int(changed.loc[0, "digit1"]) + 1) % 10
            changed.to_csv(self.data / "pls_history.csv", index=False)
            rejected = gradio_app.restore_saved_session("排列3", lambda *a, **k: None)
            self.assertIn("恢复失败", rejected[0])
            self.assertIs(gradio_app._state["pls_train_info"], current_info)

    def test_failed_training_invalidates_stale_model_info(self):
        gradio_app.load_local_data("排列3", lambda *a, **k: None)
        gradio_app._state["pls_train_info"] = {"old": True}
        with patch.object(gradio_app, "train_model", side_effect=OSError("disk full")):
            result = gradio_app.train("排列3", 10, 1, 0.001, 16, 32, progress=lambda *a, **k: None)
        self.assertIn("训练失败", result[0])
        self.assertIsNone(gradio_app._state["pls_train_info"])

    def test_data_range_fraction_duplicate_and_missing_checks(self):
        for change in (lambda df: df.assign(digit2=0.5), lambda df: df.assign(date="bad-date"),
                       lambda df: df.drop(columns="digit3"), lambda df: df.assign(issue="")):
            with self.assertRaises(ValueError):
                local_history.prepare_history(change(self.frame.copy()), "pls")
        conflicting = pd.concat([self.frame, self.frame.iloc[[0]].assign(digit1=99)], ignore_index=True)
        conflicting.loc[len(conflicting)-1, "digit1"] = (int(self.frame.iloc[0]["digit1"]) + 1) % 10
        with self.assertRaisesRegex(ValueError, "同一期号"):
            local_history.prepare_history(conflicting, "pls")

    def test_model_dependent_ui_events_share_one_queue(self):
        app = gradio_app.build_ui()
        names = {"fetch_data", "load_local_data", "restore_saved_session", "train", "predict",
                 "run_backtest", "coverage_bet", "one_click_predict", "run_portfolio",
                 "generate_top_five", "evaluate_top_five"}
        events = [event for event in app.fns.values() if event.fn and event.fn.__name__ in names]
        self.assertEqual({event.fn.__name__ for event in events}, names)
        for event in events:
            self.assertEqual(event.concurrency_id, "training_session")
            self.assertEqual(event.concurrency_limit, 1)


if __name__ == "__main__":
    unittest.main()
