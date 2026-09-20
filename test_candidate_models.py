import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from candidate_models import MarkovHeads, XGB_CONFIG, XGBoostHeads, summarize_windows
from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from jackpot_selection import _game_heads, select_top_k


def _targets(game, rows=40):
    return {
        name: (np.arange(rows, dtype=np.int64) + position) % classes
        for position, (name, classes) in enumerate(_game_heads(game))
    }


class WindowSummaryTests(unittest.TestCase):
    def test_exact_window_summary_excludes_target_and_future(self):
        features = np.arange(50, dtype=np.float32).reshape(10, 5)
        actual = summarize_windows(features, [3, 7], seq_len=3)
        expected = np.vstack([
            np.concatenate((features[2], features[0:3].mean(0), features[0:3].std(0))),
            np.concatenate((features[6], features[4:7].mean(0), features[4:7].std(0))),
        ]).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)

    def test_window_summary_rejects_future_or_invalid_boundaries(self):
        values = np.ones((10, 2), dtype=np.float32)
        for indices in ([2], [11], [3.0], [True]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    summarize_windows(values, indices, seq_len=3)
        with self.assertRaises(ValueError):
            summarize_windows(np.array([[np.nan]]), [1], seq_len=1)


class MarkovHeadsTests(unittest.TestCase):
    def test_all_five_games_and_seven_plays_produce_five_legal_tickets(self):
        observed_plays = 0
        for game in ALL_GAME_CODES:
            targets = _targets(game)
            model = MarkovHeads(game).fit(targets, start=30)
            previous = {name: values[-2:] for name, values in targets.items()}
            probabilities = model.predict_heads(previous)
            for name, classes in _game_heads(game):
                self.assertEqual(probabilities[name].shape, (2, classes))
                self.assertTrue(np.all(probabilities[name] > 0))
                np.testing.assert_allclose(probabilities[name].sum(axis=1), 1.0, rtol=1e-6)
            for play in GAME_PLAY_OPTIONS[game]:
                tickets = select_top_k(game, {name: values[0] for name, values in probabilities.items()}, 5, play)
                self.assertEqual(len(tickets), 5)
                observed_plays += 1
        self.assertEqual(observed_plays, 7)

    def test_constant_labels_smoothing_and_json_roundtrip(self):
        targets = {name: np.zeros(36, dtype=np.int64) for name, _ in _game_heads("ssq")}
        model = MarkovHeads("ssq", smoothing=1.0).fit(targets, start=30)
        previous = {name: np.array([0], dtype=np.int64) for name in targets}
        expected = model.predict_heads(previous)
        self.assertTrue(np.all(expected["red1"] > 0))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "markov"
            model.save(directory)
            restored = MarkovHeads.load(directory)
            for name in expected:
                np.testing.assert_allclose(restored.predict_heads(previous)[name], expected[name])

    def test_save_rejects_any_existing_directory(self):
        model = MarkovHeads("pls").fit(_targets("pls"), start=30)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                model.save(directory)

    def test_markov_rejects_invalid_ranges_and_dimensions(self):
        targets = _targets("pls", rows=35)
        with self.assertRaises(ValueError):
            MarkovHeads("pls").fit(targets, start=0)
        broken = dict(targets)
        broken["digit1"] = broken["digit1"][:-1]
        with self.assertRaises(ValueError):
            MarkovHeads("pls").fit(broken, start=30)

    def test_markov_end_excludes_later_labels_and_is_deterministic(self):
        targets = _targets("pls", rows=40)
        altered = {name: values.copy() for name, values in targets.items()}
        for name, classes in _game_heads("pls"):
            altered[name][35:] = (altered[name][35:] + 4) % classes
        first = MarkovHeads("pls").fit(targets, start=30, end=35)
        second = MarkovHeads("pls").fit(altered, start=30, end=35)
        previous = {name: np.array([0], dtype=np.int64) for name in targets}
        for name in targets:
            np.testing.assert_array_equal(
                first.predict_heads(previous)[name], second.predict_heads(previous)[name]
            )


@unittest.skipUnless(importlib.util.find_spec("xgboost"), "xgboost未安装；真实XGBoost测试跳过")
class XGBoostHeadsTests(unittest.TestCase):
    def test_fixed_config_missing_classes_and_json_roundtrip(self):
        self.assertEqual(XGB_CONFIG["n_estimators"], 80)
        self.assertEqual(XGB_CONFIG["random_state"], 20260912)
        rng = np.random.default_rng(20260912)
        X = rng.normal(size=(12, 6)).astype(np.float32)
        targets = _targets("ssq", rows=12)
        targets["blue"] = np.zeros(12, dtype=np.int64)  # constant head
        model = XGBoostHeads("ssq").fit(X, targets)
        expected = model.predict_heads(X[:3])
        for name, classes in _game_heads("ssq"):
            self.assertEqual(expected[name].shape, (3, classes))
            self.assertTrue(np.all(expected[name] > 0))
            np.testing.assert_allclose(expected[name].sum(axis=1), 1.0, rtol=1e-6)
        self.assertEqual(np.argmax(expected["blue"][0]), 0)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "xgboost"
            model.save(directory)
            restored = XGBoostHeads.load(directory)
            for name in expected:
                np.testing.assert_allclose(restored.predict_heads(X[:3])[name], expected[name], rtol=1e-6)

    def test_xgb_load_rejects_path_traversal_and_repeated_class_mapping(self):
        X = np.arange(72, dtype=np.float32).reshape(12, 6)
        model = XGBoostHeads("pls").fit(X, _targets("pls", rows=12))
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "xgboost"
            model.save(directory)
            metadata_path = directory / "candidate_xgboost_heads.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["heads"][0]["model_file"] = "../outside.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                XGBoostHeads.load(directory)

        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "xgboost"
            model.save(directory)
            metadata_path = directory / "candidate_xgboost_heads.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["heads"][0]["observed_classes"] = [0, 0]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                XGBoostHeads.load(directory)

    def test_xgb_save_rejects_existing_directory(self):
        X = np.arange(72, dtype=np.float32).reshape(12, 6)
        model = XGBoostHeads("pls").fit(X, _targets("pls", rows=12))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                model.save(directory)

    def test_xgb_rejects_alignment_and_feature_dimension_errors(self):
        X = np.ones((4, 3), dtype=np.float32)
        targets = _targets("pls", rows=4)
        targets["digit1"] = targets["digit1"][:-1]
        with self.assertRaises(ValueError):
            XGBoostHeads("pls").fit(X, targets)
