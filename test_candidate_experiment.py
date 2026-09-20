import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from jackpot_selection import _game_heads
from run_candidate_experiment import (holm_adjust, fair_head_probabilities, head_log_loss,
                                      summarize_game, SCOPE)
from candidate_results import read_report, _relative_name


class CandidateExperimentTests(unittest.TestCase):
    def test_holm_unsorted_missing_and_bounds(self):
        self.assertEqual(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
        self.assertEqual(holm_adjust([None, 0.01, 0.9]), [None, 0.03, 1.0])
        for bad in (True, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                holm_adjust([bad])

    def test_fair_sorted_marginals_exhaustive_back_pool(self):
        actual = np.zeros((2, 12))
        pairs = list(itertools.combinations(range(12), 2))
        for a, b in pairs:
            actual[0, a] += 1 / len(pairs)
            actual[1, b] += 1 / len(pairs)
        fair = fair_head_probabilities("dlt")
        np.testing.assert_allclose(fair["back1"], actual[0])
        np.testing.assert_allclose(fair["back2"], actual[1])
        self.assertEqual(fair["back1"][-1], 0)
        self.assertEqual(fair["back2"][0], 0)
        for game in ALL_GAME_CODES:
            for name, size in _game_heads(game):
                vector = fair_head_probabilities(game)[name]
                self.assertEqual(vector.shape, (size,))
                self.assertAlmostEqual(vector.sum(), 1)

    def test_log_loss_and_invalid_inputs(self):
        probs = {name: np.full((4, size), 1 / size) for name, size in _game_heads("pls")}
        targets = {name: np.arange(4) for name in probs}
        self.assertAlmostEqual(head_log_loss(probs, targets, "pls"), np.log(10))
        with self.assertRaises(ValueError):
            head_log_loss({**probs, "digit1": np.ones((4, 10))}, targets, "pls")

    def test_all_plays_are_scored_as_periods_not_individual_numbers(self):
        total = []
        for game in ALL_GAME_CODES:
            plays = {play: [{"hits": {"xgboost": True, "markov": False, "lstm": False, "uniform": False}},
                            {"hits": {"xgboost": False, "markov": True, "lstm": False, "uniform": True}}]
                     for play in GAME_PLAY_OPTIONS[game]}
            summary = summarize_game(game, plays)
            total.extend(summary)
            for row in summary:
                self.assertEqual(row["periods"], 2)
                self.assertEqual(row["random_wins"], 1)
                self.assertFalse(row["verified_advantage"])
        self.assertEqual(len(total), 21)

    def test_read_report_rejects_outside_directory_and_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            with patch("candidate_results.RUNS", base / "runs"):
                with self.assertRaises(ValueError):
                    read_report(base)
                folder = base / "runs" / "123"
                folder.mkdir(parents=True)
                (folder / "report.json").write_text(json.dumps({"scope": SCOPE, "summary": []}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_report(folder)

    def test_portable_artifact_path_separators_and_traversal(self):
        self.assertEqual(_relative_name("snapshots\\ssq_history.csv"), "snapshots/ssq_history.csv")
        self.assertEqual(_relative_name("models/pls/markov/table.json"), "models/pls/markov/table.json")
        for bad in ("../escape", "..\\escape", "C:\\escape", "/escape", "x:stream", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _relative_name(bad)

    def test_reader_accepts_windows_manifest_and_rejects_corruption(self):
        from run_evidence_experiment import digest
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            folder = base / "run"
            folder.mkdir()
            names = ["protocol.json"] + [f"{g}_result.json" for g in ALL_GAME_CODES] + [f"snapshots/{g}_history.csv" for g in ALL_GAME_CODES]
            artifacts = {}
            for name in names:
                target = folder / name
                target.parent.mkdir(exist_ok=True)
                target.write_text("{}", encoding="utf-8")
                artifacts[name.replace("/", "\\")] = digest(target)
            report = {"scope": SCOPE, "summary": [{"game": g, "play": p, "strategy": s}
                      for g in ALL_GAME_CODES for p in GAME_PLAY_OPTIONS[g] for s in ("xgboost", "markov", "lstm")],
                      "bias_diagnostics": [{}] * 38, "head_scores": [{}] * 20,
                      "artifact_sha256": artifacts, "protocol_sha256": artifacts["protocol.json"]}
            (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
            with patch("candidate_results.RUNS", base):
                self.assertEqual(len(read_report(folder)["summary"]), 21)
                (folder / "snapshots" / "ssq_history.csv").write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_report(folder)

    def test_empty_view_does_not_create_experiment(self):
        from candidate_ui import comparison_view
        with tempfile.TemporaryDirectory() as temp:
            absent = Path(temp) / "missing"
            with patch("candidate_ui.RUNS", absent):
                view = comparison_view()
            self.assertFalse(absent.exists())
            self.assertIsNone(view[-1])
            self.assertEqual(view[1:4], ([], [], []))


if __name__ == "__main__":
    unittest.main()
