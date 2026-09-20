from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evidence_service as service
import run_evidence_experiment as experiment


class ResearchBoundaryTests(unittest.TestCase):
    def test_scaler_boundary_excludes_validation_and_test(self):
        self.assertEqual(experiment.feature_training_boundary(130), 109)
        with self.assertRaises(ValueError):
            experiment.feature_training_boundary(31)

    def test_baseline_reproducible_and_unique_for_seven_plays(self):
        from game_config import GAME_PLAY_OPTIONS
        from portfolio_engine import _ticket_key
        for game, plays in GAME_PLAY_OPTIONS.items():
            for play in plays:
                values = experiment.baseline_tickets(game, play, "26106")
                self.assertEqual(values, experiment.baseline_tickets(game, play, "26106"))
                self.assertEqual(len({_ticket_key(t, game, play) for t in values}), 5)

    def test_target_future_fresh_immediate_and_same_year(self):
        today = date(2026, 9, 11)
        service.validate_target("26105", "2026-09-10", "26106", "2026-09-13", today)
        cases = [("26105", "2026-09-10", "26106", "2026-09-11"),
                 ("26105", "2026-09-01", "26106", "2026-09-13"),
                 ("26105", "2026-09-12", "26106", "2026-09-13"),
                 ("26105", "2026-09-10", "26107", "2026-09-13"),
                 ("26105", "2026-09-10", "26106", "2026-09-20"),
                 ("26153", "2026-12-31", "26154", "2027-01-01")]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                service.validate_target(*args, today)

    def test_target_feature_is_excluded_from_all_predictions(self):
        frame = pd.DataFrame({"issue": ["1", "2", "3", "4"], "date": ["2026-01-01"] * 4,
                              "digit1": [0, 0, 0, 0], "digit2": [0, 0, 0, 0], "digit3": [0, 0, 0, 0]})
        features = np.arange(8).reshape(4, 2)
        observed = []
        def probabilities(model, window, seq_len):
            observed.append(window.copy())
            return {"digit1": np.ones(10) / 10, "digit2": np.ones(10) / 10, "digit3": np.ones(10) / 10}
        with patch.object(experiment, "_get_probabilities", side_effect=probabilities):
            result = experiment.evaluate_windows("pls", frame, features, None, 2, 2)
        np.testing.assert_array_equal(observed[0], features[0:2])
        np.testing.assert_array_equal(observed[1], features[1:3])
        self.assertEqual(result["直选"][0]["input_last_issue"], "2")

    def test_source_study_paths_never_silently_create_on_read(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(service, "STUDY", Path(folder) / "study"):
            with self.assertRaises(ValueError):
                service.checked_study()
            self.assertFalse((Path(folder) / "study").exists())


if __name__ == "__main__":
    unittest.main()
