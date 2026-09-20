import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from model_session import load_training_session, save_training_session


class ModelSessionTest(unittest.TestCase):
    def _ssq_df(self):
        return pd.DataFrame({
            "issue": ["26001", "26002", "26003"],
            "date": ["2026/01/01", "2026-01-03", "2026-01-05"],
            "red1": [1, 2, 3], "red2": [4, 5, 6], "red3": [7, 8, 9],
            "red4": [10, 11, 12], "red5": [13, 14, 15], "red6": [16, 17, 18],
            "blue": [1, 2, 3], "sales": [100, 200, 300], "pool": [10, 20, 30],
        })

    def _info(self, model_path, *, ensemble=False, input_size=3):
        return {
            "model_path": str(model_path), "input_size": input_size, "hidden_size": 8,
            "num_layers": 1, "dropout": 0.25, "seq_len": 1,
            "bidirectional": False, "use_attention": False, "attn_heads": 4,
            "n_ensemble": 2 if ensemble else 1,
            "ensemble_seeds": [42, 43] if ensemble else [], "holdout_size": 0,
            "train_losses": [2.0, 1.0], "val_losses": [2.5, 1.5], "best_epoch": 1,
        }

    def test_single_model_round_trip_and_csv_date_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            weight = models / "ssq_lstm.pth"
            weight.write_bytes(b"single-weight")
            df = self._ssq_df()
            features = np.arange(9, dtype=np.float64).reshape(3, 3)
            path = save_training_session("ssq", df, features, self._info(weight), models)
            self.assertTrue(Path(path).is_file())
            # Re-read uses a different textual date representation but the same dates.
            csv = models / "history.csv"
            df.to_csv(csv, index=False)
            reread = pd.read_csv(csv, dtype={"issue": str})
            restored = load_training_session("双色球", reread, features.astype(np.float32), models)
            self.assertEqual(restored["model_path"], str(weight.resolve()))
            self.assertEqual(restored["n_ensemble"], 1)

    def test_ensemble_round_trip_and_integrity_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            base = models / "pls_lstm.pth"
            for name, content in ((base, b"base"), (models / "pls_lstm_seed42.pth", b"seed42"),
                                  (models / "pls_lstm_seed43.pth", b"seed43")):
                name.write_bytes(content)
            df = pd.DataFrame({
                "issue": ["1", "2", "3"], "date": ["2026-01-01"] * 3,
                "digit1": [0, 1, 2], "digit2": [1, 2, 3], "digit3": [2, 3, 4],
                "sales": [10, 20, 30],
            })
            features = np.arange(12, dtype=np.float32).reshape(3, 4)
            save_training_session("pls", df, features, self._info(base, ensemble=True, input_size=4), models)
            self.assertEqual(load_training_session("pls", df, features, models)["ensemble_seeds"], [42, 43])
            (models / "pls_lstm_seed43.pth").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_training_session("pls", df, features, models)
            (models / "pls_lstm_seed43.pth").unlink()
            with self.assertRaisesRegex(ValueError, "缺少权重"):
                load_training_session("pls", df, features, models)

    def test_rejects_missing_manifest_changed_inputs_and_unsafe_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            df = self._ssq_df()
            features = np.ones((3, 2), dtype=np.float32)
            (models / "ssq_lstm.pth").write_bytes(b"old-but-unproven")
            with self.assertRaisesRegex(ValueError, "旧权重"):
                load_training_session("ssq", df, features, models)
            save_training_session("ssq", df, features,
                                  self._info(models / "ssq_lstm.pth", input_size=2), models)
            changed = features.copy(); changed[0, 0] = 2
            with self.assertRaisesRegex(ValueError, "特征"):
                load_training_session("ssq", df, changed, models)
            changed_df = df.copy(); changed_df.loc[0, "red1"] = 33
            with self.assertRaisesRegex(ValueError, "历史数据"):
                load_training_session("ssq", changed_df, features, models)
            manifest = models / "ssq_training_session.json"
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["train_info"]["model_path"] = "../ssq_lstm.pth"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "路径遍历"):
                load_training_session("ssq", df, features, models)

    def test_rejects_bool_and_nonfinite_external_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            weight = models / "ssq_lstm.pth"; weight.write_bytes(b"weight")
            info = self._info(weight); info["seq_len"] = True
            with self.assertRaisesRegex(ValueError, "seq_len"):
                save_training_session("ssq", self._ssq_df(), np.ones((3, 2)), info, models)
            with self.assertRaisesRegex(ValueError, "features"):
                save_training_session("ssq", self._ssq_df(), np.array([[np.nan, 1]], dtype=np.float32),
                                      self._info(weight, input_size=2), models)

    def test_portable_manifest_and_training_window_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, moved = root / "original", root / "moved"
            original.mkdir(); moved.mkdir()
            weight = original / "ssq_lstm.pth"; weight.write_bytes(b"portable-weight")
            df = self._ssq_df()
            features = np.arange(9, dtype=np.float32).reshape(3, 3)
            manifest = Path(save_training_session("ssq", df, features, self._info(weight), original))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["train_info"]["model_path"], "ssq_lstm.pth")
            shutil.copy2(weight, moved / weight.name)
            shutil.copy2(manifest, moved / manifest.name)
            restored = load_training_session("ssq", df, features, moved)
            self.assertEqual(restored["model_path"], str((moved / weight.name).resolve()))
            # Returned info is directly saveable again after restoring/moving.
            save_training_session("ssq", df, features, restored, moved)
            bad = self._info(weight); bad["seq_len"] = 2
            with self.assertRaisesRegex(ValueError, "至少需要 2"):
                save_training_session("ssq", df, features, bad, original)
            bad = self._info(weight); bad["train_losses"] = [1.0]
            with self.assertRaisesRegex(ValueError, "长度"):
                save_training_session("ssq", df, features, bad, original)


if __name__ == "__main__":
    unittest.main()
