"""Offline release acceptance; run with the bundle's Python, never source venv.

User data is read-only; tiny test models/scalers/reports go to bundle cache.
This is an application-level offline simulation, not a clean Windows VM.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
APP = ROOT / "app"
QA = ROOT / "cache" / "release_qa"
QA.mkdir(parents=True, exist_ok=True)
os.chdir(APP)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(QA / "matplotlib")
os.environ["GRADIO_TEMP_DIR"] = str(QA / "gradio")
assert Path(sys.executable).resolve().is_relative_to(ROOT / "runtime")
assert all(Path(p).resolve().is_relative_to(ROOT) for p in sys.path)


def hashes():
    result = {}
    for directory in ("data", "models", "results"):
        for path in (APP / directory).rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    result[str(path.relative_to(APP))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def no_network(*args, **kwargs):
    raise OSError("Offline release QA: network intentionally unavailable")


original_connect = socket.socket.connect
original_create_connection = socket.create_connection


def local_connect(sock, address):
    if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1", "localhost"):
        return original_connect(sock, address)
    return no_network()


def local_create_connection(address, *args, **kwargs):
    if address[0] in ("127.0.0.1", "::1", "localhost"):
        return original_create_connection(address, *args, **kwargs)
    return no_network()


before = hashes()
with patch.object(socket.socket, "connect", local_connect), patch.object(socket, "create_connection", local_create_connection):
    import requests
    import torch
    import gradio_app
    import train
    from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
    from feature_engineering import build_features_ssq, build_features_dlt, build_features_digit
    from portfolio_engine import generate_portfolio
    from portfolio_ui import _session_probabilities
    from run_top5_experiment import infer_legacy_architecture
    import pandas as pd

    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(APP), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Portable unit tests failed")
    print(f"UNIT_TESTS_PASSED {result.testsRun}", flush=True)
    model_runs = []
    for game in ALL_GAME_CODES:
        df = pd.read_csv(APP / "data" / f"{game}_history.csv", dtype={"issue": str})
        info = infer_legacy_architecture(game)
        if game == "ssq":
            features, _, _ = build_features_ssq(df, fit_end=len(df) - 50)
        elif game == "dlt":
            features, _, _ = build_features_dlt(df, fit_end=len(df) - 50)
        else:
            features, _, _ = build_features_digit(df, game, fit_end=len(df) - 50)
        probabilities, _ = _session_probabilities(game, {f"{game}_features": features,
            f"{game}_df": df, f"{game}_train_info": info})
        for play in GAME_PLAY_OPTIONS[game]:
            assert len(generate_portfolio(game, play, 10, "model", probs=probabilities)["tickets"]) == 10
            model_runs.append({"game": game, "play": play, "tickets": 10})
        print(f"EXISTING_MODEL_OK {game}", flush=True)

    test_models = QA / "test_models"
    test_models.mkdir(exist_ok=True)
    train.MODELS_DIR = str(test_models)
    import predictor
    predictor.MODELS_DIR = str(test_models)
    gradio_app.SCALER_DIR = str(test_models)
    offline_runs = []
    with patch.object(requests.sessions.Session, "request", no_network):
        for game in ALL_GAME_CODES:
            summary, recent, figure = gradio_app.fetch_data(game, lambda *a, **k: None)
            assert summary.startswith("✅"), summary
            assert recent is not None
            local, _, _ = gradio_app.load_local_data(game, lambda *a, **k: None)
            assert "未联网" in local, local
            training, loss_figure = gradio_app.train(game, 30, 1, 0.001, 64, 128,
                use_attention=False, n_ensemble=2 if game == "pls" else 1,
                progress=lambda *a, **k: None)
            assert "训练档案已保存" in training, training
            probabilities, _ = _session_probabilities(game, gradio_app._state)
            gradio_app._state.update({f"{game}_{suffix}": None for suffix in
                ("df", "features", "labels", "scaler", "train_info")})
            with patch.object(gradio_app, "train_model", side_effect=AssertionError("Restore must not retrain")):
                restored = gradio_app.restore_saved_session(game, lambda *a, **k: None)
                assert "已恢复" in restored[3], restored
            restored_probs, _ = _session_probabilities(game, gradio_app._state)
            import numpy as np
            for head in probabilities:
                np.testing.assert_array_equal(probabilities[head], restored_probs[head])
            for play in GAME_PLAY_OPTIONS[game]:
                assert len(generate_portfolio(game, play, 5, "model", probs=probabilities)["tickets"]) == 5
            offline_runs.append({"game": game, "offline_data": True, "epochs": 1,
                                 "training_and_prediction": True, "restore_without_training": True,
                                 "restored_probabilities_identical": True})
            print(f"OFFLINE_FETCH_TRAIN_RESTORE_PREDICT_OK {game}", flush=True)
            import matplotlib.pyplot as plt
            plt.close("all")

after = hashes()
assert before == after, "Release user data/model/result files changed during QA"
report = {"created": datetime.now().isoformat(), "executable": sys.executable, "sys_path": sys.path,
          "unit_tests": result.testsRun, "existing_model_runs": model_runs,
          "offline_runs": offline_runs, "protected_files": len(before),
          "protected_hashes_unchanged": True, "clean_windows_vm_tested": False,
          "offline_test_method": "Non-loopback socket connections blocked; requests mocked to fail during data/training tests. Not a physical network disconnect."}
path = QA / "portable_acceptance.json"
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"PORTABLE_ACCEPTANCE_OK {path}", flush=True)
