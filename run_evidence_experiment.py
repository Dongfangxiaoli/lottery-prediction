"""Fixed-configuration chronological research; never retrospectively proves an edge.

Run with the project Python. New artifacts are isolated under evidence_runs;
the existing data/models/results and portable releases are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import torch

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS, game_name
from jackpot_evaluation import _build_features, _draw_at
from jackpot_selection import select_top_k
from local_history import read_local_history
from lstm_model import build_game_model, device
from portfolio_engine import (_canonical_ticket, _outcome_count, _permutation_count,
                              _prize, _uniform_tickets)
from predictor import _get_probabilities
import train as training

ROOT = Path(__file__).resolve().parent
EVIDENCE_ROOT = ROOT / "evidence_runs"
SOURCE_FILES = ("run_evidence_experiment.py", "feature_engineering.py", "train.py",
                "lstm_model.py", "game_config.py", "predictor.py", "jackpot_selection.py",
                "jackpot_evaluation.py", "portfolio_engine.py", "local_history.py")
TRAINING = dict(seq_len=30, num_epochs=30, batch_size=128, learning_rate=0.001,
                hidden_size=64, num_layers=2, dropout=0.3, bidirectional=False,
                use_attention=False, attn_heads=4, patience=15, n_ensemble=1, holdout_size=0)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def baseline_tickets(game, play, issue, count=5):
    seed = int.from_bytes(hashlib.sha256(f"evidence-v1|{game}|{play}|{issue}".encode()).digest()[:8], "big")
    return _uniform_tickets(game, play, count, np.random.default_rng(seed))


def feature_training_boundary(prefix_rows, seq_len=30):
    split = int(0.8 * (prefix_rows - seq_len))
    if split < 1 or split >= prefix_rows - seq_len:
        raise ValueError("训练前缀太短，无法保留独立早停段。")
    # Last training target is seq_len+split-1; its window ends one row earlier.
    return seq_len + split - 1


def load_isolated_model(game, info, path):
    model = build_game_model(game, info["input_size"], info["hidden_size"], info["num_layers"],
                             info["dropout"], info["bidirectional"], info["use_attention"], info["attn_heads"])
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


def evaluate_windows(game, df, features, model, seq_len, start, count=5):
    result = {play: [] for play in GAME_PLAY_OPTIONS[game]}
    for idx in range(start, len(df)):
        # Exclude the target row, including its own features, from inference.
        probs = _get_probabilities(model, features[idx - seq_len:idx], seq_len)
        draw = _draw_at(df, idx, game)
        issue = str(df.iloc[idx]["issue"])
        for play in GAME_PLAY_OPTIONS[game]:
            selected = [_canonical_ticket(t, game, play) for t in select_top_k(game, probs, count, play=play)]
            baseline = baseline_tickets(game, play, issue, count)
            result[play].append({"issue": issue, "date": str(df.iloc[idx]["date"])[:10],
                                 "input_last_issue": str(df.iloc[idx - 1]["issue"]),
                                 "target_index": idx, "model_tickets": selected,
                                 "baseline_tickets": baseline,
                                 "model_hit": any(_prize(t, draw, game, play) == 1 for t in selected),
                                 "baseline_hit": any(_prize(t, draw, game, play) == 1 for t in baseline)})
        if (idx - start + 1) % 100 == 0:
            print(f"EVALUATED {game} {idx-start+1}/{len(df)-start}", flush=True)
    return result


def run_experiment(output=None, n_test=300):
    if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test < 1:
        raise ValueError("n_test必须为正整数。")
    output = Path(output) if output else EVIDENCE_ROOT / datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "snapshots").mkdir()
    protected = [p for folder in ("data", "models", "results") for p in (ROOT / folder).rglob("*") if p.is_file()]
    protected_hashes = {str(p.relative_to(ROOT)): digest(p) for p in sorted(protected)}
    snapshots = {}
    for game in ALL_GAME_CODES:
        src = ROOT / "data" / f"{game}_history.csv"
        shutil.copy2(src, output / "snapshots" / src.name)
        snapshots[game] = digest(src)
    protocol = {"schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "historical_exploration_not_prospective", "primary_strategy": "lstm_joint_topk",
                "control": "uniform_unique_sha256_seed_v1", "n_tickets": 5, "n_test": n_test,
                "family_size": 7, "alpha": 0.05, "training": TRAINING,
                "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
                "data_sha256": snapshots, "protected_sha256": protected_hashes,
                "notes": ["This history may already have been inspected: no independent confirmation.",
                          "No grid search; no seed selection; all seven results must be reported.",
                          "Scaler fits only training-input rows, excluding validation and outer test.",
                          "Weights fixed throughout test; earlier test draws may enter later input windows.",
                          "Five simulated tickets per draw is a measurement setting, not purchase advice."]}
    write_new(output / "protocol.json", protocol)
    print(f"PROTOCOL_FROZEN {output}", flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    all_results = []
    for game in ALL_GAME_CODES:
        df = read_local_history(game, output / "snapshots")
        start = len(df) - n_test
        fit_end = feature_training_boundary(start)
        prefix = df.iloc[:start].copy()
        train_features, labels, _ = _build_features(prefix, game, fit_end)
        print(f"TRAINING {game} prefix={start} test={n_test} scaler_fit_end={fit_end}", flush=True)
        def progress(_, message):
            print(f"{game} {message}", flush=True)
        with patch.object(training, "MODELS_DIR", str(output / "models")):
            info = training.train_model(train_features, labels, game, progress_callback=progress, **TRAINING)
        features, _, _ = _build_features(df, game, fit_end)
        if not np.array_equal(train_features, features[:start]):
            raise RuntimeError("评估后缀改变了训练前缀特征，拒绝继续。")
        model = load_isolated_model(game, info, output / "models" / f"{game}_lstm.pth")
        rows = evaluate_windows(game, df, features, model, info["seq_len"], start)
        info["model_path"] = Path(info["model_path"]).name
        item = {"game": game, "game_name": game_name(game), "n_total": len(df),
                "train_prefix_rows": start, "feature_fit_end": fit_end,
                "weight_sha256": digest(output / "models" / f"{game}_lstm.pth"),
                "train_info": info, "training_last_issue": str(df.iloc[start - 1]["issue"]),
                "data_cutoff_issue": str(df.iloc[-1]["issue"]),
                "data_cutoff_date": str(df.iloc[-1]["date"])[:10], "plays": rows}
        write_new(output / f"{game}_result.json", item)
        all_results.append(item)
        print(f"GAME_DONE {game}", flush=True)
    if any(digest(ROOT / name) != value for name, value in protected_hashes.items()):
        raise RuntimeError("原始输入文件发生变化，实验无效。")
    if any(digest(ROOT / name) != value for name, value in protocol["source_sha256"].items()):
        raise RuntimeError("实验运行期间核心代码发生变化，实验无效。")
    summary = []
    for item in all_results:
        for play, rows in item["plays"].items():
            p0 = protocol["n_tickets"] * _permutation_count(item["game"], play) / _outcome_count(item["game"])
            summary.append({"game": item["game"], "play": play, "periods": len(rows),
                            "model_wins": sum(r["model_hit"] for r in rows),
                            "baseline_wins": sum(r["baseline_hit"] for r in rows), "p0": p0})
    report = {"scope": protocol["scope"], "protocol_sha256": digest(output / "protocol.json"),
              "protected_files": len(protected_hashes), "protected_unchanged": True, "summary": summary,
              "conclusion": "历史探索结果，不是开奖前预测记录，不能证明真实一等奖优势。"}
    write_new(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-test", type=int, default=300)
    args = parser.parse_args()
    run_experiment(args.output, args.n_test)
