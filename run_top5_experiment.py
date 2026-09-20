"""只读既有模型/数据的固定五注对照；旧模型缺少训练清单，结果仅为探索性。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import torch

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from jackpot_evaluation import backtest_top_five, format_top_five_backtest

ROOT = Path(__file__).resolve().parent


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def infer_legacy_architecture(game):
    weights = torch.load(ROOT / "models" / f"{game}_lstm.pth", map_location="cpu", weights_only=True)
    layers = [int(re.fullmatch(r"lstm\.weight_ih_l(\d+)", key).group(1))
              for key in weights if re.fullmatch(r"lstm\.weight_ih_l(\d+)", key)]
    if not layers:
        raise ValueError(f"{game}: 无法确认已有模型结构")
    return {"seq_len": 30, "holdout_size": 50, "input_size": weights["lstm.weight_ih_l0"].shape[1],
            "hidden_size": weights["lstm.weight_hh_l0"].shape[1], "num_layers": max(layers) + 1,
            "dropout": 0.3, "bidirectional": any("_reverse" in key for key in weights),
            "use_attention": any(key.startswith("attention.") for key in weights),
            "attn_heads": 4, "n_ensemble": 1, "ensemble_seeds": [],
            "legacy_metadata_verified": False,
            "metadata_note": "结构来自权重；窗口30/冻结50仅沿用项目旧记录，未由模型清单独立证明"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--n-test", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = [ROOT / folder / f"{game}_{suffix}" for game in ALL_GAME_CODES
              for folder, suffix in (("models", "lstm.pth"), ("data", "history.csv"))]
    hashes = {str(path.relative_to(ROOT)): _digest(path) for path in inputs}
    results = []
    for game in ALL_GAME_CODES:
        frame = pd.read_csv(ROOT / "data" / f"{game}_history.csv", dtype={"issue": str})
        info = infer_legacy_architecture(game)
        for play in GAME_PLAY_OPTIONS[game]:
            print(f"正在对照 {game} {play}，每期5注，{args.n_test}期...", flush=True)
            result = backtest_top_five(frame, info, game, play, args.n_test)
            result["legacy_metadata_verified"] = False
            result["train_info_assumptions"] = info
            result["data_cutoff_issue"] = str(frame.iloc[-1]["issue"])
            result["data_cutoff_date"] = str(frame.iloc[-1]["date"])
            results.append(result)
            print(format_top_five_backtest(result), flush=True)
    after = {str(path.relative_to(ROOT)): _digest(path) for path in inputs}
    if after != hashes:
        raise RuntimeError("原模型或数据在评估期间发生变化，本次结果不予接受")
    report = {"created_at": datetime.now().isoformat(), "evaluation_scope": "exploratory_existing_holdout",
              "legacy_metadata_verified": False, "protected_hashes_unchanged": True,
              "input_sha256": hashes, "results": results}
    output = args.output or ROOT / "docs" / "report" / (datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + "_top5_experiment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"REPORT: {output}", flush=True)


if __name__ == "__main__":
    main()
