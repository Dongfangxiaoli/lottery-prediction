"""Read-only model smoke + fixed-seed synthetic coverage comparison; no training."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from feature_engineering import build_features_ssq, build_features_dlt, build_features_digit
from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from portfolio_engine import generate_portfolio
from portfolio_ui import _session_probabilities
from run_top5_experiment import infer_legacy_architecture

ROOT = Path(__file__).resolve().parent


def protected_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("data", "models") for path in sorted((ROOT / folder).iterdir()) if path.is_file()}


def main():
    before = protected_hashes()
    torch.set_num_threads(2)
    model_runs, simulations = [], []
    for game in ALL_GAME_CODES:
        df = pd.read_csv(ROOT / "data" / f"{game}_history.csv", dtype={"issue": str})
        info = infer_legacy_architecture(game)
        if game == "ssq":
            features, _, _ = build_features_ssq(df, fit_end=len(df) - 50)
        elif game == "dlt":
            features, _, _ = build_features_dlt(df, fit_end=len(df) - 50)
        else:
            features, _, _ = build_features_digit(df, game, fit_end=len(df) - 50)
        state = {f"{game}_features": features, f"{game}_train_info": info, f"{game}_df": df}
        probs, source = _session_probabilities(game, state)
        for play in GAME_PLAY_OPTIONS[game]:
            start = perf_counter()
            result = generate_portfolio(game, play, 10, "model", probs=probs)
            assert len(result["tickets"]) == 10
            model_runs.append({"game": game, "play": play, "count": 10,
                               "seconds": perf_counter() - start, **source,
                               "exact_jackpot_probability": result["exact_jackpot_probability"],
                               "legacy_metadata_verified": False})
        print(f"MODEL_INFERENCE_OK {game}", flush=True)
    for game, target in (("ssq", 6), ("dlt", 7), ("qxc", 6)):
        start = perf_counter()
        result = generate_portfolio(game, n_tickets=10, strategy="coverage", target_prize=target,
                                    seed=42, n_eval=10000)
        result["elapsed_seconds"] = perf_counter() - start
        simulations.append(result)
        a, b = result["comparison"]
        print(f"SIMULATION {game} {a['hit_rate']:.4%} vs {b['hit_rate']:.4%}; "
              f"delta={a['paired_delta_vs_uniform']:.4%}", flush=True)
    after = protected_hashes()
    if before != after:
        raise RuntimeError("Protected model/history input changed during verification")
    report = {"version": "0.7.0", "created_at": datetime.now().isoformat(),
              "scope": "software_smoke_and_uniform_draw_simulation_not_future_prediction",
              "protected_file_count": len(before), "protected_hashes_unchanged": True,
              "sha256": before, "model_smoke": model_runs, "coverage_simulations": simulations,
              "interpretation": "每例仅比较一份固定随机基线；不证明真实最高奖预测优势或盈利。"}
    path = ROOT / "docs" / "report" / (datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + "_v070_verification.json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"REPORT {path}", flush=True)


if __name__ == "__main__":
    main()
