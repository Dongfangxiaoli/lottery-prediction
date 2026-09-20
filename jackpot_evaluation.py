"""Frozen-holdout, exact-jackpot comparison for five-ticket strategies.

This deliberately measures only whether at least one of five complete tickets
matches a draw.  It is exploratory: a holdout repeatedly inspected while
developing a selector is not evidence of future predictive advantage.
"""
from __future__ import annotations

from math import comb, sqrt
from typing import Callable

import numpy as np
import pandas as pd

from backtest import _validate_backtest_inputs
from feature_engineering import build_features_ssq, build_features_dlt, build_features_digit
from game_config import DIGIT_GAME_CONFIGS, game_code, game_name, is_digit_game, normalize_play
from predictor import (
    _empirical_frequency, _get_probabilities, _get_probabilities_ensemble, _load_model,
    _sample_pl3_play, _weighted_sample_digits, _weighted_sample_dlt_merged,
    _weighted_sample_ssq_merged,
)

N_TICKETS = 5
EVALUATION_SCOPE = "exploratory_existing_holdout"
CONCLUSION = "未验证真实头奖优势；重复查看的留出集仅用于探索，最终需未来新开奖"


def _ticket_key(ticket: dict, game: str, play: str) -> tuple:
    if game == "ssq":
        return tuple(sorted(ticket["red"])) + (int(ticket["blue"]),)
    if game == "dlt":
        return tuple(sorted(ticket["front"])) + tuple(sorted(ticket["back"]))
    digits = tuple(map(int, ticket["digits"]))
    return tuple(sorted(digits)) if game == "pls" and play != "直选" else digits


def _require_five_unique(tickets: list[dict], game: str, play: str, label: str) -> list[dict]:
    if len(tickets) != N_TICKETS:
        raise RuntimeError(f"{label}必须恰好生成 {N_TICKETS} 注，实际为 {len(tickets)} 注。")
    keys = [_ticket_key(ticket, game, play) for ticket in tickets]
    if len(set(keys)) != N_TICKETS:
        raise RuntimeError(f"{label}生成了重复整票，无法作为五注比较。")
    return tickets


def _is_jackpot(ticket: dict, draw: np.ndarray | dict, game: str, play: str) -> bool:
    if game == "ssq":
        return set(ticket["red"]) == set(draw["red"]) and int(ticket["blue"]) == draw["blue"]
    if game == "dlt":
        return set(ticket["front"]) == set(draw["front"]) and set(ticket["back"]) == set(draw["back"])
    values = tuple(map(int, ticket["digits"]))
    actual = tuple(map(int, draw))
    return tuple(sorted(values)) == tuple(sorted(actual)) if game == "pls" and play != "直选" else values == actual


def _uniform_tickets(game: str, play: str, rng: np.random.Generator) -> list[dict]:
    """Uniformly draw five legal, distinct tickets in the relevant ticket space."""
    result, seen = [], set()
    while len(result) < N_TICKETS:
        if game == "ssq":
            item = {"red": sorted(map(int, rng.choice(np.arange(1, 34), 6, replace=False))),
                    "blue": int(rng.integers(1, 17))}
        elif game == "dlt":
            item = {"front": sorted(map(int, rng.choice(np.arange(1, 36), 5, replace=False))),
                    "back": sorted(map(int, rng.choice(np.arange(1, 13), 2, replace=False)))}
        elif game == "pls" and play != "直选":
            if play == "组选3":
                repeated, single = rng.choice(10, 2, replace=False)
                item = {"digits": sorted([int(repeated), int(repeated), int(single)])}
            else:
                item = {"digits": sorted(map(int, rng.choice(10, 3, replace=False)))}
        else:
            item = {"digits": [int(rng.integers(0, count)) for count in DIGIT_GAME_CONFIGS[game]["classes"]]}
        key = _ticket_key(item, game, play)
        if key not in seen:
            seen.add(key); result.append(item)
    return result


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for period-level hits (not five-ticket pseudo-samples)."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _random_period_probability(game: str, play: str) -> float:
    if game == "ssq":
        return N_TICKETS / (comb(33, 6) * 16)
    if game == "dlt":
        return N_TICKETS / (comb(35, 5) * comb(12, 2))
    if game == "pls" and play == "组选3":
        return 15 / 1000
    if game == "pls" and play == "组选6":
        return 30 / 1000
    return N_TICKETS / int(np.prod(DIGIT_GAME_CONFIGS[game]["classes"]))


def _feature_builder(game: str) -> Callable:
    return {"ssq": build_features_ssq, "dlt": build_features_dlt}.get(game, build_features_digit)


def _build_features(df: pd.DataFrame, game: str, fit_end: int):
    builder = _feature_builder(game)
    return builder(df, fit_end=fit_end) if game in ("ssq", "dlt") else builder(df, game, fit_end=fit_end)


def _probabilities_for_window(model, features, game: str, train_info: dict, seeds: list[int]):
    seq_len, input_size = int(train_info["seq_len"]), int(train_info["input_size"])
    if seeds:
        return _get_probabilities_ensemble(
            game, features, seq_len, input_size, train_info["hidden_size"], train_info["num_layers"],
            train_info["dropout"], train_info.get("bidirectional", False), train_info.get("use_attention", True),
            train_info.get("attn_heads", 4), seeds)
    return _get_probabilities(model, features, seq_len)


def _legacy_tickets(probs, df: pd.DataFrame, idx: int, game: str, play: str, rng):
    if game == "ssq":
        red_cols = [f"red{i}" for i in range(1, 7)]
        main = df[red_cols].to_numpy(dtype=int)[:idx]
        side = df["blue"].to_numpy(dtype=int)[:idx].reshape(-1, 1)
        return _weighted_sample_ssq_merged(probs, N_TICKETS, 1.0, 0.0,
                                           _empirical_frequency(main, 33), _empirical_frequency(side, 16),
                                           1.0, 0.0, 0.0, main, side, rng=rng)
    if game == "dlt":
        main = df[[f"front{i}" for i in range(1, 6)]].to_numpy(dtype=int)[:idx]
        side = df[[f"back{i}" for i in range(1, 3)]].to_numpy(dtype=int)[:idx]
        return _weighted_sample_dlt_merged(probs, N_TICKETS, 1.0, 0.0,
                                           _empirical_frequency(main, 35), _empirical_frequency(side, 12),
                                           1.0, 0.0, 0.0, main, side, rng=rng)
    history = df[list(DIGIT_GAME_CONFIGS[game]["digits"])].to_numpy(dtype=int)[:idx]
    return (_sample_pl3_play(probs, play, N_TICKETS, 1.0, 0.0, history, 1.0, rng)
            if game == "pls" and play != "直选"
            else _weighted_sample_digits(probs, game, N_TICKETS, 1.0, 0.0, history, 1.0, rng))


def _draw_at(df: pd.DataFrame, idx: int, game: str):
    if game == "ssq":
        return {"red": [int(df.iloc[idx][f"red{i}"]) for i in range(1, 7)], "blue": int(df.iloc[idx]["blue"])}
    if game == "dlt":
        return {"front": [int(df.iloc[idx][f"front{i}"]) for i in range(1, 6)],
                "back": [int(df.iloc[idx][f"back{i}"]) for i in range(1, 3)]}
    return df.iloc[idx][list(DIGIT_GAME_CONFIGS[game]["digits"])].to_numpy(dtype=int)


def backtest_top_five(df: pd.DataFrame, train_info: dict, game: str, play: str | None = None,
                      n_test: int = 50) -> dict:
    """Compare exact joint top-five, old weighted samples, and uniform five-ticket baseline."""
    game = game_code(game)
    effective_play = normalize_play(game, play) if is_digit_game(game) else ""
    n, n_test, _, holdout_size = _validate_backtest_inputs(df, train_info, n_test, N_TICKETS)
    features, _, _ = _build_features(df, game, n - holdout_size)
    seq_len = int(train_info["seq_len"])
    seeds = (train_info.get("ensemble_seeds") or [] if train_info.get("n_ensemble", 1) > 1 else [])
    model = None if seeds else _load_model(
        game, int(train_info["input_size"]), train_info["hidden_size"], train_info["num_layers"],
        train_info["dropout"], train_info.get("bidirectional", False), train_info.get("use_attention", True),
        train_info.get("attn_heads", 4))

    strategies = {name: {"rows": []} for name in ("joint_top5", "legacy_weighted", "uniform_random")}
    # Local import keeps this module importable until the selector is added, while making its absence explicit on use.
    from jackpot_selection import select_top_five

    for idx in range(n - n_test, n):
        window = features[:idx]  # the target draw at idx is strictly excluded
        if len(window) < seq_len:
            raise RuntimeError("冻结回测窗口不足，不能隐式跳过目标期。")
        probs = _probabilities_for_window(model, window, game, train_info, seeds)
        candidates = {
            "joint_top5": select_top_five(game, probs, effective_play or None),
            "legacy_weighted": _legacy_tickets(probs, df, idx, game, effective_play, np.random.default_rng(42 + idx)),
            "uniform_random": _uniform_tickets(game, effective_play, np.random.default_rng(42 + idx)),
        }
        draw = _draw_at(df, idx, game)
        issue = str(df.iloc[idx]["issue"])
        for name, tickets in candidates.items():
            tickets = _require_five_unique(tickets, game, effective_play, name)
            hit = any(_is_jackpot(ticket, draw, game, effective_play) for ticket in tickets)
            strategies[name]["rows"].append({"期号": issue, "目标索引": idx, "最高奖整注命中": bool(hit),
                                               "每期注数": N_TICKETS})

    theoretical = _random_period_probability(game, effective_play)
    for item in strategies.values():
        rows = item["rows"]
        wins = sum(row["最高奖整注命中"] for row in rows)
        tested = len(rows)
        item["summary"] = {"winning_periods": wins, "n_test": tested, "hit_rate": wins / tested if tested else 0.0,
                           "wilson95": _wilson_interval(wins, tested), "tickets_per_period": N_TICKETS,
                           "random_theoretical_period_probability": theoretical}
    result = {"game": game, "game_name": game_name(game), "play": effective_play or None, "n_test": n_test,
              "legacy_metadata_verified": train_info.get("legacy_metadata_verified", True),
              "holdout_size": holdout_size, "evaluation_scope": EVALUATION_SCOPE, "conclusion": CONCLUSION,
              "strategies": strategies, "rows": {key: value["rows"] for key, value in strategies.items()},
              "summary": {key: value["summary"] for key, value in strategies.items()}}
    result.update(strategies)  # convenient stable access for callers/tests
    return result


def format_top_five_backtest(result: dict) -> str:
    """Short UI-facing report which never promotes a holdout hit as predictive proof."""
    play = f"（{result['play']}）" if result.get("play") else ""
    lines = [f"{result.get('game_name', result['game'])}{play} 最高奖五注回测", "=" * 42,
             f"冻结留出期: {result['n_test']}；每期固定 {N_TICKETS} 注。"]
    for label, display in (("joint_top5", "联合 Top-5"), ("legacy_weighted", "旧加权采样"),
                           ("uniform_random", "均匀随机")):
        summary = result["summary"][label]
        low, high = summary["wilson95"]
        lines.append(f"{display}: 最高奖命中期 {summary['winning_periods']}/{summary['n_test']} "
                     f"({summary['hit_rate']:.4%}; Wilson95% {low:.4%}–{high:.4%})")
    p = result["summary"]["uniform_random"]["random_theoretical_period_probability"]
    lines.extend([f"五注均匀随机理论期命中概率: {p:.8%}", "", f"范围: {result['evaluation_scope']}",
                  f"结论: {result['conclusion']}"])
    if not result.get("legacy_metadata_verified", True):
        lines.append("旧模型无独立训练清单，训练截止边界未由权重证明；不得把该结果称为新的严格样本外证据。")
    return "\n".join(lines)
