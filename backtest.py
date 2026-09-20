"""
历史回测模块

模型训练时先冻结历史末端 holdout，回测只在未参与训练和早停的区间内逐期预测，
统计实际命中数分布，并与纯随机基准对比，客观评估预测效果。

注意：
  - 彩票为独立随机事件，回测仅用于评估"预测策略相对随机的优劣"，
    不代表未来表现。
  - 回测复用冻结 holdout 之前训练的模型；测试过程中不重训，但允许使用已揭晓的
    较早测试期作为后续期的输入。这是严格的 frozen-model 样本外评估。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import build_features_ssq, build_features_dlt, build_features_digit
from game_config import (DIGIT_GAME_CONFIGS, game_name as configured_game_name,
                         is_digit_game, normalize_play)
from predictor import (_load_model, _get_probabilities, _get_probabilities_ensemble,
                       _weighted_sample_ssq, _weighted_sample_dlt,
                       _weighted_sample_ssq_merged, _weighted_sample_dlt_merged,
                       _weighted_sample_digits, _sample_pl3_play, _empirical_frequency)


def _hit_ssq(pred: dict, draw: dict) -> tuple[int, bool]:
    """返回 (红球命中数, 蓝球是否命中)"""
    red_hits = len(set(pred["red"]) & set(draw["reds"]))
    blue_hit = pred["blue"] == draw["blue"]
    return red_hits, blue_hit


def _hit_dlt(pred: dict, draw: dict) -> tuple[int, int]:
    """返回 (前区命中数, 后区命中数)"""
    f_hits = len(set(pred["front"]) & set(draw["fronts"]))
    b_hits = len(set(pred["back"]) & set(draw["backs"]))
    return f_hits, b_hits


def _prize_level_ssq(red_hits: int, blue_hit: bool, special_prize: bool = False) -> int:
    """双色球常规六奖级；特别规定生效时 7=福运奖。"""
    if red_hits == 6 and blue_hit:
        return 1
    if red_hits == 6:
        return 2
    if red_hits == 5 and blue_hit:
        return 3
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return 4
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return 5
    if blue_hit:
        return 6
    if special_prize and red_hits == 3:
        return 7
    return 0


def _ssq_special_prize_flags(df: pd.DataFrame) -> list[bool]:
    """按现行规则的奖池阈值重放每期是否执行福运奖。"""
    active = False
    flags = []
    dates = pd.to_datetime(df["date"])
    pool_values = df["pool"] if "pool" in df else pd.Series(0, index=df.index)
    pools = pd.to_numeric(pool_values, errors="coerce").fillna(0)
    for date, pool in zip(dates, pools):
        if date < pd.Timestamp("2026-02-01"):
            flags.append(False)
            continue
        if not active and pool >= 1_500_000_000:
            active = True
        flags.append(active)
        if active and pool < 300_000_000:
            active = False
    return flags


def _prize_level_dlt(f_hits: int, b_hits: int) -> int:
    """大乐透现行七奖级（自 26014 期起，0=未中奖）。"""
    if f_hits == 5 and b_hits == 2:
        return 1
    if f_hits == 5 and b_hits == 1:
        return 2
    if f_hits == 5 or (f_hits == 4 and b_hits == 2):
        return 3
    if f_hits == 4 and b_hits == 1:
        return 4
    if f_hits == 4 or (f_hits == 3 and b_hits == 2):
        return 5
    if (f_hits == 3 and b_hits == 1) or (f_hits == 2 and b_hits == 2):
        return 6
    if f_hits == 3 or (f_hits == 2 and b_hits == 1) or (f_hits == 1 and b_hits == 2) or b_hits == 2:
        return 7
    return 0


def _random_baseline_hits(game: str, play: str = None) -> dict:
    """均匀随机选号的精确理论期望，无蒙特卡洛噪声。"""
    if game == "ssq":
        return {
            "mean_red_hits": 6 * 6 / 33,
            "blue_hit_rate": 1 / 16,
        }
    if is_digit_game(game):
        classes = DIGIT_GAME_CONFIGS[game]["classes"]
        effective_play = normalize_play(game, play)
        exact_rate = ({"直选": 0.001, "组选3": 0.003, "组选6": 0.006}[effective_play]
                      if game == "pls" else float(np.prod([1 / n for n in classes])))
        return {
            "position_hit_rate": [1 / n for n in classes],
            "mean_position_hits": float(sum(1 / n for n in classes)),
            "exact_hit_rate": exact_rate,
            "play": effective_play,
        }
    return {
        "mean_front_hits": 5 * 5 / 35,
        "mean_back_hits": 2 * 2 / 12,
    }


def _mean_ci(values: list[float], upper: float) -> tuple[float, float]:
    """按开奖期聚合后的均值 95% 正态近似区间。"""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        mean = float(arr.mean()) if len(arr) else 0.0
        return mean, mean
    mean = float(arr.mean())
    margin = 1.96 * float(arr.std(ddof=1)) / np.sqrt(len(arr))
    return max(0.0, mean - margin), min(upper, mean + margin)


def _validate_backtest_inputs(df: pd.DataFrame, train_info: dict,
                              n_test: int, n_groups: int) -> tuple[int, int, int, int]:
    """校验回测样本、冻结区间和每期号码组数，避免悄悄缩短请求的样本外区间。"""
    if not isinstance(train_info, dict):
        raise ValueError("缺少有效的模型训练信息，无法确认冻结样本外区间。")
    n = len(df)
    if isinstance(n_test, bool) or isinstance(n_groups, bool):
        raise ValueError("回测期数 n_test 和每期采样组数 n_groups 必须为正整数。")
    if any(isinstance(value, (float, np.floating)) and not float(value).is_integer()
           for value in (n_test, n_groups)):
        raise ValueError("回测期数 n_test 和每期采样组数 n_groups 必须为正整数。")
    try:
        n_test = int(n_test)
        n_groups = int(n_groups)
        holdout_size = int(train_info.get("holdout_size", 0))
        seq_len = int(train_info["seq_len"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("回测参数或模型训练信息无效。") from exc
    if n_groups < 1:
        raise ValueError("每期采样组数 n_groups 必须至少为 1。")
    if n_test < 1:
        raise ValueError("回测期数 n_test 必须至少为 1。")
    if seq_len < 1:
        raise ValueError("模型输入窗口 seq_len 必须至少为 1。")
    if n < 2 or n_test >= n:
        raise ValueError(f"历史数据仅有 {n} 期，n_test 必须小于历史期数。")
    if holdout_size < 1 or holdout_size > n - seq_len:
        raise ValueError(
            f"冻结期数 holdout_size 必须在 1 到 {max(0, n - seq_len)} 之间，实际为 {holdout_size}。"
        )
    if n_test > holdout_size:
        raise ValueError(
            f"模型仅冻结了 {holdout_size} 期，不能严格回测 {n_test} 期；请用新版流程重新训练。"
        )
    # 最早目标期的输入窗口必须完整，并且窗口严格止于目标期之前。
    if n - n_test < seq_len:
        raise ValueError(
            f"历史数据不足：回测 {n_test} 期且模型窗口为 {seq_len} 期时，最早目标期没有完整输入窗口。"
        )
    return n, n_test, n_groups, holdout_size


def backtest_ssq(
    df: pd.DataFrame,
    train_info: dict,
    n_test: int = 20,
    n_groups: int = 5,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    top_p: float = 1.0,
    missing_alpha: float = 0.0,
    hot_alpha: float = 0.0,
    sampling_mode: str = "merged",
) -> dict:
    """
    双色球回测：用末端 n_test 期作为测试集。
    对每期：用截至该期前的数据构建特征 + 模型推理 + 采样 n_groups 组，
    统计该期最佳命中（取 n_groups 中命中最多的一组）。
    返回回测统计字典。
    """
    if sampling_mode not in ("merged", "per_position"):
        raise ValueError(f"不支持的采样模式: {sampling_mode}")
    n, n_test, n_groups, holdout_size = _validate_backtest_inputs(
        df, train_info, n_test, n_groups)

    # 缩放器也只能在外层训练前缀拟合。
    features, _, _ = build_features_ssq(df, fit_end=n - holdout_size)
    seq_len = train_info["seq_len"]
    input_size = train_info["input_size"]

    ensemble_seeds = train_info.get("ensemble_seeds") or []
    use_ensemble = train_info.get("n_ensemble", 1) > 1 and bool(ensemble_seeds)
    model = None
    if not use_ensemble:
        model = _load_model("ssq", input_size, train_info["hidden_size"],
                            train_info["num_layers"], train_info["dropout"],
                            train_info.get("bidirectional", False),
                            train_info.get("use_attention", True),
                            train_info.get("attn_heads", 4))

    red_cols = [f"red{i}" for i in range(1, 7)]
    rows = []
    red_hit_dist = []
    blue_hit_cnt = 0
    best_red_total = 0.0
    per_ticket_red_all = []   # 无偏：所有期的所有组的红球命中
    per_ticket_blue_all = []  # 无偏：所有期的所有组的蓝球命中
    per_draw_red_means = []
    per_draw_blue_rates = []
    rng = np.random.default_rng(42)
    special_prize_flags = _ssq_special_prize_flags(df)

    for target_idx in range(n - n_test, n):
        if target_idx < seq_len:
            continue
        # 目标期本身绝不进入输入；较早且已经开奖的 holdout 期可用于预测后续期。
        feat_window = features[:target_idx]
        # 推理概率（用末 seq_len 期）
        if len(feat_window) < seq_len:
            continue
        if use_ensemble:
            probs = _get_probabilities_ensemble(
                "ssq", feat_window, seq_len, input_size, train_info["hidden_size"],
                train_info["num_layers"], train_info["dropout"],
                train_info.get("bidirectional", False),
                train_info.get("use_attention", True),
                train_info.get("attn_heads", 4), ensemble_seeds)
        else:
            probs = _get_probabilities(model, feat_window, seq_len)

        hist_main = df[red_cols].values[:target_idx]
        hist_side = df["blue"].values[:target_idx].reshape(-1, 1)
        red_freq = _empirical_frequency(hist_main, 33)
        blue_freq = _empirical_frequency(hist_side, 16)

        # 采样（不保存到文件，直接调用采样器）
        if sampling_mode == "merged":
            numbers = _weighted_sample_ssq_merged(
                probs, n_groups, temperature, freq_alpha, red_freq, blue_freq, top_p,
                missing_alpha, hot_alpha, hist_main, hist_side, rng=rng)
        else:
            numbers = _weighted_sample_ssq(
                probs, n_groups, temperature, rng=rng)
        if len(numbers) != n_groups:
            raise RuntimeError(f"只生成了 {len(numbers)}/{n_groups} 组号码，无法公平回测。")

        draw = {
            "reds": [int(df.iloc[target_idx][c]) for c in red_cols],
            "blue": int(df.iloc[target_idx]["blue"]),
            "issue": str(df.iloc[target_idx]["issue"]),
        }

        # 统计该期命中：同时记录"最佳组"(有偏置，反映多注覆盖能力) 和 "单注平均"(无偏，反映预测本身)
        best_red = 0
        best_blue = False
        best_level = 0
        per_ticket_red = []  # 无偏：每组的红球命中
        per_ticket_blue = []  # 无偏：每组蓝球是否命中
        for pred in numbers:
            rh, bh = _hit_ssq(pred, draw)
            per_ticket_red.append(rh)
            per_ticket_blue.append(1 if bh else 0)
            level = _prize_level_ssq(rh, bh, special_prize_flags[target_idx])
            if level > 0 and (best_level == 0 or level < best_level):
                best_level = level
            if rh > best_red:
                best_red = rh
            if bh:
                best_blue = True
        red_hit_dist.append(best_red)
        per_ticket_red_all.extend(per_ticket_red)  # 无偏汇总
        per_ticket_blue_all.extend(per_ticket_blue)
        per_draw_red_means.append(float(np.mean(per_ticket_red)))
        per_draw_blue_rates.append(float(np.mean(per_ticket_blue)))
        if best_blue:
            blue_hit_cnt += 1
        best_red_total += best_red
        rows.append({
            "期号": draw["issue"],
            "开奖红球": " ".join(f"{r:02d}" for r in draw["reds"]),
            "蓝球": f"{draw['blue']:02d}",
            "单注平均红球命中": round(float(np.mean(per_ticket_red)), 3),
            "最佳红球命中": best_red,
            "蓝球命中": "✓" if best_blue else "✗",
            "中奖等级": _level_name_ssq(best_level),
        })

    n_valid = len(rows)
    # 红球命中分布（最佳组）
    dist = {k: red_hit_dist.count(k) for k in range(7)}
    # 无偏单注命中分布
    ub_red_dist = {k: per_ticket_red_all.count(k) for k in range(7)}
    # 随机基准
    baseline = _random_baseline_hits("ssq")

    summary = {
        "game": "ssq",
        "n_test": n_valid,
        "n_groups": n_groups,
        "rows": rows,
        "red_hit_distribution": dist,
        "mean_best_red_hits": best_red_total / max(1, n_valid),
        "blue_hit_rate": blue_hit_cnt / max(1, n_valid),
        # 无偏指标（单注平均，对比随机基准才真实）
        "unbiased_red_distribution": ub_red_dist,
        "unbiased_mean_red_hits": float(np.mean(per_ticket_red_all)) if per_ticket_red_all else 0.0,
        "unbiased_blue_hit_rate": float(np.mean(per_ticket_blue_all)) if per_ticket_blue_all else 0.0,
        "red_ci95": _mean_ci(per_draw_red_means, 6.0),
        "blue_ci95": _mean_ci(per_draw_blue_rates, 1.0),
        "baseline": baseline,
        "holdout_size": holdout_size,
        "sampling_mode": sampling_mode,
        "n_ensemble": len(ensemble_seeds) if use_ensemble else 1,
        # 每期只保留最佳奖级，故此处是“中奖期数”，不是所有中奖票的注数。
        "winning_period_count": sum(1 for r in rows if r["中奖等级"] != "未中"),
    }
    return summary


def backtest_dlt(
    df: pd.DataFrame,
    train_info: dict,
    n_test: int = 20,
    n_groups: int = 5,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    top_p: float = 1.0,
    missing_alpha: float = 0.0,
    hot_alpha: float = 0.0,
    sampling_mode: str = "merged",
) -> dict:
    """大乐透回测，结构同 backtest_ssq"""
    if sampling_mode not in ("merged", "per_position"):
        raise ValueError(f"不支持的采样模式: {sampling_mode}")
    n, n_test, n_groups, holdout_size = _validate_backtest_inputs(
        df, train_info, n_test, n_groups)

    features, _, _ = build_features_dlt(df, fit_end=n - holdout_size)
    seq_len = train_info["seq_len"]
    input_size = train_info["input_size"]

    ensemble_seeds = train_info.get("ensemble_seeds") or []
    use_ensemble = train_info.get("n_ensemble", 1) > 1 and bool(ensemble_seeds)
    model = None
    if not use_ensemble:
        model = _load_model("dlt", input_size, train_info["hidden_size"],
                            train_info["num_layers"], train_info["dropout"],
                            train_info.get("bidirectional", False),
                            train_info.get("use_attention", True),
                            train_info.get("attn_heads", 4))

    front_cols = [f"front{i}" for i in range(1, 6)]
    back_cols = [f"back{i}" for i in range(1, 3)]
    rows = []
    f_hit_dist = []
    b_hit_dist = []
    best_f_total = 0.0
    best_b_total = 0.0
    per_ticket_f_all = []   # 无偏：所有期所有组的前区命中
    per_ticket_b_all = []   # 无偏：所有期所有组的后区命中
    per_draw_f_means = []
    per_draw_b_means = []
    rng = np.random.default_rng(42)

    for target_idx in range(n - n_test, n):
        if target_idx < seq_len:
            continue
        feat_window = features[:target_idx]
        if len(feat_window) < seq_len:
            continue
        if use_ensemble:
            probs = _get_probabilities_ensemble(
                "dlt", feat_window, seq_len, input_size, train_info["hidden_size"],
                train_info["num_layers"], train_info["dropout"],
                train_info.get("bidirectional", False),
                train_info.get("use_attention", True),
                train_info.get("attn_heads", 4), ensemble_seeds)
        else:
            probs = _get_probabilities(model, feat_window, seq_len)

        hist_main = df[front_cols].values[:target_idx]
        hist_side = df[back_cols].values[:target_idx]
        front_freq = _empirical_frequency(hist_main, 35)
        back_freq = _empirical_frequency(hist_side, 12)

        if sampling_mode == "merged":
            numbers = _weighted_sample_dlt_merged(
                probs, n_groups, temperature, freq_alpha, front_freq, back_freq, top_p,
                missing_alpha, hot_alpha, hist_main, hist_side, rng=rng)
        else:
            numbers = _weighted_sample_dlt(
                probs, n_groups, temperature, rng=rng)
        if len(numbers) != n_groups:
            raise RuntimeError(f"只生成了 {len(numbers)}/{n_groups} 组号码，无法公平回测。")

        draw = {
            "fronts": [int(df.iloc[target_idx][c]) for c in front_cols],
            "backs": [int(df.iloc[target_idx][c]) for c in back_cols],
            "issue": str(df.iloc[target_idx]["issue"]),
        }

        best_f = 0
        best_b = 0
        best_level = 0
        per_ticket_f = []  # 该期各组前区命中
        per_ticket_b = []  # 该期各组后区命中
        for pred in numbers:
            fh, bh = _hit_dlt(pred, draw)
            per_ticket_f.append(fh)
            per_ticket_b.append(bh)
            level = _prize_level_dlt(fh, bh)
            if level > 0 and (best_level == 0 or level < best_level):
                best_level = level
            if fh > best_f:
                best_f = fh
            if bh > best_b:
                best_b = bh
        f_hit_dist.append(best_f)
        b_hit_dist.append(best_b)
        per_ticket_f_all.extend(per_ticket_f)
        per_ticket_b_all.extend(per_ticket_b)
        per_draw_f_means.append(float(np.mean(per_ticket_f)))
        per_draw_b_means.append(float(np.mean(per_ticket_b)))
        best_f_total += best_f
        best_b_total += best_b
        rows.append({
            "期号": draw["issue"],
            "开奖前区": " ".join(f"{r:02d}" for r in draw["fronts"]),
            "后区": " ".join(f"{r:02d}" for r in draw["backs"]),
            "单注平均前区命中": round(float(np.mean(per_ticket_f)), 3),
            "最佳前区命中": best_f,
            "最佳后区命中": best_b,
            "中奖等级": _level_name_dlt(best_level),
        })

    n_valid = len(rows)
    f_dist = {k: f_hit_dist.count(k) for k in range(6)}
    b_dist = {k: b_hit_dist.count(k) for k in range(3)}
    ub_f_dist = {k: per_ticket_f_all.count(k) for k in range(6)}
    ub_b_dist = {k: per_ticket_b_all.count(k) for k in range(3)}
    baseline = _random_baseline_hits("dlt")

    summary = {
        "game": "dlt",
        "n_test": n_valid,
        "n_groups": n_groups,
        "rows": rows,
        "front_hit_distribution": f_dist,
        "back_hit_distribution": b_dist,
        "mean_best_front_hits": best_f_total / max(1, n_valid),
        "mean_best_back_hits": best_b_total / max(1, n_valid),
        # 无偏指标（单注平均，对比随机基准才真实）
        "unbiased_front_distribution": ub_f_dist,
        "unbiased_back_distribution": ub_b_dist,
        "unbiased_mean_front_hits": float(np.mean(per_ticket_f_all)) if per_ticket_f_all else 0.0,
        "unbiased_mean_back_hits": float(np.mean(per_ticket_b_all)) if per_ticket_b_all else 0.0,
        "front_ci95": _mean_ci(per_draw_f_means, 5.0),
        "back_ci95": _mean_ci(per_draw_b_means, 2.0),
        "baseline": baseline,
        "holdout_size": holdout_size,
        "sampling_mode": sampling_mode,
        "n_ensemble": len(ensemble_seeds) if use_ensemble else 1,
        # 每期只保留最佳奖级，故此处是“中奖期数”，不是所有中奖票的注数。
        "winning_period_count": sum(1 for r in rows if r["中奖等级"] != "未中"),
    }
    return summary


def _prize_level_digit(matches: list[bool], game: str) -> int:
    """数字型直选奖级；QXC 按前六位与最后一位的现行规则判定。"""
    if game not in DIGIT_GAME_CONFIGS:
        raise ValueError(f"不是数字型彩种: {game}")
    n = len(DIGIT_GAME_CONFIGS[game]["classes"])
    if len(matches) != n:
        raise ValueError(f"{game} 命中位数应为 {n}，实际为 {len(matches)}")
    hits = int(sum(bool(x) for x in matches))
    if game != "qxc":
        return 1 if hits == n else 0
    front_hits = sum(bool(x) for x in matches[:6])
    last_hit = bool(matches[6])
    if front_hits == 6 and last_hit:
        return 1
    if front_hits == 6:
        return 2
    if front_hits == 5 and last_hit:
        return 3
    if hits == 5:
        return 4
    if hits == 4:
        return 5
    if hits == 3 or (last_hit and front_hits <= 1):
        return 6
    return 0


def backtest_digit(
    df: pd.DataFrame,
    train_info: dict,
    game: str,
    n_test: int = 20,
    n_groups: int = 5,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    top_p: float = 1.0,
    play: str = None,
) -> dict:
    """排列3/排列5/7星彩严格 frozen holdout 回测。"""
    if not is_digit_game(game):
        raise ValueError(f"不是数字型彩种: {game}")
    effective_play = normalize_play(game, play)
    grouped_pls = game == "pls" and effective_play != "直选"
    n, n_test, n_groups, holdout_size = _validate_backtest_inputs(
        df, train_info, n_test, n_groups)
    features, _, _ = build_features_digit(df, game, fit_end=n - holdout_size)
    seq_len = int(train_info["seq_len"])
    input_size = int(train_info["input_size"])
    seeds = train_info.get("ensemble_seeds") or []
    use_ensemble = train_info.get("n_ensemble", 1) > 1 and bool(seeds)
    model = None if use_ensemble else _load_model(
        game, input_size, train_info["hidden_size"], train_info["num_layers"],
        train_info["dropout"], train_info.get("bidirectional", False),
        train_info.get("use_attention", True), train_info.get("attn_heads", 4))
    cols = list(DIGIT_GAME_CONFIGS[game]["digits"])
    classes = list(DIGIT_GAME_CONFIGS[game]["classes"])
    rng = np.random.default_rng(42)
    rows, hit_totals, exact_count, prize_counts = [], np.zeros(len(cols)), 0, {}
    per_draw_means = []
    for target_idx in range(n - n_test, n):
        if target_idx < seq_len:
            continue
        window = features[:target_idx]
        if use_ensemble:
            probs = _get_probabilities_ensemble(
                game, window, seq_len, input_size, train_info["hidden_size"],
                train_info["num_layers"], train_info["dropout"],
                train_info.get("bidirectional", False), train_info.get("use_attention", True),
                train_info.get("attn_heads", 4), seeds)
        else:
            probs = _get_probabilities(model, window, seq_len)
        hist = df[cols].to_numpy(dtype=int)[:target_idx]
        numbers = (_sample_pl3_play(probs, effective_play, n_groups, temperature,
                                    freq_alpha, hist, top_p, rng)
                   if grouped_pls else _weighted_sample_digits(
                       probs, game, n_groups, temperature, freq_alpha, hist, top_p, rng))
        if len(numbers) != n_groups:
            raise RuntimeError(f"只生成了 {len(numbers)}/{n_groups} 组号码。")
        draw = df.iloc[target_idx][cols].to_numpy(dtype=int)
        per = [np.asarray(x["digits"], dtype=int) == draw for x in numbers]
        hits = np.asarray(per, dtype=int).sum(axis=1)
        group_hits = ([int(np.array_equal(np.sort(x["digits"]), np.sort(draw))) for x in numbers]
                      if grouped_pls else [int(value == len(cols)) for value in hits])
        best = int(max(group_hits)) if grouped_pls else int(hits.max())
        exact_count += int(sum(group_hits))
        per_draw_means.append(float(np.mean(group_hits) if grouped_pls else hits.mean()))
        for index, h in enumerate(per):
            if not grouped_pls:
                hit_totals += h
            level = 1 if grouped_pls and group_hits[index] else _prize_level_digit(h.tolist(), game)
            if level:
                prize_counts[str(level)] = prize_counts.get(str(level), 0) + 1
        row = {"期号": str(df.iloc[target_idx]["issue"]),
               "开奖号码": "".join(str(int(x)) for x in draw),
               "整注命中": "✓" if max(group_hits) else "✗"}
        row["最佳组命中"] = best if grouped_pls else None
        if not grouped_pls:
            row["最佳逐位命中"] = best
        rows.append(row)
    valid = len(rows)
    baseline = _random_baseline_hits(game, effective_play)
    summary = {
        "game": game, "n_test": valid, "n_groups": n_groups, "rows": rows,
        "play": effective_play,
        "exact_hit_rate": exact_count / max(1, valid * n_groups),
        "baseline": baseline, "prize_count": sum(prize_counts.values()),
        "prize_counts": prize_counts, "holdout_size": holdout_size,
        "sampling_mode": "grouped_joint" if grouped_pls else "per_position",
        "n_ensemble": len(seeds) if use_ensemble else 1,
    }
    if not grouped_pls:
        summary.update({
            "position_hit_rates": (hit_totals / max(1, valid * n_groups)).tolist(),
            "mean_position_hits": float(hit_totals.sum() / max(1, valid * n_groups)),
            "position_ci95": _mean_ci(per_draw_means, float(len(cols))),
        })
    return summary


def _level_name_ssq(level: int) -> str:
    return {0: "未中", 1: "一等奖", 2: "二等奖", 3: "三等奖",
            4: "四等奖", 5: "五等奖", 6: "六等奖", 7: "福运奖"}.get(level, "未中")


def _level_name_dlt(level: int) -> str:
    return {0: "未中", 1: "一等奖", 2: "二等奖", 3: "三等奖", 4: "四等奖",
            5: "五等奖", 6: "六等奖", 7: "七等奖"}.get(level, "未中")


def format_backtest_summary(summary: dict) -> str:
    """格式化回测结果为文本摘要"""
    g = summary["game"]
    game_name = configured_game_name(g) if is_digit_game(g) else ("双色球" if g == "ssq" else "大乐透")
    lines = []
    lines.append("=" * 64)
    lines.append(f"  {game_name} 样本外回测报告")
    lines.append("=" * 64)
    lines.append(
        f"回测期数: {summary['n_test']}  |  每期采样组数: {summary['n_groups']}  |  "
        f"训练冻结末端: {summary.get('holdout_size', 0)} 期"
    )
    lines.append(
        f"策略: {summary.get('sampling_mode', 'merged')}  |  "
        f"模型数: {summary.get('n_ensemble', 1)}"
    )
    lines.append("-" * 64)

    if g == "ssq":
        # 无偏核心指标（单注平均，可公平对比随机）
        ub_red = summary.get("unbiased_mean_red_hits", 0.0)
        ub_blue = summary.get("unbiased_blue_hit_rate", 0.0)
        base_red = summary["baseline"]["mean_red_hits"]
        base_blue = summary["baseline"]["blue_hit_rate"]
        lines.append("【核心：单注平均命中 vs 纯随机】（无偏，唯一公平对比）")
        red_ci = summary.get("red_ci95", (ub_red, ub_red))
        blue_ci = summary.get("blue_ci95", (ub_blue, ub_blue))
        lines.append(
            f"  红球  实测 {ub_red:.3f}（95%CI {red_ci[0]:.3f}~{red_ci[1]:.3f}）"
            f" vs 精确随机 {base_red:.3f}"
        )
        diff = ub_red - base_red
        verdict = "样本外高于随机" if red_ci[0] > base_red else (
            "样本外低于随机" if red_ci[1] < base_red else "未发现显著差异")
        lines.append(f"        差值 {diff:+.3f}  {verdict}")
        lines.append(
            f"  蓝球  实测 {ub_blue:.2%}（95%CI {blue_ci[0]:.2%}~{blue_ci[1]:.2%}）"
            f" vs 精确随机 {base_blue:.2%}"
        )
        diff_b = ub_blue - base_blue
        verdict_b = "高于随机" if blue_ci[0] > base_blue else (
            "低于随机" if blue_ci[1] < base_blue else "未发现显著差异")
        lines.append(f"        差值 {diff_b:+.2%}  {verdict_b}")
        lines.append("-" * 64)
        lines.append("【单注红球命中分布】（无偏）")
        ub_dist = summary.get("unbiased_red_distribution", {})
        for k in range(7):
            cnt = ub_dist.get(k, 0)
            lines.append(f"  命中{k}个: {cnt:3d}  {'█' * cnt}")
        lines.append(f"\n【最佳组红球命中】(有偏置：多注取max，非预测能力，反映覆盖广度)")
        lines.append(f"  平均: {summary['mean_best_red_hits']:.3f}")
    elif is_digit_game(g):
        if g == "pls" and summary.get("play") != "直选":
            base = summary["baseline"]
            rate = summary.get("exact_hit_rate", 0.0)
            lines.append(f"【核心：{summary.get('play')}整注命中率 vs 理论随机】")
            lines.append(f"  实测 {rate:.6%} vs 理论 {base['exact_hit_rate']:.6%}")
            lines.append("  组选按号码多重集判定；不使用按位置命中评价预测能力。")
        else:
            rates = summary.get("position_hit_rates", [])
            base = summary["baseline"]
            mean_hits = summary.get("mean_position_hits", 0.0)
            ci = summary.get("position_ci95", (mean_hits, mean_hits))
            expected_mean = base["mean_position_hits"]
            verdict = "高于随机" if ci[0] > expected_mean else (
                "低于随机" if ci[1] < expected_mean else "未发现显著差异")
            lines.append("【核心：单注平均按位命中 vs 精确随机】")
            lines.append(
                f"  实测 {mean_hits:.3f}（95%CI {ci[0]:.3f}~{ci[1]:.3f}）"
                f" vs 随机 {expected_mean:.3f}：{verdict}")
            lines.append("【按位命中率 vs 精确随机】")
            for i, (rate, expected) in enumerate(zip(rates, base["position_hit_rate"]), 1):
                lines.append(f"  第{i}位: 实测 {rate:.2%} vs 随机 {expected:.2%}")
            lines.append(f"  整注命中率: {summary.get('exact_hit_rate', 0.0):.6%}"
                         f" vs 随机 {base['exact_hit_rate']:.6%}")
            if summary.get("prize_counts"):
                lines.append(f"  奖级统计: {summary['prize_counts']}")
    else:
        ub_f = summary.get("unbiased_mean_front_hits", 0.0)
        ub_b = summary.get("unbiased_mean_back_hits", 0.0)
        base_f = summary["baseline"]["mean_front_hits"]
        base_b = summary["baseline"]["mean_back_hits"]
        lines.append("【核心：单注平均命中 vs 纯随机】（无偏，唯一公平对比）")
        front_ci = summary.get("front_ci95", (ub_f, ub_f))
        back_ci = summary.get("back_ci95", (ub_b, ub_b))
        lines.append(
            f"  前区  实测 {ub_f:.3f}（95%CI {front_ci[0]:.3f}~{front_ci[1]:.3f}）"
            f" vs 精确随机 {base_f:.3f}"
        )
        diff = ub_f - base_f
        verdict = "样本外高于随机" if front_ci[0] > base_f else (
            "样本外低于随机" if front_ci[1] < base_f else "未发现显著差异")
        lines.append(f"        差值 {diff:+.3f}  {verdict}")
        lines.append(
            f"  后区  实测 {ub_b:.3f}（95%CI {back_ci[0]:.3f}~{back_ci[1]:.3f}）"
            f" vs 精确随机 {base_b:.3f}"
        )
        diff_b = ub_b - base_b
        verdict_b = "高于随机" if back_ci[0] > base_b else (
            "低于随机" if back_ci[1] < base_b else "未发现显著差异")
        lines.append(f"        差值 {diff_b:+.3f}  {verdict_b}")
        lines.append("-" * 64)
        lines.append("【单注前区命中分布】（无偏）")
        ub_fdist = summary.get("unbiased_front_distribution", {})
        for k in range(6):
            cnt = ub_fdist.get(k, 0)
            lines.append(f"  命中{k}个: {cnt:3d}  {'█' * cnt}")
        lines.append(f"\n【最佳组前区命中】(有偏置：多注取max，非预测能力，反映覆盖广度)")
        lines.append(f"  平均: {summary['mean_best_front_hits']:.3f}")

    lines.append("-" * 64)
    if is_digit_game(g):
        denominator = summary["n_test"] * summary["n_groups"]
        lines.append(f"中奖注数（任意等级）: {summary['prize_count']}/{denominator}")
    else:
        lines.append(
            f"中奖期数（至少一注中奖）: {summary.get('winning_period_count', 0)}/{summary['n_test']}"
        )
    lines.append("=" * 64)
    lines.append("📌 结论解读：")
    lines.append("  - 若反复查看同一留出集并调参，本结果只能视为探索性，最终验证需等待未来新开奖。")
    lines.append("  - '无偏单注命中'与随机'≈持平'是正常且预期的结果——")
    lines.append("    彩票为独立随机事件，任何方法都无法稳定超过随机。")
    lines.append("  - '最佳组'数字偏高是统计偏置(N组取max)，不代表预测准。")
    if not is_digit_game(g):
        lines.append("  - 覆盖投注只改善票组重叠；保证仅在核心池命中且覆盖率为 100% 时成立。")
    return "\n".join(lines)
