"""
彩票号码预测与推荐号码生成器
加载训练好的 LSTM 模型，输出概率分布，加权采样生成 10 组推荐号码。
"""
import os
import json
import numpy as np
import torch
import itertools
from datetime import datetime

from game_config import DIGIT_GAME_CONFIGS, game_name, is_digit_game, normalize_play
from lstm_model import build_game_model, device

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _validate_sampling_options(n_groups: int, temperature: float,
                               freq_alpha: float = 0.0,
                               top_p: float = 1.0) -> int:
    """校验采样参数，避免无效值在 numpy 采样时产生难以定位的错误。"""
    if isinstance(n_groups, (bool, np.bool_)):
        raise ValueError("生成注数必须为正整数。")
    try:
        requested = int(n_groups)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("生成注数必须为正整数。") from exc
    if requested < 1 or requested != n_groups:
        raise ValueError("生成注数必须为正整数。")
    for name, value, lower, upper, lower_open in (
        ("采样温度", temperature, 0.0, None, True),
        ("频率先验系数", freq_alpha, 0.0, 1.0, False),
        ("Top-p 核采样阈值", top_p, 0.0, 1.0, True),
    ):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}必须是有限数值。") from exc
        if not np.isfinite(numeric) or (numeric <= lower if lower_open else numeric < lower) \
                or (upper is not None and numeric > upper):
            bound = "(0, 1]" if name.startswith("Top-p") else ("(0, +∞)" if name == "采样温度" else "[0, 1]")
            raise ValueError(f"{name}必须在 {bound} 范围内。")
    return requested


def _validated_probability_vector(values, expected_size: int, label: str) -> np.ndarray:
    """返回可采样的非负、有限且总质量为正的概率向量。"""
    p = np.asarray(values, dtype=float).copy()
    if p.ndim != 1 or len(p) != expected_size:
        actual = len(p) if p.ndim == 1 else p.shape
        raise ValueError(f"{label}概率维度应为 {expected_size}，实际为 {actual}。")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{label}概率必须全部为有限数值。")
    if np.any(p < 0):
        raise ValueError(f"{label}概率不能为负数。")
    if p.sum() <= 0:
        raise ValueError(f"{label}概率总和必须大于 0。")
    return p


def _validate_probability_map(game: str, probs: dict) -> None:
    """在主入口一次性检查各输出头，拒绝 NaN、inf、负概率和全零概率。"""
    if game == "ssq":
        heads = [(f"red{i}", 33) for i in range(1, 7)] + [("blue", 16)]
    elif game == "dlt":
        heads = [(f"front{i}", 35) for i in range(1, 6)] + [(f"back{i}", 12) for i in range(1, 3)]
    elif is_digit_game(game):
        heads = [(f"digit{i + 1}", count) for i, count in enumerate(DIGIT_GAME_CONFIGS[game]["classes"])]
    else:
        raise ValueError(f"不支持的彩种: {game}")
    for name, size in heads:
        if name not in probs:
            raise ValueError(f"缺少概率输出头: {name}。")
        _validated_probability_vector(probs[name], size, name)


def _load_model(game: str, input_size: int, hidden_size: int = 128,
                num_layers: int = 2, dropout: float = 0.3,
                bidirectional: bool = False, use_attention: bool = True,
                attn_heads: int = 4):
    """加载训练好的模型"""
    model_path = os.path.join(MODELS_DIR, f"{game}_lstm.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先训练模型。")

    model = build_game_model(
        game, input_size, hidden_size, num_layers, dropout,
        bidirectional, use_attention, attn_heads)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _get_probabilities(model, features: np.ndarray, seq_len: int) -> dict[str, np.ndarray]:
    """
    用最近 seq_len 期特征进行推理，获取每个号码位的概率分布。
    返回: {head_name: probabilities array}
    """
    if len(features) < seq_len:
        raise ValueError(f"特征数据不足 {seq_len} 期，当前仅 {len(features)} 期。")

    # 取最后 seq_len 期
    x = features[-seq_len:]
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq, feat)

    with torch.no_grad():
        logits = model(x_tensor)

    probs = {}
    for name, logit in logits.items():
        p = torch.softmax(logit, dim=-1).cpu().numpy().flatten()
        probs[name] = p
    return probs


def _get_probabilities_ensemble(game: str, features: np.ndarray, seq_len: int,
                                 input_size: int, hidden_size: int,
                                 num_layers: int, dropout: float,
                                 bidirectional: bool, use_attention: bool,
                                 attn_heads: int,
                                 ensemble_seeds: list[int]) -> dict[str, np.ndarray]:
    """集成推理：对多个 seed 模型分别推理，概率平均后返回"""
    all_probs = []
    for seed in ensemble_seeds:
        model_path = os.path.join(MODELS_DIR, f"{game}_lstm_seed{seed}.pth")
        if not os.path.exists(model_path):
            print(f"  [集成] seed {seed} 模型不存在: {model_path}，跳过")
            continue
        model = build_game_model(
            game, input_size, hidden_size, num_layers, dropout,
            bidirectional, use_attention, attn_heads)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        probs = _get_probabilities(model, features, seq_len)
        all_probs.append(probs)

    if not all_probs:
        raise RuntimeError("所有集成模型均加载失败，无法进行集成推理。")

    # 概率平均（softmax 后的概率平均）
    avg_probs = {}
    for key in all_probs[0].keys():
        avg_probs[key] = np.mean([p[key] for p in all_probs], axis=0)
        avg_probs[key] = avg_probs[key] / avg_probs[key].sum()
    return avg_probs


def _make_key(sample: dict, game: str) -> tuple:
    """为号码组生成去重键"""
    if game == "ssq":
        return tuple(sample["red"] + [sample["blue"]])
    if game == "dlt":
        return tuple(sample["front"] + sample["back"])
    if is_digit_game(game):
        return tuple(sample["digits"])
    raise ValueError(f"不支持的彩种: {game}")


def _require_unique_samples(results: list[dict], requested: int, game: str,
                            label: str) -> list[dict]:
    """确保采样器返回足量且整票唯一的号码，不用重复或无依据号码补齐。"""
    selected = results[:requested]
    keys = [_make_key(sample, game) for sample in selected]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{label}生成了重复整票，拒绝返回重复注。")
    if len(selected) < requested:
        raise RuntimeError(
            f"{label}在当前概率/温度设置下仅生成 {len(selected)}/{requested} 注互不重复号码；"
            "请调高采样温度或 Top-p 后重试。")
    return selected


# ====== 新增：频率先验 + 核采样 + 合并多头 ======

def _empirical_frequency(history: np.ndarray, max_num: int) -> np.ndarray:
    """
    从原始历史号码计算每个号码的经验出现频率。
    history: shape (n_draws, n_balls)，每行是一期的号码（1-based）
    返回: shape (max_num,) 归一化概率
    """
    counts = np.zeros(max_num, dtype=np.float64)
    for row in history:
        for num in row:
            if 1 <= num <= max_num:
                counts[num - 1] += 1
    total = counts.sum()
    if total == 0:
        return np.full(max_num, 1.0 / max_num)
    return counts / total


def _nucleus_filter(probs: np.ndarray, top_p: float = 0.9) -> np.ndarray:
    """
    Top-p 核采样：保留概率累计 >= top_p 的最小核心集合，其余置零后重新归一化。
    过滤长尾噪声，集中采样于高概率核心区。
    """
    _validate_sampling_options(1, 1.0, 0.0, top_p)
    probs = _validated_probability_vector(probs, len(probs), "核采样")
    probs = probs / probs.sum()
    if top_p >= 1.0:
        return probs
    # 按概率降序排序
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    # 找到累计概率首次达到 top_p 的位置
    cutoff = np.searchsorted(cumsum, top_p) + 1
    cutoff = min(cutoff, len(probs))
    keep_idx = sorted_idx[:cutoff]
    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    s = filtered.sum()
    if s <= 0:
        return probs
    return filtered / s


def _merge_position_heads(probs: dict, head_names: list[str],
                          num_classes: int) -> np.ndarray:
    """
    合并多个位置头的分布为单一"号码被选倾向"分布。
    解决序统计量多头独立采样的系统性偏差（red1 偏小、red6 偏大相互冲突的问题）。
    返回: shape (num_classes,)
    """
    merged = np.zeros(num_classes, dtype=np.float64)
    for name in head_names:
        merged += probs[name]
    merged /= len(head_names)
    s = merged.sum()
    if s > 0:
        merged /= s
    return merged


def _missing_value_weights(history: np.ndarray, max_num: int,
                            missing_alpha: float = 0.0) -> np.ndarray:
    """
    遗漏值加权：遗漏越久，权重越高（"冷号回补"假说）。
    返回归一化的权重向量 (max_num,)。
    missing_alpha<=0 时不加权（返回均匀分布），>0 时按遗漏值幂次加权。
    """
    if missing_alpha <= 0:
        return np.full(max_num, 1.0 / max_num)
    # 计算每个号码距上次出现的期数
    missing = np.full(max_num, len(history), dtype=np.float64)
    for i in range(len(history) - 1, -1, -1):
        for num in history[i]:
            if 1 <= num <= max_num:
                idx = num - 1
                if missing[idx] == len(history):
                    missing[idx] = len(history) - 1 - i
    # 遗漏越久权重越高，+1 避免遗漏为0时权重为0
    weights = np.power(missing + 1.0, missing_alpha)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights


def _hot_cold_weights(history: np.ndarray, max_num: int, window: int = 30,
                       hot_alpha: float = 0.0) -> np.ndarray:
    """
    冷热号加权：基于最近 window 期出现频率加权。
    hot_alpha>0 偏向热号（频率高），<0 偏向冷号（频率低）。
    返回归一化权重 (max_num,)。
    """
    if hot_alpha == 0:
        return np.full(max_num, 1.0 / max_num)
    recent = history[-window:] if len(history) >= window else history
    counts = np.zeros(max_num, dtype=np.float64)
    for row in recent:
        for num in row:
            if 1 <= num <= max_num:
                counts[num - 1] += 1
    freq = counts / max(1.0, counts.sum())
    # 用 +1e-3 平滑避免 0 概率
    weights = np.power(freq + 1e-3, hot_alpha)
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights


def _sample_with_annealing(probs: dict, n_groups: int, temps: list[float],
                           game: str) -> list[dict]:
    """温度退火：对最低温和最高温各采样 n_groups 组，合并去重后返回前 n_groups"""
    sampler = _weighted_sample_ssq if game == "ssq" else _weighted_sample_dlt
    all_results = []
    seen = set()
    for t in (min(temps), max(temps)):
        # 每个温度只请求最终所需注数，避免严格数量校验将“足够 n 注、
        # 但不足 2n 注”的合法候选空间误判为不足。
        samples = sampler(probs, n_groups, t)
        for s in samples:
            key = _make_key(s, game)
            if key not in seen:
                seen.add(key)
                all_results.append(s)
    return _require_unique_samples(all_results, n_groups, game, "温度退火采样")


# ====== 新增：合并多头采样器（消除位置偏差）======

def _weighted_sample_ssq_merged(
    probs: dict,
    n_groups: int = 10,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    red_freq: np.ndarray = None,
    blue_freq: np.ndarray = None,
    top_p: float = 1.0,
    missing_alpha: float = 0.0,
    hot_alpha: float = 0.0,
    history_main: np.ndarray = None,
    history_side: np.ndarray = None,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    合并多头版双色球采样。
    将 red1..red6 六个位置分布融合为单一号码倾向分布，再无放回采样 6 个红球，
    消除位置多头独立采样的系统性偏差。
    可选融合：经验频率先验(freq_alpha)、遗漏值加权(missing_alpha)、冷热号加权(hot_alpha)。
    """
    results = []
    seen = set()
    rng = np.random.default_rng() if rng is None else rng

    # 合并 6 个红球位置头
    red_merged = _merge_position_heads(probs, [f"red{i+1}" for i in range(6)], 33)
    # 频率先验融合
    if red_freq is not None and freq_alpha > 0:
        red_merged = (1 - freq_alpha) * red_merged + freq_alpha * red_freq
        red_merged = red_merged / red_merged.sum()
    # 遗漏值加权融合（冷号回补假说）
    if missing_alpha > 0 and history_main is not None:
        mw = _missing_value_weights(history_main, 33, missing_alpha)
        red_merged = red_merged * mw
        red_merged = red_merged / red_merged.sum()
    # 冷热号加权融合
    if hot_alpha != 0 and history_main is not None:
        hw = _hot_cold_weights(history_main, 33, 30, hot_alpha)
        red_merged = red_merged * hw
        red_merged = red_merged / red_merged.sum()
    blue_p = probs["blue"].copy()
    if blue_freq is not None and freq_alpha > 0:
        blue_p = (1 - freq_alpha) * blue_p + freq_alpha * blue_freq
        blue_p = blue_p / blue_p.sum()
    if missing_alpha > 0 and history_side is not None:
        mw = _missing_value_weights(history_side, 16, missing_alpha)
        blue_p = blue_p * mw
        blue_p = blue_p / blue_p.sum()
    if hot_alpha != 0 and history_side is not None:
        hw = _hot_cold_weights(history_side, 16, 30, hot_alpha)
        blue_p = blue_p * hw
        blue_p = blue_p / blue_p.sum()

    for _ in range(n_groups * 10):
        if len(results) >= n_groups:
            break

        # 红球：温度调整 + 核采样 + 无放回采样
        p = np.power(red_merged, 1.0 / temperature)
        p = p / p.sum()
        p = _nucleus_filter(p, top_p)
        red_balls = []
        p_work = p.copy()
        for i in range(6):
            p_cur = p_work.copy()
            for chosen in red_balls:
                p_cur[chosen] = 0
            s = p_cur.sum()
            if s <= 0:
                p_cur = np.ones(33)
                for chosen in red_balls:
                    p_cur[chosen] = 0
                s = p_cur.sum()
            p_cur = p_cur / s
            red_balls.append(int(rng.choice(33, p=p_cur)))

        red_sorted = sorted(x + 1 for x in red_balls)

        # 蓝球
        bp = np.power(blue_p, 1.0 / temperature)
        bp = bp / bp.sum()
        bp = _nucleus_filter(bp, top_p)
        blue_ball = int(rng.choice(16, p=bp)) + 1

        key = tuple(red_sorted + [blue_ball])
        if key not in seen:
            seen.add(key)
            # 用融合后分布计算近似置信度
            prob_list = []
            p_temp = np.power(red_merged, 1.0 / temperature)
            p_temp = p_temp / p_temp.sum()
            p_work2 = p_temp.copy()
            for idx in red_balls:
                prob_list.append(float(p_work2[idx]))
                p_work2[idx] = 0
                s2 = p_work2.sum()
                if s2 > 0:
                    p_work2 = p_work2 / s2
            prob_list.append(float(blue_p[blue_ball - 1]))
            group_prob = float(np.exp(np.mean(np.log(np.clip(prob_list, 1e-12, None)))))
            results.append({
                "red": red_sorted,
                "blue": blue_ball,
                "prob": round(group_prob, 6),
            })

    return _require_unique_samples(results, n_groups, "ssq", "双色球合并采样")


def _weighted_sample_dlt_merged(
    probs: dict,
    n_groups: int = 10,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    front_freq: np.ndarray = None,
    back_freq: np.ndarray = None,
    top_p: float = 1.0,
    missing_alpha: float = 0.0,
    hot_alpha: float = 0.0,
    history_main: np.ndarray = None,
    history_side: np.ndarray = None,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    合并多头版大乐透采样。
    将 front1..front5 融合为单一前区倾向分布，back1..back2 融合为单一后区倾向分布，
    再无放回采样，消除位置多头独立采样的系统性偏差。
    可选融合：经验频率先验、遗漏值加权、冷热号加权。
    """
    results = []
    seen = set()
    rng = np.random.default_rng() if rng is None else rng

    front_merged = _merge_position_heads(probs, [f"front{i+1}" for i in range(5)], 35)
    back_merged = _merge_position_heads(probs, [f"back{i+1}" for i in range(2)], 12)
    if front_freq is not None and freq_alpha > 0:
        front_merged = (1 - freq_alpha) * front_merged + freq_alpha * front_freq
        front_merged = front_merged / front_merged.sum()
    if back_freq is not None and freq_alpha > 0:
        back_merged = (1 - freq_alpha) * back_merged + freq_alpha * back_freq
        back_merged = back_merged / back_merged.sum()
    # 遗漏值加权
    if missing_alpha > 0 and history_main is not None:
        front_merged = front_merged * _missing_value_weights(history_main, 35, missing_alpha)
        front_merged = front_merged / front_merged.sum()
    if missing_alpha > 0 and history_side is not None:
        back_merged = back_merged * _missing_value_weights(history_side, 12, missing_alpha)
        back_merged = back_merged / back_merged.sum()
    # 冷热号加权
    if hot_alpha != 0 and history_main is not None:
        front_merged = front_merged * _hot_cold_weights(history_main, 35, 30, hot_alpha)
        front_merged = front_merged / front_merged.sum()
    if hot_alpha != 0 and history_side is not None:
        back_merged = back_merged * _hot_cold_weights(history_side, 12, 30, hot_alpha)
        back_merged = back_merged / back_merged.sum()

    for _ in range(n_groups * 10):
        if len(results) >= n_groups:
            break

        # 前区
        p = np.power(front_merged, 1.0 / temperature)
        p = p / p.sum()
        p = _nucleus_filter(p, top_p)
        front_balls = []
        p_work = p.copy()
        for i in range(5):
            p_cur = p_work.copy()
            for chosen in front_balls:
                p_cur[chosen] = 0
            s = p_cur.sum()
            if s <= 0:
                p_cur = np.ones(35)
                for chosen in front_balls:
                    p_cur[chosen] = 0
                s = p_cur.sum()
            p_cur = p_cur / s
            front_balls.append(int(rng.choice(35, p=p_cur)))
        front_sorted = sorted(x + 1 for x in front_balls)

        # 后区
        bp = np.power(back_merged, 1.0 / temperature)
        bp = bp / bp.sum()
        bp = _nucleus_filter(bp, top_p)
        back_balls = []
        bp_work = bp.copy()
        for i in range(2):
            p_cur = bp_work.copy()
            for chosen in back_balls:
                p_cur[chosen] = 0
            s = p_cur.sum()
            if s <= 0:
                p_cur = np.ones(12)
                for chosen in back_balls:
                    p_cur[chosen] = 0
                s = p_cur.sum()
            p_cur = p_cur / s
            back_balls.append(int(rng.choice(12, p=p_cur)))
        back_sorted = sorted(x + 1 for x in back_balls)

        key = tuple(front_sorted + back_sorted)
        if key not in seen:
            seen.add(key)
            prob_list = []
            p_temp = np.power(front_merged, 1.0 / temperature)
            p_temp = p_temp / p_temp.sum()
            p_work2 = p_temp.copy()
            for idx in front_balls:
                prob_list.append(float(p_work2[idx]))
                p_work2[idx] = 0
                s2 = p_work2.sum()
                if s2 > 0:
                    p_work2 = p_work2 / s2
            bp_temp = np.power(back_merged, 1.0 / temperature)
            bp_temp = bp_temp / bp_temp.sum()
            bp_work2 = bp_temp.copy()
            for idx in back_balls:
                prob_list.append(float(bp_work2[idx]))
                bp_work2[idx] = 0
                s2 = bp_work2.sum()
                if s2 > 0:
                    bp_work2 = bp_work2 / s2
            group_prob = float(np.exp(np.mean(np.log(np.clip(prob_list, 1e-12, None)))))
            results.append({
                "front": front_sorted,
                "back": back_sorted,
                "prob": round(group_prob, 6),
            })

    return _require_unique_samples(results, n_groups, "dlt", "大乐透合并采样")


def _weighted_sample_ssq(
    probs: dict,
    n_groups: int = 10,
    temperature: float = 1.0,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    根据概率分布加权采样生成双色球号码组。
    每组: 6 个不重复红球(1-33, 升序) + 1 个蓝球(1-16)
    """
    results = []
    seen = set()
    rng = np.random.default_rng() if rng is None else rng

    for _ in range(n_groups * 10):  # 多试几次避免重复
        if len(results) >= n_groups:
            break

        # 采样 6 个红球（不重复）
        red_balls = []
        for i in range(6):
            head = f"red{i+1}"
            p = probs[head].copy()
            # 提高温度以增加多样性
            p = np.power(p, 1 / temperature)
            # 排除已选号码
            for chosen in red_balls:
                p[chosen] = 0
            if p.sum() == 0:
                p = np.ones(33)
                for chosen in red_balls:
                    p[chosen] = 0
            p = p / p.sum()
            chosen_idx = int(rng.choice(33, p=p))
            red_balls.append(chosen_idx)

        red_balls_sorted = sorted([x + 1 for x in red_balls])  # 0-based -> 1-based

        # 采样蓝球
        blue_p = np.power(probs["blue"], 1 / temperature)
        blue_p = blue_p / blue_p.sum()
        blue_ball = int(rng.choice(16, p=blue_p)) + 1

        key = tuple(red_balls_sorted + [blue_ball])
        if key not in seen:
            seen.add(key)
            group_prob = _calc_ssq_group_prob(probs, red_balls, blue_ball, temperature)
            results.append({
                "red": red_balls_sorted,
                "blue": blue_ball,
                "prob": round(float(group_prob), 6),
            })

    return _require_unique_samples(results, n_groups, "ssq", "双色球采样")


def _calc_ssq_group_prob(probs: dict, red_balls: list, blue_ball: int,
                          temperature: float) -> float:
    """计算一组 SSQ 号码的几何平均概率"""
    prob_list = []
    for i in range(6):
        p = probs[f"red{i+1}"].copy()
        p = np.power(p, 1.0 / temperature)
        for j in range(i):
            p[red_balls[j]] = 0.0
        s = p.sum()
        if s > 0:
            p = p / s
        prob_list.append(float(p[red_balls[i]]))
    bp = np.power(probs["blue"], 1.0 / temperature)
    bp = bp / bp.sum()
    prob_list.append(float(bp[blue_ball - 1]))
    return float(np.exp(np.mean(np.log(prob_list))))


def _weighted_sample_dlt(
    probs: dict,
    n_groups: int = 10,
    temperature: float = 1.0,
    rng: np.random.Generator = None,
) -> list[dict]:
    """
    大乐透号码采样。
    每组: 5 个不重复前区(1-35, 升序) + 2 个不重复后区(1-12, 升序)
    """
    results = []
    seen = set()
    rng = np.random.default_rng() if rng is None else rng

    for _ in range(n_groups * 10):
        if len(results) >= n_groups:
            break

        # 采样 5 个前区
        front_balls = []
        for i in range(5):
            head = f"front{i+1}"
            p = probs[head].copy()
            p = np.power(p, 1 / temperature)
            for chosen in front_balls:
                p[chosen] = 0
            if p.sum() == 0:
                p = np.ones(35)
                for chosen in front_balls:
                    p[chosen] = 0
            p = p / p.sum()
            chosen_idx = int(rng.choice(35, p=p))
            front_balls.append(chosen_idx)

        front_sorted = sorted([x + 1 for x in front_balls])

        # 采样 2 个后区
        back_balls = []
        for i in range(2):
            head = f"back{i+1}"
            p = probs[head].copy()
            p = np.power(p, 1 / temperature)
            for chosen in back_balls:
                p[chosen] = 0
            if p.sum() == 0:
                p = np.ones(12)
                for chosen in back_balls:
                    p[chosen] = 0
            p = p / p.sum()
            chosen_idx = int(rng.choice(12, p=p))
            back_balls.append(chosen_idx)

        back_sorted = sorted([x + 1 for x in back_balls])

        key = tuple(front_sorted + back_sorted)
        if key not in seen:
            seen.add(key)
            group_prob = _calc_dlt_group_prob(probs, front_balls, back_balls, temperature)
            results.append({
                "front": front_sorted,
                "back": back_sorted,
                "prob": round(float(group_prob), 6),
            })

    return _require_unique_samples(results, n_groups, "dlt", "大乐透采样")


def _calc_dlt_group_prob(probs: dict, front_balls: list, back_balls: list,
                          temperature: float) -> float:
    """计算一组 DLT 号码的几何平均概率"""
    prob_list = []
    for i in range(5):
        p = probs[f"front{i+1}"].copy()
        p = np.power(p, 1.0 / temperature)
        for j in range(i):
            p[front_balls[j]] = 0.0
        s = p.sum()
        if s > 0:
            p = p / s
        prob_list.append(float(p[front_balls[i]]))
    for i in range(2):
        p = probs[f"back{i+1}"].copy()
        p = np.power(p, 1.0 / temperature)
        for j in range(i):
            p[back_balls[j]] = 0.0
        s = p.sum()
        if s > 0:
            p = p / s
        prob_list.append(float(p[back_balls[i]]))
    return float(np.exp(np.mean(np.log(prob_list))))


def _digit_position_frequencies(
    history: np.ndarray | None,
    class_counts: list[int],
) -> list[np.ndarray] | None:
    """按位置统计数字频率；数字 0 合法，不能复用 1-based 球号统计。"""
    if history is None or len(history) == 0:
        return None
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[1] != len(class_counts):
        raise ValueError("排列型历史号码维度不匹配。")
    result = []
    for pos, count in enumerate(class_counts):
        values = history[:, pos].astype(int)
        valid = values[(values >= 0) & (values < count)]
        counts = np.bincount(valid, minlength=count).astype(float)
        result.append(counts / counts.sum() if counts.sum() else np.full(count, 1 / count))
    return result


def _weighted_sample_digits(
    probs: dict,
    game: str,
    n_groups: int = 10,
    temperature: float = 1.0,
    freq_alpha: float = 0.0,
    history: np.ndarray | None = None,
    top_p: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """排列型彩票按位采样；各位置独立，允许重复数字并保留 0。"""
    requested = _validate_sampling_options(n_groups, temperature, freq_alpha, top_p)
    class_counts = DIGIT_GAME_CONFIGS[game]["classes"]
    frequencies = _digit_position_frequencies(history, class_counts) if freq_alpha > 0 else None
    rng = np.random.default_rng() if rng is None else rng
    results, seen = [], set()
    for _ in range(requested * 20):
        if len(results) >= requested:
            break
        digits, selected = [], []
        for pos, count in enumerate(class_counts):
            p = _validated_probability_vector(probs[f"digit{pos + 1}"], count, f"第 {pos + 1} 位")
            if frequencies is not None:
                p = (1 - freq_alpha) * p + freq_alpha * frequencies[pos]
            p = np.power(np.clip(p, 1e-12, None), 1 / temperature)
            p /= p.sum()
            p = _nucleus_filter(p, top_p)
            value = int(rng.choice(count, p=p))
            digits.append(value)
            selected.append(float(p[value]))
        key = tuple(digits)
        if key not in seen:
            seen.add(key)
            score = float(np.exp(np.mean(np.log(np.clip(selected, 1e-12, None)))))
            results.append({"digits": digits, "prob": round(score, 6)})
    if len(results) != requested:
        raise RuntimeError(
            f"直选在当前温度/Top-p 设置下仅生成 {len(results)}/{requested} 注互不重复号码；"
            "请调高 Top-p 或采样温度后重试。")
    return results


def _sample_pl3_play(probs: dict, play: str, n_groups: int, temperature: float,
                     freq_alpha: float = 0.0, history=None, top_p: float = 1.0,
                     rng: np.random.Generator | None = None) -> list[dict]:
    """按所有唯一排列的联合概率质量采样排列3组选3/组选6。"""
    play = normalize_play("pls", play)
    requested = _validate_sampling_options(n_groups, temperature, freq_alpha, top_p)
    if play == "直选":
        return _weighted_sample_digits(probs, "pls", requested, temperature,
                                       freq_alpha, history, top_p, rng)
    rng = np.random.default_rng() if rng is None else rng
    freq = _digit_position_frequencies(history, (10, 10, 10)) if freq_alpha > 0 else None
    ps = []
    for pos in range(3):
        p = _validated_probability_vector(probs[f"digit{pos + 1}"], 10, f"排列3第 {pos + 1} 位")
        if freq is not None:
            p = (1 - freq_alpha) * p + freq_alpha * freq[pos]
        p = np.power(np.clip(p, 1e-12, None), 1 / temperature); p /= p.sum()
        ps.append(_nucleus_filter(p, top_p))
    candidates = (itertools.combinations(range(10), 3) if play == "组选6"
                  else ((a, a, b) for a in range(10) for b in range(10) if a != b))
    candidates = [tuple(sorted(c)) for c in candidates]
    masses = []
    for c in candidates:
        perms = set(itertools.permutations(c))
        masses.append(sum(float(np.prod([ps[i][d[i]] for i in range(3)])) for d in perms))
    masses = np.asarray(masses, dtype=float)
    # Top-p 可以使某些组选的联合质量严格为零；只从实际有效候选中无放回抽样。
    valid = masses > 0
    candidates = [candidate for candidate, keep in zip(candidates, valid) if keep]
    masses = masses[valid]
    if len(candidates) < requested:
        raise ValueError(
            f"排列3{play}在当前温度/Top-p 设置下仅有 {len(candidates)} 个有效唯一候选，"
            f"不足请求的 {requested} 注；请调高 Top-p 或减少注数。")
    masses = masses / masses.sum()
    take = requested
    indices = rng.choice(len(candidates), size=take, replace=False, p=masses)
    return [{"digits": list(candidates[i]), "prob": round(float(masses[i]), 6)} for i in indices]


def predict_numbers(
    game: str,
    features: np.ndarray,
    seq_len: int = 30,
    n_groups: int = 10,
    temperature: float = 1.0,
    input_size: int = None,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
    use_attention: bool = True,
    attn_heads: int = 4,
    n_ensemble: int = 1,
    ensemble_seeds: list[int] = None,
    anneal_temps: list[float] = None,
    sampling_mode: str = "merged",
    freq_alpha: float = 0.0,
    top_p: float = 1.0,
    missing_alpha: float = 0.0,
    hot_alpha: float = 0.0,
    history_main: np.ndarray = None,
    history_side: np.ndarray = None,
    source_issue: str = None,
    source_date: str = None,
    play: str = None,
    selection_strategy: str = "weighted",
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """
    主预测入口。支持集成推理、温度退火和合法号码的无放回采样。
    新增：
      sampling_mode: "merged"（合并多头，消除位置偏差，推荐）/ "per_position"（原逐位）
      freq_alpha: 经验频率先验混合系数 0-1，模型不确定时退化为历史频率
      top_p: top-p 核采样阈值 0-1，过滤长尾噪声（< 1.0 生效）
      missing_alpha: 遗漏值加权系数（>0 偏向冷号回补，需提供 history_*）
      hot_alpha: 冷热号加权系数（>0 偏向热号，<0 偏向冷号，需提供 history_*）
      history_main: 主区（红球/前区）原始历史号码数组 (n_draws, n_balls)
      history_side: 辅区（蓝球 (n,) 或后区 (n,2)）原始历史号码数组
    返回: (推荐号码列表, 概率分布字典)
    """
    n_groups = _validate_sampling_options(n_groups, temperature, freq_alpha, top_p)
    if selection_strategy not in ("weighted", "joint_top5"):
        raise ValueError(f"不支持的选号策略: {selection_strategy}")
    if selection_strategy == "joint_top5" and n_groups != 5:
        raise ValueError("最高奖联合排序固定生成 5 注，不增加预算。")
    if input_size is None:
        input_size = features.shape[1]

    # 1. 获取概率（集成推理 or 单模型）
    if n_ensemble > 1 and ensemble_seeds:
        probs = _get_probabilities_ensemble(
            game, features, seq_len, input_size, hidden_size,
            num_layers, dropout, bidirectional, use_attention,
            attn_heads, ensemble_seeds)
    else:
        model = _load_model(game, input_size, hidden_size, num_layers, dropout,
                            bidirectional, use_attention, attn_heads)
        probs = _get_probabilities(model, features, seq_len)
    _validate_probability_map(game, probs)

    if selection_strategy == "joint_top5":
        from jackpot_selection import select_top_five
        numbers = select_top_five(game, probs, play)
        _save_prediction(game, numbers, source_issue, source_date,
                         normalize_play(game, play), selection_strategy=selection_strategy)
        return numbers, probs

    # 2. 预计算频率先验（若提供历史号码）
    red_freq = blue_freq = front_freq = back_freq = None
    if freq_alpha > 0 and not is_digit_game(game):
        if game == "ssq":
            if history_main is not None:
                red_freq = _empirical_frequency(np.asarray(history_main), 33)
            if history_side is not None:
                blue_freq = _empirical_frequency(np.asarray(history_side).reshape(-1, 1), 16)
        elif game == "dlt":
            if history_main is not None:
                front_freq = _empirical_frequency(np.asarray(history_main), 35)
            if history_side is not None:
                back_freq = _empirical_frequency(np.asarray(history_side), 12)

    # 3. 采样
    if is_digit_game(game):
        if missing_alpha != 0 or hot_alpha != 0:
            raise ValueError("排列3、排列5、7星彩暂不使用遗漏/冷热假设，请将两项保持为 0。")
        temps = [temperature]
        if anneal_temps and len(anneal_temps) > 1:
            temps = [min(anneal_temps), max(anneal_temps)]
        effective_play = normalize_play(game, play)
        numbers, seen = [], set()
        rng = np.random.default_rng()
        if game == "pls" and effective_play != "直选":
            numbers = _sample_pl3_play(probs, effective_play, n_groups, temperature,
                                       freq_alpha, history_main, top_p, rng)
            _save_prediction(game, numbers, source_issue, source_date, effective_play)
            return numbers, probs
        for temp in temps:
            for sample in _weighted_sample_digits(
                probs, game, n_groups, temp, freq_alpha,
                history_main, top_p, rng):
                key = _make_key(sample, game)
                if key not in seen:
                    seen.add(key)
                    numbers.append(sample)
                if len(numbers) >= n_groups:
                    break
            if len(numbers) >= n_groups:
                break
    elif anneal_temps and len(anneal_temps) > 1 and sampling_mode != "merged":
        # 退火仅用于 per_position（避免与合并多头逻辑冲突）
        numbers = _sample_with_annealing(probs, n_groups, anneal_temps, game)
    elif sampling_mode == "merged":
        if game == "ssq":
            numbers = _weighted_sample_ssq_merged(
                probs, n_groups, temperature, freq_alpha, red_freq, blue_freq, top_p,
                missing_alpha, hot_alpha, history_main, history_side)
        elif game == "dlt":
            numbers = _weighted_sample_dlt_merged(
                probs, n_groups, temperature, freq_alpha, front_freq, back_freq, top_p,
                missing_alpha, hot_alpha, history_main, history_side)
    else:
        if game == "ssq":
            numbers = _weighted_sample_ssq(probs, n_groups, temperature)
        elif game == "dlt":
            numbers = _weighted_sample_dlt(probs, n_groups, temperature)

    # 保存预测结果
    _save_prediction(game, numbers, source_issue, source_date,
                     normalize_play(game, play) if is_digit_game(game) else play)

    return numbers, probs


def _save_prediction(
    game: str,
    numbers: list[dict],
    source_issue: str = None,
    source_date: str = None,
    play: str = None,
    selection_strategy: str = "weighted",
):
    """保存预测结果到 JSON 文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    record = {
        "game": game,
        "timestamp": timestamp,
        "source_issue": source_issue,
        "source_date": source_date,
        "play": play,
        "selection_strategy": selection_strategy,
        "numbers": numbers,
    }
    suffix = 0
    while True:
        filename = f"{game}_{timestamp}{'' if suffix == 0 else f'_{suffix:03d}'}.json"
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, "x", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            return filepath
        except FileExistsError:
            suffix += 1


def _format_digit_values(game: str, digits: list[int]) -> str:
    """保留前导 0；7星彩最后一位单独显示以兼容 10-14。"""
    if game == "qxc":
        return f"{''.join(str(x) for x in digits[:6])} | {digits[6]}"
    return "".join(str(x) for x in digits)


def format_numbers_table(game: str, numbers: list[dict], play: str = None) -> str:
    """带序号和倾向分的详细版"""
    if numbers and all("joint_log_score" in item for item in numbers):
        from math import comb
        effective_play = normalize_play(game, play)
        denominator = {"ssq": comb(33, 6) * 16, "dlt": comb(35, 5) * comb(12, 2),
                       "pls": 1000, "plw": 100000, "qxc": 15000000}[game]
        multiplicity = {"组选3": 3, "组选6": 6}.get(effective_play, 1) if game == "pls" else 1
        baseline = len(numbers) * multiplicity / denominator
        lines = [f"{game_name(game)}（{effective_play}）最高奖目标 · 联合排序5注（实验）",
                 "按模型的完整组合联合分数选择，不采用遗漏/冷热、温度或Top-p筛选。"]
        for i, item in enumerate(numbers, 1):
            line = format_numbers_copy(game, [item], effective_play)
            lines.append(f"第{i}注: {line}  [模型联合log分 {item['joint_log_score']:.6f}]")
        lines.extend([f"独立均匀开奖假设下，这 {len(numbers)} 注对应最高奖理论概率: {baseline:.9%}",
                      "联合分数越大代表模型排序越靠前，不是实际中奖概率；真实头奖优势尚未验证。",
                      "同一数据和模型重复点击得到相同5注，不应重复购买来增加覆盖。"])
        return "\n".join(lines)
    lines = []
    if game == "ssq":
        lines.append("=" * 60)
        lines.append("  双色球生成号码 (红球 + 蓝球)")
        lines.append("=" * 60)
        for i, n in enumerate(numbers, 1):
            reds = " ".join(f"{r:02d}" for r in n["red"])
            blue = f"{n['blue']:02d}"
            prob = n.get("prob", 0)
            lines.append(f"  第{i:2d}组:  {reds}  |  {blue}  (倾向分: {prob:.4%})")
    elif game == "dlt":
        lines.append("=" * 60)
        lines.append("  大乐透生成号码 (前区 + 后区)")
        lines.append("=" * 60)
        for i, n in enumerate(numbers, 1):
            fronts = " ".join(f"{f:02d}" for f in n["front"])
            backs = " ".join(f"{b:02d}" for b in n["back"])
            prob = n.get("prob", 0)
            lines.append(f"  第{i:2d}组:  {fronts}  |  {backs}  (倾向分: {prob:.4%})")
    elif is_digit_game(game):
        lines.append("=" * 60)
        effective_play = normalize_play(game, play)
        lines.append(f"  {game_name(game)}生成号码（{effective_play}）")
        if game == "pls":
            base = {"直选": 1 / 1000, "组选3": 3 / 1000, "组选6": 6 / 1000}[effective_play]
            valid_keys = set()
            for item in numbers:
                digits = item.get("digits") if isinstance(item, dict) else None
                if (not isinstance(digits, (list, tuple)) or len(digits) != 3
                        or any(not isinstance(digit, (int, np.integer)) or not 0 <= digit <= 9
                               for digit in digits)):
                    continue
                counts = sorted([digits.count(digit) for digit in set(digits)])
                if effective_play == "直选":
                    valid_keys.add(tuple(digits))
                elif effective_play == "组选3" and counts == [1, 2]:
                    valid_keys.add(tuple(sorted(digits)))
                elif effective_play == "组选6" and counts == [1, 1, 1]:
                    valid_keys.add(tuple(sorted(digits)))
            unique_count = len(valid_keys)
            lines.append(f"  单注理论命中率: {base:.2%}；本次 {unique_count} 注有效唯一号码理论覆盖率: {unique_count * base:.2%}")
            lines.append("  注：覆盖率仅为玩法组合覆盖的数学比例，奖额相应降低，不代表模型优势。")
        lines.append("=" * 60)
        for i, n in enumerate(numbers, 1):
            prob = n.get("prob", 0)
            lines.append(
                f"  第{i:2d}组:  {_format_digit_values(game, n['digits'])}  "
                f"(倾向分: {prob:.4%})")
    else:
        raise ValueError(f"不支持的彩种: {game}")
    lines.append("=" * 60)
    lines.append("注：'倾向分'仅为模型对号码的相对打分，不代表中奖概率。")
    return "\n".join(lines)


def format_numbers_copy(game: str, numbers: list[dict], play: str = None,
                        include_play: bool = False) -> str:
    """号码复制文本；默认纯号码，include_play=True 时附彩种与玩法说明。"""
    lines = []
    if include_play:
        effective_play = normalize_play(game, play) if is_digit_game(game) else play
        title = game_name(game)
        lines.append(f"{title}（{effective_play}）" if effective_play else title)
    if game == "ssq":
        for n in numbers:
            reds = " ".join(f"{r:02d}" for r in n["red"])
            lines.append(f"{reds} + {n['blue']:02d}")
    elif game == "dlt":
        for n in numbers:
            fronts = " ".join(f"{f:02d}" for f in n["front"])
            backs = " ".join(f"{b:02d}" for b in n["back"])
            lines.append(f"{fronts} + {backs}")
    elif is_digit_game(game):
        lines.extend(_format_digit_values(game, n["digits"]) for n in numbers)
    else:
        raise ValueError(f"不支持的彩种: {game}")
    return "\n".join(lines)
