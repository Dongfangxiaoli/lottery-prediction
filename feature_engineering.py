"""
彩票号码特征工程模块
将历史开奖号码转换为多维特征向量，供 LSTM 模型训练。
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from game_config import DIGIT_GAME_CONFIGS


# --------------- 通用统计函数 ---------------

def _frequency_in_window(history: np.ndarray, window: int, max_num: int) -> np.ndarray:
    """
    计算最近 window 期内每个号码的出现频率。
    history: shape (n_draws, n_balls)，每行是一期的号码
    返回: shape (max_num,) 频率向量
    """
    recent = history[-window:] if len(history) >= window else history
    counts = np.zeros(max_num, dtype=np.float32)
    for row in recent:
        for num in row:
            if 1 <= num <= max_num:
                counts[num - 1] += 1
    return counts / len(recent)


def _missing_values(history: np.ndarray, max_num: int) -> np.ndarray:
    """
    计算每个号码的遗漏值（距上次出现的期数）。
    返回: shape (max_num,)
    """
    missing = np.full(max_num, len(history), dtype=np.float32)
    for i in range(len(history) - 1, -1, -1):
        for num in history[i]:
            if 1 <= num <= max_num:
                idx = num - 1
                if missing[idx] == len(history):
                    missing[idx] = len(history) - 1 - i
    return missing


def _missing_matrix(history: np.ndarray, max_num: int) -> np.ndarray:
    """
    一次性向量化计算每期的遗漏值矩阵。
    history: shape (n, k) 每行一期的号码（1-based）
    返回: shape (n, max_num)，第 i 行是截至第 i 期每个号码的遗漏值。
    比 _missing_values 逐期调用快 ~100x（从 O(n²) 降到 O(n*k)）。
    """
    n = len(history)
    mat = np.zeros((n, max_num), dtype=np.float32)
    last_seen = np.full(max_num, -1, dtype=np.int64)  # 最近出现期号，-1=从未出现
    for i in range(n):
        # 该期所有号码的遗漏 = i - last_seen（未出现过则 = i+1，即自始至今全部期数）
        mat[i] = i - last_seen
        mat[i][last_seen == -1] = i + 1
        # 更新 last_seen：本期出现的号记为 i
        row = history[i]
        valid = row[(row >= 1) & (row <= max_num)]
        if valid.size > 0:
            last_seen[(valid.astype(np.int64) - 1)] = i
    return mat


def _odd_even_ratio(nums: np.ndarray) -> float:
    """奇数占比"""
    odds = sum(1 for n in nums if n % 2 == 1)
    return odds / len(nums)


def _big_small_ratio(nums: np.ndarray, threshold: int) -> float:
    """大号占比（>= threshold 为大号）"""
    bigs = sum(1 for n in nums if n >= threshold)
    return bigs / len(nums)


def _sum_value(nums: np.ndarray) -> float:
    return float(np.sum(nums))


def _span(nums: np.ndarray) -> float:
    return float(np.max(nums) - np.min(nums))


def _ac_value(nums: np.ndarray) -> float:
    """AC 值：不同差值个数 - (选号个数 - 1)"""
    sorted_nums = sorted(nums)
    diffs = set()
    for i in range(len(sorted_nums)):
        for j in range(i + 1, len(sorted_nums)):
            diffs.add(sorted_nums[j] - sorted_nums[i])
    return float(len(diffs) - (len(nums) - 1))


def _consecutive_count(nums: np.ndarray) -> float:
    """连号组数"""
    sorted_nums = sorted(nums)
    groups = 0
    in_consecutive = False
    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] == sorted_nums[i - 1] + 1:
            if not in_consecutive:
                groups += 1
                in_consecutive = True
        else:
            in_consecutive = False
    return float(groups)


def _repeat_count(current: np.ndarray, previous: np.ndarray) -> float:
    """与上一期的重号个数"""
    return float(len(set(current) & set(previous)))


def _zone_distribution(nums: np.ndarray, max_num: int, n_zones: int = 3) -> np.ndarray:
    """区间分布比例"""
    zone_size = max_num / n_zones
    dist = np.zeros(n_zones, dtype=np.float32)
    for n in nums:
        zone_idx = min(int((n - 1) / zone_size), n_zones - 1)
        dist[zone_idx] += 1
    return dist / len(nums)


# ====== 新增特征辅助函数 ======

PRIMES_33 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
PRIMES_12 = {2, 3, 5, 7, 11}


def _prime_ratio(nums: np.ndarray, primes: set) -> float:
    """质数占比"""
    count = sum(1 for n in nums if int(n) in primes)
    return count / len(nums)


def _tail_distribution(nums: np.ndarray) -> np.ndarray:
    """尾号分布 0-9"""
    tails = np.zeros(10, dtype=np.float32)
    for n in nums:
        tails[int(n) % 10] += 1
    return tails / len(nums)


def _freq_trend_mean_std(history: np.ndarray, max_num: int, window: int = 20) -> np.ndarray:
    """频率趋势：两段不重叠窗口间频率变化率的均值+标准差"""
    if len(history) < 2:
        return np.array([0.0, 0.0], dtype=np.float32)
    w = min(window, max(1, len(history) // 2))
    if len(history) >= 2 * w:
        recent = history[-w:]
        older = history[-2 * w:-w]
    else:
        recent = history[-w:]
        older = history[:w]

    def _freq(data, mx):
        cnt = np.zeros(mx, dtype=np.float32)
        for row in data:
            for num in row:
                if 1 <= num <= mx:
                    cnt[num - 1] += 1
        return cnt / len(data)

    freq_recent = _freq(recent, max_num)
    freq_older = _freq(older, max_num)
    slopes = (freq_recent - freq_older) / w
    return np.array([float(np.mean(slopes)), float(np.std(slopes))], dtype=np.float32)


def _hot_ball_ratio(history: np.ndarray, max_num: int, window: int, threshold: int) -> float:
    """热门号码占比：最近window期出现次数>=threshold的号码比例"""
    recent = history[-window:] if len(history) >= window else history
    counts = np.zeros(max_num, dtype=np.int32)
    for row in recent:
        for num in row:
            if 1 <= num <= max_num:
                counts[num - 1] += 1
    return float(np.sum(counts >= threshold)) / max_num


def _cold_ball_ratio(history: np.ndarray, max_num: int, window: int, threshold: int) -> float:
    """冷门号码占比：最近window期出现次数<=threshold的号码比例"""
    recent = history[-window:] if len(history) >= window else history
    counts = np.zeros(max_num, dtype=np.int32)
    for row in recent:
        for num in row:
            if 1 <= num <= max_num:
                counts[num - 1] += 1
    return float(np.sum(counts <= threshold)) / max_num


def _adjacent_diff_stats(nums: np.ndarray) -> np.ndarray:
    """排序后相邻差值的均值和标准差"""
    sorted_nums = sorted(nums)
    diffs = np.array([sorted_nums[i + 1] - sorted_nums[i] for i in range(len(sorted_nums) - 1)])
    return np.array([float(np.mean(diffs)), float(np.std(diffs))], dtype=np.float32)


def _sum_tail_value(nums: np.ndarray) -> float:
    """和值尾数归一化 (sum % 10) / 10"""
    return (float(np.sum(nums)) % 10) / 10.0


def _repeat_rate_multi(current: np.ndarray, history: np.ndarray, n_balls: int) -> np.ndarray:
    """多窗口重号率：与上一期、前5期平均、前10期平均"""
    result = np.zeros(3, dtype=np.float32)
    cur_set = set(current)
    total = len(history)
    if total >= 2:
        result[0] = len(cur_set & set(history[-2])) / n_balls
    if total >= 6:
        result[1] = (sum(len(cur_set & set(history[-(j + 1)])) for j in range(1, 6)) / 5) / n_balls
    elif total >= 2:
        n = total - 1
        result[1] = (sum(len(cur_set & set(history[-(j + 1)])) for j in range(1, n + 1)) / n) / n_balls
    if total >= 11:
        result[2] = (sum(len(cur_set & set(history[-(j + 1)])) for j in range(1, 11)) / 10) / n_balls
    elif total >= 2:
        n = min(10, total - 1)
        result[2] = (sum(len(cur_set & set(history[-(j + 1)])) for j in range(1, n + 1)) / n) / n_balls
    return result


def _span_trend_mean_std(history: np.ndarray, window: int = 5) -> np.ndarray:
    """最近N期跨度的均值和标准差"""
    n = min(window, len(history))
    spans = np.array([float(np.max(row) - np.min(row)) for row in history[-n:]])
    return np.array([float(np.mean(spans)), float(np.std(spans))], dtype=np.float32)


# ====== 第二批新增特征辅助函数 ======

def _markov_recur_prob(history: np.ndarray, max_num: int, window: int = 30) -> float:
    """
    马尔可夫一阶递推概率：当前期号码在最近 window 期内"重复出现"的平均率。
    衡量号码的"惯性"——即上一期出现的号在本期再次出现的倾向。
    返回 0-1 的标量。
    """
    n = len(history)
    if n < 3:
        return 0.0
    w = min(window, n - 1)
    recurr = 0
    total = 0
    for i in range(n - w, n):
        if i < 1:
            continue
        prev_set = set(int(x) for x in history[i - 1] if 1 <= x <= max_num)
        cur_set = set(int(x) for x in history[i] if 1 <= x <= max_num)
        if prev_set:
            recurr += len(prev_set & cur_set) / len(prev_set)
            total += 1
    return recurr / total if total > 0 else 0.0


def _same_tail_count(nums: np.ndarray) -> float:
    """同尾号组数（个位相同的号码组数）"""
    tails = {}
    for n in nums:
        t = int(n) % 10
        tails[t] = tails.get(t, 0) + 1
    # 只统计出现 >= 2 的尾数
    return float(sum(1 for c in tails.values() if c >= 2))


def _prime_sum_value(nums: np.ndarray, primes: set) -> float:
    """质数号码的和值（归一化）"""
    s = sum(int(n) for n in nums if int(n) in primes)
    return float(s)


def _markov_transition_feature(history: np.ndarray, max_num: int,
                                window: int = 20) -> np.ndarray:
    """
    一阶马尔可夫转移概率特征：最近 window 期内，
    "号码 i 的下一期是否出现号码 j"的稀疏统计聚合（均值+熵），2 维。
    用于刻画号码间的转移结构。
    """
    n = len(history)
    if n < 3:
        return np.array([0.0, 0.0], dtype=np.float32)
    w = min(window, n - 1)
    # 转移计数矩阵（稀疏用 dict）
    trans = {}
    start = max(1, n - w)
    for i in range(start, n):
        prev = set(int(x) for x in history[i - 1] if 1 <= x <= max_num)
        cur = set(int(x) for x in history[i] if 1 <= x <= max_num)
        for p in prev:
            for c in cur:
                trans[(p, c)] = trans.get((p, c), 0) + 1
    if not trans:
        return np.array([0.0, 0.0], dtype=np.float32)
    counts = np.array(list(trans.values()), dtype=np.float32)
    total = counts.sum()
    p = counts / total
    # 转移分布熵（归一化到 log(max_num) 量级）
    entropy = float(-np.sum(p * np.log(p + 1e-12))) / np.log(max_num + 1)
    mean_strength = float(np.mean(counts / total))
    return np.array([mean_strength, entropy], dtype=np.float32)


def _avg_missing_window(history: np.ndarray, max_num: int,
                         window: int = 30) -> np.ndarray:
    """
    最近 window 期内各号码的平均遗漏 + 最大遗漏（2 维）。
    刻画"冷号积压"程度。
    """
    n = len(history)
    if n == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    w = min(window, n)
    recent = history[-w:]
    last_seen = np.full(max_num, w, dtype=np.float32)  # 距 window 起点的遗漏
    # 从最近一期往前数遗漏
    missing = np.full(max_num, w, dtype=np.float32)
    for i in range(len(recent) - 1, -1, -1):
        for num in recent[i]:
            if 1 <= num <= max_num:
                idx = num - 1
                if missing[idx] == w:
                    missing[idx] = len(recent) - 1 - i
    return np.array([float(np.mean(missing)), float(np.max(missing))], dtype=np.float32)


# --------------- 主特征构建函数 ---------------


def _scale_features(features: np.ndarray, fit_end: int | None) -> tuple[np.ndarray, MinMaxScaler]:
    """仅在训练前缀拟合缩放器，再变换全部时序。fit_end 为开区间。"""
    if len(features) == 0:
        raise ValueError("没有可用于构建特征的数据。")
    fit_rows = len(features) if fit_end is None else max(1, min(int(fit_end), len(features)))
    scaler = MinMaxScaler().fit(features[:fit_rows])
    return scaler.transform(features).astype(np.float32), scaler


def build_features_ssq(
    df: pd.DataFrame,
    fit_end: int | None = None,
) -> tuple[np.ndarray, dict, MinMaxScaler]:
    """
    为双色球构建特征矩阵。
    df 列: issue, date, red1-red6, blue, sales, pool
    返回: (features, labels, scaler)
      features: shape (n_draws, feature_dim)
      labels: dict with 'red' shape (n_draws, 6) and 'blue' shape (n_draws,)
    """
    red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
    reds = df[red_cols].values  # (n, 6)
    blues = df["blue"].values   # (n,)

    # 预计算遗漏值矩阵（向量化，O(n*k) 代替逐期 O(n²)）
    red_missing_mat = _missing_matrix(reds, 33)
    blue_missing_mat = _missing_matrix(blues.reshape(-1, 1), 16)

    features_list = []
    for i in range(len(df)):
        history_red = reds[:i + 1]
        history_blue = blues[:i + 1].reshape(-1, 1)
        curr_red = reds[i]
        curr_blue = blues[i]
        prev_red = reds[i - 1] if i > 0 else curr_red

        feat = []
        # 红球频率 (窗口 20/50/100)
        for w in [20, 50, 100]:
            feat.extend(_frequency_in_window(history_red, w, 33))
        # 红球遗漏值（查表，已向量化预计算）
        red_missing = red_missing_mat[i]
        feat.extend(red_missing)
        # 蓝球频率 (窗口 20/50/100)
        for w in [20, 50, 100]:
            feat.extend(_frequency_in_window(history_blue, w, 16))
        # 蓝球遗漏值（查表）
        blue_missing = blue_missing_mat[i]
        feat.extend(blue_missing)
        # 当期红球统计
        feat.append(_odd_even_ratio(curr_red))
        feat.append(_big_small_ratio(curr_red, 17))  # 33/2 ≈ 17
        feat.append(_sum_value(curr_red))
        feat.append(_span(curr_red))
        feat.append(_ac_value(curr_red))
        feat.append(_consecutive_count(curr_red))
        feat.append(_repeat_count(curr_red, prev_red))
        # 红球三分区
        feat.extend(_zone_distribution(curr_red, 33, 3))

        # ====== 新增特征 ======
        # 1. 质数比例
        feat.append(_prime_ratio(curr_red, PRIMES_33))
        # 2. 尾号分布
        feat.extend(_tail_distribution(curr_red))
        # 3. 趋势特征
        feat.extend(_freq_trend_mean_std(history_red, 33))
        feat.extend(_freq_trend_mean_std(history_blue, 16))
        # 4. 冷热号分类
        feat.append(_hot_ball_ratio(history_red, 33, 20, 5))
        feat.append(_cold_ball_ratio(history_red, 33, 50, 3))
        feat.append(_hot_ball_ratio(history_blue, 16, 20, 5))
        feat.append(_cold_ball_ratio(history_blue, 16, 50, 3))
        # 5. 遗漏极值
        feat.append(float(np.max(red_missing)))
        feat.append(float(np.max(blue_missing)))
        # 6. 邻期差值
        feat.extend(_adjacent_diff_stats(curr_red))
        # 7. 和值尾数
        feat.append(_sum_tail_value(curr_red))
        # 8. 重号率多窗口
        feat.extend(_repeat_rate_multi(curr_red, history_red, 6))
        # 9. 跨度趋势
        feat.extend(_span_trend_mean_std(history_red))
        # ====== 第二批新增特征 ======
        # 10. 马尔可夫一阶递推率（号码惯性）
        feat.append(_markov_recur_prob(history_red, 33, 30))
        # 11. 同尾号组数
        feat.append(_same_tail_count(curr_red))
        # 12. 质数和值
        feat.append(_prime_sum_value(curr_red, PRIMES_33))
        # 13. 马尔可夫转移结构（前2维）
        feat.extend(_markov_transition_feature(history_red, 33, 20))
        # 14. 红球遗漏窗口统计（均值+最大）
        feat.extend(_avg_missing_window(history_red, 33, 30))
        # 15. 蓝球遗漏窗口统计
        feat.extend(_avg_missing_window(history_blue, 16, 30))

        features_list.append(feat)

    features = np.array(features_list, dtype=np.float32)

    # 标签：下一期的号码（用于训练时需 shift）
    labels = {
        "red": reds,    # (n, 6) 值域 1-33
        "blue": blues,  # (n,)   值域 1-16
    }

    features, scaler = _scale_features(features, fit_end)

    return features, labels, scaler


def build_features_dlt(
    df: pd.DataFrame,
    fit_end: int | None = None,
) -> tuple[np.ndarray, dict, MinMaxScaler]:
    """
    为大乐透构建特征矩阵。
    df 列: issue, date, front1-front5, back1-back2, sales
    返回: (features, labels, scaler)
    """
    front_cols = ["front1", "front2", "front3", "front4", "front5"]
    back_cols = ["back1", "back2"]
    fronts = df[front_cols].values  # (n, 5)
    backs = df[back_cols].values    # (n, 2)

    # 预计算遗漏值矩阵（向量化）
    front_missing_mat = _missing_matrix(fronts, 35)
    back_missing_mat = _missing_matrix(backs, 12)

    features_list = []
    for i in range(len(df)):
        history_front = fronts[:i + 1]
        history_back = backs[:i + 1]
        curr_front = fronts[i]
        curr_back = backs[i]
        prev_front = fronts[i - 1] if i > 0 else curr_front

        feat = []
        # 前区频率 (窗口 20/50/100)
        for w in [20, 50, 100]:
            feat.extend(_frequency_in_window(history_front, w, 35))
        # 前区遗漏值（查表，已向量化预计算）
        front_missing = front_missing_mat[i]
        feat.extend(front_missing)
        # 后区频率 (窗口 20/50/100)
        for w in [20, 50, 100]:
            feat.extend(_frequency_in_window(history_back, w, 12))
        # 后区遗漏值（查表）
        back_missing = back_missing_mat[i]
        feat.extend(back_missing)
        # 当期前区统计
        feat.append(_odd_even_ratio(curr_front))
        feat.append(_big_small_ratio(curr_front, 18))  # 35/2 ≈ 18
        feat.append(_sum_value(curr_front))
        feat.append(_span(curr_front))
        feat.append(_ac_value(curr_front))
        feat.append(_consecutive_count(curr_front))
        feat.append(_repeat_count(curr_front, prev_front))
        # 前区三分区
        feat.extend(_zone_distribution(curr_front, 35, 3))

        # ====== 新增特征 ======
        # 1. 质数比例（前区+后区）
        feat.append(_prime_ratio(curr_front, PRIMES_33))
        feat.append(_prime_ratio(curr_back, PRIMES_12))
        # 2. 尾号分布（前区+后区）
        feat.extend(_tail_distribution(curr_front))
        feat.extend(_tail_distribution(curr_back))
        # 3. 趋势特征（前区+后区）
        feat.extend(_freq_trend_mean_std(history_front, 35))
        feat.extend(_freq_trend_mean_std(history_back, 12))
        # 4. 冷热号分类（前区+后区）
        feat.append(_hot_ball_ratio(history_front, 35, 20, 3))
        feat.append(_cold_ball_ratio(history_front, 35, 50, 2))
        feat.append(_hot_ball_ratio(history_back, 12, 20, 3))
        feat.append(_cold_ball_ratio(history_back, 12, 50, 2))
        # 5. 遗漏极值（前区+后区）
        feat.append(float(np.max(front_missing)))
        feat.append(float(np.max(back_missing)))
        # 6. 邻期差值（前区+后区）
        feat.extend(_adjacent_diff_stats(curr_front))
        feat.extend(_adjacent_diff_stats(curr_back))
        # 7. 和值尾数（前区+后区）
        feat.append(_sum_tail_value(curr_front))
        feat.append(_sum_tail_value(curr_back))
        # 8. 重号率多窗口（前区+后区）
        feat.extend(_repeat_rate_multi(curr_front, history_front, 5))
        feat.extend(_repeat_rate_multi(curr_back, history_back, 2))
        # 9. 跨度趋势（前区+后区）
        feat.extend(_span_trend_mean_std(history_front))
        feat.extend(_span_trend_mean_std(history_back))
        # ====== 第二批新增特征 ======
        # 10. 马尔可夫一阶递推率（前区+后区号码惯性）
        feat.append(_markov_recur_prob(history_front, 35, 30))
        feat.append(_markov_recur_prob(history_back, 12, 30))
        # 11. 同尾号组数（前区+后区）
        feat.append(_same_tail_count(curr_front))
        feat.append(_same_tail_count(curr_back))
        # 12. 质数和值（前区+后区）
        feat.append(_prime_sum_value(curr_front, PRIMES_33))
        feat.append(_prime_sum_value(curr_back, PRIMES_12))
        # 13. 马尔可夫转移结构（前区+后区各2维）
        feat.extend(_markov_transition_feature(history_front, 35, 20))
        feat.extend(_markov_transition_feature(history_back, 12, 20))
        # 14. 遗漏窗口统计（前区+后区各2维）
        feat.extend(_avg_missing_window(history_front, 35, 30))
        feat.extend(_avg_missing_window(history_back, 12, 30))

        features_list.append(feat)

    features = np.array(features_list, dtype=np.float32)

    labels = {
        "front": fronts,  # (n, 5) 值域 1-35
        "back": backs,     # (n, 2) 值域 1-12
    }

    features, scaler = _scale_features(features, fit_end)

    return features, labels, scaler


def build_features_digit(
    df: pd.DataFrame,
    game: str,
    fit_end: int | None = None,
) -> tuple[np.ndarray, dict, MinMaxScaler]:
    """构建排列3/排列5/七星彩的最小因果特征。

    每个位置使用：当期 one-hot（用于预测下一期）+ 最近 20/50/100 期按位频率。
    QXC 的配置允许最后一位使用 15 类。
    """
    if game not in DIGIT_GAME_CONFIGS:
        raise ValueError(f"不支持的数字型彩种: {game}")
    cfg = DIGIT_GAME_CONFIGS[game]
    cols = list(cfg["digits"])
    sizes = cfg["classes"]
    n_positions = len(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{game} 数据缺少字段: {', '.join(missing)}")
    digits = df[cols].to_numpy(dtype=np.int64)
    if len(digits) and any(np.any(digits[:, i] < 0) or np.any(digits[:, i] >= sizes[i])
                           for i in range(n_positions)):
        raise ValueError(f"{game} 数字超出配置范围")

    features_list = []
    for i, row in enumerate(digits):
        feat = []
        history = digits[:i + 1]
        for pos, size in enumerate(sizes):
            one_hot = np.zeros(size, dtype=np.float32)
            one_hot[int(row[pos])] = 1.0
            feat.extend(one_hot)
            for window in (20, 50, 100):
                recent = history[-window:, pos:pos + 1]
                counts = np.bincount(recent.reshape(-1), minlength=size).astype(np.float32)
                feat.extend(counts / max(1, len(recent)))
        features_list.append(feat)

    features = np.asarray(features_list, dtype=np.float32)
    features, scaler = _scale_features(features, fit_end)
    return features, {"digits": digits}, scaler
