"""
缩水/覆盖投注模块

在固定注数预算下减少票组重叠的方向：
  当开奖号码全部来自我们选定的「核心号码池」时，
  通过集合覆盖设计保证至少有一注命中 >= guaranteed_hits 个号码。

算法：贪心集合覆盖（Greedy Set Cover）。
  - 维护"待覆盖的目标 k-组合"全集
  - 每轮贪心选取能覆盖最多未覆盖组合的一注
  - 标记已覆盖，重复 n 轮

注意：彩票为独立随机事件，本模块不提高单注中奖概率，
仅在有"号码池命中"、目标组合完整枚举且覆盖率为 100% 时提供下限保证。
"""
import itertools
import numpy as np


# ====== 核心号码池构建 ======

def build_number_pool(
    probs: dict,
    head_names: list[str],
    max_num: int,
    pool_size: int,
    freq: np.ndarray = None,
    freq_alpha: float = 0.3,
) -> list[int]:
    """
    从模型概率（合并多头）+ 经验频率融合，选出 pool_size 个核心号码。
    返回升序的 1-based 号码列表。
    """
    # 合并多头
    merged = np.zeros(max_num, dtype=np.float64)
    for name in head_names:
        if name in probs:
            merged += probs[name]
    if merged.sum() > 0:
        merged /= len(head_names) if len(head_names) > 0 else 1
    else:
        merged = np.full(max_num, 1.0 / max_num)

    # 频率先验融合
    if freq is not None and freq_alpha > 0:
        merged = (1 - freq_alpha) * merged + freq_alpha * freq
        s = merged.sum()
        if s > 0:
            merged /= s

    # 取概率最高的 pool_size 个
    pool_size = min(pool_size, max_num)
    top_idx = np.argsort(merged)[::-1][:pool_size]
    return sorted(int(i + 1) for i in top_idx)


# ====== 贪心集合覆盖 ======

def _ticket_covers(ticket: frozenset, target_combos: set,
                    guaranteed_hits: int) -> int:
    """计算一注 ticket 能新覆盖多少个目标组合。"""
    cnt = 0
    for combo in target_combos:
        hits = len(ticket & combo)
        if hits >= guaranteed_hits:
            cnt += 1
    return cnt


def generate_coverage_tickets(
    main_pool: list[int],
    pick_size: int,
    n_tickets: int,
    guaranteed_hits: int,
) -> dict:
    """
    贪心集合覆盖：从 main_pool 中生成 n 注，每注 pick_size 个号码，
    保证当开奖号码都来自 main_pool 时，至少有一注命中 >= guaranteed_hits 个。

    返回:
      {
        "tickets": [[..], ...],      # 各注号码（升序）
        "pool": [...],                # 核心池
        "total_targets": int,         # 目标组合总数
        "covered_targets": int,       # 已覆盖数
        "coverage_rate": float,       # 覆盖率
        "guaranteed_hits": int,       # 保证级别
      }
    """
    pool_set = set(main_pool)
    pool_size = len(pool_set)

    # 保证级别合理范围
    guaranteed_hits = max(1, min(guaranteed_hits, pick_size))

    # 若核心池不足以选出 pick_size，直接返回单注（全选）
    if pool_size < pick_size:
        return {
            "tickets": [sorted(main_pool)],
            "pool": sorted(main_pool),
            "total_targets": 0,
            "covered_targets": 0,
            "coverage_rate": 1.0,
            "guaranteed_hits": guaranteed_hits,
            "target_space_complete": True,
            "coverage_complete": True,
        }

    # 目标：所有可能的 pick_size 组合（开奖情况）
    # 注意：当 pool 较大、pick_size 较大时组合数爆炸，需限制
    max_combos = 200000
    all_combos = list(itertools.combinations(sorted(pool_set), pick_size))
    target_space_complete = len(all_combos) <= max_combos
    if len(all_combos) > max_combos:
        # 采样目标组合以保证性能
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]

    remaining = set(frozenset(c) for c in all_combos)
    total_targets = len(remaining)
    tickets = []

    n_tickets = max(1, int(n_tickets))

    for _ in range(n_tickets):
        if not remaining:
            break
        # 贪心：从池中枚举候选注，选覆盖增益最大的
        # 候选注 = 从 pool 中选 pick_size 的组合（也可能爆炸，限制候选数）
        candidate_pool_size = pool_size
        # 候选组合数限制
        if candidate_pool_size <= 18:
            candidates = list(itertools.combinations(sorted(pool_set), pick_size))
            # 限制数量
            if len(candidates) > 5000:
                rng = np.random.default_rng(123)
                idx = rng.choice(len(candidates), 5000, replace=False)
                candidates = [candidates[i] for i in idx]
        else:
            # 大池：随机采样候选
            rng = np.random.default_rng(123)
            candidates = []
            pool_arr = sorted(pool_set)
            for _ in range(5000):
                candidates.append(tuple(sorted(rng.choice(pool_arr, pick_size, replace=False))))

        best_ticket = None
        best_gain = -1
        for cand in candidates:
            cand_set = frozenset(cand)
            # 计算增益：覆盖多少剩余目标
            gain = 0
            for combo in remaining:
                if len(cand_set & combo) >= guaranteed_hits:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_ticket = cand_set

        if best_ticket is None or best_gain <= 0:
            break

        tickets.append(sorted(best_ticket))
        # 移除已覆盖目标
        covered_now = set()
        for combo in remaining:
            if len(best_ticket & combo) >= guaranteed_hits:
                covered_now.add(combo)
        remaining -= covered_now

    covered = total_targets - len(remaining)
    rate = covered / total_targets if total_targets > 0 else 1.0

    return {
        "tickets": tickets,
        "pool": sorted(pool_set),
        "total_targets": total_targets,
        "covered_targets": covered,
        "coverage_rate": round(rate, 4),
        "guaranteed_hits": guaranteed_hits,
        "target_space_complete": target_space_complete,
        "coverage_complete": covered == total_targets,
    }


def random_coverage_baseline(main_pool: list[int], pick_size: int,
                              n_tickets: int, guaranteed_hits: int,
                              n_trials: int = 20) -> dict:
    """
    纯随机覆盖基准：用同样注数从池中随机选号，统计平均覆盖率。
    用于和贪心覆盖设计对比，量化覆盖增益。
    返回 {"mean_rate", "max_rate", "min_rate"}
    """
    pool_arr = sorted(set(main_pool))
    if len(pool_arr) < pick_size:
        return {"mean_rate": 0.0, "max_rate": 0.0, "min_rate": 0.0}

    # 复用目标组合全集（与 generate_coverage_tiles 一致的采样逻辑）
    all_combos = list(itertools.combinations(pool_arr, pick_size))
    max_combos = 200000
    if len(all_combos) > max_combos:
        rng0 = np.random.default_rng(42)
        idx = rng0.choice(len(all_combos), max_combos, replace=False)
        all_combos = [all_combos[i] for i in idx]
    targets = [frozenset(c) for c in all_combos]
    total = len(targets)

    rates = []
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        # 同一方案内不允许重复注，避免人为压低随机基准。
        ticket_count = min(n_tickets, len(all_combos))
        indices = rng.choice(len(all_combos), ticket_count, replace=False)
        rand_tickets = [frozenset(all_combos[i]) for i in indices]
        covered = 0
        for combo in targets:
            if any(len(t & combo) >= guaranteed_hits for t in rand_tickets):
                covered += 1
        rates.append(covered / total if total > 0 else 1.0)

    return {
        "mean_rate": float(np.mean(rates)),
        "max_rate": float(np.max(rates)),
        "min_rate": float(np.min(rates)),
        "total_targets": total,
    }


# ====== 高层入口 ======

def coverage_bet_ssq(
    probs: dict,
    red_freq: np.ndarray,
    blue_freq: np.ndarray,
    pool_size: int = 12,
    n_tickets: int = 10,
    guaranteed_hits: int = 4,
    freq_alpha: float = 0.3,
) -> dict:
    """
    双色球覆盖投注。
    红球：从 pool_size 个核心池中生成 n 注（每注 6 红），保证 >= guaranteed_hits 命中。
    蓝球：每注搭配概率最高的蓝球（或轮流）。
    """
    red_pool = build_number_pool(
        probs, [f"red{i+1}" for i in range(6)], 33,
        pool_size, red_freq, freq_alpha)

    cover = generate_coverage_tickets(red_pool, 6, n_tickets, guaranteed_hits)
    # 随机覆盖基准（同注数、同池、同保证级别）
    cover["random_baseline"] = random_coverage_baseline(red_pool, 6, n_tickets, guaranteed_hits)

    # 蓝球：取 top n_tickets 个（不足则循环/补最高）
    bp = probs["blue"].copy()
    top_blue = list(np.argsort(bp)[::-1][: max(n_tickets, 3)])  # 至少准备几个
    top_blue = [int(b) + 1 for b in top_blue]

    tickets_full = []
    for i, reds in enumerate(cover["tickets"]):
        blue = top_blue[i % len(top_blue)]
        tickets_full.append({"red": reds, "blue": blue})

    cover["tickets_full"] = tickets_full
    cover["blue_pool"] = sorted(set(top_blue))
    return cover


def coverage_bet_dlt(
    probs: dict,
    front_freq: np.ndarray,
    back_freq: np.ndarray,
    pool_size: int = 12,
    n_tickets: int = 10,
    guaranteed_hits: int = 3,
    freq_alpha: float = 0.3,
) -> dict:
    """
    大乐透覆盖投注。
    前区：从 pool_size 核心池中生成 n 注（每注 5 个），保证 >= guaranteed_hits 命中。
    后区：每注搭配概率最高的后区组合。
    """
    front_pool = build_number_pool(
        probs, [f"front{i+1}" for i in range(5)], 35,
        pool_size, front_freq, freq_alpha)

    cover = generate_coverage_tickets(front_pool, 5, n_tickets, guaranteed_hits)
    cover["random_baseline"] = random_coverage_baseline(front_pool, 5, n_tickets, guaranteed_hits)

    # 后区：从后区池取 top2 组合轮换
    back_merged = np.zeros(12, dtype=np.float64)
    for i in range(2):
        back_merged += probs[f"back{i+1}"]
    back_merged /= 2
    if back_freq is not None and freq_alpha > 0:
        back_merged = (1 - freq_alpha) * back_merged + freq_alpha * back_freq
    top_back = list(np.argsort(back_merged)[::-1][:4])
    top_back = [int(b) + 1 for b in top_back]
    # 生成后区 2-组合
    back_combos = [tuple(sorted(c)) for c in itertools.combinations(top_back, 2)]
    if not back_combos:
        back_combos = [(1, 2)]

    tickets_full = []
    for i, fronts in enumerate(cover["tickets"]):
        backs = list(back_combos[i % len(back_combos)])
        tickets_full.append({"front": fronts, "back": backs})

    cover["tickets_full"] = tickets_full
    cover["back_pool"] = sorted(set(top_back))
    return cover


def format_coverage_result(game: str, cover: dict) -> str:
    """格式化覆盖投注结果为文本摘要。"""
    lines = []
    game_name = "双色球" if game == "ssq" else "大乐透"
    lines.append(f"{'=' * 60}")
    lines.append(f"  {game_name} 覆盖投注方案")
    lines.append(f"{'=' * 60}")
    lines.append(f"核心号码池: {cover['pool']}")
    if game == "ssq":
        lines.append(f"蓝球池: {cover.get('blue_pool', [])}")
    else:
        lines.append(f"后区池: {cover.get('back_pool', [])}")
    exact = cover.get("target_space_complete", False)
    guaranteed = exact and cover.get("coverage_complete", False)
    if guaranteed:
        lines.append(f"已验证保证: 主区开奖号码全在核心池时，至少一注命中 >= {cover['guaranteed_hits']} 个")
    else:
        lines.append(f"覆盖目标: 至少一注命中 >= {cover['guaranteed_hits']} 个（当前未形成完整保证）")
    rate_label = "精确覆盖率" if exact else "抽样目标覆盖率估计"
    lines.append(f"{rate_label}: {cover['coverage_rate']:.2%} "
                 f"({cover['covered_targets']}/{cover['total_targets']})")
    # 与纯随机覆盖对比
    rb = cover.get("random_baseline")
    if rb:
        lines.append(f"纯随机覆盖基准(同注数): 平均 {rb['mean_rate']:.2%} "
                      f"(范围 {rb['min_rate']:.2%}~{rb['max_rate']:.2%})")
        gain = cover['coverage_rate'] - rb['mean_rate']
        tag = "↑ 覆盖设计优于随机" if gain > 0.01 else ("↓ 劣于随机" if gain < -0.01 else "≈ 持平")
        lines.append(f"覆盖增益: {gain:+.2%}  {tag}")
    lines.append("-" * 60)
    for i, t in enumerate(cover["tickets_full"], 1):
        if game == "ssq":
            reds = " ".join(f"{r:02d}" for r in t["red"])
            lines.append(f"  第{i:2d}注: {reds} | {t['blue']:02d}")
        else:
            fronts = " ".join(f"{x:02d}" for x in t["front"])
            backs = " ".join(f"{x:02d}" for x in t["back"])
            lines.append(f"  第{i:2d}注: {fronts} | {backs}")
    lines.append("=" * 60)
    lines.append("📌 覆盖投注的数学意义：在固定注数下减少票组重叠。")
    lines.append("   只有核心池命中、目标完整枚举且覆盖率100%时，条件保证才成立。")
    lines.append("   它不提高单注概率、期望收益或号码进入核心池的概率。")
    lines.append("   ⚠️ 彩票为独立随机事件，理性购彩。")
    return "\n".join(lines)
