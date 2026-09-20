"""Pure, bounded portfolio construction for complete lottery tickets.

The module intentionally has no model loading, history access, UI dependency, or
file writes.  A supplied ``probs`` mapping is only an *uncalibrated ranking
input* for ``strategy='model'``; it is not interpreted as a real draw
probability.
"""
from __future__ import annotations

from itertools import combinations, product
from math import comb, log, sqrt
from typing import Mapping

import numpy as np

from game_config import DIGIT_GAME_CONFIGS, game_code, game_name, normalize_play


OPERATIONAL_MAX_TICKETS = 500
_CANDIDATE_LIMIT = 512
_DESIGN_DRAWS = 512
_TICKET_PRICE_YUAN = 2


def _int_in_range(value, label: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label}必须是整数。")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{label}应在 {low}–{high} 之间。")
    return value


def _outcome_count(game: str) -> int:
    if game == "ssq":
        return comb(33, 6) * 16
    if game == "dlt":
        return comb(35, 5) * comb(12, 2)
    return int(np.prod(DIGIT_GAME_CONFIGS[game]["classes"]))


def _permutation_count(game: str, play: str) -> int:
    if game != "pls" or play == "直选":
        return 1
    return 3 if play == "组选3" else 6


def _ticket_space_count(game: str, play: str) -> int:
    if game == "pls" and play == "组选3":
        return 10 * 9
    if game == "pls" and play == "组选6":
        return comb(10, 3)
    if game == "ssq":
        return comb(33, 6) * 16
    if game == "dlt":
        return comb(35, 5) * comb(12, 2)
    return _outcome_count(game)


def _ticket_key(ticket: dict, game: str, play: str) -> tuple:
    if game == "ssq":
        return tuple(ticket["red"]) + (int(ticket["blue"]),)
    if game == "dlt":
        return tuple(ticket["front"]) + tuple(ticket["back"])
    digits = tuple(ticket["digits"])
    return tuple(sorted(digits)) if game == "pls" and play != "直选" else digits


def _canonical_ticket(ticket: Mapping, game: str, play: str) -> dict:
    """Validate one public ticket shape and return only its canonical fields."""
    if not isinstance(ticket, Mapping):
        raise ValueError("每注号码必须是字典。")
    def integers(values, label):
        if not isinstance(values, (list, tuple)) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
            raise ValueError(f"{label}必须全部为整数。")
        return [int(x) for x in values]
    try:
        if game == "ssq":
            red = integers(ticket["red"], "双色球红球")
            if isinstance(ticket["blue"], (bool, np.bool_)) or not isinstance(ticket["blue"], (int, np.integer)):
                raise ValueError("双色球蓝球必须为整数。")
            blue = int(ticket["blue"])
            if len(red) != 6 or len(set(red)) != 6 or sorted(red) != red or any(x < 1 or x > 33 for x in red):
                raise ValueError("双色球红球应为升序、互异的 6 个 1–33 整数。")
            if not 1 <= blue <= 16:
                raise ValueError("双色球蓝球应为 1–16 整数。")
            return {"red": red, "blue": blue}
        if game == "dlt":
            front, back = integers(ticket["front"], "大乐透前区"), integers(ticket["back"], "大乐透后区")
            if len(front) != 5 or len(set(front)) != 5 or sorted(front) != front or any(x < 1 or x > 35 for x in front):
                raise ValueError("大乐透前区应为升序、互异的 5 个 1–35 整数。")
            if len(back) != 2 or len(set(back)) != 2 or sorted(back) != back or any(x < 1 or x > 12 for x in back):
                raise ValueError("大乐透后区应为升序、互异的 2 个 1–12 整数。")
            return {"front": front, "back": back}
        digits = integers(ticket["digits"], f"{game}数字")
        classes = DIGIT_GAME_CONFIGS[game]["classes"]
        if len(digits) != len(classes) or any(x < 0 or x >= size for x, size in zip(digits, classes)):
            raise ValueError(f"{game}数字票维度或取值范围不合法。")
        if game == "pls" and play == "组选3":
            counts = sorted(digits.count(x) for x in set(digits))
            if digits != sorted(digits) or counts != [1, 2]:
                raise ValueError("排列3组选3应为升序的两同一异三个数字。")
            return {"digits": digits}
        if game == "pls" and play == "组选6" and (digits != sorted(digits) or len(set(digits)) != 3):
            raise ValueError("排列3组选6应为升序、互异的三个数字。")
        return {"digits": digits}
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc):
            raise
        raise ValueError("票面字段或数字格式无效。") from exc


def _uniform_ticket(game: str, play: str, rng: np.random.Generator) -> dict:
    if game == "ssq":
        return {"red": sorted(map(int, rng.choice(np.arange(1, 34), 6, replace=False))),
                "blue": int(rng.integers(1, 17))}
    if game == "dlt":
        return {"front": sorted(map(int, rng.choice(np.arange(1, 36), 5, replace=False))),
                "back": sorted(map(int, rng.choice(np.arange(1, 13), 2, replace=False)))}
    if game == "pls" and play == "组选3":
        repeated, single = map(int, rng.choice(10, 2, replace=False))
        return {"digits": sorted([repeated, repeated, single])}
    if game == "pls" and play == "组选6":
        return {"digits": sorted(map(int, rng.choice(10, 3, replace=False)))}
    return {"digits": [int(rng.integers(0, size)) for size in DIGIT_GAME_CONFIGS[game]["classes"]]}


def _uniform_tickets(game: str, play: str, count: int, rng: np.random.Generator) -> list[dict]:
    """Uniformly sample distinct legal tickets, including the two PL3 group spaces."""
    result, seen = [], set()
    while len(result) < count:
        ticket = _uniform_ticket(game, play, rng)
        key = _ticket_key(ticket, game, play)
        if key not in seen:
            seen.add(key)
            result.append(ticket)
    return result


def _small_space_tickets(game: str, play: str) -> list[dict] | None:
    if game != "pls":
        return None
    if play == "组选3":
        return [{"digits": [a, a, b] if a < b else [b, a, a]}
                for a in range(10) for b in range(10) if a != b]
    if play == "组选6":
        return [{"digits": list(item)} for item in combinations(range(10), 3)]
    if play == "直选":
        return [{"digits": list(item)} for item in product(range(10), repeat=3)]
    return None


def _model_tickets(game: str, play: str, count: int, probs) -> list[dict]:
    if probs is None:
        raise ValueError("strategy='model'需要调用方提供 probs；本模块不会加载模型。")
    try:
        from jackpot_selection import select_top_k
    except ImportError as exc:
        raise RuntimeError("select_top_k尚不可用，无法生成模型排序组合。") from exc
    try:
        raw = select_top_k(game, probs, count, play=play)
    except TypeError:  # tolerate the documented equivalent keyword form during integration
        raw = select_top_k(game, probs, k=count, play=play)
    tickets = [_canonical_ticket(ticket, game, play) for ticket in raw]
    if len(tickets) != count or len({_ticket_key(x, game, play) for x in tickets}) != count:
        raise RuntimeError("select_top_k未返回请求数量的互异合法整票。")
    return tickets


def _ssq_prize(red_hits: int, blue_hit: bool) -> int:
    if red_hits == 6 and blue_hit: return 1
    if red_hits == 6: return 2
    if red_hits == 5 and blue_hit: return 3
    if red_hits == 5 or (red_hits == 4 and blue_hit): return 4
    if red_hits == 4 or (red_hits == 3 and blue_hit): return 5
    return 6 if blue_hit else 0


def _dlt_prize(front_hits: int, back_hits: int) -> int:
    """大乐透现行七奖级（与 backtest._prize_level_dlt 保持一致）。"""
    if front_hits == 5 and back_hits == 2: return 1
    if front_hits == 5 and back_hits == 1: return 2
    if front_hits == 5 or (front_hits == 4 and back_hits == 2): return 3
    if front_hits == 4 and back_hits == 1: return 4
    if front_hits == 4 or (front_hits == 3 and back_hits == 2): return 5
    if (front_hits == 3 and back_hits == 1) or (front_hits == 2 and back_hits == 2): return 6
    if front_hits == 3 or (front_hits == 2 and back_hits == 1) or (front_hits == 1 and back_hits == 2) or back_hits == 2: return 7
    return 0


def _qxc_prize(matches: tuple[bool, ...]) -> int:
    front_hits, last_hit = sum(matches[:6]), bool(matches[6])
    if front_hits == 6 and last_hit: return 1
    if front_hits == 6: return 2
    if front_hits == 5 and last_hit: return 3
    if front_hits == 5 or (front_hits == 4 and last_hit): return 4
    if front_hits == 4 or (front_hits == 3 and last_hit): return 5
    # Current rule: 3+0, 2+1, 1+1, or 0+1.  It is positional, not a run rule.
    return 6 if front_hits == 3 or (last_hit and front_hits <= 2) else 0


def _prize(ticket: dict, draw: dict | tuple[int, ...], game: str, play: str) -> int:
    if game == "ssq":
        return _ssq_prize(len(set(ticket["red"]) & set(draw["red"])), ticket["blue"] == draw["blue"])
    if game == "dlt":
        return _dlt_prize(len(set(ticket["front"]) & set(draw["front"])), len(set(ticket["back"]) & set(draw["back"])))
    actual = tuple(draw)
    digits = tuple(ticket["digits"])
    if game == "pls" and play != "直选":
        return 1 if tuple(sorted(digits)) == tuple(sorted(actual)) else 0
    if game == "qxc":
        return _qxc_prize(tuple(a == b for a, b in zip(digits, actual)))
    return 1 if digits == actual else 0


def _draw(game: str, rng: np.random.Generator) -> dict | tuple[int, ...]:
    if game == "ssq":
        return {"red": sorted(map(int, rng.choice(np.arange(1, 34), 6, replace=False))), "blue": int(rng.integers(1, 17))}
    if game == "dlt":
        return {"front": sorted(map(int, rng.choice(np.arange(1, 36), 5, replace=False))),
                "back": sorted(map(int, rng.choice(np.arange(1, 13), 2, replace=False)))}
    return tuple(int(rng.integers(0, size)) for size in DIGIT_GAME_CONFIGS[game]["classes"])


def _eligible(ticket: dict, draw, game: str, play: str, target: int) -> bool:
    level = _prize(ticket, draw, game, play)
    return level != 0 and level <= target


def _single_ticket_probability(game: str, play: str, target: int) -> float:
    """Exact per-ticket probability for prize level ``<= target`` under uniform draws."""
    if game == "ssq":
        favorable = sum(
            comb(6, red) * comb(27, 6 - red) * (1 if blue else 15)
            for red in range(7) for blue in (False, True)
            if _ssq_prize(red, blue) and _ssq_prize(red, blue) <= target
        )
        return favorable / _outcome_count(game)
    if game == "dlt":
        favorable = sum(
            comb(5, front) * comb(30, 5 - front) * comb(2, back) * comb(10, 2 - back)
            for front in range(6) for back in range(3)
            if _dlt_prize(front, back) and _dlt_prize(front, back) <= target
        )
        return favorable / _outcome_count(game)
    if game == "qxc":
        probability = 0.0
        for bits in product((False, True), repeat=7):
            if _qxc_prize(bits) and _qxc_prize(bits) <= target:
                p = 1.0
                for match, classes in zip(bits, DIGIT_GAME_CONFIGS[game]["classes"]):
                    p *= 1 / classes if match else (classes - 1) / classes
                probability += p
        return probability
    return _permutation_count(game, play) / _outcome_count(game)


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0: return (0.0, 0.0)
    z, p = 1.959963984540054, successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _paired_ci(selected: np.ndarray, baseline: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Finite-sample 95% Hoeffding bound for paired outcomes in [-1, 1]."""
    deltas = selected.astype(float) - baseline.astype(float)
    mean = float(deltas.mean())
    # P(|mean-E| >= e) <= 2*exp(-n*e^2/2).  Remains non-degenerate
    # even with zero observed discordances, unlike a plug-in normal interval.
    margin = sqrt(2 * log(40) / len(deltas))
    return mean, (max(-1.0, mean - margin), min(1.0, mean + margin))


def _prize_vector(ticket: dict, draws: list, game: str) -> np.ndarray:
    """Vectorized current-rule prize levels for the lower-prize game set."""
    if game == "ssq":
        reds = np.asarray([draw["red"] for draw in draws], dtype=int)
        blue = np.asarray([draw["blue"] for draw in draws], dtype=int)
        red_hits = np.isin(reds, ticket["red"]).sum(axis=1)
        blue_hit = blue == ticket["blue"]
        return np.select([
            (red_hits == 6) & blue_hit, red_hits == 6, (red_hits == 5) & blue_hit,
            (red_hits == 5) | ((red_hits == 4) & blue_hit),
            (red_hits == 4) | ((red_hits == 3) & blue_hit), blue_hit,
        ], [1, 2, 3, 4, 5, 6], default=0).astype(int)
    if game == "dlt":
        fronts = np.asarray([draw["front"] for draw in draws], dtype=int)
        backs = np.asarray([draw["back"] for draw in draws], dtype=int)
        front_hits = np.isin(fronts, ticket["front"]).sum(axis=1)
        back_hits = np.isin(backs, ticket["back"]).sum(axis=1)
        return np.select([
            (front_hits == 5) & (back_hits == 2), (front_hits == 5) & (back_hits == 1),
            (front_hits == 5) | ((front_hits == 4) & (back_hits == 2)),
            (front_hits == 4) & (back_hits == 1),
            (front_hits == 4) | ((front_hits == 3) & (back_hits == 2)),
            ((front_hits == 3) & (back_hits == 1)) | ((front_hits == 2) & (back_hits == 2)),
            (front_hits == 3) | ((front_hits == 2) & (back_hits == 1)) |
            ((front_hits == 1) & (back_hits == 2)) | (back_hits == 2),
        ], [1, 2, 3, 4, 5, 6, 7], default=0).astype(int)
    draws_array = np.asarray(draws, dtype=int)
    matches = draws_array == np.asarray(ticket["digits"], dtype=int)
    front_hits, last_hit = matches[:, :6].sum(axis=1), matches[:, 6]
    return np.select([
        (front_hits == 6) & last_hit, front_hits == 6, (front_hits == 5) & last_hit,
        (front_hits == 5) | ((front_hits == 4) & last_hit),
        (front_hits == 4) | ((front_hits == 3) & last_hit),
        (front_hits == 3) | (last_hit & (front_hits <= 2)),
    ], [1, 2, 3, 4, 5, 6], default=0).astype(int)


def _best_prizes(tickets: list[dict], draws: list, game: str, play: str) -> np.ndarray:
    """Best (smallest) prize level per draw, vectorized across every ticket."""
    if game not in {"ssq", "dlt", "qxc"}:
        raise ValueError("仅低奖彩种需要模拟最佳奖级。")
    best = np.zeros(len(draws), dtype=int)
    for ticket in tickets:
        levels = _prize_vector(ticket, draws, game)
        replace = (levels != 0) & ((best == 0) | (levels < best))
        best[replace] = levels[replace]
    return best


def _target_hits(best_prizes: np.ndarray, target: int) -> np.ndarray:
    return (best_prizes > 0) & (best_prizes <= target)


def _tier_breakdown(best_prizes: np.ndarray, max_target: int) -> dict[str, int]:
    return {str(level): int((best_prizes == level).sum()) for level in range(1, max_target + 1)}


def _coverage_tickets(game: str, play: str, count: int, target: int, rng: np.random.Generator) -> tuple[list[dict], str]:
    # Head-prize tickets all cover disjoint outcomes: greedy would add no value.
    if target == 1:
        return _uniform_tickets(game, play, count, rng), "头奖目标：均匀去重；不进行无意义的贪心覆盖。"
    small = _small_space_tickets(game, play)
    candidate_count = min(_CANDIDATE_LIMIT, max(count, min(_CANDIDATE_LIMIT, count * 3)))
    candidates = small if small is not None and len(small) <= _CANDIDATE_LIMIT else _uniform_tickets(game, play, candidate_count, rng)
    design_draws = [_draw(game, rng) for _ in range(_DESIGN_DRAWS)]
    masks = np.asarray([[ _eligible(ticket, draw, game, play, target) for draw in design_draws] for ticket in candidates], dtype=bool)
    chosen, available = [], np.ones(len(candidates), dtype=bool)
    uncovered = np.ones(_DESIGN_DRAWS, dtype=bool)
    for _ in range(count):
        gains = (masks & uncovered).sum(axis=1)
        gains[~available] = -1
        index = int(np.argmax(gains))
        if gains[index] < 0:
            raise RuntimeError("覆盖候选不足以生成互异整票。")
        chosen.append(candidates[index]); available[index] = False
        uncovered &= ~masks[index]
    rate = 1.0 - float(uncovered.mean())
    return chosen, f"低奖目标：在独立设计样本 {_DESIGN_DRAWS} 期上贪心覆盖；设计样本覆盖率 {rate:.2%}（非保证）。"


def generate_portfolio(game, play=None, n_tickets=10, strategy="uniform", target_prize=1,
                       probs=None, seed=42, n_eval=10000) -> dict:
    """Generate and fairly simulate a fixed-count, unique full-ticket portfolio.

    ``target_prize=1`` means jackpot.  For SSQ/DLT/QXC, a larger number means
    "that prize level or better".  PL3/PL5 have no lower prize rule in this
    engine and therefore reject any value other than 1.
    """
    code = game_code(game)
    effective_play = normalize_play(code, play)
    strategy = str(strategy).strip().lower()
    if strategy not in {"uniform", "model", "coverage"}:
        raise ValueError("strategy只支持 uniform、model 或 coverage。")
    max_target = {"ssq": 6, "dlt": 7, "qxc": 6}.get(code, 1)
    target = _int_in_range(target_prize, "target_prize", 1, max_target)
    requested = _int_in_range(n_tickets, "n_tickets", 1, OPERATIONAL_MAX_TICKETS)
    ceiling = _ticket_space_count(code, effective_play)
    if requested > ceiling:
        raise ValueError(f"{game_name(code)}{effective_play}最多只有 {ceiling} 注互异合法整票。")
    seed = _int_in_range(seed, "seed", 0, 2**63 - 1)
    n_eval = _int_in_range(n_eval, "n_eval", 1, 100_000)
    selection_rng, baseline_rng, evaluation_rng = [np.random.default_rng(item)
                                                     for item in np.random.SeedSequence(seed).spawn(3)]

    if strategy == "uniform":
        tickets, provenance = _uniform_tickets(code, effective_play, requested, selection_rng), "均匀随机、去重生成。"
    elif strategy == "model":
        tickets, provenance = _model_tickets(code, effective_play, requested, probs), "模型分数仅用于未校准排序；不代表真实中奖概率。"
    else:
        tickets, provenance = _coverage_tickets(code, effective_play, requested, target, selection_rng)
    tickets = [_canonical_ticket(ticket, code, effective_play) for ticket in tickets]
    if len({_ticket_key(item, code, effective_play) for item in tickets}) != requested:
        raise RuntimeError("组合生成器产生了重复整票。")

    if strategy == "uniform":
        baseline_tickets, baseline_note = tickets, "与均匀策略同一组合（差值固定为 0）。"
    else:
        baseline_tickets, baseline_note = _uniform_tickets(code, effective_play, requested, baseline_rng), "对照是一份独立生成的同注数随机方案（非所有随机方案的平均）；共享评估开奖。"
    jackpot_exact = requested * _permutation_count(code, effective_play) / _outcome_count(code)
    single_target = _single_ticket_probability(code, effective_play, target)
    if target == 1:
        # For top prize the same-count result is exact, not a noisy Monte-Carlo
        # contest between strategies.  In particular, a sampled rare hit is not
        # allowed to look like a predictive gain.
        selected_count = baseline_count = None
        selected_rate = baseline_rate = jackpot_exact
        selected_interval = baseline_interval = (jackpot_exact, jackpot_exact)
        delta, delta_ci = 0.0, (0.0, 0.0)
        selected_tiers = baseline_tiers = None
        evaluation_note = "头奖按精确组合概率比较；未以蒙特卡洛命中次数比较策略。"
    else:
        # Fresh draws are intentionally created after (and independently of)
        # coverage design, then shared by portfolio and uniform baseline.
        evaluation_draws = [_draw(code, evaluation_rng) for _ in range(n_eval)]
        selected_best = _best_prizes(tickets, evaluation_draws, code, effective_play)
        baseline_best = _best_prizes(baseline_tickets, evaluation_draws, code, effective_play)
        selected_hits, baseline_hits = _target_hits(selected_best, target), _target_hits(baseline_best, target)
        delta, delta_ci = _paired_ci(selected_hits, baseline_hits)
        if strategy == "uniform":
            delta, delta_ci = 0.0, (0.0, 0.0)  # Identical tickets, mathematically identical events.
        selected_count, baseline_count = int(selected_hits.sum()), int(baseline_hits.sum())
        selected_rate, baseline_rate = selected_count / n_eval, baseline_count / n_eval
        selected_interval, baseline_interval = _wilson(selected_count, n_eval), _wilson(baseline_count, n_eval)
        selected_tiers, baseline_tiers = _tier_breakdown(selected_best, max_target), _tier_breakdown(baseline_best, max_target)
        evaluation_note = "低奖组合以新的独立均匀开奖样本模拟；两个组合共享同一批评估开奖。"
    rows = [
        {"strategy": strategy, "role": "portfolio", "n_tickets": requested, "winning_draws": selected_count,
         "n_eval": n_eval, "hit_rate": selected_rate, "wilson95": selected_interval,
         "interval_method": "exact" if target == 1 else "wilson95_simulation",
         "exact_jackpot_probability": jackpot_exact, "exact_single_ticket_target_probability": single_target,
         "paired_delta_vs_uniform": delta, "paired_delta_ci95": delta_ci,
         "paired_ci_method": "exact_identity" if target == 1 or strategy == "uniform" else "hoeffding95_paired_conservative",
         "best_prize_counts": selected_tiers},
        {"strategy": "uniform", "role": "same_count_baseline", "n_tickets": requested, "winning_draws": baseline_count,
         "n_eval": n_eval, "hit_rate": baseline_rate, "wilson95": baseline_interval,
         "interval_method": "exact" if target == 1 else "wilson95_simulation",
         "exact_jackpot_probability": jackpot_exact, "exact_single_ticket_target_probability": single_target,
         "paired_delta_vs_uniform": 0.0, "paired_delta_ci95": (0.0, 0.0),
         "paired_ci_method": "reference",
         "best_prize_counts": baseline_tiers},
    ]
    selected_ci, baseline_ci = selected_interval, baseline_interval
    label = "头奖" if target == 1 else f"{target}等奖及以上"
    report = "\n".join([
        f"{game_name(code)} {effective_play} 全票组合：{requested} 注，成本 {requested * _TICKET_PRICE_YUAN} 元。",
        f"目标：{label}；互异合法票数：{requested}/{requested}。",
        f"头奖精确概率（任一注）：{jackpot_exact:.12%}；同注数任一策略在均匀开奖假设下完全相等。",
        f"单注{label}精确概率：{single_target:.12%}（组合整体低奖事件因票面重叠以模拟评估）。",
        (f"组合精确头奖概率：{selected_rate:.12%}（非模拟区间）。" if target == 1 else
         f"组合模拟：{selected_count}/{n_eval} = {selected_rate:.4%}，Wilson95% {selected_ci[0]:.4%}–{selected_ci[1]:.4%}。"),
        (f"均匀基线精确头奖概率：{baseline_rate:.12%}（非模拟区间）。" if target == 1 else
         f"均匀基线：{baseline_count}/{n_eval} = {baseline_rate:.4%}，Wilson95% {baseline_ci[0]:.4%}–{baseline_ci[1]:.4%}。"),
        ("同注数最高奖概率差：精确为0。" if target == 1 else
         f"配对差值（组合−基线）：{delta * 100:+.4f}个百分点，95%保守区间 "
         f"{delta_ci[0] * 100:+.4f}至{delta_ci[1] * 100:+.4f}个百分点（Hoeffding界；同一方案自比为精确0）。"),
        ("最高奖使用精确计数；未运行模拟奖级统计。" if target == 1 else
         f"模拟最佳奖级分布（组合）：{rows[0]['best_prize_counts']}；（基线）：{rows[1]['best_prize_counts']}。"),
        provenance, baseline_note,
        evaluation_note,
        "仅含常规奖级，不含双色球福运奖、临时派奖、追加或倍投；中奖不等于盈利。",
        "反复更换复现编号并挑选评估最高的方案会产生选择偏差，本轮评估不能再作独立验证。",
        "模拟结果、即使样本内为 100%，也不构成保证，更不证明真实预测优势。",
    ])
    return {"game": code, "game_name": game_name(code), "play": effective_play, "strategy": strategy,
            "target_prize": target, "n_tickets": requested, "cost_yuan": requested * _TICKET_PRICE_YUAN,
            "tickets": tickets, "baseline_tickets": baseline_tickets, "exact_jackpot_probability": jackpot_exact,
            "exact_single_ticket_target_probability": single_target,
            "evaluation_scope": "exact_uniform_jackpot" if target == 1 else "fresh_uniform_simulation",
            "report": report, "comparison_rows": rows, "comparison": rows, "provenance": provenance,
            "seed": seed, "n_eval": n_eval, "actual_eval_draw_count": 0 if target == 1 else n_eval,
            "design_draw_count": _DESIGN_DRAWS if strategy == "coverage" and target > 1 else 0,
            "random_streams": "SeedSequence(seed).spawn(3): selection_or_design, baseline, evaluation",
            "rule_scope": "SSQ常规六奖级；DLT现行七奖级；QXC现行位置匹配六奖级；PL3/PL5仅头奖。",
            "limits": {"operational_max_tickets": OPERATIONAL_MAX_TICKETS, "ticket_space_ceiling": ceiling}}
