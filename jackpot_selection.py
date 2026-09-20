"""Deterministic top-five full-ticket selection from model output heads.

The scores produced here are *uncalibrated model joint quality scores*.  They
rank legal tickets under the supplied classifier heads; they are not real-world
winning probabilities and do not establish any predictive advantage.
"""
from __future__ import annotations

from itertools import combinations, permutations, product
from math import exp, log
from numbers import Integral
from typing import Iterable, Mapping, Sequence

import numpy as np

from game_config import DIGIT_GAME_CONFIGS, game_code, is_digit_game, normalize_play


TOP_FIVE = 5
_LOG_FLOAT_TINY = log(np.finfo(float).tiny)


def _validated_head(values, expected_size: int, label: str) -> np.ndarray:
    """Validate and normalize one classifier head without changing its ranking."""
    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}概率必须是一维数值向量。") from exc
    if vector.ndim != 1 or vector.size != expected_size:
        actual = vector.size if vector.ndim == 1 else vector.shape
        raise ValueError(f"{label}概率维度应为 {expected_size}，实际为 {actual}。")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label}概率必须全部为有限数值。")
    if np.any(vector < 0):
        raise ValueError(f"{label}概率不能为负数。")
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"{label}概率总和必须大于 0。")
    return vector / total


def _game_heads(code: str) -> list[tuple[str, int]]:
    if code == "ssq":
        return [(f"red{i}", 33) for i in range(1, 7)] + [("blue", 16)]
    if code == "dlt":
        return [(f"front{i}", 35) for i in range(1, 6)] + [
            (f"back{i}", 12) for i in range(1, 3)
        ]
    if is_digit_game(code):
        return [(f"digit{i}", size) for i, size in enumerate(
            DIGIT_GAME_CONFIGS[code]["classes"], start=1
        )]
    raise ValueError(f"不支持的彩种: {code}")


def _normalized_heads(code: str, probs: Mapping[str, object]) -> dict[str, np.ndarray]:
    if not isinstance(probs, Mapping):
        raise ValueError("probs必须是按输出头名称索引的映射。")
    normalized: dict[str, np.ndarray] = {}
    for name, size in _game_heads(code):
        if name not in probs:
            raise ValueError(f"缺少概率输出头: {name}。")
        normalized[name] = _validated_head(probs[name], size, name)
    return normalized


def _path_sort_key(item: tuple[float, tuple[int, ...]]) -> tuple[float, tuple[int, ...]]:
    """Descending log score, then lexicographic ticket order for stable ties."""
    return (-item[0], item[1])


def _take_top(paths: Iterable[tuple[float, tuple[int, ...]]], k: int = TOP_FIVE) -> list[tuple[float, tuple[int, ...]]]:
    return sorted(paths, key=_path_sort_key)[:k]


def _validated_k(k: int) -> int:
    """Validate the public top-k request without silently coercing values."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral):
        raise ValueError("k必须是1到500之间的整数。")
    k = int(k)
    if not 1 <= k <= 500:
        raise ValueError("k必须是1到500之间的整数。")
    return k


def _top_k_sorted(heads: Sequence[np.ndarray], k: int = TOP_FIVE) -> list[tuple[float, tuple[int, ...]]]:
    """Exact top-k strictly increasing paths for ordered ball-position heads.

    The DP keeps k paths ending in each number.  Future legality depends only on
    that ending number, so retaining k there cannot remove a global top-k path.
    Indices in returned paths are zero based.
    """
    if not heads:
        return [(0.0, ())]
    states: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(len(heads[0]))]
    for value, probability in enumerate(heads[0]):
        if probability > 0:
            states[value] = [(float(np.log(probability)), (value,))]
    for head in heads[1:]:
        next_states: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(len(head))]
        for value, probability in enumerate(head):
            if probability <= 0:
                continue
            log_probability = float(np.log(probability))
            extensions = (
                (score + log_probability, path + (value,))
                for previous in range(value)
                for score, path in states[previous]
            )
            next_states[value] = _take_top(extensions, k)
        states = next_states
    return _take_top((path for ending in states for path in ending), k)


def _top_k_independent(heads: Sequence[np.ndarray], k: int = TOP_FIVE) -> list[tuple[float, tuple[int, ...]]]:
    """Exact top-k independent positional paths (a small deterministic beam)."""
    paths: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for head in heads:
        paths = _take_top(
            ((score + float(np.log(probability)), prefix + (value,))
             for score, prefix in paths
             for value, probability in enumerate(head)
             if probability > 0),
            k,
        )
    return paths


def _logsumexp(values: Iterable[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + log(sum(exp(value - maximum) for value in finite))


def _require_k(candidates: Sequence[object], label: str, k: int) -> None:
    if len(candidates) < k:
        if k == TOP_FIVE:
            raise ValueError(
                f"{label}在当前概率支持下仅有 {len(candidates)} 个合法整票候选，不足固定的 {TOP_FIVE} 注。"
            )
        raise ValueError(
            f"{label}在当前概率支持下仅有 {len(candidates)} 个合法整票候选，不足请求的 {k} 注。"
        )


def _require_five(candidates: Sequence[object], label: str) -> None:
    _require_k(candidates, label, TOP_FIVE)


def _compat_score(joint_log_score: float) -> float:
    """Compatibility-only exponential score; underflow is truncated, not calibrated."""
    return float(np.exp(max(joint_log_score, _LOG_FLOAT_TINY)))


def _assemble_ball_tickets(
    primary: Sequence[tuple[float, tuple[int, ...]]],
    secondary: Sequence[tuple[float, tuple[int, ...]]],
    primary_name: str,
    secondary_name: str,
    k: int = TOP_FIVE,
    strategy: str = "joint_top5",
) -> list[dict]:
    ranked = sorted(
        ((primary_score + secondary_score, primary_path, secondary_path)
         for primary_score, primary_path in primary for secondary_score, secondary_path in secondary),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:k]
    _require_k(ranked, "球彩", k)
    result = []
    for score, primary_path, secondary_path in ranked:
        ticket = {
            primary_name: [value + 1 for value in primary_path],
            secondary_name: ([value + 1 for value in secondary_path]
                             if len(secondary_path) > 1 else secondary_path[0] + 1),
            "joint_log_score": float(score),
            "prob": _compat_score(float(score)),
            "selection_strategy": strategy,
        }
        result.append(ticket)
    return result


def _top_pl3_grouped(heads: Sequence[np.ndarray], play: str, k: int = TOP_FIVE) -> list[tuple[float, tuple[int, ...]]]:
    if play == "组选6":
        groups = combinations(range(10), 3)
    elif play == "组选3":
        groups = ((repeat, repeat, single) for repeat in range(10) for single in range(10) if repeat != single)
    else:
        raise ValueError(f"排列3玩法不支持组选计算: {play}")

    ranked: list[tuple[float, tuple[int, ...]]] = []
    for group in groups:
        canonical = tuple(sorted(group))
        perm_scores = []
        for ordered in set(permutations(canonical)):
            if all(heads[position][digit] > 0 for position, digit in enumerate(ordered)):
                perm_scores.append(sum(float(np.log(heads[position][digit]))
                                       for position, digit in enumerate(ordered)))
        score = _logsumexp(perm_scores)
        if np.isfinite(score):
            ranked.append((score, canonical))
    return _take_top(ranked, k)


def select_top_k(game, probs, k, play=None) -> list[dict]:
    """Return exactly ``k`` legal tickets ranked by uncalibrated joint score.

    ``k`` is deliberately bounded to keep full-ticket cross products small and
    deterministic.  Scores are model ranking values, not winning probabilities.
    """
    k = _validated_k(k)
    code = game_code(game)
    heads = _normalized_heads(code, probs)
    strategy = "joint_topk"

    if code == "ssq":
        red = _top_k_sorted([heads[f"red{i}"] for i in range(1, 7)], k)
        blue = _top_k_independent([heads["blue"]], k)
        return _assemble_ball_tickets(red, blue, "red", "blue", k, strategy)
    if code == "dlt":
        front = _top_k_sorted([heads[f"front{i}"] for i in range(1, 6)], k)
        back = _top_k_sorted([heads[f"back{i}"] for i in range(1, 3)], k)
        return _assemble_ball_tickets(front, back, "front", "back", k, strategy)

    effective_play = normalize_play(code, play)
    digit_heads = [heads[f"digit{i}"] for i in range(1, len(DIGIT_GAME_CONFIGS[code]["classes"]) + 1)]
    if code == "pls" and effective_play in ("组选3", "组选6"):
        ranked = _top_pl3_grouped(digit_heads, effective_play, k)
    else:
        ranked = _top_k_independent(digit_heads, k)
    _require_k(ranked, f"{code}{effective_play}", k)
    return [
        {"digits": list(path), "joint_log_score": float(score),
         "prob": _compat_score(float(score)), "selection_strategy": strategy}
        for score, path in ranked
    ]


def select_top_five(game, probs, play=None) -> list[dict]:
    """Return exactly five legal full tickets with the largest model joint score.

    `prob` is retained only for legacy UI compatibility.  Both it and
    `joint_log_score` are uncalibrated model-derived ranking values, never a
    real probability of winning.
    """
    # Keep the legacy strategy label and error wording while sharing the exact
    # variable-k implementation.
    result = select_top_k(game, probs, TOP_FIVE, play)
    for ticket in result:
        ticket["selection_strategy"] = "joint_top5"
    return result
