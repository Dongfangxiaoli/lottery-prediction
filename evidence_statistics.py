"""Pre-registered, future-only evidence statistics for jackpot hit rates.

This module deliberately has no file I/O.  A caller must freeze the game,
play, ticket count and future horizon *before* observing the corresponding
draws, then pass only the resulting period-level jackpot hits here.  It cannot
turn historical exploration or a local timestamp into independent evidence.

Statistical references (accessed 2026-09-11):
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from scipy.stats import binom, binomtest

from game_config import game_code, game_name, normalize_play
from portfolio_engine import _outcome_count, _permutation_count, _ticket_space_count


FAMILYWISE_ALPHA = 0.05
BONFERRONI_TEST_COUNT = 7
BONFERRONI_ALPHA = FAMILYWISE_ALPHA / BONFERRONI_TEST_COUNT
TARGET_POWER = 0.80
RELATIVE_LIFT = 2.0
MAX_HORIZON = 1_000_000_000
SCIPY_SOURCES = {
    "binomtest": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html",
    "clopper_pearson": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html",
    "binom": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html",
}


def _positive_int(value: object, label: str, *, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}必须是整数。")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        upper = f"–{maximum}" if maximum is not None else "以上"
        raise ValueError(f"{label}应在 {minimum}{upper} 之间。")
    return value


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是 0 到 1 之间的数值。")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{label}应大于0且不超过1。")
    return value


def _open_probability(value: object, label: str) -> float:
    """Validate a finite probability strictly inside (0, 1)."""
    value = _probability(value, label)
    if not value < 1.0:
        raise ValueError(f"{label}应大于0且小于1。")
    return value


def _relative_lift(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("relative_lift必须是大于1的有限数值。")
    value = float(value)
    if not isfinite(value) or value <= 1.0:
        raise ValueError("relative_lift必须是大于1的有限数值。")
    return value


def exact_jackpot_probability(game: str, n_tickets: int, play: str | None = None) -> dict[str, Any]:
    """Return the exact period-level jackpot probability for legal distinct tickets.

    The calculation reuses ``portfolio_engine``'s ticket-space and outcome
    counting rules.  It is valid only under uniform independent draws and only
    when the same number of distinct legal full tickets is used every period.
    """
    code = game_code(game)
    effective_play = normalize_play(code, play)
    count = _positive_int(n_tickets, "n_tickets")
    ceiling = _ticket_space_count(code, effective_play)
    if count > ceiling:
        raise ValueError(f"{game_name(code)}{effective_play}最多只有 {ceiling} 注互异合法整票。")
    favorable_per_ticket = _permutation_count(code, effective_play)
    outcome_count = _outcome_count(code)
    probability = count * favorable_per_ticket / outcome_count
    return {
        "game": code,
        "game_name": game_name(code),
        "play": effective_play,
        "n_tickets": count,
        "ticket_space_ceiling": ceiling,
        "favorable_outcomes_per_ticket": favorable_per_ticket,
        "outcome_count": outcome_count,
        "p0": probability,
        "assumption": "每期开奖均匀且相互独立；每期使用固定数量的互异合法整票。",
    }


def _clopper_pearson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval, separate from the one-sided test."""
    result = binomtest(successes, n=trials, alternative="two-sided")
    interval = result.proportion_ci(confidence_level=0.95, method="exact")
    return (float(interval.low), float(interval.high))


def _critical_successes(trials: int, p0: float, alpha: float) -> int:
    """Smallest k for which P_0[X >= k] <= alpha; n+1 means no rejection."""
    candidate = min(trials + 1, max(1, int(binom.isf(alpha, trials, p0)) + 1))
    # Quantile conventions and floating-point endpoints are normalized against
    # the actual tail inequality used by the registered test.
    while candidate > 1 and binom.sf(candidate - 2, trials, p0) <= alpha:
        candidate -= 1
    while candidate <= trials and binom.sf(candidate - 1, trials, p0) > alpha:
        candidate += 1
    return candidate


def power_for_horizon(trials: int, p0: float, *, alpha: float = BONFERRONI_ALPHA,
                      relative_lift: float = RELATIVE_LIFT) -> dict[str, float | int]:
    """Exact power of the registered one-sided binomial test at a given horizon."""
    n = _positive_int(trials, "trials", maximum=MAX_HORIZON)
    null_probability = _probability(p0, "p0")
    alpha = _open_probability(alpha, "alpha")
    lift = _relative_lift(relative_lift)
    alternative_probability = null_probability * lift
    if alternative_probability > 1.0:
        raise ValueError("相对提升后的命中概率超过1，无法计算该替代假设。")
    critical = _critical_successes(n, null_probability, alpha)
    power = 0.0 if critical > n else float(binom.sf(critical - 1, n, alternative_probability))
    return {"trials": n, "p0": null_probability, "p1": alternative_probability,
            "alpha": alpha, "critical_successes": critical, "power": power}


def estimate_required_horizon(p0: float, *, alpha: float = BONFERRONI_ALPHA,
                              target_power: float = TARGET_POWER,
                              relative_lift: float = RELATIVE_LIFT,
                              max_horizon: int = MAX_HORIZON) -> dict[str, Any]:
    """Find the smallest horizon meeting target power by exact integer search.

    Search progresses through ranges with the same critical rejection count.
    Power is monotone within each such range, so the function checks every
    possible rejection region in ascending ``n`` rather than assuming that an
    exact discrete-test power curve is globally monotone.  This keeps calls to
    SciPy logarithmic per critical-count range and remains bounded at 1e9.
    """
    null_probability = _probability(p0, "p0")
    alpha = _open_probability(alpha, "alpha")
    target_power = _probability(target_power, "target_power")
    upper = _positive_int(max_horizon, "max_horizon", maximum=MAX_HORIZON)
    lift = _relative_lift(relative_lift)
    p1 = null_probability * lift
    if p1 > 1.0:
        raise ValueError("相对提升后的命中概率超过1，无法计算该替代假设。")

    start, segments = 1, 0
    while start <= upper:
        segments += 1
        if segments > 100_000:  # Defensive bound; normal 2x-lift cases use few ranges.
            raise RuntimeError("精确搜索的临界区间过多，已停止以控制计算开销。")
        critical = _critical_successes(start, null_probability, alpha)
        low, high = start, upper
        # Critical counts never decrease with n.  Locate this count's final n.
        while low < high:
            middle = (low + high + 1) // 2
            if _critical_successes(middle, null_probability, alpha) <= critical:
                low = middle
            else:
                high = middle - 1
        end = low
        end_power = power_for_horizon(end, null_probability, alpha=alpha, relative_lift=lift)["power"]
        if end_power >= target_power:
            low, high = start, end
            while low < high:
                middle = (low + high) // 2
                if power_for_horizon(middle, null_probability, alpha=alpha, relative_lift=lift)["power"] >= target_power:
                    high = middle
                else:
                    low = middle + 1
            result = power_for_horizon(low, null_probability, alpha=alpha, relative_lift=lift)
            return {
                **result,
                "target_power": target_power,
                "relative_lift": lift,
                "search_method": "exact_integer_by_critical_region",
                "searched_critical_regions": segments,
                "max_horizon": upper,
                "assumption": "每期开奖均匀且相互独立，且p1 = relative_lift * p0。",
            }
        start = end + 1
    raise ValueError(f"在不超过 {upper:,} 期的上界内，达不到目标把握度。")


def estimate_game_horizon(game: str, n_tickets: int, play: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Convenience wrapper that derives p0 from the legal-ticket exact count."""
    baseline = exact_jackpot_probability(game, n_tickets, play)
    return {"baseline": baseline, "power_plan": estimate_required_horizon(baseline["p0"], **kwargs)}


def evaluate_preregistered_horizon(game: str, n_tickets: int, successes: int, trials: int,
                                   planned_horizon: int, play: str | None = None, *,
                                   preregistered: bool = True,
                                   historical_exploration: bool = False,
                                   evidence_provenance: str = "local_only") -> dict[str, Any]:
    """Evaluate one frozen future horizon without overstating evidence.

    ``successes`` is the count of *draw periods* with at least one jackpot hit,
    not the count of winning tickets.  Results are descriptive until all of the
    pre-registered horizon is observed.  Historical/exploratory input is never
    elevated to verified advantage.  ``local_only`` time records are likewise
    never described as independent evidence.
    """
    total = _positive_int(trials, "trials", minimum=0)
    horizon = _positive_int(planned_horizon, "planned_horizon")
    if total > horizon:
        raise ValueError("trials不能超过预先冻结的planned_horizon。")
    hits = _positive_int(successes, "successes", minimum=0)
    if hits > total:
        raise ValueError("successes不能超过trials。")
    if not isinstance(preregistered, bool) or not isinstance(historical_exploration, bool):
        raise ValueError("preregistered和historical_exploration必须为布尔值。")
    if evidence_provenance not in {"local_only", "independent_archive"}:
        raise ValueError("evidence_provenance只支持local_only或independent_archive。")
    baseline = exact_jackpot_probability(game, n_tickets, play)
    p0 = baseline["p0"]
    test = binomtest(hits, n=total, p=p0, alternative="greater") if total else None
    pvalue = float(test.pvalue) if test is not None else None
    ci95 = _clopper_pearson_interval(hits, total) if total else (0.0, 1.0)
    completed = total == horizon
    signal = bool(completed and pvalue is not None and pvalue <= BONFERRONI_ALPHA)
    registered_statistical_signal = bool(signal and preregistered and not historical_exploration)
    # This pure calculation module cannot inspect an archive, signature, or
    # third-party time authority.  A caller's provenance label is therefore a
    # claim, not verified independent preservation.
    eligible_for_independent_verification = registered_statistical_signal
    verified_advantage = False

    if not completed:
        status = "DESCRIPTIVE_PENDING_FROZEN_HORIZON"
        conclusion = "尚未达到预先冻结的未来期数；仅显示描述性结果，不宣布预测优势。"
    elif historical_exploration or not preregistered:
        status = "EXPLORATORY_OR_UNREGISTERED_NOT_VERIFIED"
        conclusion = "历史探索或未预注册结果不能升级为已验证的预测优势。"
    elif registered_statistical_signal and evidence_provenance == "local_only":
        status = "STATISTICAL_SIGNAL_LOCAL_ONLY_NOT_INDEPENDENTLY_VERIFIED"
        conclusion = "达到统计阈值，但本地时间记录不是独立存证；结论仍不能称已验证优势。"
    elif registered_statistical_signal:
        status = "REGISTERED_STATISTICAL_SIGNAL_INDEPENDENT_VERIFICATION_PENDING"
        conclusion = "达到登记统计阈值，但本工具未核验所声称的外部档案；仅可进入独立核验，不能称已验证优势。"
    else:
        status = "NO_REGISTERED_STATISTICAL_ADVANTAGE"
        conclusion = "完成预先冻结期数后未达到校正后的单侧检验阈值，未显示可验证的头奖预测优势。"

    return {
        "baseline": baseline,
        "successes": hits,
        "trials": total,
        "planned_horizon": horizon,
        "horizon_completed": completed,
        "observed_rate": (hits / total) if total else None,
        "p0": p0,
        "test": "scipy.stats.binomtest(alternative='greater')",
        "pvalue_greater": pvalue,
        "ci95_clopper_pearson": ci95,
        "ci_method": "scipy BinomTestResult.proportion_ci(method='exact'); two-sided 95%",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_test_count": BONFERRONI_TEST_COUNT,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "statistical_signal": signal,
        "registered_statistical_signal": registered_statistical_signal,
        "eligible_for_independent_verification": eligible_for_independent_verification,
        "verified_advantage": verified_advantage,
        "preregistered": preregistered,
        "historical_exploration": historical_exploration,
        "evidence_provenance": evidence_provenance,
        "status": status,
        "conclusion": conclusion,
        "assumptions": [
            "每期开奖均匀且相互独立。",
            "每期固定使用同一数量的互异合法整票；successes按每期任一注头奖命中计数。",
            "七个固定彩种玩法同时检验，使用Bonferroni校正。",
            "统计显著不等于可盈利，也不证明开奖机制存在可利用模式。",
        ],
        "sources": SCIPY_SOURCES,
    }
