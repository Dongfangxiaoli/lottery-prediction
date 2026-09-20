"""Exploratory marginal and lag-one lottery diagnostics, not prediction evidence.

The Joe (1993) correction below is for *set* margins: a lottery pool is not
treated as six (or five) independent ordered positions.  These diagnostics do
not prove complete randomness and must not be used to rank tickets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from game_config import DIGIT_GAME_CONFIGS, game_code

SCOPE = "exploratory_diagnostic_not_prediction"
_SEED = 20260912
_PERMUTATIONS = 999


def _result(game, component, label, n_draws, statistic, p_value, method, effect):
    return {
        "game": game, "component": component, "label": label,
        "n_draws": int(n_draws), "statistic": None if statistic is None else float(statistic),
        "p_value": None if p_value is None else float(p_value), "method": method,
        "effect_summary": effect, "scope": SCOPE,
    }


def _integer_column(frame: pd.DataFrame, column: str, lower: int, upper: int) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"历史数据缺少列: {column}")
    series = frame[column]
    if series.isna().any() or series.map(lambda x: isinstance(x, (bool, np.bool_))).any():
        raise ValueError(f"{column} 含空缺或非法布尔值")
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
            or not ((lower <= values) & (values <= upper)).all()):
        raise ValueError(f"{column} 必须是 {lower}–{upper} 范围内的有限整数")
    return values.astype(int)


def _validated_pool(frame: pd.DataFrame, columns: tuple[str, ...], upper: int) -> np.ndarray:
    values = np.column_stack([_integer_column(frame, col, 1, upper) for col in columns])
    # Deliberately do not require a presentation order; a pool is a set.
    if any(len(set(row)) != len(row) for row in values):
        raise ValueError("球彩每一区号码必须不重复")
    return values


def _joe(counts: np.ndarray, n_draws: int, k: int) -> tuple[float, float]:
    """Joe (1993) one-number-margin statistic: (N-1)/(N-k) times Pearson X².

    Boland and Pawitan (1999), Eq. (1), explicitly gives this correction and
    its asymptotic chi-square(N-1) reference for lotto sets.
    """
    categories = len(counts)
    expected = n_draws * k / categories
    raw = np.sum((counts - expected) ** 2 / expected)
    return (categories - 1) / (categories - k) * raw, expected


def _category_counts(rows: np.ndarray, categories: int, lower: int) -> np.ndarray:
    return np.bincount(rows.ravel() - lower, minlength=categories)[:categories]


def _marginal(game: str, component: str, label: str, rows: np.ndarray,
              categories: int, k: int, lower: int = 1) -> dict:
    n_draws = len(rows)
    counts = _category_counts(rows, categories, lower)
    statistic, expected = _joe(counts, n_draws, k)
    if expected < 5:
        method = ("Joe (1993) corrected set-margin chi-square; asymptotic p-value "
                  "withheld because expected cell count is below 5")
        p_value = None
    else:
        method = "Joe (1993) corrected set-margin chi-square; chi-square asymptotic reference"
        p_value = chi2.sf(statistic, categories - 1)
    prefix = "insufficient_sample; " if p_value is None else ""
    effect = (f"{prefix}marginal counts range {int(counts.min())}–{int(counts.max())}; "
              f"uniform expectation {expected:.3g} per number")
    return _result(game, component, f"{label}边际均匀性", n_draws, statistic, p_value, method, effect)


def _lag_stat_sets(rows: np.ndarray) -> float:
    return float(np.mean([len(set(a).intersection(b)) for a, b in zip(rows[:-1], rows[1:])]))


def _lag_stat_digits(values: np.ndarray) -> float:
    return float(np.mean(values[:-1] == values[1:]))


def _lag_diagnostic(game: str, component: str, label: str, rows: np.ndarray,
                    categories: int, is_set: bool, lower: int = 1) -> dict:
    n_draws = len(rows)
    if n_draws < 20:
        return _result(game, component, f"{label}跨期相邻独立性", n_draws, None, None,
                       "lag-one order-permutation test; insufficient sample (requires n_draws >= 20)",
                       "insufficient_sample")
    values = rows if is_set else rows.ravel()
    counts = _category_counts(values, categories, lower)
    expected = float(np.sum(counts * (counts - 1)) / (n_draws * (n_draws - 1)))
    observed = _lag_stat_sets(rows) if is_set else _lag_stat_digits(values)
    rng = np.random.default_rng(_SEED)  # local, reproducible, and does not alter caller/global RNG state
    simulated = np.empty(_PERMUTATIONS, dtype=float)
    for index in range(_PERMUTATIONS):
        permuted = rows[rng.permutation(n_draws)]
        simulated[index] = _lag_stat_sets(permuted) if is_set else _lag_stat_digits(permuted.ravel())
    tail = np.count_nonzero(np.abs(simulated - expected) >= abs(observed - expected))
    p_value = (tail + 1) / (_PERMUTATIONS + 1)
    metric = "mean adjacent-set overlap" if is_set else "adjacent equal-value rate"
    method = "lag-one order permutation, 999 fixed-seed permutations, two-sided conditional tail"
    effect = f"{metric} {observed:.4g}; conditional random-order expectation {expected:.4g}"
    return _result(game, component, f"{label}跨期相邻独立性", n_draws, observed, p_value, method, effect)


def _pool_results(game: str, frame: pd.DataFrame, component: str, label: str,
                  columns: tuple[str, ...], categories: int) -> list[dict]:
    rows = _validated_pool(frame, columns, categories)
    return [_marginal(game, component, label, rows, categories, rows.shape[1]),
            _lag_diagnostic(game, component, label, rows, categories, True)]


def _digit_results(game: str, frame: pd.DataFrame) -> list[dict]:
    results: list[dict] = []
    config = DIGIT_GAME_CONFIGS[game]
    for position, (column, categories) in enumerate(zip(config["digits"], config["classes"]), 1):
        rows = _integer_column(frame, column, 0, categories - 1).reshape(-1, 1)
        component, label = column, f"第{position}位"
        results.extend((_marginal(game, component, label, rows, categories, 1, lower=0),
                        _lag_diagnostic(game, component, label, rows, categories, False, lower=0)))
    return results


def analyze_game_bias(df: pd.DataFrame, game: str) -> list[dict]:
    """Diagnose only ``df`` (usually a training prefix); never read global history.

    Returned p-values are unadjusted exploratory diagnostics.  A caller that
    presents several games/tests must apply one family-level correction (for
    example Holm) before discussing evidence.  Margins and lag one do not test
    all forms of lottery randomness.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须是 pandas.DataFrame")
    if df.empty:
        raise ValueError("没有可诊断的历史数据")
    code = game_code(game)
    if code == "ssq":
        return (_pool_results(code, df, "red", "红球", tuple(f"red{i}" for i in range(1, 7)), 33)
                + _pool_results(code, df, "blue", "蓝球", ("blue",), 16))
    if code == "dlt":
        return (_pool_results(code, df, "front", "前区", tuple(f"front{i}" for i in range(1, 6)), 35)
                + _pool_results(code, df, "back", "后区", ("back1", "back2"), 12))
    return _digit_results(code, df)
