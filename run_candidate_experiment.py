"""Isolated, fixed-protocol XGBoost/Markov comparison; historical exploration only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from math import comb
from pathlib import Path
import shutil

import numpy as np
from scipy.stats import binomtest

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS, game_name
from jackpot_evaluation import _build_features, _draw_at
from jackpot_selection import _game_heads, select_top_k
from local_history import read_local_history
from portfolio_engine import _canonical_ticket, _prize
from predictor import _get_probabilities
from train import _labels_to_targets
from run_evidence_experiment import digest, load_isolated_model, baseline_tickets, write_new
from evidence_service import checked_study
from evidence_statistics import exact_jackpot_probability

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "candidate_runs"
STRATEGIES = ("xgboost", "markov", "lstm")
STRATEGY_NAMES = {"xgboost": "XGBoost", "markov": "位置马尔可夫", "lstm": "冻结LSTM"}
SOURCE_FILES = ("run_candidate_experiment.py", "candidate_models.py", "bias_diagnostics.py")
SCOPE = "historical_exploration_not_prospective"
N_TEST, N_TICKETS, SEQ_LEN = 300, 5, 30


def holm_adjust(values):
    """Holm correction, keeping missing tests in the prespecified family size."""
    adjusted = [None] * len(values)
    for value in values:
        if value is not None and (isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("p值须为有限的0到1数值或None。")
    ranked = sorted((float(p), i) for i, p in enumerate(values) if p is not None)
    previous = 0.0
    for rank, (p, index) in enumerate(ranked):
        previous = max(previous, min(1.0, (len(values) - rank) * p))
        adjusted[index] = previous
    return adjusted


def fair_head_probabilities(game):
    """Correct order-statistic marginals for sorted balls, not uniform slot heads."""
    result = {}
    if game == "ssq":
        groups = [("red", 33, 6), ("blue", 16, 1)]
    elif game == "dlt":
        groups = [("front", 35, 5), ("back", 12, 2)]
    else:
        return {name: np.full(size, 1 / size) for name, size in _game_heads(game)}
    for prefix, size, count in groups:
        for position in range(1, count + 1):
            values = np.zeros(size, dtype=float)
            for ball in range(position, size - count + position + 1):
                values[ball - 1] = comb(ball - 1, position - 1) * comb(size - ball, count - position) / comb(size, count)
            result[prefix if prefix == "blue" else f"{prefix}{position}"] = values
    return result


def head_log_loss(heads, targets, game):
    """Mean marginal log loss only; never call the product a jackpot probability."""
    losses = []
    for name, size in _game_heads(game):
        probs = np.asarray(heads[name], dtype=float)
        labels = np.asarray(targets[name])
        if probs.ndim != 2 or probs.shape != (len(labels), size) or not len(labels):
            raise ValueError("位置概率/目标维度不一致。")
        if (not np.isfinite(probs).all() or np.any(probs < 0) or
                not np.allclose(probs.sum(axis=1), 1) or labels.dtype.kind not in "iu" or
                np.any(labels < 0) or np.any(labels >= size)):
            raise ValueError("位置概率或目标非法。")
        losses.append(-np.log(np.maximum(probs[np.arange(len(labels)), labels], np.finfo(float).tiny)))
    return float(np.mean(losses))


def summarize_game(game, plays):
    summaries = []
    for play, rows in plays.items():
        p0 = exact_jackpot_probability(game, N_TICKETS, play)["p0"]
        for strategy in STRATEGIES:
            successes = sum(row["hits"][strategy] for row in rows)
            test = binomtest(successes, len(rows), p0, alternative="greater")
            interval = binomtest(successes, len(rows)).proportion_ci(method="exact")
            summaries.append({"game": game, "play": play, "strategy": strategy,
                              "periods": len(rows), "wins": successes,
                              "random_wins": sum(row["hits"]["uniform"] for row in rows),
                              "p0": p0, "p_value": float(test.pvalue),
                              "ci95": [float(interval.low), float(interval.high)],
                              "verified_advantage": False})
    return summaries


def protected_snapshot():
    return {str(path.relative_to(ROOT)): digest(path)
            for folder in ("data", "models", "results", "evidence_runs")
            for path in sorted((ROOT / folder).rglob("*")) if path.is_file()}


def _canonical(tickets, game, play):
    result = [_canonical_ticket(t, game, play) for t in tickets]
    if len(result) != N_TICKETS or len({json.dumps(t, sort_keys=True) for t in result}) != N_TICKETS:
        raise ValueError("每种策略必须恰好产生5组互异合法号码。")
    return result


def _emit(callback, text):
    print(text, flush=True)
    if callback:
        callback(text)


def run_experiment(output=None, progress=None):
    # Lazy imports keep the rest of the app readable if an optional install is incomplete.
    from candidate_models import XGBoostHeads, MarkovHeads, summarize_windows, XGB_CONFIG
    from bias_diagnostics import analyze_game_bias

    xgb_version = importlib.metadata.version("xgboost")
    state, reference = checked_study()
    protected = protected_snapshot()
    reference_protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    if reference_protocol["n_test"] != N_TEST or reference_protocol["n_tickets"] != N_TICKETS:
        raise ValueError("参照研究必须使用300期、每期5组口径。")
    output = Path(output).resolve() if output else RUNS / datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    if not output.is_relative_to(RUNS.resolve()):
        raise ValueError("新实验只能写入独立candidate_runs目录。")
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshots").mkdir()
    for game in ALL_GAME_CODES:
        shutil.copy2(reference / "snapshots" / f"{game}_history.csv", output / "snapshots" / f"{game}_history.csv")
    protocol = {"schema": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": SCOPE, "n_test": N_TEST, "n_tickets": N_TICKETS, "seq_len": SEQ_LEN,
                "reference_run": reference.name, "reference_protocol_sha256": digest(reference / "protocol.json"),
                "old_prospective_protocol_hash": state["protocol"]["protocol_hash"],
                "reference_artifact_sha256": state["protocol"]["configuration"]["artifact_sha256"],
                "reference_source_sha256": state["protocol"]["configuration"]["source_sha256"],
                "strategies": list(STRATEGIES), "random_seed_rule": "evidence-v1 sha256 of game/play/issue",
                "xgboost": {"version": xgb_version, "params": XGB_CONFIG,
                            "features": "previous30: last, mean, std; no current target row"},
                "markov": {"smoothing": 1.0, "state": "previous draw same sorted/digit position", "update_during_test": False},
                "training": "Same chronological 80% prefix training targets as reference LSTM; no hyperparameter search or early stopping for new candidates.",
                "primary_test": {"family_size": 21, "method": "one-sided exact binomial vs uniform null; Holm", "alpha": 0.05},
                "bias_test": {"family_size": 38, "data": "training rows only, excluding validation and outer test",
                              "correction": "Holm across all component tests from all five games; separate exploratory family"},
                "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
                "data_sha256": {game: digest(output / "snapshots" / f"{game}_history.csv") for game in ALL_GAME_CODES},
                "protected_sha256": protected,
                "notes": ["Already inspected historical data; no independent confirmation, even if p is small.",
                          "Bias tests never select models, features or parameters in this run.",
                          "Model marginal log loss is diagnostic, not a jackpot hit probability.",
                          "No changes to existing prospective protocol, records, model files or official-data inputs.",
                          "No purchases, data uploads or automatic candidate promotion."]}
    write_new(output / "protocol.json", protocol)
    _emit(progress, f"协议已保存：{output.name}；将完整报告21个模型/玩法结果。")
    all_summaries, all_bias, score_rows = [], [], []
    for game in ALL_GAME_CODES:
        old = json.loads((reference / f"{game}_result.json").read_text(encoding="utf-8"))
        frame = read_local_history(game, output / "snapshots")
        start = len(frame) - N_TEST
        if start != old["train_prefix_rows"] or old["train_info"]["seq_len"] != SEQ_LEN:
            raise ValueError("研究数据长度或序列窗口与参照不符。")
        train_end = SEQ_LEN + int(0.8 * (start - SEQ_LEN))
        if train_end - 1 != old["feature_fit_end"]:
            raise ValueError("训练输入缩放边界与参照不符。")
        _emit(progress, f"{game_name(game)}：训练前缀偏差检验、训练两个固定候选模型。")
        all_bias.extend(analyze_game_bias(frame.iloc[:train_end].copy(), game))
        features, labels, _ = _build_features(frame, game, old["feature_fit_end"])
        prefix_features, _, _ = _build_features(frame.iloc[:start].copy(), game, old["feature_fit_end"])
        if not np.array_equal(prefix_features, features[:start]):
            raise ValueError("未来后缀改变了训练前缀特征。")
        targets = _labels_to_targets(labels, game)
        train_indices = np.arange(SEQ_LEN, train_end)
        test_indices = np.arange(start, len(frame))
        train_x = summarize_windows(features, train_indices, SEQ_LEN)
        test_x = summarize_windows(features, test_indices, SEQ_LEN)
        xgb = XGBoostHeads(game).fit(train_x, {key: values[train_indices] for key, values in targets.items()})
        markov = MarkovHeads(game, smoothing=1.0).fit(targets, start=SEQ_LEN, end=train_end)
        model_dir = output / "models" / game
        xgb.save(model_dir / "xgboost")
        markov.save(model_dir / "markov")
        xgb_probs = xgb.predict_heads(test_x)
        previous = {key: values[test_indices - 1] for key, values in targets.items()}
        markov_probs = markov.predict_heads(previous)
        # Actually restore the on-disk artifacts before accepting the run.
        for expected, restored in ((xgb_probs, XGBoostHeads.load(model_dir / "xgboost").predict_heads(test_x)),
                                   (markov_probs, MarkovHeads.load(model_dir / "markov").predict_heads(previous))):
            if any(not np.array_equal(expected[name], restored[name]) for name, _ in _game_heads(game)):
                raise ValueError("独立模型保存/恢复后输出不一致。")
        lstm = load_isolated_model(game, old["train_info"], reference / "models" / f"{game}_lstm.pth")
        lstm_lists = {name: [] for name, _ in _game_heads(game)}
        plays = {play: [] for play in GAME_PLAY_OPTIONS[game]}
        for offset, index in enumerate(test_indices):
            lstm_head = _get_probabilities(lstm, features[index - SEQ_LEN:index], SEQ_LEN)
            for name in lstm_lists:
                lstm_lists[name].append(lstm_head[name])
            heads = {"xgboost": {name: values[offset] for name, values in xgb_probs.items()},
                     "markov": {name: values[offset] for name, values in markov_probs.items()}, "lstm": lstm_head}
            issue, day = str(frame.iloc[index]["issue"]), str(frame.iloc[index]["date"])[:10]
            draw = _draw_at(frame, index, game)
            for play in plays:
                tickets = {strategy: _canonical(select_top_k(game, probabilities, N_TICKETS, play), game, play)
                           for strategy, probabilities in heads.items()}
                tickets["uniform"] = _canonical(baseline_tickets(game, play, issue), game, play)
                old_row = old["plays"][play][offset]
                if (old_row["issue"] != issue or old_row["date"] != day or
                        tickets["lstm"] != _canonical(old_row["model_tickets"], game, play) or
                        tickets["uniform"] != _canonical(old_row["baseline_tickets"], game, play)):
                    raise ValueError("LSTM/随机对照未能精确重放旧研究，拒绝比较。")
                plays[play].append({"issue": issue, "date": day, "input_last_issue": str(frame.iloc[index - 1]["issue"]),
                                    "tickets": tickets, "hits": {strategy: any(_prize(t, draw, game, play) == 1 for t in group)
                                                                for strategy, group in tickets.items()}})
            if (offset + 1) % 100 == 0:
                _emit(progress, f"{game_name(game)}：已评估{offset + 1}/300期。")
        actual = {name: values[test_indices] for name, values in targets.items()}
        fair = {name: np.tile(values, (N_TEST, 1)) for name, values in fair_head_probabilities(game).items()}
        for strategy, probabilities in (("xgboost", xgb_probs), ("markov", markov_probs),
                                         ("lstm", {name: np.asarray(values) for name, values in lstm_lists.items()}), ("fair_marginals", fair)):
            score_rows.append({"game": game, "strategy": strategy, "head_log_loss": head_log_loss(probabilities, actual, game),
                               "interpretation": "位置边际概率诊断，非整注中奖概率；越低越好。"})
        write_new(output / f"{game}_result.json", {"game": game, "train_target_start": SEQ_LEN,
                  "train_target_end_exclusive": train_end, "validation_start": train_end, "outer_test_start": start,
                  "training_last_issue": str(frame.iloc[train_end - 1]["issue"]), "plays": plays,
                  "model_restore_identical": True, "reference_replay_identical": True})
        all_summaries.extend(summarize_game(game, plays))
    if len(all_summaries) != 21 or len(all_bias) != 38:
        raise ValueError("必须报告全部21项模型结果与38项诊断，不能挑选玩法或缩小检验家族。")
    for row, adjusted in zip(all_summaries, holm_adjust([row["p_value"] for row in all_summaries])):
        row["p_holm"] = adjusted
        row["historical_signal_only"] = adjusted < 0.05
    for row, adjusted in zip(all_bias, holm_adjust([row["p_value"] for row in all_bias])):
        row["p_holm"] = adjusted
        row["diagnostic_flag_only"] = adjusted is not None and adjusted < 0.05
    if protected_snapshot() != protected:
        raise ValueError("原数据/模型/结果/旧研究文件被改动；本轮不能发布成功报告。")
    checked_study()
    if any(digest(ROOT / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("实验期间新候选源码被改动，不能发布成功报告。")
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    report = {"schema": 1, "scope": SCOPE, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": digest(output / "protocol.json"), "artifact_sha256": artifacts,
              "summary": all_summaries, "bias_diagnostics": all_bias, "head_scores": score_rows,
              "protected_files": len(protected), "protected_unchanged": True, "verified_advantage": False,
              "conclusion": "历史探索不构成未来一等奖预测优势；不自动替换算法或改变旧前瞻协议。"}
    write_new(output / "report.json", report)
    _emit(progress, f"已完成：{output.name}；原数据/模型/记录未改动。")
    return output


def read_report(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(RUNS.resolve()):
        raise ValueError("报告目录必须位于candidate_runs。")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    expected_keys = {(game, play, strategy) for game in ALL_GAME_CODES
                     for play in GAME_PLAY_OPTIONS[game] for strategy in STRATEGIES}
    actual_keys = {(row.get("game"), row.get("play"), row.get("strategy")) for row in report.get("summary", [])}
    if (report.get("scope") != SCOPE or len(report.get("summary", [])) != 21 or actual_keys != expected_keys
            or len(report.get("bias_diagnostics", [])) != 38 or len(report.get("head_scores", [])) != 20):
        raise ValueError("研究报告结构或范围不完整。")
    required = {"protocol.json", *[f"{game}_result.json" for game in ALL_GAME_CODES],
                *[f"snapshots/{game}_history.csv" for game in ALL_GAME_CODES]}
    if not required.issubset(report.get("artifact_sha256", {})):
        raise ValueError("研究工件清单不完整。")
    if digest(folder / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("报告协议指纹不匹配。")
    for name, expected in report["artifact_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or digest(path) != expected:
            raise ValueError("研究工件缺失、变化或路径非法。")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_experiment(args.output)
