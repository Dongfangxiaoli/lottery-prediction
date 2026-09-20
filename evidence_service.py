"""Application bridge for isolated, fixed-model prospective research."""
from datetime import date, datetime
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

from evidence_registry import create_study, read_study, commit_prediction, score_study, REQUIRED_CONFIGURATION
from evidence_statistics import evaluate_preregistered_horizon, estimate_game_horizon
from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS, game_code, game_name
from jackpot_evaluation import _build_features
from jackpot_selection import select_top_k
from local_history import read_local_history
from model_session import _data_fingerprint
from portfolio_engine import _canonical_ticket
from predictor import _get_probabilities, format_numbers_copy
from run_evidence_experiment import digest, load_isolated_model, baseline_tickets, SOURCE_FILES

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "evidence_runs"
STUDY = RUNS / "studies" / "default"
SERVICE_SOURCES = ("evidence_service.py", "evidence_registry.py", "evidence_statistics.py", "model_session.py")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_completed_experiment():
    matches = sorted(p for p in RUNS.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]{17}", p.name)
                     and (p / "report.json").is_file()) if RUNS.is_dir() else []
    if not matches:
        raise ValueError("尚无完成的固定参数实验，请先运行 run_evidence_experiment.py。")
    return matches[0]  # Never automatically select the run with the best score.


def freeze_study():
    if STUDY.exists():
        state, _ = checked_study()
        return f"已有冻结协议，未覆盖。记录数：{len(state['records'])}。本地校验不是第三方时间戳。"
    folder = first_completed_experiment()
    report, protocol = _json(folder / "report.json"), _json(folder / "protocol.json")
    if digest(folder / "protocol.json") != report["protocol_sha256"]:
        raise ValueError("实验报告与协议不匹配。")
    if any(digest(ROOT / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("实验核心算法已变化，不能冻结为同一版本。")
    artifacts = {}
    for game in ALL_GAME_CODES:
        for relative in (f"{game}_result.json", f"models/{game}_lstm.pth", f"snapshots/{game}_history.csv"):
            artifacts[relative] = digest(folder / relative)
        result = _json(folder / f"{game}_result.json")
        if artifacts[f"models/{game}_lstm.pth"] != result["weight_sha256"]:
            raise ValueError("研究权重已变化，拒绝冻结。")
        if artifacts[f"snapshots/{game}_history.csv"] != protocol["data_sha256"][game]:
            raise ValueError("研究历史快照已变化，拒绝冻结。")
    state = create_study(STUDY, {**REQUIRED_CONFIGURATION, "experiment_id": folder.name,
                                "experiment_protocol_sha256": report["protocol_sha256"],
                                "artifact_sha256": artifacts,
                                "source_sha256": {**protocol["source_sha256"],
                                                  **{name: digest(ROOT / name) for name in SERVICE_SOURCES}}})
    return (f"已冻结7种玩法：每期5组实验号码，每玩法1000个预先登记目标期；不是建议购买量。\n"
            f"模型不重训、不挑选种子或最佳历史运行。协议哈希：{state['protocol']['protocol_hash']}\n"
            "1000期不代表足以证明微小优势，尤其百万分之一的头奖。尚无未来开奖证据；本地文件不能替代第三方存证。")


def checked_study():
    state = read_study(STUDY)
    config = state["protocol"]["configuration"]
    run_id = config.get("experiment_id", "")
    if not re.fullmatch(r"[0-9]{17}", run_id):
        raise ValueError("研究运行目录标识无效。")
    folder = RUNS / run_id
    if digest(folder / "protocol.json") != config["experiment_protocol_sha256"]:
        raise ValueError("冻结的实验协议已变化。")
    expected_sources = set(SOURCE_FILES) | set(SERVICE_SOURCES)
    if set(config["source_sha256"]) != expected_sources:
        raise ValueError("冻结源码文件清单不完整。")
    for name, value in config["source_sha256"].items():
        if digest(ROOT / name) != value:
            raise ValueError("冻结算法源码已变化，请保留旧实验，不要混入新版本成绩。")
    expected_artifacts = {name for g in ALL_GAME_CODES for name in
                          (f"{g}_result.json", f"models/{g}_lstm.pth", f"snapshots/{g}_history.csv")}
    if set(config["artifact_sha256"]) != expected_artifacts:
        raise ValueError("冻结工件清单不完整。")
    for name, value in config["artifact_sha256"].items():
        if digest(folder / name) != value:
            raise ValueError(f"冻结研究工件已变化：{name}。")
    return state, folder


def current_history(game):
    """Prefer the newer complete local snapshot; never update either input."""
    candidates = []
    for directory in (ROOT / "data", RUNS / "live_data"):
        if (directory / f"{game}_history.csv").is_file():
            frame = read_local_history(game, directory)
            candidates.append((str(frame.iloc[-1]["date"])[:10], int(frame.iloc[-1]["issue"]), frame,
                               directory / f"{game}_history.csv"))
    if not candidates:
        raise ValueError("缺少历史数据。")
    chosen = max(candidates, key=lambda x: x[:2])
    return chosen[2], chosen[3]


def validate_target(cutoff_issue, cutoff_date, target_issue, target_date, today):
    target_day = date.fromisoformat(target_date)
    if target_day <= today:
        raise ValueError("为避免已开奖后补录，当前版本只登记明天或以后的目标日期。")
    if not 0 <= (today - date.fromisoformat(cutoff_date)).days <= 4:
        raise ValueError(f"数据截止{cutoff_date}，过旧或晚于今天，请先联网更新并核对。")
    if (not re.fullmatch(r"[0-9]{5}", target_issue)
            or int(target_issue) != int(cutoff_issue) + 1
            or target_day.year != date.fromisoformat(cutoff_date).year):
        raise ValueError("仅支持当前数据紧接的下一期五位期号；跨年首期暂需人工审查，不允许跳期。")
    if not 1 <= (target_day - date.fromisoformat(cutoff_date)).days <= 4:
        raise ValueError("目标日期应紧接现有数据，不能跳过较近开奖。")


def record_next_prediction(game, play, target_issue, target_date):
    code = game_code(game)
    if play not in GAME_PLAY_OPTIONS[code]:
        raise ValueError("请选择对应彩种的有效玩法。")
    state, folder = checked_study()
    info = _json(folder / f"{code}_result.json")
    df, history_path = current_history(code)
    input_digest = digest(history_path)
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    cutoff_date, cutoff_issue = str(df.iloc[-1]["date"])[:10], str(df.iloc[-1]["issue"])
    target_issue, target_date = str(target_issue).strip(), str(target_date).strip()
    validate_target(cutoff_issue, cutoff_date, target_issue, target_date, today)
    start = info["train_prefix_rows"]
    snapshot = read_local_history(code, folder / "snapshots")
    if _data_fingerprint(code, snapshot.iloc[:start]) != _data_fingerprint(code, df.iloc[:start]):
        raise ValueError("训练前缀被修订，不能继续这个冻结模型实验。")
    features, _, _ = _build_features(df, code, info["feature_fit_end"])
    if not np.isfinite(features).all():
        raise ValueError("推断特征存在非有限值。")
    model = load_isolated_model(code, info["train_info"], folder / "models" / f"{code}_lstm.pth")
    probs = _get_probabilities(model, features, info["train_info"]["seq_len"])
    count = state["protocol"]["configuration"]["n_tickets"]
    tickets = [_canonical_ticket(t, code, play) for t in select_top_k(code, probs, count, play=play)]
    baseline = baseline_tickets(code, play, target_issue, count)
    # Verify files again after inference, before the irreversible append.
    checked_study()
    if digest(history_path) != input_digest:
        raise ValueError("推断过程中历史文件发生变化，未写入记录。")
    record = commit_prediction(STUDY, code, play, target_issue, target_date, cutoff_issue, cutoff_date,
                               tickets, baseline, {"experiment_id": folder.name,
                                                   "weight_sha256": info["weight_sha256"],
                                                   "input_sha256": input_digest,
                                                   "data_source": str(history_path.relative_to(ROOT)),
                                                   "latest_official_data_verified": False})
    return (f"已登记第{target_issue}期，目标日期{target_date}；禁止覆盖或反复选号。\n"
            f"记录哈希：{record['record_hash']}\n本地时间不是可信第三方存证；请将原始协议和记录提前交由独立方保存。\n"
            f"数据截止{cutoff_date}，目标期日期仍需官方核对。\n\n模型实验组：\n"
            f"{format_numbers_copy(code, tickets, play)}\n\n同注数随机对照：\n{format_numbers_copy(code, baseline, play)}")


def research_summary():
    folder = first_completed_experiment()
    report = _json(folder / "report.json")
    rows = []
    for row in report["summary"]:
        stats = evaluate_preregistered_horizon(row["game"], 5, row["model_wins"], row["periods"],
                                               row["periods"], row["play"], historical_exploration=True,
                                               preregistered=False)
        rows.append([game_name(row["game"]), row["play"], row["periods"], row["model_wins"],
                     row["baseline_wins"], f"{row['p0']:.9%}",
                     f"{stats['pvalue_greater']:.6g}", "历史探索，非独立验证"])
    return ("固定参数的历史时间切分对照；训练、早停和缩放器均不使用目标期。\n"
            "过去的数据已经被查看，以下p值仅作探索；不挑最优彩种，不以小奖/部分号码命中代替一等奖。", rows)


def sample_size_summary():
    rows = []
    for code in ALL_GAME_CODES:
        for play in GAME_PLAY_OPTIONS[code]:
            plan = estimate_game_horizon(code, 5, play)
            power = plan["power_plan"]
            rows.append([game_name(code), play, f"{plan['baseline']['p0']:.9%}",
                         power["trials"], power["critical_successes"], f"{power['power']:.2%}"])
    return rows


def prospective_summary():
    state, _ = checked_study()
    histories = {code: current_history(code)[0] for code in ALL_GAME_CODES}
    report = score_study(STUDY, histories)
    rows = []
    for key, item in report["per_play"].items():
        code, play = key.split(":", 1)
        stats = evaluate_preregistered_horizon(code, 5, item["model_first_prize_wins"],
                                               item["scored_periods"], item["horizon"], play,
                                               evidence_provenance="local_only")
        problems = sorted(set(item["statuses"]) - {"scored"})
        rows.append([game_name(code), play, item["selected_periods"], item["scored_periods"],
                     item["model_first_prize_wins"], item["baseline_first_prize_wins"],
                     ", ".join(problems) if problems else stats["conclusion"]])
    return f"本地前瞻记录共{len(state['records'])}条；尚无独立第三方开奖前存证，不能宣布已验证优势。", rows
