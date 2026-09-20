"""Local-only, append-only registry for prospective lottery experiments.

This is an audit aid, not a source of trusted timestamping.  The machine clock
and files are controlled locally: a valid chain detects accidental/intermediate
edits, but cannot prove a prediction existed before a draw, detect tail
deletion, or resist someone who rewrites the complete local chain.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from game_config import DIGIT_GAME_CONFIGS, GAME_PLAY_OPTIONS, game_code, normalize_play
from portfolio_engine import _canonical_ticket, _prize, _ticket_key


SCHEMA_VERSION = 1
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
REQUIRED_CONFIGURATION = {"n_tickets": 5, "horizon": 1000, "family_size": 7, "alpha": 0.05}
FIXED_FAMILY = tuple((game, play) for game, plays in GAME_PLAY_OPTIONS.items() for play in plays)
_RECORD_NAME = re.compile(r"\d{6}\.json\Z")


def _now_local() -> datetime:
    """Kept as a small seam so tests can mock the trusted-local clock."""
    return datetime.now(LOCAL_TZ)


def _clock_text() -> str:
    now = _now_local()
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    return now.astimezone(LOCAL_TZ).isoformat(timespec="seconds")


def _today() -> date:
    now = _now_local()
    if now.tzinfo is None:
        now = now.replace(tzinfo=LOCAL_TZ)
    return now.astimezone(LOCAL_TZ).date()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("记录只能包含可规范化的 JSON 数据。") from exc


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    copied = dict(value)
    copied.pop(key, None)
    return hashlib.sha256(_canonical_json(copied).encode("ascii")).hexdigest()


def _study_path(directory) -> Path:
    path = Path(directory)
    if ".." in path.parts:
        raise ValueError("实验目录不能包含路径穿越 '..'。")
    return path.resolve(strict=False)


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"损坏或不可读取的登记文件: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"登记文件必须是 JSON 对象: {path.name}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_json(value)
    try:
        with path.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError("登记文件已存在；拒绝覆盖或并发写入。") from exc


def _validate_configuration(configuration: Mapping[str, Any]) -> dict:
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration 必须是对象。")
    config = dict(configuration)
    for key, expected in REQUIRED_CONFIGURATION.items():
        if key not in config or config[key] != expected or isinstance(config[key], bool):
            raise ValueError(f"前瞻协议固定要求 {key}={expected!r}。")
    _canonical_json(config)
    return config


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != SCHEMA_VERSION or protocol.get("kind") != "prospective_lottery_study":
        raise ValueError("不支持或损坏的前瞻实验协议。")
    if protocol.get("local_only") is not True:
        raise ValueError("协议必须明确标记为 local_only。")
    if protocol.get("trusted_timestamp") is not False:
        raise ValueError("本地协议不能声明 trusted_timestamp。")
    _parse_local_datetime(protocol.get("created_at_local"), "created_at_local")
    if protocol.get("timezone") != "Asia/Shanghai":
        raise ValueError("协议时区必须为 Asia/Shanghai。")
    _validate_configuration(protocol.get("configuration"))
    family = protocol.get("fixed_family")
    expected = [{"game": game, "play": play} for game, play in FIXED_FAMILY]
    if family != expected:
        raise ValueError("协议中的七玩法 family 已损坏。")
    digest = protocol.get("protocol_hash")
    if not isinstance(digest, str) or digest != _hash_without(protocol, "protocol_hash"):
        raise ValueError("协议自哈希校验失败。")


def create_study(directory, configuration) -> dict:
    """Create one empty, immutable local experiment directory.

    The supplied configuration is persisted once and is not accepted by later
    operations.  A directory must be new; this function never overwrites it.
    """
    root = _study_path(directory)
    config = _validate_configuration(configuration)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("实验目录已存在；拒绝覆盖既有研究。") from exc
    records = root / "records"
    try:
        records.mkdir(exist_ok=False)
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "kind": "prospective_lottery_study",
            "created_at_local": _clock_text(),
            "timezone": "Asia/Shanghai",
            "local_only": True,
            "trusted_timestamp": False,
            "timestamp_limit": "Local clock only; not independent pre-draw timestamp proof.",
            "chain_limit": "Detects changed/deleted middle records, not tail deletion or complete-chain rewrite.",
            "configuration": config,
            "fixed_family": [{"game": game, "play": play} for game, play in FIXED_FAMILY],
        }
        protocol["protocol_hash"] = _hash_without(protocol, "protocol_hash")
        _write_exclusive(root / "protocol.json", protocol)
    except Exception:
        # Do not attempt cleanup: preserving partial evidence is safer than a
        # destructive recovery action.  It will be rejected by read_study.
        raise
    return {"directory": str(root), "protocol": protocol, "records": []}


def _load(root: Path) -> tuple[dict, list[dict]]:
    if not root.is_dir():
        raise ValueError("实验目录不存在。")
    protocol_path, records_dir = root / "protocol.json", root / "records"
    if not protocol_path.is_file() or not records_dir.is_dir():
        raise ValueError("实验目录缺少 protocol.json 或 records 目录。")
    protocol = _read_json(protocol_path)
    _validate_protocol(protocol)
    names = sorted(item.name for item in records_dir.iterdir())
    if any(not _RECORD_NAME.fullmatch(name) for name in names):
        raise ValueError("records 目录含有非规范记录文件。")
    records: list[dict] = []
    previous = protocol["protocol_hash"]
    for expected_number, name in enumerate(names, start=1):
        if name != f"{expected_number:06d}.json":
            raise ValueError("记录序号不连续；可能发生了删除或篡改。")
        record = _read_json(records_dir / name)
        if record.get("sequence") != expected_number or record.get("previous_record_hash") != previous:
            raise ValueError("记录 hash 链不连续。")
        digest = record.get("record_hash")
        if not isinstance(digest, str) or digest != _hash_without(record, "record_hash"):
            raise ValueError("记录自哈希校验失败。")
        _validate_record_shape(record, protocol["configuration"]["n_tickets"])
        previous = digest
        records.append(record)
    seen: set[tuple[str, str, str]] = set()
    latest_by_play: dict[tuple[str, str], dict] = {}
    for record in records:
        key = (record["game"], record["play"], record["target_issue"])
        if key in seen:
            raise ValueError("存在重复的彩种、玩法和目标期号。")
        seen.add(key)
        pair = key[:2]
        prior = latest_by_play.get(pair)
        if prior and (int(record["target_issue"]) <= int(prior["target_issue"])
                      or record["target_date"] <= prior["target_date"]):
            raise ValueError("同一玩法记录的目标期号或日期乱序/冲突。")
        latest_by_play[pair] = record
    return protocol, records


def read_study(directory) -> dict:
    """Read and fully validate a local study; bad JSON or a bad chain is fatal."""
    root = _study_path(directory)
    protocol, records = _load(root)
    return {"directory": str(root), "protocol": protocol, "records": records}


def _canonical_game_play(game, play) -> tuple[str, str]:
    code = game_code(game)
    if not isinstance(play, str) or play not in GAME_PLAY_OPTIONS[code]:
        raise ValueError("彩种与玩法组合不合法。")
    effective = normalize_play(code, play)
    if (code, effective) not in FIXED_FAMILY:
        raise ValueError("玩法不属于固定七玩法 family。")
    return code, effective


def _parse_issue(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{label}必须是 ASCII 数字期号。")
    # Retain leading zeroes for exact source matching, but disallow meaningless
    # all-zero issue values and unbounded integer conversions.
    if int(value) <= 0:
        raise ValueError(f"{label}必须是正数期号。")
    return value


def _parse_date(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{label}必须为 YYYY-MM-DD。")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(f"{label}不是有效日期。") from exc


def _parse_local_datetime(value, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}必须是带时区的 ISO 时间。")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label}不是有效 ISO 时间。") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label}必须包含时区偏移。")
    return parsed.astimezone(LOCAL_TZ)


def _canonical_tickets(tickets, game: str, play: str, n_tickets: int, label: str) -> list[dict]:
    if not isinstance(tickets, list) or len(tickets) != n_tickets:
        raise ValueError(f"{label}必须恰好包含 {n_tickets} 注。")
    result = [_canonical_ticket(ticket, game, play) for ticket in tickets]
    if len({_ticket_key(ticket, game, play) for ticket in result}) != n_tickets:
        raise ValueError(f"{label}包含重复整票。")
    return result


def _validate_record_shape(record: Mapping[str, Any], n_tickets: int) -> None:
    try:
        game, play = _canonical_game_play(record["game"], record["play"])
        target_issue = _parse_issue(record["target_issue"], "target_issue")
        cutoff_issue = _parse_issue(record["cutoff_issue"], "cutoff_issue")
        target_date = _parse_date(record["target_date"], "target_date")
        cutoff_date = _parse_date(record["cutoff_date"], "cutoff_date")
        if int(target_issue) <= int(cutoff_issue) or target_date <= cutoff_date:
            raise ValueError("目标必须晚于数据截止点。")
        _canonical_tickets(record["model_tickets"], game, play, n_tickets, "model_tickets")
        _canonical_tickets(record["baseline_tickets"], game, play, n_tickets, "baseline_tickets")
        _canonical_json(record["provenance"])
        created_at = _parse_local_datetime(record.get("created_at_local"), "created_at_local")
        if created_at.date().isoformat() >= target_date:
            raise ValueError("记录创建日期必须严格早于目标开奖日期。")
        if cutoff_date > created_at.date().isoformat():
            raise ValueError("数据截止日期不能晚于记录创建日期。")
        if record.get("trusted_timestamp") is not False:
            raise ValueError("本地记录不能声明 trusted_timestamp。")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("登记记录字段或票面不合法。") from exc


def commit_prediction(directory, game, play, target_issue, target_date, cutoff_issue,
                      cutoff_date, model_tickets, baseline_tickets, provenance) -> dict:
    """Append one fixed-count prospective prediction, using only the local clock."""
    root = _study_path(directory)
    protocol, records = _load(root)
    code, effective_play = _canonical_game_play(game, play)
    target_issue = _parse_issue(target_issue, "target_issue")
    cutoff_issue = _parse_issue(cutoff_issue, "cutoff_issue")
    target_date = _parse_date(target_date, "target_date")
    cutoff_date = _parse_date(cutoff_date, "cutoff_date")
    if target_date <= _today().isoformat():
        raise ValueError("target_date 必须严格晚于本机 Asia/Shanghai 当天。")
    if cutoff_date > _today().isoformat():
        raise ValueError("cutoff_date 不能晚于本机 Asia/Shanghai 当天。")
    if target_date <= cutoff_date or int(target_issue) <= int(cutoff_issue):
        raise ValueError("target 期号和日期必须严格晚于 cutoff。")
    n_tickets = protocol["configuration"]["n_tickets"]
    model = _canonical_tickets(model_tickets, code, effective_play, n_tickets, "model_tickets")
    baseline = _canonical_tickets(baseline_tickets, code, effective_play, n_tickets, "baseline_tickets")
    _canonical_json(provenance)
    same_play = [item for item in records if item["game"] == code and item["play"] == effective_play]
    if any(item["target_issue"] == target_issue for item in same_play):
        raise ValueError("同一彩种、玩法和目标期号只能提交一次。")
    if same_play:
        latest = same_play[-1]
        if int(target_issue) <= int(latest["target_issue"]) or target_date <= latest["target_date"]:
            raise ValueError("同一玩法的目标期号和日期必须按提交顺序严格递增。")
    sequence = len(records) + 1
    previous = records[-1]["record_hash"] if records else protocol["protocol_hash"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "created_at_local": _clock_text(),
        "trusted_timestamp": False,
        "game": code,
        "play": effective_play,
        "target_issue": target_issue,
        "target_date": target_date,
        "cutoff_issue": cutoff_issue,
        "cutoff_date": cutoff_date,
        "model_tickets": model,
        "baseline_tickets": baseline,
        "provenance": provenance,
        "previous_record_hash": previous,
    }
    record["record_hash"] = _hash_without(record, "record_hash")
    _validate_record_shape(record, n_tickets)
    _write_exclusive(root / "records" / f"{sequence:06d}.json", record)
    return record


def _history_row(history: pd.DataFrame, record: Mapping[str, Any]) -> tuple[str, Any | None]:
    if not isinstance(history, pd.DataFrame) or not {"issue", "date"}.issubset(history.columns):
        return "invalid_history", None
    issues = history["issue"].astype(str)
    dates = pd.to_datetime(history["date"], errors="coerce")
    if dates.isna().any():
        return "invalid_history", None
    date_text = dates.dt.date.astype(str)
    issue_matches = history.loc[issues == record["target_issue"]]
    exact = history.loc[(issues == record["target_issue"]) & (date_text == record["target_date"])]
    if len(exact) == 1 and len(issue_matches) == 1:
        return "matched", exact.iloc[0]
    if len(exact) > 1 or len(issue_matches) > 0:
        return "conflict", None
    return "missing", None


def _actual_draw(row, game: str):
    try:
        if game == "ssq":
            return _canonical_ticket({"red": [row[f"red{i}"] for i in range(1, 7)], "blue": row["blue"]}, game, "单式投注")
        if game == "dlt":
            return _canonical_ticket({"front": [row[f"front{i}"] for i in range(1, 6)], "back": [row["back1"], row["back2"]]}, game, "基本投注")
        digits = [row[column] for column in DIGIT_GAME_CONFIGS[game]["digits"]]
        # Actual numbers are always validated as direct-order draws.  Group play
        # is applied only afterwards by _prize.
        return tuple(_canonical_ticket({"digits": digits}, game, "直选")["digits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("匹配到的实际开奖号字段或号码不合法。") from exc


def _best_first_prize(tickets: list[dict], draw, game: str, play: str) -> bool:
    return any(_prize(ticket, draw, game, play) == 1 for ticket in tickets)


def score_study(directory, histories: dict[str, pd.DataFrame]) -> dict:
    """Describe resolved periods without performing inferential statistics.

    ``histories`` is the caller-supplied in-memory source.  This function never
    reads or writes source data, model files, or result files.
    """
    root = _study_path(directory)
    protocol, records = _load(root)
    if not isinstance(histories, dict):
        raise ValueError("histories 必须是按彩种代码索引的 DataFrame 字典。")
    horizon = protocol["configuration"]["horizon"]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["game"], record["play"]), []).append(record)
    outcomes, summary = [], {}
    any_problem, complete = False, True
    for pair in FIXED_FAMILY:
        ordered = sorted(grouped.get(pair, []), key=lambda item: int(item["target_issue"]))
        selected, overflow = ordered[:horizon], ordered[horizon:]
        model_wins = baseline_wins = scored = 0
        statuses = []
        if len(selected) < horizon:
            complete = False
        for record in selected:
            status, row = _history_row(histories.get(pair[0]), record)
            if record["target_date"] > _today().isoformat():
                if status == "matched":
                    status = "time_problem_future_draw_present"
                else:
                    status = "pending"
            elif status == "matched":
                try:
                    draw = _actual_draw(row, pair[0])
                    model_win = _best_first_prize(record["model_tickets"], draw, *pair)
                    baseline_win = _best_first_prize(record["baseline_tickets"], draw, *pair)
                    status, scored = "scored", scored + 1
                    model_wins += int(model_win)
                    baseline_wins += int(baseline_win)
                    outcomes.append({"sequence": record["sequence"], "game": pair[0], "play": pair[1],
                                     "target_issue": record["target_issue"], "target_date": record["target_date"],
                                     "status": status, "model_first_prize_win": model_win,
                                     "baseline_first_prize_win": baseline_win})
                    statuses.append(status)
                    continue
                except ValueError:
                    status = "invalid_actual_draw"
            if status != "scored":
                any_problem = True
            outcomes.append({"sequence": record["sequence"], "game": pair[0], "play": pair[1],
                             "target_issue": record["target_issue"], "target_date": record["target_date"], "status": status})
            statuses.append(status)
        for record in overflow:
            outcomes.append({"sequence": record["sequence"], "game": pair[0], "play": pair[1],
                             "target_issue": record["target_issue"], "target_date": record["target_date"],
                             "status": "outside_fixed_horizon"})
        summary[f"{pair[0]}:{pair[1]}"] = {"selected_periods": len(selected), "horizon": horizon,
                                              "scored_periods": scored, "model_first_prize_wins": model_wins,
                                              "baseline_first_prize_wins": baseline_wins, "statuses": statuses}
    horizon_data_complete = complete and not any_problem and all(
        item["scored_periods"] == horizon for item in summary.values())
    return {"directory": str(root), "local_only": True,
            "trust_notice": "Local-only records are not independent third-party pre-draw timestamp proof.",
            "chain_limit": "Tail deletion and complete-chain rewrite remain undetectable locally.",
            "statistical_unit": "one target issue per model-versus-baseline portfolio, never individual tickets.",
            # No local clock/file chain can prove pre-draw registration.  Even a
            # complete horizon therefore remains descriptive unless a separate,
            # independently archived record is evaluated elsewhere.
            "horizon_data_complete": horizon_data_complete,
            "eligibility_for_advantage_claim": False,
            "eligibility_reason": "local_only registry never establishes an advantage claim; horizon_data_complete only describes data completeness",
            "per_play": summary, "records": outcomes}
