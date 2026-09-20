"""Portable result reader, separate from the unchanged frozen experiment code.

The experiment artifact manifest uses native Windows separators. Normalize
relative names here without modifying the frozen protocol or any result bytes.
Local hashes detect corruption, not deliberate rewriting or independent timing.
"""
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS
from run_candidate_experiment import RUNS, STRATEGIES, SCOPE
from run_evidence_experiment import digest


def _relative_name(value):
    if not isinstance(value, str) or not value or PureWindowsPath(value).drive:
        raise ValueError("研究工件路径非法。")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("研究工件路径必须是包内相对路径。")
    return path.as_posix()


def read_report(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(RUNS.resolve()):
        raise ValueError("报告目录必须位于candidate_runs。")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    expected = {(game, play, strategy) for game in ALL_GAME_CODES
                for play in GAME_PLAY_OPTIONS[game] for strategy in STRATEGIES}
    rows = report.get("summary", [])
    actual = {(r.get("game"), r.get("play"), r.get("strategy")) for r in rows}
    if (report.get("scope") != SCOPE or len(rows) != 21 or actual != expected or
            len(report.get("bias_diagnostics", [])) != 38 or len(report.get("head_scores", [])) != 20):
        raise ValueError("研究报告结构或范围不完整。")
    entries = report.get("artifact_sha256", {})
    normalized = {_relative_name(name): sha for name, sha in entries.items()}
    required = {"protocol.json", *[f"{g}_result.json" for g in ALL_GAME_CODES],
                *[f"snapshots/{g}_history.csv" for g in ALL_GAME_CODES]}
    if len(normalized) != len(entries) or not required.issubset(normalized):
        raise ValueError("研究工件清单不完整或有重复路径。")
    if normalized["protocol.json"] != report.get("protocol_sha256"):
        raise ValueError("报告协议指纹不匹配。")
    for relative, expected_sha in normalized.items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or digest(path) != expected_sha:
            raise ValueError(f"研究工件缺失、变化或路径非法：{relative}")
    return report
