"""Portable, integrity-checked manifests for locally trained lottery models.

The manifest records that a set of local weights was produced for one exact local
history/features input.  It is an integrity and reproducibility aid only; it does
not establish independent out-of-sample predictive advantage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from game_config import ALL_GAME_CODES, DIGIT_GAME_CONFIGS, game_code


_SCHEMA_VERSION = 1
_SESSION_SUFFIX = "_training_session.json"
_TRAIN_INFO_FIELDS = (
    "model_path", "input_size", "hidden_size", "num_layers", "dropout",
    "seq_len", "bidirectional", "use_attention", "attn_heads", "n_ensemble",
    "ensemble_seeds", "holdout_size", "train_losses", "val_losses", "best_epoch",
)
_BASE_COLUMNS = {
    "ssq": ("issue", "date", "red1", "red2", "red3", "red4", "red5", "red6",
            "blue", "sales", "pool"),
    "dlt": ("issue", "date", "front1", "front2", "front3", "front4", "front5",
            "back1", "back2", "sales"),
}


def _error(message: str) -> ValueError:
    return ValueError(f"训练会话无效：{message}")


def _code(game: Any) -> str:
    if not isinstance(game, str):
        raise _error("彩种必须是字符串。")
    try:
        code = game_code(game)
    except (TypeError, ValueError) as exc:
        raise _error("不支持的彩种。") from exc
    if code not in ALL_GAME_CODES:
        raise _error("不支持的彩种。")
    return code


def _models_path(models_dir: Any) -> Path:
    if isinstance(models_dir, (bool, np.bool_)) or not isinstance(models_dir, (str, os.PathLike)):
        raise _error("models_dir 必须是路径。")
    try:
        return Path(models_dir).resolve()
    except (OSError, TypeError, ValueError) as exc:
        raise _error("models_dir 不是有效路径。") from exc


def _session_path(code: str, models_dir: Any) -> Path:
    return _models_path(models_dir) / f"{code}{_SESSION_SUFFIX}"


def _columns_for(code: str) -> tuple[str, ...]:
    if code in _BASE_COLUMNS:
        return _BASE_COLUMNS[code]
    return ("issue", "date", *DIGIT_GAME_CONFIGS[code]["digits"], "sales")


def _as_positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise _error(f"{name} 必须是整数。")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        comparator = "非负" if allow_zero else "正"
        raise _error(f"{name} 必须是{comparator}整数。")
    return result


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise _error(f"{name} 必须是有限数值。")
    result = float(value)
    if not math.isfinite(result):
        raise _error(f"{name} 必须是有限数值。")
    return result


def _canonical_train_info(code: str, train_info: Any, models_dir: Path) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(train_info, dict):
        raise _error("train_info 必须是字典。")
    missing = [field for field in _TRAIN_INFO_FIELDS if field not in train_info]
    if missing:
        raise _error(f"train_info 缺少字段：{', '.join(missing)}。")
    unknown = set(train_info) - set(_TRAIN_INFO_FIELDS)
    if unknown:
        raise _error("train_info 包含未识别字段，不能安全恢复。")

    info: dict[str, Any] = {}
    for name in ("input_size", "hidden_size", "num_layers", "seq_len", "attn_heads", "n_ensemble"):
        info[name] = _as_positive_int(train_info[name], name)
    info["holdout_size"] = _as_positive_int(train_info["holdout_size"], "holdout_size", allow_zero=True)
    info["best_epoch"] = _as_positive_int(train_info["best_epoch"], "best_epoch", allow_zero=True)
    info["dropout"] = _as_finite_float(train_info["dropout"], "dropout")
    if not 0.0 <= info["dropout"] < 1.0:
        raise _error("dropout 必须在 [0, 1) 内。")
    for name in ("bidirectional", "use_attention"):
        if not isinstance(train_info[name], (bool, np.bool_)):
            raise _error(f"{name} 必须是布尔值。")
        info[name] = bool(train_info[name])
    if info["use_attention"] and (info["hidden_size"] * (2 if info["bidirectional"] else 1)) % info["attn_heads"]:
        raise _error("attn_heads 必须整除 LSTM 输出维度。")

    for name in ("train_losses", "val_losses"):
        values = train_info[name]
        if not isinstance(values, list) or not values:
            raise _error(f"{name} 必须是非空列表。")
        info[name] = [_as_finite_float(value, name) for value in values]

    seeds = train_info["ensemble_seeds"]
    if not isinstance(seeds, list):
        raise _error("ensemble_seeds 必须是列表。")
    info["ensemble_seeds"] = [_as_positive_int(seed, "ensemble_seeds", allow_zero=True) for seed in seeds]
    if len(set(info["ensemble_seeds"])) != len(info["ensemble_seeds"]):
        raise _error("ensemble_seeds 不能重复。")
    if info["n_ensemble"] == 1:
        if info["ensemble_seeds"]:
            raise _error("单模型训练的 ensemble_seeds 必须为空。")
    elif len(info["ensemble_seeds"]) != info["n_ensemble"]:
        raise _error("n_ensemble 必须与 ensemble_seeds 数量一致。")

    base_name = f"{code}_lstm.pth"
    raw_model_path = train_info["model_path"]
    if not isinstance(raw_model_path, str) or not raw_model_path:
        raise _error("model_path 必须是模型文件路径。")
    # Only the known basename is accepted.  The passed directory is the sole
    # authority for the actual location, so a manifest cannot redirect loading.
    supplied_path = Path(raw_model_path)
    expected_path = models_dir / base_name
    if supplied_path.name != base_name:
        raise _error("model_path 不是当前彩种的受信任模型文件名。")
    if supplied_path.is_absolute():
        try:
            is_expected_path = supplied_path.resolve() == expected_path
        except (OSError, ValueError) as exc:
            raise _error("model_path 不是有效路径。") from exc
    else:
        is_expected_path = supplied_path == Path(base_name)
    if not is_expected_path:
        raise _error("model_path 不能包含目录或路径遍历。")
    # The manifest is portable: persist only this vetted basename.  Loading later
    # binds it to the caller's current models_dir after all checks have passed.
    info["model_path"] = base_name
    weight_names = [base_name]
    if info["n_ensemble"] > 1:
        weight_names.extend(f"{code}_lstm_seed{seed}.pth" for seed in info["ensemble_seeds"])
    return {field: info[field] for field in _TRAIN_INFO_FIELDS}, weight_names


def _data_fingerprint(code: str, df: Any) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise _error("df 必须是 pandas.DataFrame。")
    columns = _columns_for(code)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise _error(f"df 缺少规范列：{', '.join(missing)}。")
    if df.empty:
        raise _error("df 不能为空。")
    canonical = df.loc[:, columns].copy()
    issues = canonical["issue"]
    if issues.isna().any():
        raise _error("issue 不能缺失。")
    canonical["issue"] = issues.astype(str).str.strip()
    if (canonical["issue"] == "").any():
        raise _error("issue 不能为空。")
    # Parse one value at a time: pandas' vectorized format inference can reject a
    # valid CSV column merely because some rows use '/' and others use '-'.
    dates = canonical["date"].map(lambda value: pd.to_datetime(value, errors="coerce", utc=True))
    if dates.isna().any():
        raise _error("date 包含无法规范化的日期。")
    canonical["date"] = dates.dt.strftime("%Y-%m-%d")
    for column in columns[2:]:
        original = canonical[column]
        if pd.api.types.is_bool_dtype(original) or original.map(
                lambda value: isinstance(value, (bool, np.bool_))).any():
            raise _error(f"{column} 不能是布尔值。")
        numeric = pd.to_numeric(original, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise _error(f"{column} 必须是有限整数。")
        canonical[column] = values.astype(np.int64)
    digest = hashlib.sha256()
    digest.update(("\x1f".join(columns) + "\n").encode("utf-8"))
    for row in canonical.itertuples(index=False, name=None):
        digest.update(("\x1f".join(str(value) for value in row) + "\n").encode("utf-8"))
    return {"sha256": digest.hexdigest(), "rows": int(len(canonical)), "columns": list(columns)}


def _features_fingerprint(features: Any) -> dict[str, Any]:
    if isinstance(features, (list, tuple)):
        array = np.asarray(features)
    elif isinstance(features, np.ndarray):
        array = features
    else:
        raise _error("features 必须是 numpy 数组。")
    if array.dtype.kind in "bOScU" or array.ndim != 2 or 0 in array.shape:
        raise _error("features 必须是非空二维数值数组。")
    try:
        normalized = np.ascontiguousarray(array, dtype="<f4")
    except (TypeError, ValueError) as exc:
        raise _error("features 无法转换为 float32。") from exc
    if not np.isfinite(normalized).all():
        raise _error("features 必须全部为有限数值。")
    return {"sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
            "shape": [int(size) for size in normalized.shape], "dtype": "float32"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_records(models_dir: Path, names: list[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for name in names:
        if Path(name).name != name:
            raise _error("权重文件名不安全。")
        path = models_dir / name
        if not path.is_file():
            raise _error(f"缺少权重文件：{name}。")
        records.append({"name": name, "sha256": _sha256_file(path)})
    return records


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_training_session(game: str, df: pd.DataFrame, features: np.ndarray,
                          train_info: dict, models_dir: str | os.PathLike[str]) -> str:
    """Save a JSON-only local-training manifest and return its path.

    Weights are never loaded here.  All required weights must already exist in
    ``models_dir`` and are SHA-256 bound before the manifest is atomically replaced.
    """
    code = _code(game)
    directory = _models_path(models_dir)
    info, weight_names = _canonical_train_info(code, train_info, directory)
    data_fingerprint = _data_fingerprint(code, df)
    features_fingerprint = _features_fingerprint(features)
    _validate_training_window(info, data_fingerprint, features_fingerprint)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "game": code,
        "train_info": info,
        "data_fingerprint": data_fingerprint,
        "features_fingerprint": features_fingerprint,
        "weights": _weight_records(directory, weight_names),
        "evidence_note": "Matches local training inputs and weight files only; not independent out-of-sample evidence.",
    }
    path = _session_path(code, directory)
    _atomic_json_write(path, payload)
    return str(path)


def _require_fingerprint(value: Any, expected: dict[str, Any], name: str) -> None:
    if not isinstance(value, dict) or value != expected:
        raise _error(f"{name} 与训练会话不匹配。")


def _validate_training_window(info: dict[str, Any], data_fingerprint: dict[str, Any],
                              features_fingerprint: dict[str, Any]) -> None:
    """Validate dimensions and the same minimum fitting window as train_model."""
    rows = data_fingerprint["rows"]
    shape = features_fingerprint["shape"]
    if shape[0] != rows:
        raise _error("features 行数必须与 df 行数一致。")
    if shape[1] != info["input_size"]:
        raise _error("features 列数必须与 input_size 一致。")
    fit_count = rows - info["seq_len"] - info["holdout_size"]
    if fit_count < 2:
        raise _error("seq_len 和 holdout_size 之后至少需要 2 个训练/验证样本。")
    if len(info["train_losses"]) != len(info["val_losses"]):
        raise _error("train_losses 与 val_losses 长度必须一致。")
    if info["best_epoch"] >= len(info["val_losses"]):
        raise _error("best_epoch 必须位于验证损失记录范围内。")


def load_training_session(game: str, df: pd.DataFrame, features: np.ndarray,
                          models_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Strictly validate a saved local session and return safe ``train_info``.

    Missing manifests intentionally fail: legacy standalone ``.pth`` files cannot
    be restored without the data/feature/weight provenance recorded by this module.
    """
    code = _code(game)
    directory = _models_path(models_dir)
    path = _session_path(code, directory)
    if not path.is_file():
        raise _error("未找到训练会话清单；旧权重不能恢复。")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise _error("训练会话清单无法读取。") from exc
    expected_keys = {"schema_version", "game", "train_info", "data_fingerprint",
                     "features_fingerprint", "weights", "evidence_note"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise _error("训练会话清单结构不受支持。")
    if (isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != _SCHEMA_VERSION
            or payload["game"] != code):
        raise _error("训练会话清单与当前彩种或版本不匹配。")
    if not isinstance(payload["evidence_note"], str):
        raise _error("训练会话清单说明无效。")
    info, names = _canonical_train_info(code, payload["train_info"], directory)
    data_fingerprint = _data_fingerprint(code, df)
    features_fingerprint = _features_fingerprint(features)
    _validate_training_window(info, data_fingerprint, features_fingerprint)
    _require_fingerprint(payload["data_fingerprint"], data_fingerprint, "历史数据")
    _require_fingerprint(payload["features_fingerprint"], features_fingerprint, "特征")
    expected_weights = _weight_records(directory, names)
    if payload["weights"] != expected_weights:
        raise _error("权重文件集合或 SHA-256 校验不匹配。")
    info["model_path"] = str(directory / f"{code}_lstm.pth")
    return info
