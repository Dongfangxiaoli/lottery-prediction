"""Independent fixed-configuration candidate models for comparison experiments.

These helpers deliberately do not modify the frozen LSTM research workflow.
For ball games, ``red1``/``front1`` etc. are *sorted-number positions* from
the historical record.  They are not the physical order in which balls were
drawn.
"""
from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path
from typing import Mapping

import numpy as np

from game_config import game_code
from jackpot_selection import _game_heads


XGB_CONFIG = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "max_bin": 64,
    "n_jobs": 2,
    "random_state": 20260912,
}
"""Frozen XGBoost comparison configuration; do not tune it per experiment."""

_SCHEMA_VERSION = 1
_XGB_METADATA = "candidate_xgboost_heads.json"
_MARKOV_METADATA = "candidate_markov_heads.json"
_PROBABILITY_EPSILON = 1e-12


def _require_xgboost():
    """Import XGBoost only for a non-constant model operation."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "缺少 xgboost：请先在本项目虚拟环境安装 xgboost，然后重新运行候选模型实验。"
        ) from exc
    return XGBClassifier


def _as_feature_matrix(values, label: str) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是二维数值矩阵。") from exc
    if matrix.ndim != 2:
        raise ValueError(f"{label}必须是二维数值矩阵。")
    if matrix.shape[1] == 0:
        raise ValueError(f"{label}不能有零个特征列。")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label}必须全部为有限数值。")
    return matrix


def _as_class_vector(values, classes: int, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{label}必须是非空一维类别数组。")
    if np.issubdtype(raw.dtype, np.bool_):
        raise ValueError(f"{label}必须是0基整数类别。")
    try:
        numeric = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是0基整数类别。") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{label}必须是0基整数类别。")
    result = numeric.astype(np.int64)
    if np.any(result < 0) or np.any(result >= classes):
        raise ValueError(f"{label}类别必须在0到{classes - 1}之间。")
    return result


def _as_observed_classes(values, classes: int, label: str) -> np.ndarray:
    """Validate the persisted label encoding, including its one-to-one order."""
    result = _as_class_vector(values, classes, label)
    if result.size > 1 and not np.all(result[1:] > result[:-1]):
        raise ValueError(f"{label}必须是严格升序且不重复的类别映射。")
    return result


def _as_nonnegative_int(value, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label}必须是整数。")
    value = int(value)
    if value < 0:
        raise ValueError(f"{label}不能为负数。")
    return value


def _head_spec(game: str) -> tuple[str, list[tuple[str, int]]]:
    code = game_code(game)
    return code, _game_heads(code)


def summarize_windows(features, indices, seq_len: int = 30) -> np.ndarray:
    """Compress target-safe history windows into last/mean/std feature vectors.

    Each output at target index ``idx`` uses exactly ``features[idx-seq_len:idx]``.
    In particular, the target row itself and any future row are excluded.
    """
    matrix = _as_feature_matrix(features, "features")
    if isinstance(seq_len, (bool, np.bool_)) or not isinstance(seq_len, Integral):
        raise ValueError("seq_len必须是正整数。")
    seq_len = int(seq_len)
    if seq_len < 1:
        raise ValueError("seq_len必须是正整数。")
    try:
        requested = list(indices)
    except TypeError as exc:
        raise ValueError("indices必须是目标行索引序列。") from exc
    checked: list[int] = []
    for index in requested:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise ValueError("indices必须是整数索引。")
        index = int(index)
        if index < seq_len or index > matrix.shape[0]:
            raise ValueError("目标行索引必须满足seq_len <= idx <= len(features)。")
        checked.append(index)
    output = np.empty((len(checked), matrix.shape[1] * 3), dtype=np.float32)
    for row, index in enumerate(checked):
        window = matrix[index - seq_len:index]
        output[row] = np.concatenate((window[-1], window.mean(axis=0), window.std(axis=0)))
    return output


class XGBoostHeads:
    """One fixed XGBoost classifier per game position, with full-class output."""

    def __init__(self, game: str):
        self.game, self._heads = _head_spec(game)
        self._models: dict[str, object] = {}
        self._encodings: dict[str, np.ndarray] = {}
        self._input_size: int | None = None

    def fit(self, X, targets_dict: Mapping[str, object]):
        X = _as_feature_matrix(X, "X")
        if X.shape[0] == 0:
            raise ValueError("X至少需要一行训练样本。")
        if not isinstance(targets_dict, Mapping):
            raise ValueError("targets_dict必须是按输出头名称索引的映射。")
        expected = {name for name, _ in self._heads}
        if set(targets_dict) != expected:
            raise ValueError("targets_dict必须恰好包含该彩种的全部输出头。")
        models: dict[str, object] = {}
        encodings: dict[str, np.ndarray] = {}
        classifier_type = None
        for name, classes in self._heads:
            labels = _as_class_vector(targets_dict[name], classes, name)
            if labels.shape[0] != X.shape[0]:
                raise ValueError(f"{name}标签长度必须与X行数相同。")
            observed = np.unique(labels)
            encodings[name] = observed
            if observed.size == 1:
                models[name] = {"kind": "constant", "class": int(observed[0])}
                continue
            if classifier_type is None:
                classifier_type = _require_xgboost()
            encoded = np.searchsorted(observed, labels)
            model = classifier_type(
                **XGB_CONFIG,
                objective="multi:softprob",
                num_class=int(observed.size),
                eval_metric="mlogloss",
                verbosity=0,
            )
            model.fit(X, encoded)
            models[name] = model
        self._models = models
        self._encodings = encodings
        self._input_size = int(X.shape[1])
        return self

    def predict_heads(self, X) -> dict[str, np.ndarray]:
        if self._input_size is None:
            raise ValueError("模型尚未训练或加载。")
        X = _as_feature_matrix(X, "X")
        if X.shape[1] != self._input_size:
            raise ValueError(f"X特征列数应为{self._input_size}，实际为{X.shape[1]}。")
        result: dict[str, np.ndarray] = {}
        for name, classes in self._heads:
            model = self._models[name]
            observed = self._encodings[name]
            full = np.full((X.shape[0], classes), _PROBABILITY_EPSILON, dtype=np.float64)
            if isinstance(model, dict):
                full[:, int(model["class"])] += 1.0
            else:
                probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
                if probabilities.ndim == 1:
                    probabilities = probabilities.reshape(-1, 1)
                if probabilities.shape != (X.shape[0], observed.size):
                    raise RuntimeError(f"{name}的XGBoost概率输出维度异常。")
                if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                    raise RuntimeError(f"{name}的XGBoost概率输出无效。")
                full[:, observed] += probabilities
            full /= full.sum(axis=1, keepdims=True)
            result[name] = full.astype(np.float32)
        return result

    def save(self, directory) -> Path:
        if self._input_size is None:
            raise ValueError("模型尚未训练，不能保存。")
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"候选XGBoost保存目录必须是不存在的新目录: {directory}")
        directory.mkdir(parents=True, exist_ok=False)
        metadata_path = directory / _XGB_METADATA
        records = []
        for name, classes in self._heads:
            model = self._models[name]
            record = {"name": name, "classes": classes,
                      "observed_classes": self._encodings[name].astype(int).tolist()}
            if isinstance(model, dict):
                record.update(kind="constant", constant_class=int(model["class"]))
            else:
                filename = f"candidate_xgb_{name}.json"
                model.save_model(str(directory / filename))
                record.update(kind="xgboost", model_file=filename)
            records.append(record)
        metadata = {"schema_version": _SCHEMA_VERSION, "model_type": "xgboost_heads",
                    "game": self.game, "input_size": self._input_size,
                    "config": XGB_CONFIG, "heads": records}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        try:
            metadata = json.loads((directory / _XGB_METADATA).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("无法读取候选XGBoost JSON元数据。") from exc
        if metadata.get("schema_version") != _SCHEMA_VERSION or metadata.get("model_type") != "xgboost_heads":
            raise ValueError("候选XGBoost元数据版本或类型不匹配。")
        instance = cls(metadata.get("game", ""))
        input_size = metadata.get("input_size")
        if isinstance(input_size, bool) or not isinstance(input_size, Integral) or input_size < 1:
            raise ValueError("候选XGBoost元数据中的input_size无效。")
        records = metadata.get("heads")
        if not isinstance(records, list) or len(records) != len(instance._heads):
            raise ValueError("候选XGBoost元数据中的输出头无效。")
        models, encodings = {}, {}
        classifier_type = None
        for record, (name, classes) in zip(records, instance._heads):
            if not isinstance(record, dict) or record.get("name") != name or record.get("classes") != classes:
                raise ValueError("候选XGBoost元数据与彩种输出头不匹配。")
            observed = _as_observed_classes(record.get("observed_classes", []), classes, f"{name}元数据")
            kind = record.get("kind")
            if kind == "constant":
                constant = _as_class_vector([record.get("constant_class")], classes, f"{name}常量类别")[0]
                if observed.size != 1 or constant != observed[0]:
                    raise ValueError("候选XGBoost常量头元数据无效。")
                models[name] = {"kind": "constant", "class": int(constant)}
            elif kind == "xgboost":
                expected_file = f"candidate_xgb_{name}.json"
                if (observed.size < 2 or record.get("model_file") != expected_file):
                    raise ValueError("候选XGBoost模型头元数据无效。")
                model_path = (directory / expected_file).resolve()
                if model_path.parent != directory.resolve() or not model_path.is_file():
                    raise ValueError(f"候选XGBoost模型文件不存在: {model_path.name}")
                if classifier_type is None:
                    classifier_type = _require_xgboost()
                model = classifier_type(
                    **XGB_CONFIG, objective="multi:softprob", num_class=int(observed.size),
                    eval_metric="mlogloss", verbosity=0,
                )
                model.load_model(str(model_path))
                models[name] = model
            else:
                raise ValueError("候选XGBoost模型头类型无效。")
            encodings[name] = observed
        instance._models, instance._encodings = models, encodings
        instance._input_size = int(input_size)
        return instance


class MarkovHeads:
    """Position-wise one-step Markov baselines over chronological target labels.

    Ball-game positions mean sorted-number positions, not physical ball order.
    """

    def __init__(self, game: str, smoothing: float = 1.0):
        if isinstance(smoothing, (bool, np.bool_)) or not isinstance(smoothing, Real):
            raise ValueError("smoothing必须是有限正数。")
        smoothing = float(smoothing)
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing必须是有限正数。")
        self.game, self._heads = _head_spec(game)
        self.smoothing = smoothing
        self._transitions: dict[str, np.ndarray] = {}

    def fit(self, targets_dict: Mapping[str, object], start: int = 30, end: int | None = None):
        if not isinstance(targets_dict, Mapping):
            raise ValueError("targets_dict必须是按输出头名称索引的映射。")
        expected = {name for name, _ in self._heads}
        if set(targets_dict) != expected:
            raise ValueError("targets_dict必须恰好包含该彩种的全部输出头。")
        values = {name: _as_class_vector(targets_dict[name], classes, name)
                  for name, classes in self._heads}
        lengths = {labels.size for labels in values.values()}
        if len(lengths) != 1:
            raise ValueError("各输出头的历史标签长度必须相同。")
        length = lengths.pop()
        start = _as_nonnegative_int(start, "start")
        if end is None:
            end = length
        else:
            end = _as_nonnegative_int(end, "end")
        if start < 1 or start >= end or end > length:
            raise ValueError("训练范围必须满足1 <= start < end <= 标签长度。")
        transitions: dict[str, np.ndarray] = {}
        for name, classes in self._heads:
            counts = np.full((classes, classes), self.smoothing, dtype=np.float64)
            labels = values[name]
            np.add.at(counts, (labels[start - 1:end - 1], labels[start:end]), 1.0)
            transitions[name] = counts / counts.sum(axis=1, keepdims=True)
        self._transitions = transitions
        return self

    def predict_heads(self, previous_dict: Mapping[str, object]) -> dict[str, np.ndarray]:
        if not self._transitions:
            raise ValueError("模型尚未训练或加载。")
        if not isinstance(previous_dict, Mapping):
            raise ValueError("previous_dict必须是按输出头名称索引的映射。")
        expected = {name for name, _ in self._heads}
        if set(previous_dict) != expected:
            raise ValueError("previous_dict必须恰好包含该彩种的全部输出头。")
        result, lengths = {}, set()
        for name, classes in self._heads:
            previous = _as_class_vector(previous_dict[name], classes, f"{name}上期类别")
            result[name] = self._transitions[name][previous].astype(np.float32)
            lengths.add(previous.size)
        if len(lengths) != 1:
            raise ValueError("各输出头的上期类别数量必须相同。")
        return result

    def save(self, directory) -> Path:
        if not self._transitions:
            raise ValueError("模型尚未训练，不能保存。")
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(f"候选Markov保存目录必须是不存在的新目录: {directory}")
        directory.mkdir(parents=True, exist_ok=False)
        metadata_path = directory / _MARKOV_METADATA
        records = [{"name": name, "classes": classes,
                    "transition": self._transitions[name].tolist()}
                   for name, classes in self._heads]
        metadata = {"schema_version": _SCHEMA_VERSION, "model_type": "markov_heads",
                    "game": self.game, "smoothing": self.smoothing, "heads": records}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        try:
            metadata = json.loads((directory / _MARKOV_METADATA).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("无法读取候选Markov JSON元数据。") from exc
        if metadata.get("schema_version") != _SCHEMA_VERSION or metadata.get("model_type") != "markov_heads":
            raise ValueError("候选Markov元数据版本或类型不匹配。")
        instance = cls(metadata.get("game", ""), metadata.get("smoothing"))
        records = metadata.get("heads")
        if not isinstance(records, list) or len(records) != len(instance._heads):
            raise ValueError("候选Markov元数据中的输出头无效。")
        transitions = {}
        for record, (name, classes) in zip(records, instance._heads):
            if not isinstance(record, dict) or record.get("name") != name or record.get("classes") != classes:
                raise ValueError("候选Markov元数据与彩种输出头不匹配。")
            try:
                matrix = np.asarray(record.get("transition"), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("候选Markov转移矩阵不是数值矩阵。") from exc
            if matrix.shape != (classes, classes) or not np.all(np.isfinite(matrix)) or np.any(matrix <= 0):
                raise ValueError("候选Markov转移矩阵无效。")
            if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9):
                raise ValueError("候选Markov转移矩阵行和必须为1。")
            transitions[name] = matrix
        instance._transitions = transitions
        return instance
