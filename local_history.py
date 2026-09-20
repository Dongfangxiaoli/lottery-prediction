"""Read-only local history loading and fail-closed input validation."""
from pathlib import Path

import numpy as np
import pandas as pd

from game_config import DIGIT_GAME_CONFIGS, digit_columns, game_code, game_name

DATA_DIR = Path(__file__).resolve().parent / "data"


def prepare_history(frame, game):
    code = game_code(game)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("没有可用历史数据。请检查本地文件，或联网获取数据。")
    if code == "ssq":
        groups = [([f"red{i}" for i in range(1, 7)], 1, 33), (["blue"], 1, 16)]
    elif code == "dlt":
        groups = [([f"front{i}" for i in range(1, 6)], 1, 35), (["back1", "back2"], 1, 12)]
    else:
        groups = [([column], 0, limit - 1) for column, limit in
                  zip(digit_columns(code), DIGIT_GAME_CONFIGS[code]["classes"])]
    number_columns = [column for columns, _, _ in groups for column in columns]
    required = ["issue", "date", *number_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("历史文件缺少列：" + "、".join(missing))
    clean = frame.copy(deep=True)
    clean["issue"] = clean["issue"].astype(str).str.strip()
    if not clean["issue"].str.fullmatch(r"\d+").all():
        raise ValueError("历史文件包含空白或非法期号，请核对原文件。")
    dates = pd.to_datetime(clean["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("历史文件包含无法识别的开奖日期，请核对原文件。")
    clean["date"] = dates
    for columns, lower, upper in groups:
        for column in columns:
            if clean[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError(f"{column} 含非法布尔值。")
            values = pd.to_numeric(clean[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{column} 必须是有限整数，不能含空白、小数或乱码。")
            if not ((values >= lower) & (values <= upper)).all():
                raise ValueError(f"{column} 超出 {lower}–{upper} 范围。")
            clean[column] = values.astype(int)
        if code in ("ssq", "dlt") and len(columns) > 1:
            if not (np.diff(clean[columns].to_numpy(), axis=1) > 0).all():
                raise ValueError("球彩每一区内号码必须升序且不重复，请核对历史文件。")
    unique_rows = clean.drop_duplicates(subset=required)
    if unique_rows["issue"].duplicated().any():
        raise ValueError("同一期号出现不同日期或号码，不能自动选择其中一条，请先核对。")
    if code == "ssq":
        from data_fetcher_ssq import _normalize_history
        normalized = _normalize_history(clean)
    elif code == "dlt":
        from data_fetcher_dlt import _normalize_history
        normalized = _normalize_history(clean)
    else:
        from data_fetcher_digit import _normalize_history
        normalized = _normalize_history(clean, code)
    if normalized.empty:
        raise ValueError("没有符合当前玩法规则的数据。")
    return normalized


def read_local_history(game, directory=None):
    code = game_code(game)
    path = Path(directory if directory is not None else DATA_DIR) / f"{code}_history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"未找到{game_name(code)}本地历史文件：{path.name}。请完整解压或联网获取。")
    frame = pd.read_csv(path, dtype={"issue": str})
    return prepare_history(frame, code)
