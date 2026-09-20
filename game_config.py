"""彩票游戏的轻量公共配置。"""

from math import comb

GAME_NAMES = {
    "ssq": "双色球",
    "dlt": "大乐透",
    "pls": "排列3",
    "plw": "排列5",
    "qxc": "7星彩",
}
GAME_NAME_TO_CODE = {name: code for code, name in GAME_NAMES.items()}
ALL_GAME_CODES = tuple(GAME_NAMES)

GAME_PLAY_OPTIONS = {
    "ssq": ("单式投注",), "dlt": ("基本投注",),
    "pls": ("直选", "组选3", "组选6"), "plw": ("直选",), "qxc": ("单式投注",),
}
GAME_PLAY_DEFAULTS = {"ssq": "单式投注", "dlt": "基本投注", "pls": "组选6", "plw": "直选", "qxc": "单式投注"}

# UI 中的训练、选号与回测共用同一组冻结默认值，避免不同页面悄然漂移。
DEFAULT_TRAINING_PARAMS = {
    "seq_len": 30, "epochs": 30, "lr": 0.001, "hidden": 64, "batch_size": 128,
    "bidirectional": False, "use_attention": False, "n_ensemble": 1,
}
DEFAULT_SELECTION_PARAMS = {
    "n_groups": 5, "temperature": 1.0, "anneal_enabled": False,
    "sampling_mode": "merged", "freq_alpha": 0.0, "top_p": 1.0,
    "missing_alpha": 0.0, "hot_alpha": 0.0,
}
DEFAULT_BACKTEST_PARAMS = {"n_test": 50, **DEFAULT_SELECTION_PARAMS}


def _probability_text(numerator: int, denominator: int) -> str:
    """以可审计的整数计数表达理论命中概率，不混入奖金或收益承诺。"""
    return f"{numerator}/{denominator:,}" if numerator != 1 else f"1/{denominator:,}"


# 每项均以 (彩种代码, 玩法) 为键，不能仅按“直选”等通用玩法名查找。
GAME_PLAY_DESCRIPTIONS = {
    ("ssq", "单式投注"): (
        "红球从 01–33 中选 6 个，不重复、顺序不计；蓝球从 01–16 中选 1 个。"
        f"理论头奖命中概率：{_probability_text(1, comb(33, 6) * 16)}。"
    ),
    ("dlt", "基本投注"): (
        "前区从 01–35 中选 5 个、不重复、顺序不计；后区从 01–12 中选 2 个、不重复、顺序不计。"
        f"理论头奖命中概率：{_probability_text(1, comb(35, 5) * comb(12, 2))}。"
    ),
    ("pls", "直选"): (
        "选择 3 位数字，每位为 0–9；顺序重要，允许重复。"
        f"理论命中概率：{_probability_text(1, 10 ** 3)}。"
    ),
    ("pls", "组选3"): (
        "选择 2 个不同数字，其中一个重复一次；号码按组展示，顺序不计，覆盖 3 个有效排列。"
        f"理论命中概率：{_probability_text(3, 10 ** 3)}。"
    ),
    ("pls", "组选6"): (
        "选择 3 个不同数字；号码按组展示，顺序不计，覆盖 6 个有效排列。"
        f"理论命中概率：{_probability_text(6, 10 ** 3)}。"
    ),
    ("plw", "直选"): (
        "选择 5 位数字，每位为 0–9；顺序重要，允许重复。"
        f"理论命中概率：{_probability_text(1, 10 ** 5)}。"
    ),
    ("qxc", "单式投注"): (
        "前 6 位每位为 0–9，第 7 位为 0–14；顺序重要，允许各位置出现相同数字。"
        f"理论头奖（整注全部命中）概率：{_probability_text(1, 10 ** 6 * 15)}。"
    ),
}

# classes 是各位置可取类别数；digits 是本地历史 CSV 的号码列名。
DIGIT_GAME_CONFIGS = {
    "pls": {
        "name": GAME_NAMES["pls"],
        "digits": ("digit1", "digit2", "digit3"),
        "classes": (10, 10, 10),
        "history_url": "https://datachart.500.com/pls/history/inc/history.php",
    },
    "plw": {
        "name": GAME_NAMES["plw"],
        "digits": ("digit1", "digit2", "digit3", "digit4", "digit5"),
        "classes": (10, 10, 10, 10, 10),
        "history_url": "https://datachart.500.com/plw/history/inc/history.php",
    },
    "qxc": {
        "name": GAME_NAMES["qxc"],
        "digits": ("digit1", "digit2", "digit3", "digit4", "digit5", "digit6", "digit7"),
        "classes": (10, 10, 10, 10, 10, 10, 15),
        "history_url": "https://datachart.500.com/qxc/history/inc/history.php",
        "min_issue": 20100,
    },
}


def game_code(value: str) -> str:
    """将中文名或代码规范为代码；未知游戏直接报错。"""
    normalized = str(value).strip()
    code = normalized.lower()
    if code in GAME_NAMES:
        return code
    if normalized in GAME_NAME_TO_CODE:
        return GAME_NAME_TO_CODE[normalized]
    raise ValueError(f"不支持的彩种: {value}")


def game_name(value: str) -> str:
    return GAME_NAMES[game_code(value)]


def is_digit_game(value: str) -> bool:
    try:
        return game_code(value) in DIGIT_GAME_CONFIGS
    except ValueError:
        return False


def digit_columns(value: str) -> tuple[str, ...]:
    return DIGIT_GAME_CONFIGS[game_code(value)]["digits"]


def normalize_play(game: str, play: str | None = None) -> str:
    code = game_code(game)
    options = GAME_PLAY_OPTIONS[code]
    if code == "pls" and play is None:
        return "直选"  # 兼容旧调用；UI 默认可另行使用 GAME_PLAY_DEFAULTS
    if code != "pls":
        return options[0]
    if play not in options:
        raise ValueError(f"排列3不支持玩法: {play}")
    return play


def game_play_description(game: str, play: str | None = None) -> str:
    """返回与彩种绑定的玩法规则和组合概率说明。"""
    code = game_code(game)
    normalized_play = normalize_play(code, play)
    return GAME_PLAY_DESCRIPTIONS[(code, normalized_play)]
