"""Variable-count portfolio UI, isolated from legacy prediction records."""
from datetime import datetime
import json
import math
from numbers import Real
from pathlib import Path

import gradio as gr
import pandas as pd

from game_config import game_code, game_name, normalize_play
from predictor import format_numbers_copy

EXPORT_DIR = Path(__file__).parent / "results" / "portfolios"
STRATEGIES = {"均匀随机去重": "uniform", "模型联合排序（实验）": "model", "目标奖级覆盖优化": "coverage"}
MAX_PRIZE = {"ssq": 6, "dlt": 7, "pls": 1, "plw": 1, "qxc": 6}


def _integer_input(value, label):
    # Browser Number inputs can send 10.0; accept integral values but never round.
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or int(value) != value:
        raise ValueError(f"{label}必须是整数，不能留空或填写小数。")
    return int(value)


def portfolio_options(game, play):
    code = game_code(game)
    effective = normalize_play(code, play)
    maximum = {"组选3": 90, "组选6": 120}.get(effective, 500) if code == "pls" else 500
    choices = [("最高奖", 1)] + [(f"至少中{level}等奖（含更高奖）", level)
                                     for level in range(2, MAX_PRIZE[code] + 1)]
    return maximum, choices


def _session_probabilities(code, state):
    from predictor import _load_model, _get_probabilities, _get_probabilities_ensemble
    features, info = state.get(f"{code}_features"), state.get(f"{code}_train_info")
    if features is None or info is None:
        raise ValueError("模型排序需要先在分步操作中获取数据并训练；随机去重和覆盖优化无需训练。")
    kwargs = {key: info[key] for key in ("input_size", "hidden_size", "num_layers", "dropout")}
    kwargs.update(bidirectional=info.get("bidirectional", False),
                  use_attention=info.get("use_attention", True), attn_heads=info.get("attn_heads", 4))
    seeds = info.get("ensemble_seeds")
    if info.get("n_ensemble", 1) > 1 and seeds:
        probs = _get_probabilities_ensemble(code, features, info["seq_len"],
                                            **kwargs, ensemble_seeds=seeds)
    else:
        probs = _get_probabilities(_load_model(code, **kwargs), features, info["seq_len"])
    df = state.get(f"{code}_df")
    source = {"source_issue": None, "source_date": None}
    if df is not None and len(df):
        source = {"source_issue": str(df.iloc[-1]["issue"]), "source_date": str(df.iloc[-1]["date"])[:10]}
    return probs, source


def make_portfolio(game, play, count, strategy, target, seed, state):
    from portfolio_engine import generate_portfolio
    code = game_code(game)
    effective = normalize_play(code, play)
    if strategy not in STRATEGIES:
        raise ValueError("请选择有效的组合算法。")
    count, target, seed = (_integer_input(value, label) for value, label in
                           ((count, "注数"), (target, "目标奖级"), (seed, "复现编号")))
    probs, source = None, {}
    if STRATEGIES[strategy] == "model":
        probs, source = _session_probabilities(code, state)
    result = generate_portfolio(code, effective, n_tickets=count,
                                strategy=STRATEGIES[strategy], target_prize=target,
                                probs=probs, seed=seed, n_eval=10000)
    result.update(source)
    result["generated_at"] = datetime.now().isoformat(timespec="microseconds")
    result["schema_version"] = "portfolio-0.7.0"
    result["source_kind"] = "session_model" if probs is not None else "mathematical_design"
    result["draw_binding"] = "未绑定实际待开奖期；不是开奖前存证或未来命中证据"
    result["report"] += "\n" + result["draw_binding"]
    if source:
        result["report"] += (f"\n模型数据截止：{source['source_issue']}（{source['source_date']}）。"
                             "请核对是否最新；模型分数不是实际中奖概率。")
    copied = format_numbers_copy(code, result["tickets"], effective, include_play=True)
    rows = [{"注号": i, "号码": format_numbers_copy(code, [ticket], effective)}
            for i, ticket in enumerate(result["tickets"], 1)]
    comparison = []
    for item in result["comparison"]:
        exact = target == 1
        rate = result["exact_jackpot_probability"] if exact else item["hit_rate"]
        low, high = item["wilson95"]
        comparison.append({
            "方案": "本次组合" if item["role"] == "portfolio" else "同注数随机对照",
            "注数": item["n_tickets"], "基本成本（元）": 2 * item["n_tickets"],
            "最高奖精确概率": f"{item['exact_jackpot_probability']:.10%}",
            "目标奖级或更高奖至少中一次": f"{rate:.10%}",
            "统计口径": "组合精确计数" if exact else f"独立模拟{item['n_eval']}期",
            "模拟95%区间": "不适用（精确值）" if exact else f"{low:.4%}–{high:.4%}",
        })
    return result["report"], copied, pd.DataFrame(rows), pd.DataFrame(comparison), result, None


def export_portfolio(result, directory=None):
    if not result:
        raise ValueError("请先生成方案；更改参数后旧方案已清空。")
    directory = Path(directory) if directory is not None else EXPORT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"portfolio_{datetime.now():%Y%m%d%H%M%S%f}"
    suffix = 0
    # A clock tick can repeat. Exclusive creation also protects concurrent exports.
    while True:
        name = stem if suffix == 0 else f"{stem}_{suffix}"
        path = directory / f"{name}.json"
        try:
            stream = path.open("x", encoding="utf-8")
            break
        except FileExistsError:
            suffix += 1
    with stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return str(path.resolve())


def build_portfolio_tab(game_selector, play_selector, state, restore_button=None):
    with gr.Tab("🧩 组合方案与中奖概率", render_children=True):
        gr.Markdown(
            "### 可调注数 · 完整组合覆盖\n"
            "随机去重和覆盖优化无需训练。最高奖概率精确按不同整票计数；"
            "小奖使用独立模拟开奖评估，不是历史预测准确率。"
            "覆盖优化的目标为所选奖级或更高奖至少中一次，不保证获利。\n\n"
            "注数不再固定为5。本机每次最多500注以控制计算量；排列3组选3最多90种、组选6最多120种。"
        )
        with gr.Row():
            count = gr.Number(value=10, minimum=1, maximum=500, label="方案注数（正整数）")
            strategy = gr.Dropdown(list(STRATEGIES), value="均匀随机去重", label="组合算法")
            target = gr.Dropdown(portfolio_options("ssq", None)[1], value=1, label="优化目标奖级")
            seed = gr.Number(value=42, minimum=0, maximum=2**32 - 1, label="复现编号（整数）")
        gr.Markdown("同一参数与复现编号会生成同一方案，重复生成或重复购买不会增加不同整票覆盖。费用按每注2元基本投注估算，不含追加或倍投。")
        generate = gr.Button("生成组合方案并评估", variant="primary")
        report = gr.Textbox(label="方案、成本与概率说明", lines=15)
        comparison = gr.Dataframe(label="各奖级与同注数随机方案对照", interactive=False)
        copy = gr.Textbox(label="复制整套号码（含彩种/玩法）", lines=8)
        numbers = gr.Dataframe(label="去重后的完整号码", interactive=False)
        result = gr.State(None)
        export = gr.Button("导出本次方案与评估（JSON）")
        file = gr.File(label="独立方案文件（不写入旧预测历史）", interactive=False)
        outputs = [report, copy, numbers, comparison, result, file]

        def run_portfolio(game, play, count, strategy, target, seed, progress=gr.Progress()):
            try:
                progress(0.1, desc="生成完整组合并进行独立概率评估...")
                values = make_portfolio(game, play, count, strategy, target, seed, state)
                progress(1.0, desc="方案评估完成")
                return values
            except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
                return f"无法生成：{exc}", "", None, None, None, None

        def clear_portfolio():
            return "参数已更改，请重新生成方案。", "", None, None, None, None

        def refresh_portfolio_game(game):
            # The main app updates the play asynchronously; don't read its old value here.
            from game_config import GAME_PLAY_DEFAULTS
            return refresh_portfolio_play(game, GAME_PLAY_DEFAULTS[game_code(game)])

        def refresh_portfolio_play(game, play):
            maximum, choices = portfolio_options(game, play)
            return gr.update(maximum=maximum, value=10), gr.update(choices=choices, value=1), *clear_portfolio()

        def save_portfolio(value):
            try:
                return export_portfolio(value)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        controls = [game_selector, play_selector, count, strategy, target, seed, generate]
        if restore_button is not None:
            controls.append(restore_button)

        def lock_portfolio():
            return [gr.update(interactive=False) for _ in controls]

        def unlock_portfolio():
            return [gr.update(interactive=True) for _ in controls]

        # Prevent an in-flight old result from reappearing after a user switches
        # the game or parameters.  .then unlocks after success OR failure.
        generate.click(lock_portfolio, [], controls, queue=False).then(
            run_portfolio, [game_selector, play_selector, count, strategy, target, seed], outputs,
            concurrency_id="training_session", concurrency_limit=1,
        ).then(unlock_portfolio, [], controls, queue=False)
        export.click(save_portfolio, [result], [file])
        for control in (count, strategy, target, seed):
            control.change(clear_portfolio, [], outputs, queue=False)
        game_selector.change(refresh_portfolio_game, [game_selector], [count, target, *outputs], queue=False)
        play_selector.change(refresh_portfolio_play, [game_selector, play_selector], [count, target, *outputs], queue=False)
