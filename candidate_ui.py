"""Research-only UI for independently stored alternative-model experiments."""
from pathlib import Path
import re

import gradio as gr

from game_config import ALL_GAME_CODES, GAME_PLAY_OPTIONS, game_name
from run_candidate_experiment import RUNS, STRATEGY_NAMES, run_experiment
from candidate_results import read_report


def comparison_view(folder=None):
    if folder is None:
        completed = sorted(p for p in RUNS.iterdir() if p.is_dir() and re.fullmatch(r"[0-9]{17}", p.name)
                           and (p / "report.json").is_file()) if RUNS.exists() else []
        if not completed:
            return "尚无完成的候选实验。原有模型和前瞻研究不会被改变。", [], [], [], None
        folder = completed[-1]  # Chronological display, not selection by performance.
    folder = Path(folder)
    report = read_report(folder)
    index = {(r["game"], r["play"], r["strategy"]): r for r in report["summary"]}
    results = []
    for game in ALL_GAME_CODES:
        for play in GAME_PLAY_OPTIONS[game]:
            rows = [index[(game, play, strategy)] for strategy in ("xgboost", "markov", "lstm")]
            results.append([game_name(game), play, rows[0]["periods"], *[r["wins"] for r in rows],
                            rows[0]["random_wins"], *[format(r["p_holm"], ".6g") for r in rows]])
    bias = [[game_name(row["game"]), row["label"], row["n_draws"],
             None if row["p_value"] is None else format(row["p_value"], ".6g"),
             None if row["p_holm"] is None else format(row["p_holm"], ".6g"),
             "仅异常线索，非预测优势" if row["diagnostic_flag_only"] else
             ("样本不足" if row["p_holm"] is None else "未检出偏差，不等于证明随机")]
            for row in report["bias_diagnostics"]]
    scores = [[game_name(row["game"]), STRATEGY_NAMES.get(row["strategy"], "公平开奖位置边际"),
               round(row["head_log_loss"], 6)] for row in report["head_scores"]]
    signal = sum(r["historical_signal_only"] for r in report["summary"])
    bias_flags = sum(r["diagnostic_flag_only"] for r in report["bias_diagnostics"])
    note = (f"实验 {folder.name}：五彩种、七玩法，每期各5组。21项模型比较中{signal}项历史统计信号；"
            "无论是否显著，都没有已验证的未来一等奖优势。\n"
            "位置对数损失越低越好，但不是一等奖命中概率。偏差诊断只使用训练前缀，未用于选号或调参。\n"
            f"38项偏差诊断中{bias_flags}项异常线索；应先独立核验数据来源及规则适用期，不能直接解释为可预测。\n"
            "这批300期已被查看；重复运行不会产生新的独立证据。旧前瞻协议/记录保持不变。")
    return note, results, bias, scores, str(folder / "report.json")


def build_candidate_tab():
    queue = dict(concurrency_id="training_session", concurrency_limit=1)

    def refresh():
        try:
            return comparison_view()
        except Exception as exc:
            return f"不能读取完整对照：{exc}", [], [], [], None

    def execute(progress=gr.Progress()):
        try:
            folder = run_experiment(progress=lambda message: progress(0, desc=message))
            progress(1, desc="固定参数对照完成")
            return comparison_view(folder)
        except Exception as exc:
            return f"实验未完成，不能据此宣布优势：{exc}。旧研究未被替换。", [], [], [], None

    with gr.Tab("🧪 国际算法对照", render_children=True):
        gr.Markdown("### XGBoost / 简单马尔可夫 / 冻结LSTM / 均匀随机\n"
                    "新算法是研究候选，不是中奖承诺。本页只运行历史对照，不购买、不登记未来预测、不替换默认算法。\n"
                    "固定使用原研究快照末端300期，每期每种方法5组；完整结果写入独立 `candidate_runs`。")
        with gr.Row():
            refresh_btn = gr.Button("查看最新完整对照")
            run_btn = gr.Button("运行固定参数实验（五彩种，耗时取决于CPU）")
        with gr.Accordion("本轮固定方法与边界", open=False):
            gr.Markdown("XGBoost：80棵、深度3、学习率0.05、hist、max_bin64、2线程、固定种子20260912。"
                        "输入为此前30期特征的最后一行、均值、标准差。\n"
                        "马尔可夫：同一位置上一期到下一期，Laplace平滑1；球彩位置指排序位置，不是实际出球顺序。\n"
                        "两候选只用原LSTM相同训练目标区间，不用验证段调参，300期测试中不重训。"
                        "主检验21项统一Holm；偏差诊断另作一组Holm。"
                        "仅检验边际与相邻重复，不能证明开奖完全随机或证明可以预测。")
        note = gr.Textbox(label="实验结论与范围", lines=5, interactive=False)
        results = gr.Dataframe(headers=["彩种", "玩法", "期数", "XGB命中期", "Markov命中期", "LSTM命中期", "随机命中期",
                                        "XGB校正p", "Markov校正p", "LSTM校正p"], interactive=False)
        with gr.Accordion("训练前缀偏差诊断（不能作为中奖率）", open=False):
            bias = gr.Dataframe(headers=["彩种", "诊断", "训练前缀期数", "原始p", "Holm校正p", "范围说明"], interactive=False)
        with gr.Accordion("辅助概率诊断（越低越好，非整注概率）", open=False):
            scores = gr.Dataframe(headers=["彩种", "模型/基准", "平均位置对数损失"], interactive=False)
        report_file = gr.File(label="完整机器可读报告", interactive=False)
        outputs = [note, results, bias, scores, report_file]
        refresh_btn.click(refresh, outputs=outputs, **queue)
        run_btn.click(execute, outputs=outputs, **queue)
