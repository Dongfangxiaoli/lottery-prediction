"""Small, separate research tab; does not alter the ordinary model session."""
import gradio as gr
from evidence_service import (freeze_study, record_next_prediction, research_summary,
                              sample_size_summary, prospective_summary)


def build_evidence_tab(game_selector, play_selector):
    def safe_text(fn):
        def run(*args):
            try:
                return fn(*args)
            except Exception as exc:
                return f"未完成：{exc}"
        return run

    def refresh():
        try:
            description, rows = research_summary()
            return description, rows, sample_size_summary()
        except Exception as exc:
            return f"尚不能读取研究：{exc}", [], []

    def score():
        try:
            return prospective_summary()
        except Exception as exc:
            return f"尚不能核验前瞻记录：{exc}", []

    queue = dict(concurrency_id="training_session", concurrency_limit=1)
    with gr.Tab("🔬 预测优势验证", render_children=True):
        gr.Markdown("### 找证据，而不是预设模型有效\n"
                    "历史探索与未来验证分开。每期5组仅为固定测量口径，不是购买建议；不自动购买。\n"
                    "同一目标期只登记一次，记录不能覆盖。所有玩法公开报告，不挑中奖的结果。")
        refresh_btn = gr.Button("查看历史实验与所需样本量")
        note = gr.Textbox(label="研究范围", interactive=False, lines=3)
        historical = gr.Dataframe(headers=["彩种", "玩法", "期数", "模型命中期", "随机命中期", "精确随机基准", "单侧p值（探索）", "证据等级"], interactive=False)
        gr.Markdown("功效规划假设：真实命中概率是随机的2倍、80%把握度、七玩法合计5%误报控制。"
                    "下表是测量难度，不是预测效果承诺。1000期协议是有限试验上限，不保证能检出微小优势。")
        sizes = gr.Dataframe(headers=["彩种", "玩法", "每期5注p0", "所需期数", "临界命中期数", "把握度"], interactive=False)
        refresh_btn.click(refresh, outputs=[note, historical, sizes], **queue)
        freeze_btn = gr.Button("冻结本轮前瞻实验（仅创建一次）")
        freeze_note = gr.Textbox(label="冻结状态", interactive=False, lines=4)
        freeze_btn.click(safe_text(freeze_study), outputs=freeze_note, **queue)
        gr.Markdown("### 开奖前登记下一期\n"
                    "先在顶部选彩种/玩法，联网更新数据并核对期号日期。仅支持数据紧接的下一期；"
                    "为避免当天开奖时间歧义，仅允许明天或以后的日期，跨年首期暂不支持。"
                    "使用冻结研究权重，不读取当前训练滑块，也不会重训。")
        issue = gr.Textbox(label="目标期号（五位，例如26106）")
        target_date = gr.Textbox(label="目标开奖日期（YYYY-MM-DD，以官方为准）")
        commit_btn = gr.Button("生成并锁定模型组与随机对照")
        committed = gr.Textbox(label="登记结果（本地校验，不是独立时间戳）", lines=12, interactive=False)
        commit_btn.click(safe_text(record_next_prediction), [game_selector, play_selector, issue, target_date], committed, **queue)
        score_btn = gr.Button("开奖后读取本地数据核验")
        score_note = gr.Textbox(label="前瞻结论", interactive=False, lines=3)
        scored = gr.Dataframe(headers=["彩种", "玩法", "已登记期数", "已核验期数", "模型命中期", "随机命中期", "状态"], interactive=False)
        score_btn.click(score, outputs=[score_note, scored], **queue)
