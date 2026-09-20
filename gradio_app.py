"""
彩票预测工具 - Gradio Web UI
支持双色球、大乐透及数字型彩票的数据获取、模型训练、号码预测。
"""
import os
import sys
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings("ignore")

# 确保模块路径
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher_ssq import fetch_incremental as fetch_ssq
from data_fetcher_dlt import fetch_incremental as fetch_dlt
from data_fetcher_digit import fetch_incremental as fetch_digit
from feature_engineering import build_features_ssq, build_features_dlt, build_features_digit
from train import train_model
from predictor import predict_numbers, format_numbers_table, format_numbers_copy
from prediction_review import load_prediction_history, review_prediction, review_history_stats
from coverage_betting import (
    coverage_bet_ssq, coverage_bet_dlt, format_coverage_result,
)
from predictor import _empirical_frequency
from portfolio_ui import build_portfolio_tab
from evidence_ui import build_evidence_tab
from candidate_ui import build_candidate_tab
from backtest import backtest_ssq, backtest_dlt, backtest_digit, format_backtest_summary
from game_config import (
    DEFAULT_BACKTEST_PARAMS, DEFAULT_SELECTION_PARAMS, DEFAULT_TRAINING_PARAMS,
    GAME_NAMES, DIGIT_GAME_CONFIGS, GAME_PLAY_OPTIONS, GAME_PLAY_DEFAULTS,
    digit_columns, game_code, game_play_description, normalize_play,
)


def _code(game: str) -> str:
    return game_code(game)

# ---------- 中文字体配置 ----------
def _setup_chinese_font():
    """配置 matplotlib 中文字体"""
    font_candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "Source Han Sans CN", "Noto Sans CJK SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in font_candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # fallback
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()

# ---------- 全局状态 ----------
_state = {
    "ssq_df": None,
    "dlt_df": None,
    "ssq_features": None,
    "ssq_labels": None,
    "ssq_scaler": None,
    "dlt_features": None,
    "dlt_labels": None,
    "dlt_scaler": None,
    "ssq_train_info": None,
    "dlt_train_info": None,
}
for _digit_code in DIGIT_GAME_CONFIGS:
    _state[f"{_digit_code}_df"] = None
    _state[f"{_digit_code}_features"] = None
    _state[f"{_digit_code}_labels"] = None
    _state[f"{_digit_code}_scaler"] = None
    _state[f"{_digit_code}_train_info"] = None

SCALER_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(SCALER_DIR, exist_ok=True)
BACKTEST_HOLDOUT = 50


# ---------- 数据获取 / 只读离线载入 ----------
def _prepare_data(game, df):
    from local_history import prepare_history
    g = _code(game)
    df = prepare_history(df, g)
    fit_end = len(df) - BACKTEST_HOLDOUT
    if fit_end < 2:
        raise ValueError(f"历史数据不足：当前{len(df)}期，需保留末端{BACKTEST_HOLDOUT}期。")
    if g == "ssq":
        features, labels, scaler = build_features_ssq(df, fit_end=fit_end)
    elif g == "dlt":
        features, labels, scaler = build_features_dlt(df, fit_end=fit_end)
    else:
        features, labels, scaler = build_features_digit(df, g, fit_end=fit_end)
    if not np.isfinite(features).all():
        raise ValueError("历史数据构建出非有限特征，已拒绝替换当前会话。")
    return df, features, labels, scaler


def _describe_data(game, df, features, offline):
    from datetime import datetime
    g = _code(game)
    latest = df.iloc[-1]
    if g == "ssq":
        nums = "红球: " + " ".join(f"{int(latest[f'red{i}']):02d}" for i in range(1, 7))
        nums += f" | 蓝球: {int(latest['blue']):02d}"
    elif g == "dlt":
        nums = "前区: " + " ".join(f"{int(latest[f'front{i}']):02d}" for i in range(1, 6))
        nums += " | 后区: " + " ".join(f"{int(latest[f'back{i}']):02d}" for i in range(1, 3))
    else:
        nums = "开奖号码: " + " ".join(str(int(latest[c])) for c in digit_columns(g))
    latest_date = pd.Timestamp(latest["date"])
    source = "本地文件（未联网、未修改原文件）" if offline else "在线更新后可用数据（网络失败时可能为本地缓存）"
    summary = (
        f"✅ {GAME_NAMES[g]}数据载入完成\n来源：{source}\n"
        f"共 {len(df)} 期 | 数据截止：{latest_date:%Y-%m-%d} 第{latest['issue']}期\n"
        f"{nums}\n特征维度: {features.shape}\n号码、日期与期号基本校验通过；不等同于官方来源核验。"
    )
    age = (datetime.now().date() - latest_date.date()).days
    if age > 7:
        summary += f"\n⚠️ 本地可用数据距今 {age} 天；尚未确认最新开奖，请联网更新并核对日期。"
    elif not offline:
        summary += "\n请自行核对截止期号；本地文件无新增不代表在线已核实最新。"
    recent = df.tail(20).iloc[::-1].reset_index(drop=True).copy()
    recent["date"] = recent["date"].astype(str).str[:10]
    return summary, recent, _plot_frequency(df, g)


def _commit_data(game, prepared, train_info=None):
    g = _code(game)
    df, features, labels, scaler = prepared
    # Build/validate every item first; an invalid file never leaves a half-updated session.
    _state.update({f"{g}_df": df, f"{g}_features": features, f"{g}_labels": labels,
                   f"{g}_scaler": scaler, f"{g}_train_info": train_info})


def fetch_data(game: str, progress=gr.Progress()):
    """Try an online update, then validate before replacing in-memory session data."""
    try:
        g = _code(game)
        callback = lambda p, m: progress(p, desc=m)
        df = fetch_ssq(callback) if g == "ssq" else fetch_dlt(callback) if g == "dlt" else fetch_digit(g, callback)
        progress(0.9, desc="校验数据并构建特征...")
        prepared = _prepare_data(g, df)
        display = _describe_data(g, prepared[0], prepared[1], offline=False)
        _commit_data(g, prepared)
        return display
    except Exception as exc:
        return f"❌ 获取数据失败：{exc}\n未替换当前会话的数据与模型。", None, None


def load_local_data(game: str, progress=gr.Progress()):
    """Read bundled CSV only: no HTTP request and no scaler/model overwrite."""
    try:
        from local_history import read_local_history
        progress(0.1, desc="读取本地历史文件（不联网）...")
        prepared = _prepare_data(game, read_local_history(game))
        display = _describe_data(game, prepared[0], prepared[1], offline=True)
        _commit_data(game, prepared)
        return display
    except Exception as exc:
        return f"❌ 本地载入失败：{exc}\n未替换当前会话的数据与模型。", None, None


def restore_saved_session(game: str, progress=gr.Progress()):
    """Restore only a matching, checksummed v0.8+ training archive."""
    try:
        from local_history import read_local_history
        from model_session import load_training_session
        progress(0.1, desc="读取本地数据并检查已保存的训练档案...")
        prepared = _prepare_data(game, read_local_history(game))
        df, features, _, _ = prepared
        info = load_training_session(_code(game), df, features, SCALER_DIR)
        # Also check that the recorded architecture can load the actual model weights.
        from portfolio_ui import _session_probabilities
        _session_probabilities(_code(game), {f"{_code(game)}_df": df,
            f"{_code(game)}_features": features, f"{_code(game)}_train_info": info})
        display = _describe_data(game, df, features, offline=True)
        loss = _plot_loss_curves(info["train_losses"], info["val_losses"])
        summary = (
            f"✅ {GAME_NAMES[_code(game)]}已恢复保存的模型，无需本次重训。\n"
            f"窗口 {info['seq_len']} | 隐藏层 {info['hidden_size']} | 集成 {info['n_ensemble']}\n"
            "数据、特征和权重校验一致；恢复使用的是档案参数，不是当前滑块值。\n"
            "此校验不证明真实预测优势；历史回测仍只作探索。"
        )
        _commit_data(game, prepared, info)
        return (*display, summary, loss)
    except Exception as exc:
        return (f"❌ 恢复失败：{exc}\n未替换当前会话；可先载入数据再训练一次。",
                None, None, "模型未恢复。旧权重没有训练档案时不能自动推定训练参数。", None)


# ---------- 模型训练 ----------
def train(game: str, seq_len: int, epochs: int, lr: float, hidden: int,
          batch_size: int, bidirectional: bool = False,
          use_attention: bool = True, n_ensemble: int = 1,
          progress=gr.Progress()):
    """训练 LSTM 模型"""
    try:
        g = _code(game)
        features = _state.get(f"{g}_features")
        labels = _state.get(f"{g}_labels")
        if features is None:
            return "❌ 请先获取数据！", None

        # A failed retrain may have touched checkpoints; do not retain stale architecture info.
        _state[f"{g}_train_info"] = None
        info = train_model(
            features=features,
            labels_dict=labels,
            game=g,
            seq_len=int(seq_len),
            num_epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=lr,
            hidden_size=int(hidden),
            bidirectional=bidirectional,
            use_attention=use_attention,
            n_ensemble=int(n_ensemble),
            holdout_size=BACKTEST_HOLDOUT,
            progress_callback=lambda p, m: progress(p, desc=m),
        )

        _state[f"{g}_train_info"] = info

        archive_note = ""
        try:
            from model_session import save_training_session
            archive = save_training_session(g, _state[f"{g}_df"], features, info,
                                            os.path.dirname(info["model_path"]))
            archive_note = "\n训练档案已保存；下次启动可点击“恢复已保存模型（不重训）”。"
        except Exception as exc:
            archive_note = f"\n⚠️ 本次模型可用，但训练档案保存失败，重启后可能需要重训：{exc}"

        # Loss 曲线图
        loss_fig = _plot_loss_curves(info["train_losses"], info["val_losses"])

        summary = (
            f"✅ {game}模型训练完成\n"
            f"最佳 Epoch: {info['best_epoch'] + 1} / {len(info['train_losses'])}\n"
            f"最终 Train Loss: {info['train_losses'][-1]:.4f}\n"
            f"最佳 Val Loss: {min(info['val_losses']):.4f}\n"
        )

        if n_ensemble > 1:
            summary += f"集成模型数: {n_ensemble}\n"

        summary += f"样本外保留: 末端 {info['holdout_size']} 期（未参与训练/早停）\n"
        summary += f"模型已保存: {info['model_path']}" + archive_note

        return summary, loss_fig

    except Exception as e:
        import traceback
        return f"❌ 训练失败: {str(e)}\n{traceback.format_exc()}", None


# ---------- 预测号码 ----------
def predict(game: str, play: str, n_groups: int, temperature: float,
            anneal_enabled: bool = False,
            sampling_mode: str = "merged",
            freq_alpha: float = 0.3,
            top_p: float = 0.9,
            missing_alpha: float = 0.0,
            hot_alpha: float = 0.0,
            progress=gr.Progress(), selection_strategy="weighted"):
    """生成推荐号码"""
    try:
        g = _code(game)
        if g == "ssq":
            features = _state.get("ssq_features")
            train_info = _state.get("ssq_train_info")
            df = _state.get("ssq_df")
            g = "ssq"
        elif g == "dlt":
            features = _state.get("dlt_features")
            train_info = _state.get("dlt_train_info")
            df = _state.get("dlt_df")
            g = "dlt"
        else:
            features = _state.get(f"{g}_features")
            train_info = _state.get(f"{g}_train_info")
            df = _state.get(f"{g}_df")

        if features is None:
            return "❌ 请先获取数据！", "", None, None
        if train_info is None:
            return "❌ 请先训练模型！", "", None, None

        progress(0.3, desc="加载模型并推理...")

        n_ensemble = train_info.get("n_ensemble", 1)
        ensemble_seeds = train_info.get("ensemble_seeds", None)
        bidirectional = train_info.get("bidirectional", False)
        use_attention = train_info.get("use_attention", True)
        attn_heads = train_info.get("attn_heads", 4)

        anneal_temps = [0.5, 0.8, 1.2, 1.8, 2.5] if anneal_enabled else None

        # 提取原始历史号码用于频率/遗漏/冷热加权
        history_main = history_side = None
        if df is not None and (freq_alpha > 0 or missing_alpha > 0 or hot_alpha != 0):
            if g == "ssq":
                red_cols = [f"red{i}" for i in range(1, 7)]
                history_main = df[red_cols].values
                history_side = df["blue"].values.reshape(-1, 1)
            elif g == "dlt":
                front_cols = [f"front{i}" for i in range(1, 6)]
                back_cols = [f"back{i}" for i in range(1, 3)]
                history_main = df[front_cols].values
                history_side = df[back_cols].values
            else:
                cols = list(digit_columns(g))
                history_main = df[cols].values

        play = normalize_play(g, play)
        numbers, probs = predict_numbers(
            game=g,
            play=play,
            features=features,
            seq_len=train_info["seq_len"],
            n_groups=int(n_groups),
            temperature=temperature,
            input_size=train_info["input_size"],
            hidden_size=train_info["hidden_size"],
            num_layers=train_info["num_layers"],
            dropout=train_info["dropout"],
            bidirectional=bidirectional,
            use_attention=use_attention,
            attn_heads=attn_heads,
            n_ensemble=n_ensemble,
            ensemble_seeds=ensemble_seeds,
            anneal_temps=anneal_temps,
            sampling_mode=sampling_mode,
            freq_alpha=float(freq_alpha),
            top_p=float(top_p),
            missing_alpha=float(missing_alpha),
            hot_alpha=float(hot_alpha),
            history_main=history_main,
            history_side=history_side,
            source_issue=str(df.iloc[-1]["issue"]) if df is not None else None,
            source_date=str(df.iloc[-1]["date"])[:10] if df is not None else None,
            selection_strategy=selection_strategy,
        )

        progress(0.7, desc="生成号码...")

        # 格式化输出
        text = format_numbers_table(g, numbers, play)
        if selection_strategy == "joint_top5" and df is not None:
            latest = df.iloc[-1]
            text += (f"\n数据截止：第{latest['issue']}期（{str(latest['date'])[:10]}）。"
                     "请确认已更新至最新开奖，旧数据输出不代表当前待开奖期。")
        copy_text = format_numbers_copy(g, numbers, play, include_play=True)

        # 概率分布热力图
        prob_fig = _plot_probability_heatmap(probs, g)

        # 号码表格
        if g == "ssq":
            table_data = []
            for n in numbers:
                table_data.append({
                    "红球": " ".join(f"{n['red'][j]:02d}" for j in range(6)),
                    "蓝球": f"{n['blue']:02d}",
                })
        elif g == "dlt":
            table_data = []
            for n in numbers:
                table_data.append({
                    "前区": " ".join(f"{n['front'][j]:02d}" for j in range(5)),
                    "后区": " ".join(f"{n['back'][j]:02d}" for j in range(2)),
                })
        else:
            table_data = [{"玩法": play, "开奖号码": format_numbers_copy(g, [n], play)} for n in numbers]

        numbers_df = pd.DataFrame(table_data)
        progress(1.0, desc="预测完成！")

        return text, copy_text, numbers_df, prob_fig

    except Exception as e:
        import traceback
        return f"❌ 预测失败: {str(e)}\n{traceback.format_exc()}", "", None, None


def generate_top_five(game: str, play: str, progress=gr.Progress()):
    """面向最高奖的固定5注入口；不改变旧随机对照流程。"""
    return predict(game, play, 5, 1.0, False, "per_position", 0.0, 1.0, 0.0, 0.0,
                   progress=progress, selection_strategy="joint_top5")


def evaluate_top_five(game: str, play: str, progress=gr.Progress()):
    try:
        from jackpot_evaluation import backtest_top_five, format_top_five_backtest
        g = _code(game)
        df, info = _state.get(f"{g}_df"), _state.get(f"{g}_train_info")
        if df is None or info is None:
            return "请先在分步操作页获取数据并训练模型，再做固定5注对照。", None
        progress(0.1, desc="对比联合排序、旧加权、均匀随机的最高奖命中...")
        result = backtest_top_five(df, info, g, play, n_test=min(50, info["holdout_size"]))
        progress(1.0, desc="最高奖对照完成；不据此宣称预测优势")
        rows = result["rows"]
        table = {"期号": [row["期号"] for row in rows["joint_top5"]]}
        for key, label in (("joint_top5", "联合排序"), ("legacy_weighted", "旧加权"),
                           ("uniform_random", "均匀随机")):
            table[label + "最高奖命中"] = [row["最高奖整注命中"] for row in rows[key]]
        return format_top_five_backtest(result), pd.DataFrame(table)
    except Exception as exc:
        return f"最高奖对照失败：{exc}", None


# ---------- 一键预测 ----------
def one_click_predict(game: str, play: str, seq_len: int, epochs: int, lr: float, hidden: int,
                      batch_size: int, n_groups: int, temperature: float,
                      bidirectional: bool = False, use_attention: bool = True,
                      n_ensemble: int = 1, anneal_enabled: bool = False,
                      sampling_mode: str = "merged",
                      freq_alpha: float = 0.3,
                      top_p: float = 0.9,
                      missing_alpha: float = 0.0,
                      hot_alpha: float = 0.0,
                      progress=gr.Progress()):
    """一键完成：获取数据 → 训练 → 预测"""
    # Step 1: 获取数据
    progress(0.0, desc="Step 1/3: 获取数据...")
    data_summary, recent_df, freq_fig = fetch_data(game, progress)
    if "❌" in data_summary:
        return data_summary, None, None, None, None, None, ""

    # Step 2: 训练模型
    progress(0.33, desc="Step 2/3: 训练模型...")
    train_summary, loss_fig = train(game, seq_len, epochs, lr, hidden, batch_size,
                                    bidirectional, use_attention, n_ensemble, progress)
    if "❌" in train_summary:
        return train_summary, None, None, None, None, None, ""

    # Step 3: 预测号码
    progress(0.66, desc="Step 3/3: 生成号码...")
    pred_text, copy_text, numbers_df, prob_fig = predict(
        game, play, n_groups, temperature, anneal_enabled,
        sampling_mode, freq_alpha, top_p, missing_alpha, hot_alpha, progress)

    full_summary = f"{data_summary}\n\n{train_summary}\n\n{pred_text}"
    return full_summary, recent_df, freq_fig, loss_fig, numbers_df, prob_fig, copy_text


# ---------- 图表绘制 ----------
def _plot_frequency(df: pd.DataFrame, game: str):
    """号码出现频率分布图"""
    game = _code(game)
    if game in DIGIT_GAME_CONFIGS:
        cols = list(digit_columns(game)); classes = DIGIT_GAME_CONFIGS[game]["classes"]
        fig, axes = plt.subplots(len(cols), 1, figsize=(10, max(3, 2 * len(cols))), squeeze=False)
        for i, (col, size) in enumerate(zip(cols, classes)):
            axes[i, 0].bar(range(size), np.bincount(df[col].astype(int), minlength=size))
            axes[i, 0].set_title(f"第{i + 1}位频率 (0-{size - 1})")
        plt.tight_layout(); return fig
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if game == "ssq":
        red_cols = [f"red{i}" for i in range(1, 7)]
        all_reds = df[red_cols].values.flatten()
        red_counts = np.bincount(all_reds, minlength=34)[1:]  # 1-33
        axes[0].bar(range(1, 34), red_counts, color="#e74c3c", alpha=0.8)
        axes[0].set_title("红球频率分布 (1-33)")
        axes[0].set_xlabel("号码")
        axes[0].set_ylabel("出现次数")

        all_blues = df["blue"].values
        blue_counts = np.bincount(all_blues, minlength=17)[1:]  # 1-16
        axes[1].bar(range(1, 17), blue_counts, color="#3498db", alpha=0.8)
        axes[1].set_title("蓝球频率分布 (1-16)")
        axes[1].set_xlabel("号码")
        axes[1].set_ylabel("出现次数")
    else:
        front_cols = [f"front{i}" for i in range(1, 6)]
        all_fronts = df[front_cols].values.flatten()
        front_counts = np.bincount(all_fronts, minlength=36)[1:]
        axes[0].bar(range(1, 36), front_counts, color="#e74c3c", alpha=0.8)
        axes[0].set_title("前区频率分布 (1-35)")
        axes[0].set_xlabel("号码")
        axes[0].set_ylabel("出现次数")

        back_cols = [f"back{i}" for i in range(1, 3)]
        all_backs = df[back_cols].values.flatten()
        back_counts = np.bincount(all_backs, minlength=13)[1:]
        axes[1].bar(range(1, 13), back_counts, color="#3498db", alpha=0.8)
        axes[1].set_title("后区频率分布 (1-12)")
        axes[1].set_xlabel("号码")
        axes[1].set_ylabel("出现次数")

    plt.tight_layout()
    return fig


def _plot_loss_curves(train_losses: list, val_losses: list):
    """训练/验证 Loss 曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="#e74c3c", linewidth=1.5)
    ax.plot(val_losses, label="Val Loss", color="#3498db", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("训练损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _plot_probability_heatmap(probs: dict, game: str):
    """号码概率分布热力图"""
    if game in DIGIT_GAME_CONFIGS:
        names = [f"digit{i + 1}" for i in range(len(digit_columns(game)))]
        width = max(len(probs[name]) for name in names)
        data = np.full((len(names), width), np.nan)
        for i, name in enumerate(names):
            data[i, :len(probs[name])] = probs[name]
        fig, ax = plt.subplots(figsize=(14, 4))
        im = ax.imshow(data, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(names))); ax.set_yticklabels([f"第{i + 1}位" for i in range(len(names))])
        ax.set_xticks(range(width)); ax.set_xticklabels(range(width))
        ax.set_title(f"{GAME_NAMES.get(game, game)}各位置概率分布")
        plt.colorbar(im, ax=ax, shrink=0.7); plt.tight_layout(); return fig
    if game == "ssq":
        # 红球概率
        red_names = [f"red{i+1}" for i in range(6)]
        red_data = np.array([probs[n] for n in red_names])  # (6, 33)
        blue_data = probs["blue"].reshape(1, -1)  # (1, 16)

        fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                                  gridspec_kw={"height_ratios": [3, 1]})

        im1 = axes[0].imshow(red_data, aspect="auto", cmap="YlOrRd")
        axes[0].set_yticks(range(6))
        axes[0].set_yticklabels([f"红{i+1}" for i in range(6)])
        axes[0].set_xticks(range(33))
        axes[0].set_xticklabels(range(1, 34), fontsize=7)
        axes[0].set_title("红球各位概率分布")
        plt.colorbar(im1, ax=axes[0], shrink=0.6)

        im2 = axes[1].imshow(blue_data, aspect="auto", cmap="Blues")
        axes[1].set_yticks([0])
        axes[1].set_yticklabels(["蓝球"])
        axes[1].set_xticks(range(16))
        axes[1].set_xticklabels(range(1, 17), fontsize=8)
        axes[1].set_title("蓝球概率分布")
        plt.colorbar(im2, ax=axes[1], shrink=0.6)

    else:
        front_names = [f"front{i+1}" for i in range(5)]
        front_data = np.array([probs[n] for n in front_names])
        back_names = [f"back{i+1}" for i in range(2)]
        back_data = np.array([probs[n] for n in back_names])

        fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                                  gridspec_kw={"height_ratios": [3, 1]})

        im1 = axes[0].imshow(front_data, aspect="auto", cmap="YlOrRd")
        axes[0].set_yticks(range(5))
        axes[0].set_yticklabels([f"前区{i+1}" for i in range(5)])
        axes[0].set_xticks(range(35))
        axes[0].set_xticklabels(range(1, 36), fontsize=7)
        axes[0].set_title("前区各位概率分布")
        plt.colorbar(im1, ax=axes[0], shrink=0.6)

        im2 = axes[1].imshow(back_data, aspect="auto", cmap="Blues")
        axes[1].set_yticks(range(2))
        axes[1].set_yticklabels([f"后区{i+1}" for i in range(2)])
        axes[1].set_xticks(range(12))
        axes[1].set_xticklabels(range(1, 13), fontsize=8)
        axes[1].set_title("后区概率分布")
        plt.colorbar(im2, ax=axes[1], shrink=0.6)

    plt.tight_layout()
    return fig


# ---------- 覆盖投注 ----------
def coverage_bet(game: str, pool_size: int, n_tickets: int,
                  guaranteed_hits: int, freq_alpha: float, progress=gr.Progress()):
    """生成覆盖投注方案"""
    try:
        g = _code(game)
        if g in DIGIT_GAME_CONFIGS:
            return "ℹ️ 排列3、排列5、7星彩为按位数字型彩票，不适用双色球/大乐透的集合覆盖投注。", None
        if g == "ssq":
            features = _state.get("ssq_features")
            train_info = _state.get("ssq_train_info")
            df = _state.get("ssq_df")
            g = "ssq"
        else:
            features = _state.get("dlt_features")
            train_info = _state.get("dlt_train_info")
            df = _state.get("dlt_df")
            g = "dlt"

        if features is None or train_info is None:
            return "❌ 请先获取数据并训练模型！", None

        progress(0.3, desc="加载模型推理概率...")
        # 复用 predict 拿到概率分布（不保存号码）
        # 先做一次概率推理
        from predictor import _load_model, _get_probabilities
        n_ensemble = train_info.get("n_ensemble", 1)
        ensemble_seeds = train_info.get("ensemble_seeds", None)
        bidirectional = train_info.get("bidirectional", False)
        use_attention = train_info.get("use_attention", True)
        attn_heads = train_info.get("attn_heads", 4)
        from predictor import _get_probabilities_ensemble
        if n_ensemble > 1 and ensemble_seeds:
            probs = _get_probabilities_ensemble(
                g, features, train_info["seq_len"], train_info["input_size"],
                train_info["hidden_size"], train_info["num_layers"], train_info["dropout"],
                bidirectional, use_attention, attn_heads, ensemble_seeds)
        else:
            model = _load_model(g, train_info["input_size"], train_info["hidden_size"],
                                train_info["num_layers"], train_info["dropout"],
                                bidirectional, use_attention, attn_heads)
            probs = _get_probabilities(model, features, train_info["seq_len"])

        progress(0.6, desc="构建核心号码池与覆盖设计...")
        # 频率
        red_freq = blue_freq = front_freq = back_freq = None
        if df is not None and freq_alpha > 0:
            if g == "ssq":
                red_freq = _empirical_frequency(df[[f"red{i}" for i in range(1, 7)]].values, 33)
                blue_freq = _empirical_frequency(df["blue"].values.reshape(-1, 1), 16)
            else:
                front_freq = _empirical_frequency(df[[f"front{i}" for i in range(1, 6)]].values, 35)
                back_freq = _empirical_frequency(df[[f"back{i}" for i in range(1, 3)]].values, 12)

        if g == "ssq":
            cover = coverage_bet_ssq(probs, red_freq, blue_freq,
                                     int(pool_size), int(n_tickets), int(guaranteed_hits),
                                     float(freq_alpha))
        else:
            cover = coverage_bet_dlt(probs, front_freq, back_freq,
                                     int(pool_size), int(n_tickets), int(guaranteed_hits),
                                     float(freq_alpha))

        progress(1.0, desc="覆盖方案生成完成！")
        text = format_coverage_result(g, cover)

        # 表格
        if g == "ssq":
            table_data = [{
                "红球": " ".join(f"{t['red'][j]:02d}" for j in range(6)),
                "蓝球": f"{t['blue']:02d}",
            } for t in cover["tickets_full"]]
        else:
            table_data = [{
                "前区": " ".join(f"{t['front'][j]:02d}" for j in range(5)),
                "后区": " ".join(f"{t['back'][j]:02d}" for j in range(2)),
            } for t in cover["tickets_full"]]
        import pandas as _pd
        numbers_df = _pd.DataFrame(table_data)

        return text, numbers_df

    except Exception as e:
        import traceback
        return f"❌ 覆盖投注失败: {str(e)}\n{traceback.format_exc()}", None


# ---------- 历史回测 ----------
def run_backtest(game: str, play: str, n_test: int, n_groups: int,
                  temperature: float, sampling_mode: str,
                  freq_alpha: float, top_p: float,
                  missing_alpha: float, hot_alpha: float,
                  progress=gr.Progress()):
    """运行历史回测"""
    try:
        g = _code(game)
        if g in DIGIT_GAME_CONFIGS:
            df = _state.get(f"{g}_df"); train_info = _state.get(f"{g}_train_info")
            if df is None or train_info is None:
                return "❌ 请先获取数据并训练模型！", None
            if float(missing_alpha) != 0 or float(hot_alpha) != 0:
                return "ℹ️ 排列型数字彩票不使用遗漏/冷热假设，请将这两项保持为 0。", None
            summary = backtest_digit(
                df, train_info, g, int(n_test), int(n_groups),
                play=normalize_play(g, play),
                temperature=float(temperature), freq_alpha=float(freq_alpha),
                top_p=float(top_p))
            return format_backtest_summary(summary), pd.DataFrame(summary["rows"])
        if g == "ssq":
            df = _state.get("ssq_df")
            train_info = _state.get("ssq_train_info")
            g = "ssq"
        else:
            df = _state.get("dlt_df")
            train_info = _state.get("dlt_train_info")
            g = "dlt"

        if df is None:
            return "❌ 请先获取数据！", None
        if train_info is None:
            return "❌ 请先训练模型！", None

        progress(0.1, desc="构建特征...")
        if g == "ssq":
            summary = backtest_ssq(df, train_info, int(n_test), int(n_groups),
                                   temperature=float(temperature),
                                   sampling_mode=sampling_mode,
                                   freq_alpha=float(freq_alpha),
                                   top_p=float(top_p),
                                   missing_alpha=float(missing_alpha),
                                   hot_alpha=float(hot_alpha))
        else:
            summary = backtest_dlt(df, train_info, int(n_test), int(n_groups),
                                   temperature=float(temperature),
                                   sampling_mode=sampling_mode,
                                   freq_alpha=float(freq_alpha),
                                   top_p=float(top_p),
                                   missing_alpha=float(missing_alpha),
                                   hot_alpha=float(hot_alpha))

        progress(1.0, desc="回测完成！")
        text = format_backtest_summary(summary)
        df_result = pd.DataFrame(summary["rows"])
        return text, df_result

    except Exception as e:
        import traceback
        return f"❌ 回测失败: {str(e)}\n{traceback.format_exc()}", None


# ---------- Gradio 界面 ----------
def build_ui():
    with gr.Blocks(
        title="彩票预测工具 - 五种彩票",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            "# 🎰 彩票随机选号 + 覆盖投注工具\n"
            "**支持双色球、大乐透、排列3、排列5、7星彩** | LSTM 与候选算法研究\n\n"
            "> ⚠️ **重要声明**：在**公平、独立、均匀开奖**前提下，历史号码不能提高下一期单注的真实中奖概率。\n"
            "> 本工具**尚未证实预测优势**；偏差检验只是研究诊断。组选覆盖不同排列，不代表预测能力。工具价值包括：\n"
            "> 1. 按所选规则生成合法号码；\n"
            "> 2. 生成可调注数的去重组合，分别报告最高奖精确概率与小奖独立模拟评估；\n"
            "> 3. 客观回测并与理论随机基准对比。\n"
            "> 请理性购彩，量力而行。"
        )

        with gr.Row():
            game_selector = gr.Radio(
                choices=["双色球", "大乐透", "排列3", "排列5", "7星彩"],
                value="双色球",
                label="选择彩种",
            )
            play_selector = gr.Radio(
                choices=GAME_PLAY_OPTIONS.get("ssq", ["直选"]),
                value=GAME_PLAY_DEFAULTS.get("ssq", "直选"),
                label="选择玩法",
            )
            play_info = gr.Markdown(
                "当前玩法：单式投注。红球从 01–33 中选 6 个，不重复、顺序不计；蓝球从 01–16 中选 1 个。")
        restore_defaults_btn = gr.Button("↺ 恢复默认训练/选号参数", size="sm")
        gr.Markdown("*此按钮只恢复界面参数，不会重训或覆盖已有模型。*")

        def describe_play(game, play):
            code = _code(game)
            normalized_play = normalize_play(code, play)
            return (
                f"当前玩法：{normalized_play}。{game_play_description(code, normalized_play)}\n\n"
                "旧训练/选号页默认：窗口30、训练30轮、学习率0.001、隐藏层64、Batch 128、5注、"
                "温度1、单模型、无 Attention、频率先验0、Top-p 1、遗漏/冷热0。"
            )

        def update_play_options(game):
            code = _code(game)
            options = GAME_PLAY_OPTIONS.get(code, ["直选"])
            default = GAME_PLAY_DEFAULTS.get(code, options[0])
            return gr.update(choices=options, value=default), describe_play(code, default)

        with gr.Tabs():
            build_portfolio_tab(game_selector, play_selector, _state, restore_defaults_btn)
            build_evidence_tab(game_selector, play_selector)
            build_candidate_tab()
            with gr.Tab("🏆 旧5注实验对照", render_children=True):
                gr.Markdown(
                    "### 每期固定5注 · 对应玩法最高奖\n"
                    "先在「分步操作」获取数据并训练模型，再生成5注。"
                    "这里按完整号码的模型联合分数排序；不增加注数、不倍投、不以小奖率代替最高奖。\n\n"
                    "球彩按合法升序组合搜索；数字直选保留顺序；排列3组选汇总全部有效排列。"
                    "这是**实验性选号算法**，优化模型分数不等于真实中奖率提高。"
                )
                jp_generate = gr.Button("生成最高奖目标5注（实验）", variant="primary")
                jp_text = gr.Textbox(label="5注方案与概率边界", lines=12)
                jp_copy = gr.Textbox(label="复制5注（含彩种/玩法）", lines=6)
                jp_numbers = gr.Dataframe(label="最高奖目标5注")
                jp_prob = gr.Plot(label="模型位置分布（非真实中奖概率）")
                jp_generate.click(generate_top_five, inputs=[game_selector, play_selector],
                                  outputs=[jp_text, jp_copy, jp_numbers, jp_prob],
                                  concurrency_id="training_session", concurrency_limit=1)
                gr.Markdown("对照组同样每期5注；只计对应最高奖。旧留出集已查看，结果仅用于探索，不自动选赢家或调参。")
                jp_evaluate = gr.Button("对比三种算法的最高奖命中")
                jp_report = gr.Textbox(label="最高奖对照报告", lines=14)
                jp_rows = gr.Dataframe(label="逐期最高奖命中对照")
                jp_evaluate.click(evaluate_top_five, inputs=[game_selector, play_selector],
                                  outputs=[jp_report, jp_rows],
                                  concurrency_id="training_session", concurrency_limit=1)
            # ---- Tab 1: 一键选号 ----
            with gr.Tab("🚀 一键选号", render_children=True) as one_click_tab:
                gr.Markdown("自动完成：获取最新数据 → 训练模型 → 生成号码（本质为随机加权选号，非预测）")

                with gr.Row():
                    with gr.Column(scale=1):
                        oc_seq_len = gr.Slider(10, 60, value=DEFAULT_TRAINING_PARAMS["seq_len"], step=5, label="窗口大小（期数）")
                        oc_epochs = gr.Slider(20, 300, value=DEFAULT_TRAINING_PARAMS["epochs"], step=10, label="训练轮数")
                        oc_lr = gr.Number(value=DEFAULT_TRAINING_PARAMS["lr"], label="学习率")
                        oc_hidden = gr.Slider(64, 256, value=DEFAULT_TRAINING_PARAMS["hidden"], step=32, label="隐藏层大小")
                        oc_batch = gr.Slider(16, 128, value=DEFAULT_TRAINING_PARAMS["batch_size"], step=16, label="Batch Size")
                        oc_n_groups = gr.Slider(3, 20, value=DEFAULT_SELECTION_PARAMS["n_groups"], step=1, label="生成号码组数")
                        oc_temp = gr.Slider(0.5, 3.0, value=DEFAULT_SELECTION_PARAMS["temperature"], step=0.1, label="采样温度（越高越随机）")
                        oc_bidirectional = gr.Checkbox(value=DEFAULT_TRAINING_PARAMS["bidirectional"], label="双向LSTM")
                        oc_use_attention = gr.Checkbox(value=DEFAULT_TRAINING_PARAMS["use_attention"], label="启用Attention（实验）")
                        oc_n_ensemble = gr.Slider(1, 5, value=DEFAULT_TRAINING_PARAMS["n_ensemble"], step=1, label="集成模型数")
                        oc_anneal = gr.Checkbox(value=DEFAULT_SELECTION_PARAMS["anneal_enabled"], label="温度退火")
                        gr.Markdown("**采样优化**")
                        oc_sampling_mode = gr.Radio(
                            choices=["merged", "per_position"],
                            value=DEFAULT_SELECTION_PARAMS["sampling_mode"], label="采样模式",
                            info="merged: 合并多头消除位置偏差(推荐)")
                        oc_freq_alpha = gr.Slider(0.0, 1.0, value=DEFAULT_SELECTION_PARAMS["freq_alpha"], step=0.05,
                                                    label="频率先验混合系数α")
                        oc_top_p = gr.Slider(0.5, 1.0, value=DEFAULT_SELECTION_PARAMS["top_p"], step=0.05,
                                              label="Top-p核采样阈值")
                        oc_missing_alpha = gr.Slider(0.0, 2.0, value=DEFAULT_SELECTION_PARAMS["missing_alpha"], step=0.1,
                                                       label="遗漏值加权系数（>0偏向冷号回补）")
                        oc_hot_alpha = gr.Slider(-1.0, 1.0, value=DEFAULT_SELECTION_PARAMS["hot_alpha"], step=0.1,
                                                   label="冷热号系数（>0热号/<0冷号）")
                        oc_btn = gr.Button("🎯 一键选号", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        oc_summary = gr.Textbox(label="执行摘要", lines=15)
                        oc_numbers = gr.Dataframe(label="📋 生成号码（随机加权）")

                with gr.Row():
                    oc_recent = gr.Dataframe(label="最近开奖记录")

                with gr.Row():
                    oc_freq_plot = gr.Plot(label="号码频率分布")
                    oc_loss_plot = gr.Plot(label="训练损失曲线")

                with gr.Row():
                    oc_prob_plot = gr.Plot(label="号码概率热力图")

                with gr.Row():
                    oc_copy = gr.Textbox(label="📋 一键复制（含彩种/玩法）", lines=6)

                oc_btn.click(
                    one_click_predict,
                    inputs=[game_selector, play_selector, oc_seq_len, oc_epochs, oc_lr, oc_hidden,
                            oc_batch, oc_n_groups, oc_temp, oc_bidirectional,
                            oc_use_attention, oc_n_ensemble, oc_anneal,
                            oc_sampling_mode, oc_freq_alpha, oc_top_p,
                            oc_missing_alpha, oc_hot_alpha],
                    outputs=[oc_summary, oc_recent, oc_freq_plot, oc_loss_plot,
                             oc_numbers, oc_prob_plot, oc_copy],
                    concurrency_id="training_session", concurrency_limit=1,
                )

            # ---- Tab 2: 分步操作 ----
            with gr.Tab("🔧 分步操作", render_children=True) as steps_tab:
                with gr.Row():
                    # 数据获取
                    with gr.Column():
                        gr.Markdown("### 📥 Step 1: 获取数据")
                        fetch_btn = gr.Button("获取/更新数据", variant="primary")
                        local_btn = gr.Button("读取本地数据（不联网）")
                        restore_btn = gr.Button("恢复已保存模型（不重训）")
                        gr.Markdown("离线直接读取本地数据；v0.8起训练成功后保存档案。数据或权重已变化时，恢复会拒绝并提示重新训练。")
                        fetch_summary = gr.Textbox(label="数据摘要", lines=5)
                        fetch_recent = gr.Dataframe(label="最近 20 期")
                        fetch_freq = gr.Plot(label="频率分布")

                        fetch_btn.click(
                            fetch_data,
                            inputs=[game_selector],
                            outputs=[fetch_summary, fetch_recent, fetch_freq],
                            concurrency_id="training_session", concurrency_limit=1,
                        )
                        local_btn.click(load_local_data, inputs=[game_selector],
                                        outputs=[fetch_summary, fetch_recent, fetch_freq],
                                        concurrency_id="training_session", concurrency_limit=1)

                with gr.Row():
                    # 模型训练
                    with gr.Column():
                        gr.Markdown("### 🏋️ Step 2: 训练模型")
                        with gr.Row():
                            s_seq_len = gr.Slider(10, 60, value=DEFAULT_TRAINING_PARAMS["seq_len"], step=5, label="窗口大小")
                            s_epochs = gr.Slider(20, 300, value=DEFAULT_TRAINING_PARAMS["epochs"], step=10, label="训练轮数")
                        with gr.Row():
                            s_lr = gr.Number(value=DEFAULT_TRAINING_PARAMS["lr"], label="学习率")
                            s_hidden = gr.Slider(64, 256, value=DEFAULT_TRAINING_PARAMS["hidden"], step=32, label="隐藏层大小")
                            s_batch = gr.Slider(16, 128, value=DEFAULT_TRAINING_PARAMS["batch_size"], step=16, label="Batch Size")
                        with gr.Row():
                            s_bidirectional = gr.Checkbox(value=DEFAULT_TRAINING_PARAMS["bidirectional"], label="双向LSTM")
                            s_use_attention = gr.Checkbox(value=DEFAULT_TRAINING_PARAMS["use_attention"], label="启用Attention（实验）")
                            s_n_ensemble = gr.Slider(1, 5, value=DEFAULT_TRAINING_PARAMS["n_ensemble"], step=1, label="集成模型数")
                        train_btn = gr.Button("开始训练", variant="primary")
                        train_summary = gr.Textbox(label="训练摘要", lines=5)
                        train_loss = gr.Plot(label="Loss 曲线")

                        train_btn.click(
                            train,
                            inputs=[game_selector, s_seq_len, s_epochs, s_lr, s_hidden, s_batch,
                                    s_bidirectional, s_use_attention, s_n_ensemble],
                            outputs=[train_summary, train_loss],
                            concurrency_id="training_session", concurrency_limit=1,
                        )
                        restore_btn.click(restore_saved_session, inputs=[game_selector],
                                          outputs=[fetch_summary, fetch_recent, fetch_freq, train_summary, train_loss],
                                          concurrency_id="training_session", concurrency_limit=1)

                with gr.Row():
                    # 选号
                    with gr.Column():
                        gr.Markdown("### 🎯 Step 3: 生成号码\n*（随机加权选号，非预测。调参只影响选号分布，不改变中奖概率）*")
                        with gr.Row():
                            p_n_groups = gr.Slider(3, 20, value=DEFAULT_SELECTION_PARAMS["n_groups"], step=1, label="生成组数")
                            p_temp = gr.Slider(0.5, 3.0, value=DEFAULT_SELECTION_PARAMS["temperature"], step=0.1, label="采样温度")
                        with gr.Row():
                            p_sampling_mode = gr.Radio(
                                choices=["merged", "per_position"],
                                value=DEFAULT_SELECTION_PARAMS["sampling_mode"], label="采样模式",
                                info="merged: 合并多头消除位置偏差")
                            p_anneal = gr.Checkbox(value=DEFAULT_SELECTION_PARAMS["anneal_enabled"], label="温度退火")
                        with gr.Row():
                            p_freq_alpha = gr.Slider(0.0, 1.0, value=DEFAULT_SELECTION_PARAMS["freq_alpha"], step=0.05,
                                                       label="频率先验混合系数α")
                            p_top_p = gr.Slider(0.5, 1.0, value=DEFAULT_SELECTION_PARAMS["top_p"], step=0.05,
                                                  label="Top-p核采样阈值")
                        with gr.Row():
                            p_missing_alpha = gr.Slider(0.0, 2.0, value=DEFAULT_SELECTION_PARAMS["missing_alpha"], step=0.1,
                                                          label="遗漏值加权系数")
                            p_hot_alpha = gr.Slider(-1.0, 1.0, value=DEFAULT_SELECTION_PARAMS["hot_alpha"], step=0.1,
                                                      label="冷热号系数")
                        pred_btn = gr.Button("生成号码", variant="primary")
                        pred_text = gr.Textbox(label="生成号码（随机加权）", lines=15)
                        pred_numbers = gr.Dataframe(label="号码表格")
                        pred_prob = gr.Plot(label="号码倾向热力图")
                        pred_copy = gr.Textbox(label="📋 一键复制（含彩种/玩法）", lines=6)

                        pred_btn.click(
                            predict,
                            inputs=[game_selector, play_selector, p_n_groups, p_temp, p_anneal,
                                    p_sampling_mode, p_freq_alpha, p_top_p,
                                    p_missing_alpha, p_hot_alpha],
                            outputs=[pred_text, pred_copy, pred_numbers, pred_prob],
                            concurrency_id="training_session", concurrency_limit=1,
                        )

            # ---- Tab: 历史回测 ----
            with gr.Tab("📈 历史回测", render_children=True) as backtest_tab:
                gr.Markdown(
                    "**样本外回测**：模型训练时冻结末端 50 期，本页只在该未见区间逐期预测，"
                    "并与精确随机基准对比。\n\n"
                    "> 需先训练，或恢复与当前数据一致的v0.8训练档案；无档案旧模型不能用于严格回测。\n"
                    "⚠️ 回测耗时取决于 N 与组数；彩票为随机事件，回测不代表未来表现。\n\n"
                    "⚠️ 若反复查看结果并调参，该留出集就成为调参集，结果只能视为探索性；"
                    "最终验证需等待未来新开奖。"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        bt_n_test = gr.Slider(5, 50, value=DEFAULT_BACKTEST_PARAMS["n_test"], step=5,
                                                label="回测期数（末端测试集大小）")
                        bt_n_groups = gr.Slider(3, 15, value=DEFAULT_BACKTEST_PARAMS["n_groups"], step=1,
                                                  label="每期采样组数")
                        bt_temperature = gr.Slider(0.5, 3.0, value=DEFAULT_BACKTEST_PARAMS["temperature"], step=0.1,
                                                   label="采样温度")
                        bt_sampling_mode = gr.Radio(
                            choices=["merged", "per_position"], value=DEFAULT_BACKTEST_PARAMS["sampling_mode"],
                            label="采样模式")
                        bt_freq_alpha = gr.Slider(0.0, 1.0, value=DEFAULT_BACKTEST_PARAMS["freq_alpha"], step=0.05,
                                                    label="频率先验混合系数α")
                        bt_top_p = gr.Slider(0.5, 1.0, value=DEFAULT_BACKTEST_PARAMS["top_p"], step=0.05,
                                               label="Top-p核采样阈值")
                        bt_missing_alpha = gr.Slider(0.0, 2.0, value=DEFAULT_BACKTEST_PARAMS["missing_alpha"], step=0.1,
                                                       label="遗漏值加权系数")
                        bt_hot_alpha = gr.Slider(-1.0, 1.0, value=DEFAULT_BACKTEST_PARAMS["hot_alpha"], step=0.1,
                                                   label="冷热号系数")
                        bt_btn = gr.Button("📈 开始回测", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        bt_summary = gr.Textbox(label="回测报告", lines=28)
                        bt_table = gr.Dataframe(label="逐期命中明细")

                bt_btn.click(
                    run_backtest,
                    inputs=[game_selector, play_selector, bt_n_test, bt_n_groups,
                            bt_temperature, bt_sampling_mode,
                            bt_freq_alpha, bt_top_p, bt_missing_alpha, bt_hot_alpha],
                    outputs=[bt_summary, bt_table],
                    concurrency_id="training_session", concurrency_limit=1,
                )

            # ---- Tab: 缩水/覆盖投注 ----
            with gr.Tab("🎰 缩水/覆盖投注", render_children=True) as coverage_tab:
                gr.Markdown(
                    "**覆盖投注**：选定核心号码池后，用集合覆盖设计生成多注，\n"
                    "减少票组重叠；结果只报告在核心池命中等明确前提下的条件覆盖率。\n\n"
                    "> ⚠️ 它不提高单注中奖概率或期望收益。\n"
                    "需先在「分步操作」或「一键预测」中训练模型以获取概率。"
                )
                coverage_notice = gr.Markdown(visible=False)
                with gr.Row():
                    with gr.Column(scale=1):
                        cb_pool_size = gr.Slider(7, 18, value=12, step=1,
                                                   label="核心号码池大小")
                        cb_n_tickets = gr.Slider(3, 20, value=10, step=1,
                                                   label="生成注数")
                        cb_guaranteed = gr.Slider(2, 5, value=4, step=1,
                                                    label="目标命中数（用于条件覆盖率计算）")
                        cb_freq_alpha = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                                    label="频率先验混合系数α")
                        cb_btn = gr.Button("🎰 生成覆盖方案", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        cb_summary = gr.Textbox(label="覆盖方案", lines=22)
                        cb_numbers = gr.Dataframe(label="📋 覆盖投注号码")

                cb_btn.click(
                    coverage_bet,
                    inputs=[game_selector, cb_pool_size, cb_n_tickets,
                            cb_guaranteed, cb_freq_alpha],
                    outputs=[cb_summary, cb_numbers],
                    concurrency_id="training_session", concurrency_limit=1,
                )

            # ---- Tab 3: 预测回顾 ----
            with gr.Tab("📊 预测回顾", render_children=True):
                gr.Markdown("按预测保存的数据截止期匹配下一期开奖；旧记录缺少该字段时不会猜测匹配。")
                with gr.Row():
                    review_btn = gr.Button("🔍 查看命中情况", variant="primary", size="lg")
                with gr.Row():
                    review_text = gr.Textbox(label="命中详情", lines=22)
                with gr.Row():
                    review_table = gr.Dataframe(label="命中排行表")

                review_btn.click(
                    review_prediction,
                    inputs=[game_selector, play_selector],
                    outputs=[review_text, review_table],
                )

                gr.Markdown("---\n### 📈 长期命中统计（汇总所有历史预测，对比理论随机期望）")
                with gr.Row():
                    hist_btn = gr.Button("📈 生成长期统计", variant="primary")
                with gr.Row():
                    hist_text = gr.Textbox(label="长期命中报告", lines=26)
                with gr.Row():
                    hist_table = gr.Dataframe(label="逐记录命中明细")
                with gr.Row():
                    hist_plot = gr.Plot(label="相对倾向分与命中对照")

                hist_btn.click(
                    review_history_stats,
                    inputs=[game_selector, play_selector],
                    outputs=[hist_text, hist_table, hist_plot],
                )

            # ---- Tab 4: 历史预测 ----
            with gr.Tab("📂 历史预测", render_children=True):
                gr.Markdown("### 历史预测记录")
                history_btn = gr.Button("刷新记录")
                history_text = gr.Textbox(label="历史记录", lines=20)

                history_btn.click(load_prediction_history, inputs=[game_selector, play_selector], outputs=[history_text])

        def clear_incompatible_and_results(game):
            code = _code(game)
            is_digit = code in DIGIT_GAME_CONFIGS
            supports_coverage = code in {"ssq", "dlt"}
            notice = f"已切换至 {GAME_NAMES[code]}，上一彩种的结果已清空。"
            coverage_notice_text = (
                f"{GAME_NAMES[code]}为按位数字型彩票，不适用双色球/大乐透的集合覆盖；"
                "相关参数已隐藏。"
            )
            return (
                gr.update(value=0.0, visible=not is_digit), gr.update(value=0.0, visible=not is_digit),
                gr.update(value=0.0, visible=not is_digit), gr.update(value=0.0, visible=not is_digit),
                gr.update(value=0.0, visible=not is_digit), gr.update(value=0.0, visible=not is_digit),
                gr.update(value="per_position" if is_digit else DEFAULT_SELECTION_PARAMS["sampling_mode"], visible=not is_digit),
                gr.update(value=False, visible=not is_digit),
                gr.update(value="per_position" if is_digit else DEFAULT_SELECTION_PARAMS["sampling_mode"], visible=not is_digit),
                gr.update(value=False, visible=not is_digit),
                gr.update(value="per_position" if is_digit else DEFAULT_BACKTEST_PARAMS["sampling_mode"], visible=not is_digit),
                gr.update(visible=supports_coverage), gr.update(visible=supports_coverage),
                gr.update(visible=supports_coverage), gr.update(visible=supports_coverage),
                gr.update(visible=supports_coverage),
                gr.update(value=notice if supports_coverage else "", visible=supports_coverage),
                gr.update(value=None, visible=supports_coverage),
                gr.update(value=coverage_notice_text if not supports_coverage else "", visible=not supports_coverage),
                notice, None, None, None, None, None, "",
                notice, None, None,
                notice, None,
                notice, "", None, None,
                notice, None,
                notice, None,
                notice, None, None,
                notice,
                notice, "", None, None, notice, None,
            )

        incompatible_outputs = [
            oc_missing_alpha, oc_hot_alpha, p_missing_alpha, p_hot_alpha, bt_missing_alpha, bt_hot_alpha,
            oc_sampling_mode, oc_anneal, p_sampling_mode, p_anneal, bt_sampling_mode,
            cb_pool_size, cb_n_tickets, cb_guaranteed, cb_freq_alpha, cb_btn, cb_summary, cb_numbers,
            coverage_notice,
            oc_summary, oc_numbers, oc_recent, oc_freq_plot, oc_loss_plot, oc_prob_plot, oc_copy,
            fetch_summary, fetch_recent, fetch_freq,
            train_summary, train_loss,
            pred_text, pred_copy, pred_numbers, pred_prob,
            bt_summary, bt_table,
            review_text, review_table,
            hist_text, hist_table, hist_plot,
            history_text,
            jp_text, jp_copy, jp_numbers, jp_prob, jp_report, jp_rows,
        ]

        stale_result_outputs = [
            oc_summary, oc_numbers, oc_recent, oc_freq_plot, oc_loss_plot, oc_prob_plot, oc_copy,
            fetch_summary, fetch_recent, fetch_freq,
            train_summary, train_loss,
            pred_text, pred_copy, pred_numbers, pred_prob,
            bt_summary, bt_table,
            cb_summary, cb_numbers,
            review_text, review_table,
            hist_text, hist_table, hist_plot,
            history_text,
            jp_text, jp_copy, jp_numbers, jp_prob, jp_report, jp_rows,
        ]

        def clear_results(game, play):
            code = _code(game)
            normalized_play = normalize_play(code, play)
            notice = f"已切换至 {GAME_NAMES[code]}（{normalized_play}），此前结果已清空。"
            return (
                notice, None, None, None, None, None, "",
                notice, None, None,
                notice, None,
                notice, "", None, None,
                notice, None,
                notice, None,
                notice, None,
                notice, None, None,
                notice,
                notice, "", None, None, notice, None,
            )

        def restore_defaults(game):
            code = _code(game)
            default_play = GAME_PLAY_DEFAULTS[code]
            train_defaults = DEFAULT_TRAINING_PARAMS
            selection_defaults = DEFAULT_SELECTION_PARAMS
            backtest_defaults = DEFAULT_BACKTEST_PARAMS
            return (
                gr.update(choices=GAME_PLAY_OPTIONS[code], value=default_play), describe_play(code, default_play),
                train_defaults["seq_len"], train_defaults["epochs"], train_defaults["lr"],
                train_defaults["hidden"], train_defaults["batch_size"], selection_defaults["n_groups"],
                selection_defaults["temperature"], train_defaults["bidirectional"],
                train_defaults["use_attention"], train_defaults["n_ensemble"], selection_defaults["anneal_enabled"],
                selection_defaults["sampling_mode"], selection_defaults["freq_alpha"], selection_defaults["top_p"],
                selection_defaults["missing_alpha"], selection_defaults["hot_alpha"],
                train_defaults["seq_len"], train_defaults["epochs"], train_defaults["lr"],
                train_defaults["hidden"], train_defaults["batch_size"], train_defaults["bidirectional"],
                train_defaults["use_attention"], train_defaults["n_ensemble"],
                selection_defaults["n_groups"], selection_defaults["temperature"],
                selection_defaults["sampling_mode"], selection_defaults["anneal_enabled"],
                selection_defaults["freq_alpha"], selection_defaults["top_p"],
                selection_defaults["missing_alpha"], selection_defaults["hot_alpha"],
                backtest_defaults["n_test"], backtest_defaults["n_groups"], backtest_defaults["temperature"],
                backtest_defaults["sampling_mode"], backtest_defaults["freq_alpha"], backtest_defaults["top_p"],
                backtest_defaults["missing_alpha"], backtest_defaults["hot_alpha"],
                12, 10, 4, 0.0,
            )

        default_outputs = [
            play_selector, play_info,
            oc_seq_len, oc_epochs, oc_lr, oc_hidden, oc_batch, oc_n_groups, oc_temp, oc_bidirectional,
            oc_use_attention, oc_n_ensemble, oc_anneal, oc_sampling_mode, oc_freq_alpha, oc_top_p,
            oc_missing_alpha, oc_hot_alpha,
            s_seq_len, s_epochs, s_lr, s_hidden, s_batch, s_bidirectional, s_use_attention, s_n_ensemble,
            p_n_groups, p_temp, p_sampling_mode, p_anneal, p_freq_alpha, p_top_p, p_missing_alpha, p_hot_alpha,
            bt_n_test, bt_n_groups, bt_temperature, bt_sampling_mode, bt_freq_alpha, bt_top_p,
            bt_missing_alpha, bt_hot_alpha,
            cb_pool_size, cb_n_tickets, cb_guaranteed, cb_freq_alpha,
        ]

        def refresh_applicability(game):
            # 首次打开页签时客户端可能重新挂载子控件。只同步可见性，
            # 不重置用户参数，也不清空已有结果。
            is_digit = _code(game) in DIGIT_GAME_CONFIGS
            return tuple(gr.update(visible=not is_digit) for _ in range(18)) + (
                gr.update(visible=is_digit),
            )

        for page in (one_click_tab, steps_tab, backtest_tab, coverage_tab):
            page.select(
                refresh_applicability, inputs=[game_selector],
                outputs=incompatible_outputs[:19], queue=False,
            )

        game_selector.change(update_play_options, inputs=[game_selector], outputs=[play_selector, play_info]).then(
            clear_incompatible_and_results, inputs=[game_selector], outputs=incompatible_outputs,
        )
        play_selector.change(describe_play, inputs=[game_selector, play_selector], outputs=[play_info]).then(
            clear_results, inputs=[game_selector, play_selector], outputs=stale_result_outputs,
        )
        restore_defaults_btn.click(restore_defaults, inputs=[game_selector], outputs=default_outputs).then(
            clear_incompatible_and_results, inputs=[game_selector], outputs=incompatible_outputs,
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    import socket
    port = 7861
    for p in range(7861, 7881):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p
                break

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        inbrowser=True,
    )
