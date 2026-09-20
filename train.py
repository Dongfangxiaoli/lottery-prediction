"""
彩票 LSTM 模型训练流程
支持双色球和大乐透，时间序列划分（不 shuffle），早停。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from game_config import ALL_GAME_CODES, DIGIT_GAME_CONFIGS
from lstm_model import build_game_model, device

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _prepare_sequences(features: np.ndarray, labels_dict: dict, seq_len: int, game: str):
    """
    将特征和标签切分为滑动窗口序列。
    features: (n_draws, feat_dim)
    labels_dict: {"red": (n,6), "blue": (n,)} 或 {"front": (n,5), "back": (n,2)}
    返回: X (n_samples, seq_len, feat_dim), Y dict of arrays
    """
    n = len(features)
    X = []
    Y = {k: [] for k in labels_dict}

    for i in range(seq_len, n):
        X.append(features[i - seq_len:i])
        for k, v in labels_dict.items():
            Y[k].append(v[i])  # 预测第 i 期的号码

    X = np.array(X, dtype=np.float32)
    for k in Y:
        Y[k] = np.array(Y[k])
    return X, Y


def _labels_to_targets(Y: dict, game: str) -> dict[str, np.ndarray]:
    """
    将原始号码标签转换为 head_name -> class_index 的映射。
    号码值 1-based 转为 0-based index。
    """
    targets = {}
    if game == "ssq":
        reds = Y["red"]  # (n, 6)
        for i in range(6):
            targets[f"red{i+1}"] = reds[:, i] - 1  # 1-33 -> 0-32
        targets["blue"] = Y["blue"] - 1  # 1-16 -> 0-15
    elif game == "dlt":
        fronts = Y["front"]  # (n, 5)
        for i in range(5):
            targets[f"front{i+1}"] = fronts[:, i] - 1  # 1-35 -> 0-34
        backs = Y["back"]  # (n, 2)
        for i in range(2):
            targets[f"back{i+1}"] = backs[:, i] - 1  # 1-12 -> 0-11
    elif game in DIGIT_GAME_CONFIGS:
        digits = Y["digits"]
        for i in range(digits.shape[1]):
            targets[f"digit{i+1}"] = digits[:, i]
    return targets


def train_model(
    features: np.ndarray,
    labels_dict: dict,
    game: str,          # "ssq" or "dlt"
    seq_len: int = 30,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = False,
    use_attention: bool = True,
    attn_heads: int = 4,
    patience: int = 15,
    n_ensemble: int = 1,
    holdout_size: int = 50,
    progress_callback=None,
) -> dict:
    """
    训练 LSTM 模型。
    返回: {"model_path": str, "train_losses": list, "val_losses": list, "best_epoch": int}
    """
    if game not in ALL_GAME_CODES:
        raise ValueError(f"不支持的彩种: {game}")
    if seq_len < 1:
        raise ValueError("seq_len 必须大于 0。")

    # 1. 准备序列；末端 holdout 只用于真正的样本外回测，训练和早停都不可见。
    X, Y = _prepare_sequences(features, labels_dict, seq_len, game)
    targets = _labels_to_targets(Y, game)
    head_names = list(targets.keys())
    holdout_size = max(0, int(holdout_size))
    fit_count = len(X) - holdout_size
    if fit_count < 2:
        raise ValueError(
            f"数据不足：序列样本 {len(X)}，样本外保留 {holdout_size}，至少需要 2 个训练/验证样本。"
        )
    X_fit = X[:fit_count]
    targets_fit = {k: v[:fit_count] for k, v in targets.items()}

    # 2. 在外层训练段内再做时间序列划分 (80% train, 20% val)
    split = int(0.8 * len(X_fit))
    if split < 1 or split >= len(X_fit):
        raise ValueError("训练/验证划分后存在空数据集。")
    X_train, X_val = X_fit[:split], X_fit[split:]
    t_train = {k: v[:split] for k, v in targets_fit.items()}
    t_val = {k: v[split:] for k, v in targets_fit.items()}

    # 3. 构建 DataLoader
    train_tensors = [torch.tensor(X_train, dtype=torch.float32)]
    val_tensors = [torch.tensor(X_val, dtype=torch.float32)]
    for name in head_names:
        train_tensors.append(torch.tensor(t_train[name], dtype=torch.long))
        val_tensors.append(torch.tensor(t_val[name], dtype=torch.long))

    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. 构建模型（支持多 seed 集成训练）
    input_size = X.shape[2]

    ensemble_seeds = []
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0
    model_path = os.path.join(MODELS_DIR, f"{game}_lstm.pth")

    num_models = n_ensemble if n_ensemble >= 1 else 1
    for i in range(num_models):
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if num_models > 1:
            ensemble_seeds.append(seed)

        model = build_game_model(game, input_size, hidden_size, num_layers, dropout,
                                 bidirectional, use_attention, attn_heads).to(device)

        # label smoothing 抑制过度自信的尖峰分布，让低概率号码保留合理权重，采样更稳
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        # AdamW + weight decay 正则化，防止过拟合历史噪声
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # 5. 训练循环
        model_best_val = float("inf")
        model_best_epoch = 0
        for epoch in range(num_epochs):
            # Train
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                x_batch = batch[0].to(device)
                label_batches = {head_names[i]: batch[i + 1].to(device) for i in range(len(head_names))}

                outputs = model(x_batch)
                loss = sum(criterion(outputs[name], label_batches[name]) for name in head_names)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(device)
                    label_batches = {head_names[i]: batch[i + 1].to(device) for i in range(len(head_names))}
                    outputs = model(x_batch)
                    loss = sum(criterion(outputs[name], label_batches[name]) for name in head_names)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < model_best_val:
                model_best_val = avg_val_loss
                model_best_epoch = epoch
                if num_models > 1:
                    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{game}_lstm_seed{seed}.pth"))
                    if i == 0:
                        torch.save(model.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                if i == 0:
                    best_val_loss = model_best_val
                    best_epoch = model_best_epoch

            if epoch - model_best_epoch >= patience:
                break

            if progress_callback:
                label = f"Model {i+1}/{num_models}" if num_models > 1 else ""
                msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                if label:
                    msg = f"{label} | {msg}"
                overall_progress = ((i * num_epochs) + (epoch + 1)) / (num_models * num_epochs)
                progress_callback(overall_progress, msg)

    # 加载最佳模型
    load_path = model_path if n_ensemble <= 1 else os.path.join(MODELS_DIR, f"{game}_lstm_seed42.pth")
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))

    return {
        "model_path": model_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "seq_len": seq_len,
        "bidirectional": bidirectional,
        "use_attention": use_attention,
        "attn_heads": attn_heads,
        "n_ensemble": n_ensemble,
        "ensemble_seeds": ensemble_seeds,
        "holdout_size": holdout_size,
    }
