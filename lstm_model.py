"""
彩票 LSTM 预测模型
每个号码位输出一个概率分布，通过多头 FC 实现。
"""
import torch
import torch.nn as nn

from game_config import DIGIT_GAME_CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LotteryLSTM(nn.Module):
    """
    通用彩票 LSTM 模型。
    输入: (batch, seq_len, feature_dim) 的特征序列
    输出: 多个号码位的概率分布

    参数:
        input_size: 特征维度
        hidden_size: LSTM 隐藏层大小
        num_layers: LSTM 层数
        heads: list of (name, num_classes)
            例如双色球: [("red1",33), ("red2",33), ..., ("blue",16)]
            大乐透: [("front1",35), ..., ("back1",12), ("back2",12)]
        dropout: Dropout 比率
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 heads: list[tuple[str, int]] = None, dropout: float = 0.3,
                 bidirectional: bool = False, use_attention: bool = True,
                 attn_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.attn_heads = attn_heads
        self.num_directions = 2 if bidirectional else 1
        self.lstm_out_dim = hidden_size * self.num_directions
        self.head_names = [h[0] for h in heads]

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        # Multi-Head Self-Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.lstm_out_dim, num_heads=attn_heads,
                dropout=dropout, batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(self.lstm_out_dim)

        # 每个号码位一个分类头
        self.fc_heads = nn.ModuleDict()
        for name, num_classes in heads:
            self.fc_heads[name] = nn.Sequential(
                nn.Linear(self.lstm_out_dim, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (batch, seq_len, feature_dim)
        返回: {head_name: (batch, num_classes) logits}
        """
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out: (batch, seq_len, lstm_out_dim)

        if self.use_attention:
            attn_out, _ = self.attention(out, out, out)  # self-attention
            out = self.attn_norm(out + attn_out)  # residual + norm

        last_hidden = self.dropout(out[:, -1, :])  # (batch, lstm_out_dim)

        results = {}
        for name in self.head_names:
            results[name] = self.fc_heads[name](last_hidden)
        return results


def build_ssq_model(input_size: int, hidden_size: int = 128, num_layers: int = 2,
                    dropout: float = 0.3, bidirectional: bool = False,
                    use_attention: bool = True, attn_heads: int = 4) -> LotteryLSTM:
    """创建双色球模型：6 个红球头(33类) + 1 个蓝球头(16类)"""
    heads = [(f"red{i+1}", 33) for i in range(6)] + [("blue", 16)]
    return LotteryLSTM(input_size, hidden_size, num_layers, heads, dropout,
                       bidirectional=bidirectional, use_attention=use_attention,
                       attn_heads=attn_heads)


def build_dlt_model(input_size: int, hidden_size: int = 128, num_layers: int = 2,
                    dropout: float = 0.3, bidirectional: bool = False,
                    use_attention: bool = True, attn_heads: int = 4) -> LotteryLSTM:
    """创建大乐透模型：5 个前区头(35类) + 2 个后区头(12类)"""
    heads = [(f"front{i+1}", 35) for i in range(5)] + [(f"back{i+1}", 12) for i in range(2)]
    return LotteryLSTM(input_size, hidden_size, num_layers, heads, dropout,
                       bidirectional=bidirectional, use_attention=use_attention,
                       attn_heads=attn_heads)


def build_digit_model(game: str, input_size: int, hidden_size: int = 128,
                      num_layers: int = 2, dropout: float = 0.3,
                      bidirectional: bool = False, use_attention: bool = True,
                      attn_heads: int = 4) -> LotteryLSTM:
    """创建按位数字模型；每个位独立分类，允许重复数字。"""
    if game not in DIGIT_GAME_CONFIGS:
        raise ValueError(f"不支持的数字型彩种: {game}")
    cfg = DIGIT_GAME_CONFIGS[game]
    sizes = cfg["classes"]
    heads = [(f"digit{i + 1}", int(size)) for i, size in enumerate(sizes)]
    return LotteryLSTM(input_size, hidden_size, num_layers, heads, dropout,
                       bidirectional=bidirectional, use_attention=use_attention,
                       attn_heads=attn_heads)


def build_game_model(game: str, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.3,
                     bidirectional: bool = False, use_attention: bool = True,
                     attn_heads: int = 4) -> LotteryLSTM:
    """显式按彩种构建模型，避免未知彩种误落到大乐透。"""
    if game == "ssq":
        return build_ssq_model(input_size, hidden_size, num_layers, dropout,
                               bidirectional, use_attention, attn_heads)
    if game == "dlt":
        return build_dlt_model(input_size, hidden_size, num_layers, dropout,
                               bidirectional, use_attention, attn_heads)
    if game in DIGIT_GAME_CONFIGS:
        return build_digit_model(game, input_size, hidden_size, num_layers, dropout,
                                 bidirectional, use_attention, attn_heads)
    raise ValueError(f"不支持的彩种: {game}")
