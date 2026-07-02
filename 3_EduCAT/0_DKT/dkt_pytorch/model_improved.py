"""
DKT 模型改进版 - 接近论文实现

主要改进:
1. 更接近论文的输入编码方式
2. Dropout 只在输出层应用
3. 更好的数据预处理
4. 批次大小增加到 100 (如论文)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DKTModelImproved(nn.Module):
    """
    改进版 DKT 模型

    改进点:
    1. 直接使用 concat(one-hot, correct) 作为输入，而非嵌入
    2. Dropout 只在输出层应用 (如论文所述)
    3. 使用更大的批次
    """

    def __init__(
        self,
        n_questions: int,
        n_hidden: int = 200,
        n_layers: int = 1,
        dropout: float = 0.5,
        use_lstm: bool = True
    ):
        super().__init__()

        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.use_lstm = use_lstm

        # 输入维度: one-hot(题目) + 正确性
        # 题目 one-hot: n_questions 维
        # 正确性: 2 维 (one-hot)
        self.n_input = n_questions + 2

        # LSTM 层
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=self.n_input,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=self.n_input,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )

        # Dropout 只在输出层 (论文3.3节)
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.output_layer = nn.Linear(n_hidden, n_questions)

        self._init_hidden()

    def _init_hidden(self, batch_size: int = 1, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        self.hidden = (h0, c0) if self.use_lstm else h0

    def _prepare_input(self, question_ids: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """
        准备输入: concat(one-hot, correct)

        论文3.2节: "xt to be a one-hot encoding of the student interaction tuple"
        """
        batch_size, seq_len = question_ids.shape

        # one-hot 编码题目ID
        question_one_hot = F.one_hot(question_ids, num_classes=self.n_questions).float()

        # one-hot 编码正确性 (0 或 1)
        correct_one_hot = F.one_hot(correct.long(), num_classes=2).float()

        # 拼接
        x = torch.cat([question_one_hot, correct_one_hot], dim=-1)

        return x

    def forward(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = question_ids.shape

        # 准备输入
        x = self._prepare_input(question_ids, correct)

        # 初始化隐藏状态
        if hidden is None:
            self._init_hidden(batch_size, x.device)
            hidden = self.hidden

        # LSTM 前向传播
        rnn_output, hidden = self.rnn(x, hidden)
        self.hidden = hidden

        # Dropout 只在输出层应用 (关键改进!)
        rnn_output = self.dropout(rnn_output)

        # 输出: 每个题目的正确概率
        all_outputs = torch.sigmoid(self.output_layer(rnn_output))

        # 提取对应题目的预测
        predictions = self._get_predictions(all_outputs, question_ids)

        return {
            "predictions": predictions,
            "all_outputs": all_outputs,
            "hidden": hidden
        }

    def _get_predictions(self, all_outputs: torch.Tensor, question_ids: torch.Tensor) -> torch.Tensor:
        """选取每个位置的下一题预测"""
        batch_size, seq_len = question_ids.shape

        # all_outputs[:, t, :] 预测的是 t+1 时刻
        # 选取对应的题目概率
        relevant_outputs = all_outputs[:, :-1, :]
        next_question_ids = question_ids[:, 1:]

        predictions = torch.gather(
            relevant_outputs,
            dim=2,
            index=next_question_ids.unsqueeze(-1)
        ).squeeze(-1)

        return predictions

    def predict(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测接口"""
        result = self.forward(question_ids, correct)
        predictions = result["predictions"]
        targets = correct[:, 1:]

        if mask is not None:
            mask = mask[:, 1:]
        else:
            mask = torch.ones_like(targets)

        return predictions, targets, mask


def test_improved_model():
    """测试改进版模型"""
    print("=" * 50)
    print("测试改进版 DKT 模型")
    print("=" * 50)

    model = DKTModelImproved(
        n_questions=110,
        n_hidden=200,
        dropout=0.5
    )

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试
    batch_size, seq_len = 32, 50
    question_ids = torch.randint(1, 110, (batch_size, seq_len))
    correct = torch.randint(0, 2, (batch_size, seq_len)).float()

    result = model(question_ids, correct)

    print(f"\n输入形状: question_ids={question_ids.shape}, correct={correct.shape}")
    print(f"输出形状: predictions={result['predictions'].shape}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type != "cpu":
        model = model.to(device)
        question_ids = question_ids.to(device)
        correct = correct.to(device)
        result = model(question_ids, correct)
        print(f"\nGPU 测试通过!")


if __name__ == "__main__":
    test_improved_model()
