"""
DKT 模型定义

Deep Knowledge Tracing (深度知识追踪) 模型实现

核心思想:
    DKT 使用循环神经网络 (LSTM/RNN) 来建模学生的知识状态。
    在每个时间步，模型接收学生当前答题的信息（题目ID + 是否正确），
    然后预测学生下一题的答题情况。

模型架构:
    输入层: [题目ID, 正确性] -> 嵌入层 -> LSTM -> 全连接层 -> Sigmoid输出
    输出层: 预测下一题的正确概率

参考论文:
    Piech et al., "Deep Knowledge Tracing", NeurIPS 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CompressedSensing(nn.Module):
    """
    压缩感知层

    将高维的输入向量压缩到低维，可以加速训练同时保持大部分信息。
    原理是使用一个随机矩阵进行投影。

    Args:
        input_dim: 输入维度 (通常是 n_questions * 2)
        output_dim: 输出维度 (压缩后的维度)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 随机投影矩阵
        # 使用正交随机矩阵可以更好地保持信息
        self.register_buffer(
            "basis",
            torch.randn(input_dim, output_dim) / torch.sqrt(torch.tensor(output_dim).float())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: x @ basis

        Args:
            x: 输入张量 (batch, seq_len, input_dim)

        Returns:
            压缩后的张量 (batch, seq_len, output_dim)
        """
        return torch.matmul(x, self.basis)


class DKTModel(nn.Module):
    """
    Deep Knowledge Tracing 模型

    模型接收学生的历史答题序列，预测每个时间步学生答对下一题的概率。

    输入格式:
        - question_ids: 题目ID序列 (batch, seq_len)
        - correct: 正确性序列 (batch, seq_len)

    输出格式:
        - predictions: 预测概率 (batch, seq_len - 1)
          只预测前 seq_len-1 个位置，因为最后一个位置没有"下一题"

    工作流程:
        1. 将题目ID和正确性分别嵌入
        2. 将两个嵌入向量拼接
        3. 可选：压缩感知降维
        4. 通过 LSTM 编码历史信息
        5. 全连接层预测每道题的正确概率
    """

    def __init__(
        self,
        n_questions: int,
        n_hidden: int = 200,
        n_layers: int = 1,
        dropout: float = 0.5,
        use_lstm: bool = True,
        use_compressed_sensing: bool = False,
        compressed_dim: int = 100
    ):
        """
        初始化 DKT 模型

        Args:
            n_questions: 数据集中题目的总数量
            n_hidden: 隐藏层神经元数量
            n_layers: LSTM 层数
            dropout: Dropout 概率
            use_lstm: 是否使用 LSTM (False 则使用普通 RNN)
            use_compressed_sensing: 是否使用压缩感知
            compressed_dim: 压缩感知的输出维度
        """
        super().__init__()

        # 保存参数
        self.n_questions = n_questions
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.use_lstm = use_lstm
        self.use_compressed_sensing = use_compressed_sensing

        # 输入维度 = 题目ID one-hot + 正确性
        # 题目ID: n_questions 维
        # 正确性: 2 维 (one-hot)
        self.n_input = n_questions + 2

        # 嵌入层
        # 将 one-hot 的题目ID转换为密集向量
        self.question_embed = nn.Linear(n_questions, n_hidden, bias=False)

        # 正确性嵌入
        # 将 0/1 正确性转换为向量
        self.correct_embed = nn.Embedding(2, 2)

        # 输入投影层: 将 (n_hidden + 2) 维映射到 n_hidden 维
        self.input_proj = nn.Linear(n_hidden + 2, n_hidden)

        # 压缩感知层 (可选)
        if use_compressed_sensing:
            self.compressed_sensing = CompressedSensing(n_hidden + 2, compressed_dim)
            lstm_input_dim = compressed_dim
        else:
            self.compressed_sensing = None
            lstm_input_dim = n_hidden

        # LSTM/RNN 层
        # bidirectional=False 因为我们只关心之前的序列
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.RNN(
                input_size=lstm_input_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 输出层
        # 将隐藏状态映射到每个题目的正确概率
        self.output_layer = nn.Linear(n_hidden, n_questions)

        # 初始化隐藏状态
        self._init_hidden()

    def _init_hidden(self, batch_size: int = 1, device: torch.device = None):
        """
        初始化隐藏状态

        Args:
            batch_size: 批次大小
            device: 设备 (cuda/mps/cpu)
        """
        if device is None:
            device = next(self.parameters()).device

        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)

        self.hidden = (h0, c0) if self.use_lstm else h0

    def _prepare_input(self, question_ids: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """
        准备模型输入

        将题目ID和正确性转换为嵌入向量并拼接。

        Args:
            question_ids: 题目ID (batch, seq_len)
            correct: 正确性 (batch, seq_len)

        Returns:
            拼接后的输入 (batch, seq_len, n_hidden)
        """
        batch_size, seq_len = question_ids.shape

        # 将题目ID转换为 one-hot 向量，然后嵌入
        # question_ids: (batch, seq_len) -> one_hot: (batch, seq_len, n_questions)
        question_one_hot = F.one_hot(question_ids, num_classes=self.n_questions).float()
        # 嵌入到 n_hidden 维
        question_embedded = self.question_embed(question_one_hot)  # (batch, seq_len, n_hidden)

        # 将正确性嵌入
        # correct: (batch, seq_len) -> (batch, seq_len, 2)
        correct_embedded = self.correct_embed(correct.long())  # 需要是 long 类型

        # 拼接题目嵌入和正确性嵌入
        # question_embedded: (batch, seq_len, n_hidden)
        # correct_embedded: (batch, seq_len, 2)
        # 拼接后得到 (batch, seq_len, n_hidden + 2)，但由于 linear 层会调整维度
        x = torch.cat([question_embedded, correct_embedded], dim=-1)  # (batch, seq_len, n_hidden + 2)

        # 再次投影到 n_hidden 维，与 LSTM 的输入维度匹配
        # 需要展平然后投影
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size * seq_len, -1)  # (batch * seq_len, n_hidden + 2)
        x = torch.relu(self.input_proj(x))  # (batch * seq_len, n_hidden)
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, n_hidden)

        return x

    def forward(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        hidden: Optional[Tuple] = None,
        return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            question_ids: 题目ID序列 (batch, seq_len)
            correct: 正确性序列 (batch, seq_len)
            hidden: 隐藏状态 (可选)
            return_hidden: 是否返回最后的隐藏状态

        Returns:
            包含以下键的字典:
            - predictions: 预测概率 (batch, seq_len - 1)
            - hidden: 最后的隐藏状态 (可选)
            - all_outputs: 所有时间步的输出 (batch, seq_len, n_questions)
        """
        batch_size, seq_len = question_ids.shape

        # 准备输入
        x = self._prepare_input(question_ids, correct)  # (batch, seq_len, n_hidden + 2)

        # 压缩感知 (可选)
        if self.use_compressed_sensing:
            x = self.compressed_sensing(x)

        # 初始化隐藏状态
        if hidden is None:
            self._init_hidden(batch_size, x.device)
            hidden = self.hidden

        # 通过 RNN
        # rnn_output: (batch, seq_len, n_hidden)
        # hidden: 最后的隐藏状态
        rnn_output, hidden = self.rnn(x, hidden)

        # 存储隐藏状态供下次调用
        self.hidden = hidden

        # Dropout
        rnn_output = self.dropout(rnn_output)

        # 输出层
        # all_outputs: (batch, seq_len, n_questions)
        all_outputs = torch.sigmoid(self.output_layer(rnn_output))

        # DKT 的输出是预测下一题，所以需要 shift
        # predictions[t] 预测的是第 t+1 题
        # 即: predictions[:, t] = all_outputs[:, t, question_ids[:, t+1]]
        predictions = self._get_predictions(all_outputs, question_ids)

        result = {
            "predictions": predictions,
            "all_outputs": all_outputs,
        }

        if return_hidden:
            result["hidden"] = hidden

        return result

    def _get_predictions(
        self,
        all_outputs: torch.Tensor,
        question_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        从所有输出中提取对应题目的预测概率

        all_outputs 包含了对所有题目的预测概率，我们需要根据实际的题目ID
        来选取对应的概率。

        Args:
            all_outputs: 所有题目的预测概率 (batch, seq_len, n_questions)
            question_ids: 题目ID (batch, seq_len)

        Returns:
            实际题目的预测概率 (batch, seq_len - 1)
        """
        batch_size, seq_len = question_ids.shape

        # 选取每个位置上"下一题"的预测
        # 预测序列比输入序列少一个 (最后一个位置没有下一题)
        # next_question_ids: (batch, seq_len - 1)
        next_question_ids = question_ids[:, 1:]

        # 使用 gather 选取对应题目的预测
        # all_outputs[:, :-1, :] 中存储的是对"下一题"的预测
        # 我们需要根据 next_question_ids 来选取
        relevant_outputs = all_outputs[:, :-1, :]  # (batch, seq_len - 1, n_questions)

        # gather: 根据 question_ids 选取对应的概率
        # question_ids[:, 1:] 形状是 (batch, seq_len - 1)
        predictions = torch.gather(
            relevant_outputs,
            dim=2,
            index=next_question_ids.unsqueeze(-1)  # (batch, seq_len - 1, 1)
        ).squeeze(-1)  # (batch, seq_len - 1)

        return predictions

    def predict(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测接口 (供评估使用)

        Args:
            question_ids: 题目ID (batch, seq_len)
            correct: 正确性 (batch, seq_len)
            mask: 有效位置掩码 (batch, seq_len)

        Returns:
            (predictions, targets, mask) 元组
            - predictions: 预测概率
            - targets: 真实标签
            - mask: 有效位置掩码
        """
        # 获取预测
        result = self.forward(question_ids, correct)
        predictions = result["predictions"]  # (batch, seq_len - 1)

        # 获取目标 (真实正确性)
        # 目标序列比输入序列少一个
        targets = correct[:, 1:]  # (batch, seq_len - 1)

        # 获取掩码
        if mask is not None:
            mask = mask[:, 1:]  # (batch, seq_len - 1)
        else:
            mask = torch.ones_like(targets)

        return predictions, targets, mask


def test_model():
    """测试模型"""
    print("=" * 50)
    print("测试 DKT 模型")
    print("=" * 50)

    # 创建模型
    model = DKTModel(
        n_questions=110,
        n_hidden=200,
        n_layers=1,
        dropout=0.5,
        use_lstm=True
    )

    # 打印模型结构
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试输入
    batch_size, seq_len = 4, 20
    question_ids = torch.randint(1, 110, (batch_size, seq_len))
    correct = torch.randint(0, 2, (batch_size, seq_len)).float()

    # 前向传播
    result = model(question_ids, correct)

    print(f"\n输入形状:")
    print(f"  question_ids: {question_ids.shape}")
    print(f"  correct: {correct.shape}")

    print(f"\n输出形状:")
    print(f"  predictions: {result['predictions'].shape}")
    print(f"  all_outputs: {result['all_outputs'].shape}")

    # 测试预测
    predictions, targets, mask = model.predict(question_ids, correct)

    print(f"\n预测结果:")
    print(f"  预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  目标值范围: [{targets.min():.4f}, {targets.max():.4f}]")

    # 检查 CUDA/MPS 可用性
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n可用设备: {device}")

    # 移动到设备测试
    if device.type != "cpu":
        model = model.to(device)
        question_ids = question_ids.to(device)
        correct = correct.to(device)
        result = model(question_ids, correct)
        print(f"GPU 测试通过!")


if __name__ == "__main__":
    test_model()
