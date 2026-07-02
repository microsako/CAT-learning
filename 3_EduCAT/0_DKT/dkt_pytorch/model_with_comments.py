"""
Deep Knowledge Tracing (DKT) - 详细注释版

本代码参考论文: "Deep Knowledge Tracing"
Chris Piech et al., NIPS 2015

论文链接: https://papers.nips.cc/paper/5653-deep-knowledge-tracing

================================================================================
一、DKT 核心思想
================================================================================

传统知识追踪 (Bayesian Knowledge Tracing) 的问题:
- 需要专家手工标注知识点
- 不能自动发现知识点之间的关系
- 模型容量有限

DKT 的解决方案:
- 使用 RNN/LSTM 自动学习知识点
- 不需要手工标注
- 可以发现知识点之间的依赖关系

================================================================================
二、DKT 的输入输出
================================================================================

输入: 学生的答题序列
  x_1 = (q_1, a_1)   # 第1题: 题目ID + 是否正确
  x_2 = (q_2, a_2)   # 第2题
  x_3 = (q_3, a_3)   # 第3题
  ...

输出: 预测学生下一题的正确概率
  P(a_{t+1} = 1 | q_{t+1})  # 预测第 t+1 题的正确概率

================================================================================
三、模型架构 (参考论文 Figure 2)
================================================================================

  x_1        x_2        x_3        x_t
   |          |          |          |
   v          v          v          v
+-------+  +-------+  +-------+  +-------+
| Embed |  | Embed |  | Embed |  | Embed |   # 嵌入层
+-------+  +-------+  +-------+  +-------+
   |          |          |          |
   +----------+----------+----------+
              |
              v
         +---------+
         |   LSTM  |  ← 记忆学生之前的学习历史
         +---------+
              |
              v
         +---------+
         |  Output |  → 输出每个题目的正确概率
         +---------+
              |
              v
         y_1        y_2        y_3        y_t

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DKTSimplifiedWithComments(nn.Module):
    """
    DKT 模型 - 详细注释版

    这个类实现了 Deep Knowledge Tracing 的核心逻辑。
    代码结构完全按照论文描述来实现。

    论文参考:
    - Section 3.1 Model: RNN/LSTM 的数学公式
    - Section 3.2 Input and Output Time Series: 输入输出编码
    - Section 3.3 Optimization: 训练技巧
    """

    def __init__(
        self,
        n_questions: int = 110,
        n_hidden: int = 200,
        n_layers: int = 1,
        dropout: float = 0.5,
        use_lstm: bool = True
    ):
        """
        模型初始化

        参数说明:
        ----------
        n_questions : int
            题目总数。在 ASSISTments 数据集中有 110 道题目。
            模型需要知道这个数字来:
            1. 创建输入编码 (one-hot)
            2. 创建输出层 (预测每个题目的正确概率)

        n_hidden : int
            隐藏层维度。论文默认使用 200。
            这个向量用来表示学生的"知识状态"。
            维度越大，模型能表示的知识越复杂，但训练也越慢。

        n_layers : int
            LSTM 层数。论文使用单层 LSTM。
            多层可以学习更复杂的模式，但这里用单层。

        dropout : float
            Dropout 比率。论文使用 0.5。
            训练时随机关闭 50% 的神经元，防止过拟合。

        use_lstm : bool
            是否使用 LSTM。论文发现 LSTM 比普通 RNN 效果更好。
        """
        super().__init__()

        # =====================================================================
        # 第一步: 保存配置参数
        # =====================================================================
        self.n_questions = n_questions      # 题目数量
        self.n_hidden = n_hidden            # 隐藏层维度
        self.n_layers = n_layers            # LSTM 层数
        self.use_lstm = use_lstm            # 是否用 LSTM

        # =====================================================================
        # 第二步: 定义输入编码
        # =====================================================================
        #
        # 论文 3.2 节描述了输入编码方式:
        # "xt to be a one-hot encoding of the student interaction tuple"
        #
        # 我们的输入包含两部分:
        # 1. 题目ID (one-hot 编码, 110 维)
        # 2. 是否正确 (one-hot 编码, 2 维: 0=错误, 1=正确)
        #
        # 总输入维度 = 110 + 2 = 112
        self.n_input = n_questions + 2

        # =====================================================================
        # 第三步: 定义 LSTM 层
        # =====================================================================
        #
        # LSTM 是 DKT 的核心，它负责:
        # 1. 记住学生之前做过的题目
        # 2. 根据历史更新知识状态
        # 3. 预测下一题的正确概率
        #
        # 论文 3.1 节:
        # "ht = tanh(Whx·xt + Whh·h_{t-1} + bh)"
        # "yt = σ(Wyh·ht + by)"
        #
        # LSTM 相比普通 RNN 的优势:
        # - 可以记住很久以前的信息 (梯度消失问题)
        # - 训练更稳定
        #
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=self.n_input,     # 输入维度: 112
                hidden_size=n_hidden,        # 隐藏层维度: 200
                num_layers=n_layers,         # 层数: 1
                batch_first=True             # 输入格式: (batch, seq, features)
            )
        else:
            self.rnn = nn.RNN(
                input_size=self.n_input,
                hidden_size=n_hidden,
                num_layers=n_layers,
                batch_first=True
            )

        # =====================================================================
        # 第四步: 定义 Dropout (论文 3.3 节)
        # =====================================================================
        #
        # 论文 3.3 节:
        # "dropout was applied to ht when computing the readout yt,
        #  but not when computing the next hidden state ht+1"
        #
        # 关键点: Dropout 只在输出层使用，不在循环连接中使用！
        # 这样可以防止过拟合，同时不影响模型的记忆能力。
        #
        self.output_dropout = nn.Dropout(dropout)

        # =====================================================================
        # 第五步: 定义输出层
        # =====================================================================
        #
        # 输出层把隐藏状态转换为每个题目的正确概率。
        #
        # 论文 3.2 节:
        # "The output yt is a vector of length equal to the number of problems,
        #  where each entry represents the predicted probability that the student
        #  would answer that particular problem correctly."
        #
        # 所以输出是 110 维 (每个题目一个概率值)。
        #
        self.output_layer = nn.Linear(n_hidden, n_questions)

        # 初始化隐藏状态 (稍后会在 forward 中重新初始化)
        self._init_hidden()

    def _init_hidden(self, batch_size: int = 1, device: torch.device = None):
        """
        初始化 LSTM 的隐藏状态

        为什么需要隐藏状态?
        ------------------------
        LSTM 需要维护两个状态:
        - h: 短期状态 (hidden state)
        - c: 长期状态 (cell state)

        这两个状态会在时间步之间传递，
        用来记住学生之前的学习历史。

        初始化为 0 表示:
        "在学生开始答题之前，我们假设他什么都不会"
        """
        if device is None:
            device = next(self.parameters()).device

        # h_0: 初始隐藏状态 (短期记忆)
        # 形状: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)

        if self.use_lstm:
            # c_0: 初始细胞状态 (长期记忆)
            # LSTM 用这个来传递长期信息
            c0 = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
            self.hidden = (h0, c0)  # LSTM 需要两个状态
        else:
            self.hidden = h0  # 普通 RNN 只需要一个状态

    def _prepare_input(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor
    ) -> torch.Tensor:
        """
        第三步: 构建输入向量

        这个方法把 (题目ID, 正确性) 转换为模型可以处理的向量。

        参数:
        ----------
        question_ids : torch.Tensor
            形状: (batch_size, seq_len)
            每行是一个学生的题目ID序列

        correct : torch.Tensor
            形状: (batch_size, seq_len)
            每行是一个学生的正确性序列 (0 或 1)

        返回:
        ----------
        torch.Tensor
            形状: (batch_size, seq_len, n_questions + 2)
            拼接后的输入向量
        """
        # =====================================================================
        # 3.1 题目ID的 One-Hot 编码
        # =====================================================================
        #
        # One-Hot 编码示例 (假设有 5 道题):
        #
        # 题目ID=2 → [0, 1, 0, 0, 0]
        # 题目ID=4 → [0, 0, 0, 1, 0]
        #
        # 为什么用 One-Hot?
        # - 每个题目都是独立的，没有大小关系
        # - 模型可以学习每个题目的独特特征
        #
        # F.one_hot 的输出形状: (batch, seq, n_questions)
        # 例如: torch.Size([32, 100, 110])
        #
        question_one_hot = F.one_hot(
            question_ids.long(),          # 需要是整数类型
            num_classes=self.n_questions  # 类别数 = 题目数
        ).float()  # 转换为浮点数

        # =====================================================================
        # 3.2 正确性的 One-Hot 编码
        # =====================================================================
        #
        # 正确性只有两种: 0 (错误) 或 1 (正确)
        #
        # 编码示例:
        # 正确=0 → [1, 0]  (one-hot: 第0位是1)
        # 正确=1 → [0, 1]  (one-hot: 第1位是1)
        #
        # 注意: 这里不是简单的 [0] 或 [1]，
        # 而是 2 维的 one-hot 向量。
        #
        correct_one_hot = F.one_hot(
            correct.long(),   # 需要是整数类型
            num_classes=2     # 0 或 1，共2类
        ).float()

        # =====================================================================
        # 3.3 拼接两部分
        # =====================================================================
        #
        # 论文 3.2 节:
        # "we set xt to be a one-hot encoding of the student interaction tuple
        #  {qt, at} that represents the combination of which exercise was
        #  answered and if the exercise was answered correctly"
        #
        # 把题目ID和正确性拼接在一起:
        # [one-hot题目, one-hot正确性]
        #
        # 形状变化:
        # question_one_hot: (batch, seq, 110)
        # correct_one_hot:  (batch, seq, 2)
        # concat 后:        (batch, seq, 112)
        #
        x = torch.cat([question_one_hot, correct_one_hot], dim=-1)

        return x

    def forward(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        hidden: Optional[Tuple] = None
    ) -> Dict[str, torch.Tensor]:
        """
        第四步: 前向传播

        这是模型的核心逻辑，把输入转换为预测。

        参数:
        ----------
        question_ids : torch.Tensor
            形状: (batch_size, seq_len)
            题目ID序列

        correct : torch.Tensor
            形状: (batch_size, seq_len)
            正确性序列

        hidden : tuple, 可选
            LSTM 的隐藏状态。如果不提供，使用初始状态。

        返回:
        ----------
        Dict[str, torch.Tensor]
            - predictions: 形状 (batch, seq-1), 预测的下一题正确概率
            - all_outputs: 形状 (batch, seq, n_questions), 所有题目的概率
            - hidden: 最后的隐藏状态
        """
        batch_size, seq_len = question_ids.shape

        # =====================================================================
        # 4.1 构建输入向量
        # =====================================================================
        # 详细解释见 _prepare_input 方法
        x = self._prepare_input(question_ids, correct)
        # 形状: (batch, seq, 112)

        # =====================================================================
        # 4.2 初始化隐藏状态
        # =====================================================================
        #
        # 第一次调用或需要新的批次时，重新初始化隐藏状态。
        # 隐藏状态初始化为 0 表示"学生从零开始学习"。
        #
        if hidden is None:
            self._init_hidden(batch_size, x.device)
            hidden = self.hidden

        # =====================================================================
        # 4.3 LSTM 前向传播
        # =====================================================================
        #
        # 这是 DKT 的核心步骤！
        #
        # LSTM 会处理整个序列，维护和更新隐藏状态。
        #
        # 论文 3.1 节:
        # "ht = tanh(Whx·xt + Whh·h_{t-1} + bh)"
        #
        # 这个公式说的是:
        # 新的隐藏状态 = f(当前输入, 之前的隐藏状态)
        #
        # 直观理解:
        # - 如果学生做对了当前的题 (a_t = 1)，
        #   LSTM 会增强相关知识点的状态
        # - 如果学生做错了 (a_t = 0)，
        #   LSTM 会降低相关知识点的状态
        # - LSTM 会记住学生之前做过的所有题目
        #
        rnn_output, hidden = self.rnn(x, hidden)
        self.hidden = hidden  # 保存状态用于下一个批次

        # rnn_output 形状: (batch, seq, n_hidden)
        # 每一列是时间步 t 的隐藏状态 h_t
        # h_t 包含了学生前 t 道题目的学习历史

        # =====================================================================
        # 4.4 应用 Dropout (论文 3.3 节)
        # =====================================================================
        #
        # 关键: Dropout 只在输出层使用！
        # 这样可以防止过拟合，但不影响模型的记忆能力。
        #
        rnn_output = self.output_dropout(rnn_output)

        # =====================================================================
        # 4.5 输出层: 计算每个题目的正确概率
        # =====================================================================
        #
        # 论文 3.1 节:
        # "yt = σ(Wyh·ht + by)"
        #
        # Sigmoid 函数把任意实数映射到 (0, 1) 区间，
        # 正好适合表示概率。
        #
        # 输出形状: (batch, seq, n_questions)
        # 含义: 每个时间步，预测每个题目的正确概率
        #
        all_outputs = torch.sigmoid(self.output_layer(rnn_output))

        # =====================================================================
        # 4.6 提取对应的预测
        # =====================================================================
        #
        # DKT 的预测逻辑:
        # - 在时间步 t，我们用前 t 个答题记录
        # - 预测第 t+1 题的正确概率
        #
        # 也就是说:
        # - all_outputs[:, t, :] 包含对所有题目的预测
        # - 我们需要用 question_ids[:, t+1] 来选取对应的预测
        #
        predictions = self._get_predictions(all_outputs, question_ids)

        return {
            "predictions": predictions,
            "all_outputs": all_outputs,
            "hidden": hidden
        }

    def _get_predictions(
        self,
        all_outputs: torch.Tensor,
        question_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        第五步: 提取对应的预测

        这个方法实现 DKT 的核心预测逻辑。

        举例说明:
        --------
        假设有 3 道题 (A, B, C)，某个学生的答题序列是:

        时间步 1: 做 A，对了
        时间步 2: 做 B，错了
        时间步 3: 做 C，对了

        question_ids = [A, B, C]  # 题目ID
        all_outputs[1] = [0.9, 0.5, 0.7]  # 对 A,B,C 的预测概率

        DKT 预测:
        - predictions[1] 应该预测下一题 (B) 的正确概率
        - 选取 all_outputs[1, B] = 0.5

        所以 predictions = [?, ?, ?]
        其中 predictions[t] = all_outputs[t, question_ids[t+1]]
        """
        batch_size, seq_len = question_ids.shape

        # all_outputs 的形状: (batch, seq, n_questions)
        # 我们要选取 all_outputs[:, t, question_ids[:, t+1]]

        # 去掉最后一个时间步 (因为没有下一题)
        relevant_outputs = all_outputs[:, :-1, :]  # (batch, seq-1, n_questions)
        next_question_ids = question_ids[:, 1:]    # (batch, seq-1)

        # torch.gather: 根据索引选取元素
        # dim=2 表示在最后一维 (n_questions) 上选取
        predictions = torch.gather(
            relevant_outputs,
            dim=2,
            index=next_question_ids.unsqueeze(-1)  # 需要 (batch, seq-1, 1) 形状
        ).squeeze(-1)  # 压缩回 (batch, seq-1)

        return predictions

    def predict(
        self,
        question_ids: torch.Tensor,
        correct: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测接口 - 供训练和评估使用

        这个方法把模型的输出整理成标准格式。

        参数:
        ----------
        question_ids : torch.Tensor
            形状: (batch_size, seq_len)
            题目ID序列

        correct : torch.Tensor
            形状: (batch_size, seq_len)
            正确性序列

        mask : torch.Tensor, 可选
            形状: (batch_size, seq_len)
            有效位置掩码 (1=有效, 0=无效)

        返回:
        ----------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (predictions, targets, mask)
            - predictions: 模型预测的概率
            - targets: 真实的正确性标签
            - mask: 有效位置掩码
        """
        result = self.forward(question_ids, correct)

        # predictions 形状: (batch, seq-1)
        predictions = result["predictions"]

        # targets 是 correct 序列的下一个
        # 如果 correct = [1, 0, 1]，则 targets = [0, 1]
        targets = correct[:, 1:]

        # mask 也要对应地移动
        if mask is not None:
            mask = mask[:, 1:]
        else:
            mask = torch.ones_like(targets)

        return predictions, targets, mask


# =============================================================================
# 测试代码
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DKT 模型测试 - 详细注释版")
    print("=" * 60)

    # 创建模型
    model = DKTSimplifiedWithComments(
        n_questions=110,  # ASSISTments 数据集有 110 道题
        n_hidden=200,     # 论文默认值
        dropout=0.5      # 论文默认值
    )

    print("\n模型参数量:", sum(p.numel() for p in model.parameters()))
    print("  - LSTM 参数: 可学习权重和偏置")
    print("  - Output 层参数: 200*110 + 110 = 22,110")

    # 创建虚拟输入
    batch_size = 4
    seq_len = 10

    # question_ids: 每行是一个学生的题目ID序列 (0-109 之间)
    question_ids = torch.randint(0, 110, (batch_size, seq_len))

    # correct: 每行是一个学生的正确性序列 (0 或 1)
    correct = torch.randint(0, 2, (batch_size, seq_len)).float()

    print(f"\n虚拟输入:")
    print(f"  question_ids 形状: {question_ids.shape}")
    print(f"  correct 形状: {correct.shape}")

    # 前向传播
    result = model(question_ids, correct)

    print(f"\n输出:")
    print(f"  predictions 形状: {result['predictions'].shape}")
    print(f"  all_outputs 形状: {result['all_outputs'].shape}")

    # 解释输出
    print("\n输出解释:")
    print("  - predictions[t] = 预测第 t+1 题的正确概率")
    print("  - all_outputs[t] = 对所有 110 道题的预测概率")

    # 示例
    print(f"\n示例 (第一个学生):")
    print(f"  题目ID: {question_ids[0].tolist()}")
    print(f"  正确性: {correct[0].long().tolist()}")
    print(f"  预测:   {result['predictions'][0][:5].tolist()}")

    print("\n" + "=" * 60)
    print("测试通过!")
    print("=" * 60)
