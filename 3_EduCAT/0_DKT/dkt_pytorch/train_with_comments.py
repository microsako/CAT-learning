"""
DKT 训练脚本 - 详细注释版

本脚本实现完整的 DKT 训练流程。
参考论文: "Deep Knowledge Tracing", NIPS 2015

================================================================================
一、训练流程概述
================================================================================

1. 加载数据
   - 读取学生的答题记录
   - 转换为模型需要的格式

2. 创建模型
   - 初始化 DKT 模型
   - 设置优化器和学习率

3. 训练循环
   for epoch in range(epochs):
       for batch in train_loader:
           # 前向传播
           predictions = model(inputs)

           # 计算损失
           loss = criterion(predictions, targets)

           # 反向传播
           loss.backward()

           # 更新参数
           optimizer.step()

4. 评估
   - 在测试集上计算 AUC

================================================================================
二、损失函数 (论文公式 3)
================================================================================

论文公式 (3):
  L = Σ l(y_t·δ(q_{t+1}), a_{t+1})

其中:
- y_t: 模型预测的概率向量
- δ(q_{t+1}): 下一题的 one-hot 编码
- a_{t+1}: 真实的正确性 (0 或 1)
- l: 二元交叉熵损失

直观理解:
- 如果学生做对了 (a=1)，损失 = -log(p)，p越大损失越小
- 如果学生做错了 (a=0)，损失 = -log(1-p)，p越小损失越小

================================================================================
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

# =============================================================================
# 第一步: 导入模块
# =============================================================================

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_with_comments import DKTSimplifiedWithComments


# =============================================================================
# 第二步: 设置超参数
# =============================================================================

# 这些超参数都参考论文 3.3 节
class Config:
    """训练配置"""

    # 数据相关
    N_QUESTIONS = 110       # ASSISTments 数据集有 110 道题目
    BATCH_SIZE = 100         # 论文使用批次大小 100
    MAX_SEQ_LEN = 200       # 最大序列长度 (防止内存溢出)

    # 模型相关
    N_HIDDEN = 200           # 隐藏层维度 (论文默认值)
    DROPOUT = 0.5            # Dropout 比率 (论文默认值)

    # 训练相关
    EPOCHS = 50              # 最大训练轮数
    LEARNING_RATE = 0.1      # 学习率 (论文使用 0.1)
    MOMENTUM = 0.9           # SGD 动量
    MAX_GRAD_NORM = 5.0     # 梯度裁剪阈值 (论文使用)
    LR_DECAY_STEP = 15       # 学习率衰减步长
    LR_DECAY_GAMMA = 0.5    # 学习率衰减比率

    # 早停
    PATIENCE = 15           # 早停耐心值

    # 路径
    TRAIN_DATA = "data/assistments/builder_train.csv"
    TEST_DATA = "data/assistments/builder_test.csv"
    MODEL_DIR = "output/models/"


# =============================================================================
# 第三步: 数据加载
# =============================================================================

class ASSISTmentsDatasetWithComments:
    """
    ASSISTments 数据集加载器 - 详细注释版

    数据格式 (每4行一个学生):
    ------------------------
    第1行: 答题数量 (n)
    第2行: n 个题目ID，用逗号分隔
    第3行: n 个正确性 (0 或 1)，用逗号分隔
    第4行: 空行 (分隔符)
    """

    def __init__(self, filepath: str, n_questions: int = 110, max_seq_len: int = 200):
        """
        初始化数据集

        参数:
        ----------
        filepath : str
            CSV 文件路径

        n_questions : int
            题目总数

        max_seq_len : int
            最大序列长度，超过则截断
        """
        self.filepath = filepath
        self.n_questions = n_questions
        self.max_seq_len = max_seq_len

        # 读取数据
        self._load_data()

    def _load_data(self):
        """加载数据文件"""
        self.students = []
        self.num_students = 0
        self.max_sequence_length = 0

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # 每4行是一个学生
        i = 0
        while i < len(lines):
            # 第1行: 答题数量
            try:
                n_answers = int(lines[i].strip())
            except:
                i += 1
                continue

            # 第2行: 题目ID
            try:
                questions = [int(q) for q in lines[i + 1].strip().split(',')]
            except:
                i += 4
                continue

            # 第3行: 正确性
            try:
                correct = [int(c) for c in lines[i + 2].strip().split(',')]
            except:
                i += 4
                continue

            # 确保数据长度一致
            n_answers = min(n_answers, len(questions), len(correct))

            # 截断过长序列
            if n_answers > self.max_seq_len:
                questions = questions[:self.max_seq_len]
                correct = correct[:self.max_seq_len]
                n_answers = self.max_seq_len

            # 保存学生数据
            self.students.append({
                "n_answers": n_answers,
                "questions": questions,
                "correct": correct
            })

            self.max_sequence_length = max(self.max_sequence_length, n_answers)
            self.num_students += 1

            i += 4  # 跳到下一个学生

    def __len__(self):
        return self.num_students

    def __getitem__(self, idx):
        """获取单个学生的数据"""
        return self.students[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批处理函数 - 将多个学生数据合并为一个批次

    参数:
    ----------
    batch : List[Dict]
        多个学生的数据列表

    返回:
    ----------
    Dict[str, torch.Tensor]
        包含:
        - question_ids: (batch_size, max_seq_len)
        - correct: (batch_size, max_seq_len)
        - mask: (batch_size, max_seq_len) - 有效位置为1
        - lengths: (batch_size,) - 每个序列的实际长度
    """
    batch_size = len(batch)

    # 找到批次中最长的序列
    max_len = max(student["n_answers"] for student in batch)

    # 初始化张量
    question_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    correct = torch.zeros(batch_size, max_len, dtype=torch.float)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    # 填充数据
    for i, student in enumerate(batch):
        n = student["n_answers"]
        question_ids[i, :n] = torch.tensor(student["questions"])
        correct[i, :n] = torch.tensor(student["correct"])
        mask[i, :n] = 1.0  # 前 n 个位置是有效的
        lengths[i] = n

    return {
        "question_ids": question_ids,
        "correct": correct,
        "mask": mask,
        "lengths": lengths
    }


def get_data_loaders_with_comments(
    train_path: str,
    test_path: str,
    n_questions: int,
    batch_size: int
):
    """
    创建数据加载器

    参数:
    ----------
    train_path : str
        训练数据路径

    test_path : str
        测试数据路径

    n_questions : int
        题目数量

    batch_size : int
        批次大小

    返回:
    ----------
    (train_loader, test_loader)
    """
    # 创建数据集
    train_dataset = ASSISTmentsDatasetWithComments(train_path, n_questions)
    test_dataset = ASSISTmentsDatasetWithComments(test_path, n_questions)

    print(f"训练集: {len(train_dataset)} 学生")
    print(f"测试集: {len(test_dataset)} 学生")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # 打乱数据
        num_workers=0,       # 单进程
        collate_fn=collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


# =============================================================================
# 第四步: 评估指标
# =============================================================================

def calculate_auc(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    计算 AUC (Area Under the ROC Curve)

    AUC 衡量模型区分正负样本的能力。
    - AUC = 0.5: 随机猜测
    - AUC = 1.0: 完美预测
    - AUC > 0.7: 不错的模型

    参数:
    ----------
    predictions : np.ndarray
        预测概率

    targets : np.ndarray
        真实标签 (0 或 1)

    mask : np.ndarray
        有效位置掩码

    返回:
    ----------
    float: AUC 值
    """
    from sklearn.metrics import roc_auc_score

    # 只考虑有效位置
    valid_idx = mask > 0
    valid_preds = predictions[valid_idx]
    valid_targets = targets[valid_idx]

    if len(np.unique(valid_targets)) < 2:
        # 如果只有一个类别，返回 0.5
        return 0.5

    return roc_auc_score(valid_targets, valid_preds)


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    """
    计算准确率

    参数:
    ----------
    predictions : np.ndarray
        预测概率

    targets : np.ndarray
        真实标签

    mask : np.ndarray
        有效位置掩码

    返回:
    ----------
    float: 准确率
    """
    valid_idx = mask > 0
    valid_preds = (predictions[valid_idx] > 0.5).astype(float)
    valid_targets = targets[valid_idx]

    return np.mean(valid_preds == valid_targets)


class EarlyStopping:
    """
    早停策略

    当验证集性能连续多轮没有提升时，停止训练。
    这样可以防止过拟合。
    """

    def __init__(self, patience: int = 10, mode: str = "max"):
        """
        参数:
        ----------
        patience : int
            耐心值。连续多少轮没有提升就停止。

        mode : str
            "max": 指标越大越好 (如 AUC)
            "min": 指标越小越好 (如 Loss)
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        判断是否应该停止训练

        参数:
        ----------
        value : float
            当前指标值

        返回:
        ----------
        bool: 是否应该停止
        """
        if self.best_value is None:
            # 第一轮
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            # 有提升，重置计数器
            self.best_value = value
            self.counter = 0
        else:
            # 没有提升
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =============================================================================
# 第五步: 训练函数
# =============================================================================

def train():
    """
    完整的训练流程
    """
    # 设置随机种子 (保证可复现)
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建输出目录
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # =====================================================================
    # 5.1 选择设备
    # =====================================================================
    # 优先使用 MPS (Apple Silicon GPU)，其次是 CUDA，最后是 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"使用设备: {device}")

    # =====================================================================
    # 5.2 加载数据
    # =====================================================================
    print("\n[1/5] 加载数据...")
    train_loader, test_loader = get_data_loaders_with_comments(
        train_path=Config.TRAIN_DATA,
        test_path=Config.TEST_DATA,
        n_questions=Config.N_QUESTIONS,
        batch_size=Config.BATCH_SIZE
    )
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # =====================================================================
    # 5.3 创建模型
    # =====================================================================
    print("\n[2/5] 创建模型...")
    model = DKTSimplifiedWithComments(
        n_questions=Config.N_QUESTIONS,
        n_hidden=Config.N_HIDDEN,
        dropout=Config.DROPOUT,
        use_lstm=True
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # =====================================================================
    # 5.4 设置优化器和学习率调度器
    # =====================================================================
    #
    # 论文 3.3 节:
    # "This objective was minimized using stochastic gradient descent"
    #
    # 我们使用 SGD + 动量，这是论文推荐的方法
    #
    criterion = nn.BCELoss(reduction='none')  # 不进行平均，后面手动处理

    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM
    )

    # 学习率衰减
    # 每 15 轮，学习率减半
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.LR_DECAY_STEP,
        gamma=Config.LR_DECAY_GAMMA
    )

    # 早停
    early_stopping = EarlyStopping(patience=Config.PATIENCE, mode="max")

    # =====================================================================
    # 5.5 训练循环
    # =====================================================================
    print("\n[3/5] 开始训练...")
    print("=" * 60)

    best_auc = 0.0

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
        num_batches = 0

        # 用于计算指标的列表
        all_preds = []
        all_targets = []
        all_masks = []

        # -----------------------------------------------------------------
        # 遍历每个批次
        # -----------------------------------------------------------------
        for batch in train_loader:
            # 获取数据
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)

            # -----------------------------------------------------------------
            # 前向传播
            # -----------------------------------------------------------------
            # 模型预测下一题的正确概率
            predictions, targets, mask_out = model.predict(question_ids, correct, mask)

            # -----------------------------------------------------------------
            # 计算损失 (论文公式 3)
            # -----------------------------------------------------------------
            #
            # L = Σ l(y_t·δ(q_{t+1}), a_{t+1})
            #
            # 这里的实现:
            # 1. 预测值 predictions 已经选取了对应当前题目的概率
            # 2. 目标值 targets 是真实的正确性
            # 3. mask_out 只在有效位置计算损失
            #
            # BCE Loss: l(p, y) = -[y*log(p) + (1-y)*log(1-p)]
            #
            loss = (criterion(predictions.flatten(), targets.flatten())
                    * mask_out.flatten()).sum() / mask_out.sum()

            # -----------------------------------------------------------------
            # 反向传播
            # -----------------------------------------------------------------
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 计算梯度

            # -----------------------------------------------------------------
            # 梯度裁剪 (论文 3.3 节)
            # -----------------------------------------------------------------
            #
            # "We prevent gradients from 'exploding' as we backpropagate
            #  through time by truncating the length of gradients whose
            #  norm is above a threshold."
            #
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                Config.MAX_GRAD_NORM
            )

            # -----------------------------------------------------------------
            # 更新参数
            # -----------------------------------------------------------------
            optimizer.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1

            # 保存用于计算 AUC
            all_preds.append(predictions.detach().cpu().numpy().flatten())
            all_targets.append(targets.cpu().numpy().flatten())
            all_masks.append(mask_out.cpu().numpy().flatten())

        # 更新学习率
        scheduler.step()

        # -----------------------------------------------------------------
        # 计算训练指标
        # -----------------------------------------------------------------
        train_auc = calculate_auc(
            np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_masks)
        )
        train_loss = total_loss / num_batches

        # -----------------------------------------------------------------
        # 验证
        # -----------------------------------------------------------------
        model.eval()
        val_preds = []
        val_targets = []
        val_masks = []

        with torch.no_grad():
            for batch in test_loader:
                question_ids = batch["question_ids"].to(device)
                correct = batch["correct"].to(device)
                mask = batch["mask"].to(device)

                predictions, targets, mask_out = model.predict(
                    question_ids, correct, mask
                )

                val_preds.append(predictions.cpu().numpy().flatten())
                val_targets.append(targets.cpu().numpy().flatten())
                val_masks.append(mask_out.cpu().numpy().flatten())

        val_auc = calculate_auc(
            np.concatenate(val_preds),
            np.concatenate(val_targets),
            np.concatenate(val_masks)
        )
        val_acc = calculate_accuracy(
            np.concatenate(val_preds),
            np.concatenate(val_targets),
            np.concatenate(val_masks)
        )

        epoch_time = time.time() - epoch_start

        # 打印结果
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train AUC: {train_auc:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # -----------------------------------------------------------------
        # 保存最佳模型
        # -----------------------------------------------------------------
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "auc": val_auc,
                "accuracy": val_acc
            }, f"{Config.MODEL_DIR}best_model_commented.pt")
            print(f"         >>> 保存最佳模型 (AUC: {best_auc:.4f})")

        # -----------------------------------------------------------------
        # 早停检查
        # -----------------------------------------------------------------
        if early_stopping(val_auc):
            print(f"\n早停触发! 连续{early_stopping.patience}轮无改善")
            break

    # =====================================================================
    # 5.6 测试集评估
    # =====================================================================
    print("\n" + "=" * 60)
    print(f"[4/5] 训练完成! 最佳验证 AUC: {best_auc:.4f}")

    print("\n[5/5] 测试集评估...")
    checkpoint = torch.load(f"{Config.MODEL_DIR}best_model_commented.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []
    test_masks = []

    with torch.no_grad():
        for batch in test_loader:
            question_ids = batch["question_ids"].to(device)
            correct = batch["correct"].to(device)
            mask = batch["mask"].to(device)

            predictions, targets, mask_out = model.predict(
                question_ids, correct, mask
            )

            test_preds.append(predictions.cpu().numpy().flatten())
            test_targets.append(targets.cpu().numpy().flatten())
            test_masks.append(mask_out.cpu().numpy().flatten())

    test_auc = calculate_auc(
        np.concatenate(test_preds),
        np.concatenate(test_targets),
        np.concatenate(test_masks)
    )
    test_acc = calculate_accuracy(
        np.concatenate(test_preds),
        np.concatenate(test_targets),
        np.concatenate(test_masks)
    )

    print(f"\n测试集结果:")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print("=" * 60)

    return test_auc, test_acc


# =============================================================================
# 第六步: 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DKT 训练脚本 - 详细注释版")
    print("=" * 60)

    # 运行训练
    test_auc, test_acc = train()

    print("\n训练完成!")
    print(f"最终测试 AUC: {test_auc:.4f}")
