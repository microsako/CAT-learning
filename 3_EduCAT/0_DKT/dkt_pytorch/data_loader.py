"""
数据加载器模块

负责加载和预处理 ASSISTments 数据集。

ASSISTments 数据格式 (每4行为一个学生):
    第1行: 答题数量 n
    第2行: 题目ID列表 (n个，用逗号分隔)
    第3行: 正确性列表 (n个，0或1，用逗号分隔)
    (空行)

示例:
    7
    7,7,7,7,7,7,8
    1,1,0,1,1,1,1

表示该学生回答了7道题，题目ID为 [7,7,7,7,7,7,8]，正确性为 [1,1,0,1,1,1,1]
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional


class ASSISTmentsDataset(Dataset):
    """
    ASSISTments 数据集类

    将原始数据转换为适合 DKT 模型训练的格式。
    DKT 的核心思想是：根据学生历史答题序列，预测下一题的答题情况。
    """

    def __init__(self, data_path: str, n_questions: int = 110, max_seq_len: Optional[int] = None):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            n_questions: 数据集中的题目总数
            max_seq_len: 最大序列长度，超过则截断
        """
        self.data_path = data_path
        self.n_questions = n_questions
        self.max_seq_len = max_seq_len

        # 加载原始数据
        self.students = self._load_data()

        # 统计信息
        self.num_students = len(self.students)
        self.max_sequence_length = max(len(s["question_ids"]) for s in self.students)

        print(f"加载数据: {data_path}")
        print(f"学生数量: {self.num_students}")
        print(f"最大序列长度: {self.max_sequence_length}")

    def _load_data(self) -> List[Dict]:
        """
        从文件加载数据

        原始数据格式是每4行一个学生:
        - 第1行: 答题数量
        - 第2行: 题目ID序列
        - 第3行: 正确性序列
        - 第4行: 空行

        Returns:
            学生数据列表，每个元素是一个字典，包含:
            - question_ids: 题目ID列表
            - correct: 正确性列表
            - n_answers: 答题数量
        """
        students = []

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        with open(self.data_path, 'r') as f:
            lines = f.readlines()

        # 每4行是一个学生 (3行数据 + 1个空行)
        for i in range(0, len(lines), 4):
            if i + 2 >= len(lines):
                break

            try:
                # 解析行数据
                n_answers = int(lines[i].strip())
                question_ids_str = lines[i + 1].strip().rstrip(',')
                correct_str = lines[i + 2].strip().rstrip(',')

                # 处理空行情况
                if not question_ids_str or not correct_str:
                    continue

                # 转换为列表
                question_ids = [int(x) + 1 for x in question_ids_str.split(',') if x.strip()]  # +1 因为题目ID从1开始
                correct = [int(x) for x in correct_str.split(',') if x.strip()]

                # 验证数据完整性
                if len(question_ids) != n_answers or len(correct) != n_answers:
                    continue

                # 过滤掉序列太短的样本 (DKT需要至少2个答题记录来形成一对输入-输出)
                if len(question_ids) < 2:
                    continue

                students.append({
                    "question_ids": question_ids,
                    "correct": correct,
                    "n_answers": len(question_ids)
                })

            except (ValueError, IndexError):
                # 跳过格式错误的数据行
                continue

        return students

    def __len__(self) -> int:
        """返回学生数量"""
        return self.num_students

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个学生的数据

        DKT 的数据处理逻辑:
        对于长度为 T 的序列，我们有 T-1 个训练样本:
        - 输入: 前 t 个答题记录 (题目ID + 正确性)
        - 输出: 预测第 t+1 题的正确性

        Args:
            idx: 学生索引

        Returns:
            包含以下键的字典:
            - question_ids: 题目ID序列 (长T)
            - correct: 正确性序列 (长T)
            - mask: 有效位置掩码 (长T)
            - n_answers: 答题数量
        """
        student = self.students[idx]
        question_ids = student["question_ids"]
        correct = student["correct"]

        # 序列长度
        seq_len = len(question_ids)

        # 截断到最大长度
        if self.max_seq_len is not None and seq_len > self.max_seq_len:
            question_ids = question_ids[:self.max_seq_len]
            correct = correct[:self.max_seq_len]
            seq_len = self.max_seq_len

        # 创建掩码 (所有位置都有效)
        mask = torch.ones(seq_len, dtype=torch.float32)

        return {
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "correct": torch.tensor(correct, dtype=torch.float32),
            "mask": mask,
            "n_answers": seq_len
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    批次整理函数

    将多个学生的数据整理成一个批次。
    由于不同学生的序列长度不同，需要进行填充。

    Args:
        batch: 单个学生的数据列表

    Returns:
        批次数据，包含填充后的序列
    """
    # 找出批次中的最大序列长度
    max_len = max(item["n_answers"] for item in batch)
    batch_size = len(batch)

    # 初始化填充后的张量
    question_ids_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
    correct_padded = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask_padded = torch.zeros(batch_size, max_len, dtype=torch.float32)

    # 填充
    for i, item in enumerate(batch):
        seq_len = item["n_answers"]
        question_ids_padded[i, :seq_len] = item["question_ids"]
        correct_padded[i, :seq_len] = item["correct"]
        mask_padded[i, :seq_len] = item["mask"]

    return {
        "question_ids": question_ids_padded,
        "correct": correct_padded,
        "mask": mask_padded,
        "n_answers": [item["n_answers"] for item in batch],
        "batch_size": batch_size,
        "max_len": max_len
    }


def get_data_loaders(
    train_path: str,
    test_path: str,
    n_questions: int = 110,
    max_seq_len: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和测试数据加载器

    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        n_questions: 题目数量
        max_seq_len: 最大序列长度
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        (train_loader, test_loader) 元组
    """
    # 创建数据集
    train_dataset = ASSISTmentsDataset(train_path, n_questions, max_seq_len)
    test_dataset = ASSISTmentsDataset(test_path, n_questions, max_seq_len)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不打乱顺序
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 测试数据加载
    print("=" * 50)
    print("测试数据加载器")
    print("=" * 50)

    dataset = ASSISTmentsDataset(
        "data/assistments/builder_train.csv",
        n_questions=110
    )

    print(f"\n第一个学生的数据:")
    sample = dataset[0]
    print(f"题目ID: {sample['question_ids']}")
    print(f"正确性: {sample['correct']}")
    print(f"答题数量: {sample['n_answers']}")

    # 测试 DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\n批次数据形状:")
    print(f"题目ID: {batch['question_ids'].shape}")
    print(f"正确性: {batch['correct'].shape}")
    print(f"掩码: {batch['mask'].shape}")
