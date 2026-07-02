"""
训练数据集类 - 用于模型训练

继承自Dataset类，增加了PyTorch Dataset接口的支持，
用于在训练过程中批量加载数据。

数据集格式说明：
- data: list, [(学生ID, 题目ID, 得分)]
- concept_map: dict, {题目ID: [知识点ID列表]}
"""

import torch
from torch.utils import data

try:
    # 用于Python模块导入
    from .dataset import Dataset
except (ImportError, SystemError):
    # 用于Python脚本直接运行
    from dataset import Dataset


class TrainDataset(Dataset, data.dataset.Dataset):
    """
    训练数据集
    
    继承自Dataset和PyTorch的Dataset，支持：
    - 按索引获取数据项
    - 自动构建知识点嵌入向量
    
    返回格式：[(学生ID, 题目ID, 知识点嵌入向量, 得分), ...]
    """

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        初始化训练数据集
        
        Args:
            data: list - 原始数据，格式为 [(学生ID, 题目ID, 得分), ...]
            concept_map: dict - 题目-知识点映射
            num_students: int - 学生总数
            num_questions: int - 题目总数
            num_concepts: int - 知识点总数
        """
        super().__init__(data, concept_map,
                        num_students, num_questions, num_concepts)

    def __getitem__(self, item):
        """
        获取单个数据项
        
        Args:
            item: int - 数据索引
            
        Returns:
            tuple: (学生ID, 题目ID, 知识点嵌入向量, 得分)
        """
        sid, qid, score = self.raw_data[item]
        # 获取题目涉及的知识点
        concepts = self.concept_map[qid]
        # 构建知识点嵌入向量（one-hot形式）
        concepts_emb = [0.] * self.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        return sid, qid, torch.Tensor(concepts_emb), score

    def __len__(self):
        """返回数据集大小"""
        return len(self.raw_data)
