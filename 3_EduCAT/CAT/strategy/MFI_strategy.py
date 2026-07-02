"""
最大费舍尔信息（MFI）选题策略实现

MFI是一种经典的选题策略，基于Fisher信息量来选择能够最大程度
减少能力估计方差的题目。

适用于：
- IRT模型（一维情况）
- MIRT模型（多维情况，此时也称为D-opt策略）
"""

import numpy as np
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MFIStrategy(AbstractStrategy):
    """
    最大费舍尔信息（Maximum Fisher Information）选题策略
    
    原理：
    选择使 Fisher 信息矩阵行列式最大的题目，这样可以最大程度
    减少学生能力估计的置信椭球体积（提高估计精度）。
    
    数学表达：
    选择题目 q* = argmax det(I(θ) + F(q))
    其中 I(θ) 是已选题目的累积信息，F(q) 是候选题目的Fisher信息
    """

    def __init__(self):
        super().__init__()
        # 存储每个学生的累积Fisher信息矩阵
        self.I = None

    @property
    def name(self):
        return 'Maximum Fisher Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        """
        执行选题
        
        Args:
            model: AbstractModel - 认知诊断模型（需要实现get_fisher和get_pred方法）
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {学生ID: 题目ID}
        """
        assert hasattr(model, 'get_fisher'), \
            '模型必须实现get_fisher方法'
        assert hasattr(model, 'get_pred'), \
            '模型必须实现get_pred方法用于加速计算'
        
        # 获取所有题目的预测概率
        pred_all = model.get_pred(adaptest_data)
        
        # 初始化累积信息矩阵（每个学生一个）
        if self.I is None:
            self.I = [np.zeros((model.model.num_dim, model.model.num_dim)) 
                     for _ in range(adaptest_data.num_students)]
        
        selection = {}
        # 获取已测题目数量（用于置信区间计算）
        n = len(adaptest_data.tested[0])
        
        for sid in range(adaptest_data.num_students):
            # 获取该学生的未测题目
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            
            # 如果没有未测题目，跳过
            if len(untested_questions) == 0:
                continue
            
            untested_dets = []
            untested_fisher = []
            
            # 计算每个候选题目的Fisher信息
            for qid in untested_questions:
                fisher_info = model.get_fisher(sid, qid, pred_all)
                untested_fisher.append(fisher_info)
                untested_dets.append(np.linalg.det(self.I[sid] + fisher_info))
            
            # 选择使行列式最大的题目
            j = np.argmax(untested_dets)
            selection[sid] = untested_questions[j]
            self.I[sid] += untested_fisher[j]
        
        return selection


class DoptStrategy(MFIStrategy):
    """
    D最优性（D-Optimality）选题策略
    
    继承自MFIStrategy，本质上是多维情况下的Fisher信息选题。
    用于MIRT模型的选题。
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'D-Optimality Strategy'
