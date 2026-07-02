"""
Kullback-Leibler信息（KLI）选题策略实现

KLI是一种基于KL散度的选题策略，选择能够提供最多关于学生能力信息的题目。
相比Fisher信息，KLI直接度量的是能力分布的变化。

适用于：
- IRT模型（一维情况，也称为KLI）
- MIRT模型（多维情况，也称为MKLI）
"""

import numpy as np
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class KLIStrategy(AbstractStrategy):
    """
    Kullback-Leibler信息（KLI）选题策略
    
    原理：
    选择能够最大程度区分当前能力估计与能力先验分布的题目。
    通过计算候选题目对能力估计的KL散度来量化这种区分度。
    
    数学表达：
    选择题目 q* = argmax KL(p(θ|s,q) || p(θ|s))
    即选择使后验分布与先验分布KL散度最大的题目。
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Kullback-Leibler Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        """
        执行选题
        
        Args:
            model: AbstractModel - 认知诊断模型（需要实现get_kli和get_pred方法）
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {学生ID: 题目ID}
        """
        assert hasattr(model, 'get_kli'), \
            '模型必须实现get_kli方法'
        assert hasattr(model, 'get_pred'), \
            '模型必须实现get_pred方法用于加速计算'
        
        # 获取所有题目的预测概率
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        # 获取已测题目数量（用于积分边界计算）
        n = len(adaptest_data.tested[0])
        
        for sid in range(adaptest_data.num_students):
            # 获取学生的当前能力估计
            theta = model.get_theta(sid)
            # 获取未测题目
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            # 计算每个候选题目的KL信息量
            untested_kli = [model.get_kli(sid, qid, n, pred_all) for qid in untested_questions]
            # 选择KL信息量最大的题目
            j = np.argmax(untested_kli)
            selection[sid] = untested_questions[j]
        
        return selection


class MKLIStrategy(KLIStrategy):
    """
    多元Kullback-Leibler信息（MKLI）选题策略
    
    继承自KLIStrategy，用于MIRT模型的多维情况。
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Multivariate Kullback-Leibler Information Strategy'
