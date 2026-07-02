"""
边界估计自适应测试（BECAT）选题策略实现

BECAT是一种基于能力边界估计的选题策略，选择能够最有效缩小能力估计边界的题目。
该策略通过计算题目对能力边界的影响程度来选择题目。

特点：
- 不依赖特定的认知诊断模型（模型无关）
- 通过边界估计来量化题目信息量
- 适用于NCD等深度学习模型
"""

import numpy as np
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset
import random


class BECATstrategy(AbstractStrategy):
    """
    边界估计自适应测试（BECAT）选题策略
    
    原理：
    选择能够最有效缩小学生能力估计边界的题目。
    通过计算每个题目对当前能力边界的影响来选择最优题目。
    
    算法步骤：
    1. 从未测题目中随机采样一部分用于计算
    2. 对每个候选题目，计算其对边界的影响增量
    3. 选择影响增量最大的题目
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BECAT Strategy'
    
    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset, S_set):
        """
        执行选题
        
        Args:
            model: AbstractModel - 认知诊断模型（需要实现delta_q_S_t和get_pred方法）
            adaptest_data: AdapTestDataset - 测试数据集
            S_set: dict - 每个学生的已选题目列表 {学生ID: [题目ID列表]}
            
        Returns:
            dict - {学生ID: 题目ID}
        """
        assert hasattr(model, 'delta_q_S_t'), \
            '模型必须实现delta_q_S_t方法'
        assert hasattr(model, 'get_pred'), \
            '模型必须实现get_pred方法用于加速计算'
        
        # 获取所有题目的预测概率
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        
        for sid in range(adaptest_data.num_students):
            # 已选题目数量（用于确定采样大小）
            tmplen = len(S_set[sid])
            # 获取未测题目
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            
            # 随机采样一部分未测题目用于计算（减少计算量）
            sampled_elements = np.random.choice(untested_questions, tmplen + 5)
            
            # 计算每个候选题目的边界影响增量
            untested_deltaq = [
                model.delta_q_S_t(qid, pred_all[sid], S_set[sid], sampled_elements) 
                for qid in untested_questions
            ]
            
            # 选择边界影响增量最大的题目
            j = np.argmax(untested_deltaq)
            selection[sid] = untested_questions[j]
        
        return selection
