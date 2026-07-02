"""
模型无关自适应测试（MAAT）选题策略实现

MAAT是一种结合预期模型变化和知识点覆盖率的选题策略。
它综合考虑了题目的信息量和知识点覆盖，是一个多目标优化策略。

特点：
- 模型无关（可用于IRT、NCD等多种模型）
- 综合考虑估计精度和知识点覆盖
- 使用候选集机制提高计算效率
"""

import numpy as np
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MAATStrategy(AbstractStrategy):
    """
    模型无关自适应测试（MAAT）选题策略
    
    原理：
    MAAT综合考虑两个目标：
    1. 预期模型变化（EMC）：选择能使模型参数变化最大的题目
    2. 知识点覆盖率：选择能覆盖更多未测知识点的题目
    
    算法步骤：
    1. 计算所有未测题目的预期模型变化
    2. 选择EMC最高的n个候选题目
    3. 在候选集中选择知识点覆盖率最高的题目
    """

    def __init__(self, n_candidates=10):
        """
        初始化MAAT策略
        
        Args:
            n_candidates: int - 候选集大小（默认10）
        """
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        """
        计算选择某题目后的知识点覆盖率增益
        
        使用覆盖率增益而非绝对覆盖率来评估题目价值：
        公式：覆盖率 = Σ(cnt / (cnt + 1)) / 知识点总数
        
        Args:
            sid: int - 学生ID
            qid: int - 题目ID
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            float - 覆盖率增益
        """
        concept_cnt = {}
        # 统计该学生所有题目涉及的知识点
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        
        # 统计已测题目和候选题目涉及的知识点
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        
        # 计算覆盖率
        # 使用 cnt/(cnt+1) 公式，使得已有多个题目的知识点增益递减
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        """
        执行选题
        
        Args:
            model: AbstractModel - 认知诊断模型（需要实现expected_model_change和get_pred方法）
            adaptest_data: AdapTestDataset - 测试数据集
            
        Returns:
            dict - {学生ID: 题目ID}
        """
        assert hasattr(model, 'expected_model_change'), \
            '模型必须实现expected_model_change方法'
        
        # 获取所有题目的预测概率
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        
        for sid in range(adaptest_data.num_students):
            # 获取未测题目
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            
            # 计算每个候选题目的预期模型变化
            emc_arr = [
                model.expected_model_change(sid, qid, adaptest_data, pred_all) 
                for qid in untested_questions
            ]
            
            # 选择EMC最高的n个题目作为候选集
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            
            # 在候选集中选择知识点覆盖率最高的题目
            selection[sid] = max(
                candidates, 
                key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data)
            )
        
        return selection
