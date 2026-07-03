import numpy as np
from ..strategy.abstract_strategy import AbstractStrategy
from ..model import AbstractModel
from ..utils.data import AdapTestDataset


class MAATStrategy(AbstractStrategy):
    """MAAT 选题策略 = 质量模块 + 多样性模块两级筛选:
    先按 EMC 取 top-n_candidates 个候选(质量),再在候选中选知识点覆盖增益最大的一道(多样性)。
    注意:论文的重要性模块(式 10-15,知识点权重 w_k)未随论文开源,
    这里等价于所有知识点等权 w_k = 1
    """

    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        """计算把候选题 qid 加入已测集合后的知识点覆盖(论文式 5-6 的无权重版)"""
        concept_cnt = {}
        # 该学生做过的题涉及的全部知识点,计数清零
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        # 统计"已测题 + 候选题"覆盖每个知识点的次数
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        # IncCov = cnt/(cnt+1):边际收益递减,已覆盖多次的知识点再覆盖收益变小
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            '模型必须实现 expected_model_change 方法'
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            # 第一级:质量模块,按 EMC 从大到小取前 n_candidates 道当候选
            emc_arr = [getattr(model, 'expected_model_change')(sid, qid, adaptest_data)
                       for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            # 第二级:多样性模块,候选中选覆盖增益最大的一道
            selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
        return selection
