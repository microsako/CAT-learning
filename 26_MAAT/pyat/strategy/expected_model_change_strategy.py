import numpy as np

from .abstract_strategy import AbstractStrategy
from ..model import AbstractModel
from ..utils.data import AdapTestDataset


class ExpectedModelChangeStrategy(AbstractStrategy):
    """纯质量模块策略(MAAT 去掉多样性模块的消融版):
    每步直接选期望模型变化(EMC,论文式 7-9)最大的题,不考虑知识点覆盖
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Expected Model Change Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            '模型必须实现 expected_model_change 方法'
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            # 对每道未测题算 EMC,取最大者
            emc_arr = [getattr(model, 'expected_model_change')(sid, qid, adaptest_data)
                       for qid in untested_questions]
            selection[sid] = untested_questions[np.argmax(emc_arr)]
        return selection
