import numpy as np

from .abstract_strategy import AbstractStrategy
from ..model import AbstractModel
from ..utils.data import AdapTestDataset


class RandomStrategy(AbstractStrategy):
    """随机选题策略(最弱基线):每步从未测题目中等概率随机抽一道"""

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Random Select Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            # 等概率随机挑一道未测题
            selection[sid] = np.random.choice(untested_questions)
        return selection
