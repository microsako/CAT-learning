import numpy as np

from .abstract_strategy import AbstractStrategy
from ..model import AbstractModel
from ..utils.data import AdapTestDataset


class FisherStrategy(AbstractStrategy):
    """Fisher 信息量策略(MFI,传统 CAT 最经典的基线):
    每步选在当前能力估计 theta 下 Fisher 信息量最大的题。
    IRT 下有闭式解 I(theta) = alpha^2 * p * (1 - p),其中 p 为答对概率。
    注意:该策略依赖 IRT 的参数形式,不是模型无关的,仅用于 IRT
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Fisher Information Strategy'

    def _fisher_information(self, model, sid, qid):
        theta = model.get_theta(sid)         # 学生当前能力估计
        alpha = model.get_alpha(qid)         # 题目区分度
        beta = float(model.get_beta(qid)[0])  # 题目难度(取成标量)
        pred = 1.0 / (1.0 + np.exp(-(float(np.dot(alpha, theta)) + beta)))  # 答对概率
        return float(np.dot(alpha, alpha)) * pred * (1 - pred)

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        for method in ('get_theta', 'get_alpha', 'get_beta'):
            assert hasattr(model, method), \
                'Fisher 策略要求模型提供 {} 方法(IRT 类模型)'.format(method)
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            # 对每道未测题算 Fisher 信息量,取最大者
            fisher_arr = [self._fisher_information(model, sid, qid)
                          for qid in untested_questions]
            selection[sid] = untested_questions[np.argmax(fisher_arr)]
        return selection
