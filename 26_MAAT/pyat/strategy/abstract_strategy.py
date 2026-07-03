from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    """选题策略抽象基类:所有策略只需实现 name 和 adaptest_select"""

    @property
    @abstractmethod
    def name(self):
        """策略名称

        Returns:
            name: str
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_select(self, model, adaptest_data):
        """为每个学生从其未测题目中选出下一道题

        Args:
            model: AbstractModel,当前认知诊断模型
            adaptest_data: AdapTestDataset,自适应测试数据(含已测/未测集合)

        Returns:
            selected_questions: dict,{学生id: 选中的题目id}
        """
        raise NotImplementedError
