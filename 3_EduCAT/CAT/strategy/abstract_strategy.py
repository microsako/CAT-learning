"""
抽象选题策略基类 - 定义选题算法的接口规范

本模块定义了所有选题策略必须实现的抽象方法。
选题策略负责根据当前的模型估计，从题库中选择最合适的下一道题目。
"""

from abc import ABC, abstractmethod


class AbstractStrategy(ABC):
    """
    选题策略的抽象基类
    
    所有具体的选题策略（如MFI、KLI、MAAT等）都必须继承此类。
    """

    @property
    @abstractmethod
    def name(self):
        """
        获取策略名称
        
        Returns:
            name: str - 策略名称
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_select(self, model, adaptest_data):
        """
        选择下一道题目
        
        根据模型估计和当前测试状态，为每个学生选择下一道题目。
        
        Args:
            model: AbstractModel - 认知诊断模型，用于估计学生能力和计算题目信息量
            adaptest_data: AdapTestDataset - 自适应测试数据集，包含已测和未测题目信息
            
        Returns:
            selected_questions: dict - 选题结果，格式为 {学生ID: 题目ID}
        """
        raise NotImplementedError
