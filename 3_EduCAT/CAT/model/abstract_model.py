"""
抽象模型基类 - 定义认知诊断模型（CDM）的接口规范

本模块定义了所有认知诊断模型必须实现的抽象方法，
包括模型初始化、训练、评估、保存和加载等功能。
"""

from abc import ABC, abstractmethod
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset


class AbstractModel(ABC):
    """
    认知诊断模型的抽象基类
    
    所有具体的认知诊断模型（如IRT、NCD）都必须继承此类并实现以下抽象方法。
    """

    @property
    @abstractmethod
    def name(self):
        """
        获取模型的名称
        
        Returns:
            name: str - 模型名称，如 'Item Response Theory' 或 'Neural Cognitive Diagnosis'
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        """
        使用测试数据更新模型参数
        
        在自适应测试过程中，根据学生已作答的题目来更新学生的能力估计。
        
        Args:
            adaptest_data: AdapTestDataset - 包含学生作答记录的自适应测试数据集
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, adaptest_data: AdapTestDataset):
        """
        评估模型性能
        
        计算模型的 AUC、准确率、覆盖率等评估指标。
        
        Args:
            adaptest_data: AdapTestDataset - 包含学生作答记录的自适应测试数据集
            
        Returns:
            dict - 包含 'auc', 'cov', 'acc' 等评估指标的字典
        """
        raise NotImplementedError

    @abstractmethod
    def init_model(self, data: Dataset):
        """
        初始化模型
        
        根据数据集的规模和配置参数初始化模型结构。
        
        Args:
            data: Dataset - 训练数据集，包含学生数、题目数、知识点数等信息
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: TrainDataset):
        """
        训练模型
        
        使用训练数据来学习学生能力、题目难度等参数。
        
        Args:
            train_data: TrainDataset - 训练数据集
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_save(self, path):
        """
        保存模型
        
        保存模型参数到指定路径，通常只保存题目相关参数（题目难度、区分度等），
        而不保存学生参数。
        
        Args:
            path: str - 保存路径
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_load(self, path):
        """
        加载模型
        
        从指定路径加载模型参数。
        
        Args:
            path: str - 模型文件路径
        """
        raise NotImplementedError
