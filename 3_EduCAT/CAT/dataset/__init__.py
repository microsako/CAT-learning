"""
数据集模块

本模块提供了处理教育数据的数据集类。

提供的类：
- Dataset: 基础数据集类
- TrainDataset: 训练数据集类
- AdapTestDataset: 自适应测试数据集类

详见：
- dataset.py: 基础数据集
- train_dataset.py: 训练数据集
- adaptest_dataset.py: 自适应测试数据集
"""

from .dataset import Dataset
from .train_dataset import TrainDataset
from .adaptest_dataset import AdapTestDataset

__all__ = ['Dataset', 'TrainDataset', 'AdapTestDataset']
