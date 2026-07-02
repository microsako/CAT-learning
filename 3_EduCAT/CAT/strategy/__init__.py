"""
选题策略模块

本模块实现了各种自适应测试选题算法，用于从题库中选择最优题目。

提供的策略：
- MFIStrategy: 最大费舍尔信息策略（适用于IRT/MIRT）
- KLIStrategy: KL散度信息策略（适用于IRT/MIRT）
- MAATStrategy: 模型无关自适应测试策略（适用于所有模型）
- BECATstrategy: 边界估计自适应测试策略（适用于NCD）
- BOBCATStrategy: 双层优化CAT策略（强化学习方法）
- NCATStrategy: 神经CAT策略（神经网络方法）
"""

from .MFI_strategy import MFIStrategy, DoptStrategy
from .KLI_strategy import KLIStrategy, MKLIStrategy
from .MAAT_strategy import MAATStrategy
from .BECAT_strategy import BECATstrategy

__all__ = [
    'MFIStrategy', 
    'DoptStrategy',
    'KLIStrategy', 
    'MKLIStrategy',
    'MAATStrategy',
    'BECATstrategy',
]
