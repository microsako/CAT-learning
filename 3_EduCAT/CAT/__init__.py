"""
CAT - 计算机化自适应测试 Python 库

本库提供了快速开发计算机化自适应测试（CAT）系统的完整解决方案，
集成了传统统计方法和最新的机器学习/深度学习技术。

主要模块：
- CAT.model: 认知诊断模型（CDM），包括 IRT、NCD 等
- CAT.strategy: 选题算法，包括 MFI、KLI、MAAT、BECAT、BOBCAT、NCAT 等
- CAT.dataset: 数据处理，包括 Dataset、TrainDataset、AdapTestDataset

快速开始：
    from CAT.model import IRTModel
    from CAT.strategy import MFIStrategy
    from CAT.dataset import AdapTestDataset
    
    # 初始化模型
    model = IRTModel(num_dim=1, learning_rate=0.001, batch_size=32, num_epochs=10)
    model.init_model(data)
    
    # 训练模型
    model.train(train_data)
    
    # 自适应测试
    strategy = MFIStrategy()
    for step in range(max_steps):
        selection = strategy.adaptest_select(model, adaptest_data)
        # ... 应用选题，更新数据 ...
"""

__version__ = '0.0.1'

from .model import IRTModel, NCDModel
from .strategy import MFIStrategy, KLIStrategy, MAATStrategy, BECATstrategy
from .dataset import Dataset, TrainDataset, AdapTestDataset

__all__ = [
    'IRTModel',
    'NCDModel',
    'MFIStrategy',
    'KLIStrategy',
    'MAATStrategy',
    'BECATstrategy',
    'Dataset',
    'TrainDataset',
    'AdapTestDataset',
]
