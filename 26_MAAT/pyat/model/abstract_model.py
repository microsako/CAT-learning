from abc import ABC, abstractmethod
from ..utils.data import TrainDataset, AdapTestDataset, _Dataset


class AbstractModel(ABC):
    """认知诊断模型抽象基类:框架"模型无关"的含义就是任何实现了
    这组接口的模型(IRT/NCD/DKT 等)都能接入同一套选题策略
    """

    @property
    @abstractmethod
    def name(self):
        """模型名称"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        """考试中:根据已作答题目在线更新学生能力估计"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_evaluate(self, adaptest_data: AdapTestDataset):
        """评估当前能力估计的质量(如 AUC、知识点覆盖率)"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_init(self, data: _Dataset):
        """按数据规模初始化模型参数"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_train(self, train_data: TrainDataset):
        """离线阶段:用历史学生数据预训练题目参数"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_save(self, path):
        """保存(题目侧)模型参数"""
        raise NotImplementedError

    @abstractmethod
    def adaptest_preload(self, path):
        """加载预训练的模型参数"""
        raise NotImplementedError
