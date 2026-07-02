"""
自适应测试数据集类 - 用于自适应测试过程

继承自Dataset类，增加了：
- 已测/未测题目跟踪
- 选题操作接口
- 自适应测试状态管理

在自适应测试过程中，需要追踪每个学生已作答的题目，
以便选题算法选择下一道题目。
"""

from collections import defaultdict, deque
import torch

try:
    # 用于Python模块导入
    from .dataset import Dataset
    from .train_dataset import TrainDataset
except (ImportError, SystemError):
    # 用于Python脚本直接运行
    from dataset import Dataset
    from train_dataset import TrainDataset


class AdapTestDataset(Dataset):
    """
    自适应测试数据集
    
    在基础数据集上增加了自适应测试所需的状态跟踪：
    - _tested: 已测题目集 {学生ID: [题目ID列表], ...}
    - _untested: 未测题目集 {学生ID: {题目ID集合}, ...}
    
    主要功能：
    - apply_selection: 应用选题结果
    - reset: 重置测试状态
    - get_tested_dataset: 获取已测数据用于模型更新
    """

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        初始化自适应测试数据集
        
        Args:
            data: list - 原始数据，格式为 [(学生ID, 题目ID, 得分), ...]
            concept_map: dict - 题目-知识点映射
            num_students: int - 学生总数
            num_questions: int - 题目总数
            num_concepts: int - 知识点总数
        """
        super().__init__(data, concept_map,
                        num_students, num_questions, num_concepts)

        # 初始化已测和未测题目集
        self._tested = None
        self._untested = None
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        """
        应用选题结果 - 将某题目从未测集移到已测集
        
        当选题算法选择了一道题目后，需要调用此方法更新状态。
        
        Args:
            student_idx: int - 学生ID
            question_idx: int - 题目ID
        """
        assert question_idx in self._untested[student_idx], \
            '选择的题目不在未测集中'
        # 从未测集移除
        self._untested[student_idx].remove(question_idx)
        # 加入已测集
        self._tested[student_idx].append(question_idx)

    def reset(self):
        """
        重置测试状态 - 清空已测题目
        
        用于开始新一轮自适应测试。
        """
        # 使用deque支持append操作
        self._tested = defaultdict(deque)
        # 使用set便于快速查找
        self._untested = defaultdict(set)
        for sid in self.data:
            # 初始时，所有题目都是未测的
            self._untested[sid] = set(self.data[sid].keys())

    @property
    def tested(self):
        """
        已测题目集
        
        Returns:
            dict: {学生ID: [题目ID列表], ...}
        """
        return self._tested

    @property
    def untested(self):
        """
        未测题目集
        
        Returns:
            dict: {学生ID: {题目ID集合}, ...}
        """
        return self._untested

    def get_tested_dataset(self, last=False, ssid=None):
        """
        获取已测数据用于模型训练
        
        用于在自适应测试过程中，用已测数据更新模型。
        
        Args:
            last: bool - True: 只返回最后一道题目；False: 返回所有已测题目
            ssid: int - 学生ID（可选），指定获取特定学生的数据
            
        Returns:
            TrainDataset: 可用于模型训练的数据集
        """
        if ssid is None:
            triplets = []
            for sid, qids in self._tested.items():
                if last:
                    # 只取最后一道题目
                    qid = qids[-1]
                    triplets.append((sid, qid, self.data[sid][qid]))
                else:
                    # 取所有已测题目
                    for qid in qids:
                        triplets.append((sid, qid, self.data[sid][qid]))
            return TrainDataset(triplets, self.concept_map,
                               self.num_students, self.num_questions, self.num_concepts)
        else:
            # 指定特定学生
            triplets = []
            for sid, qids in self._tested.items():
                if ssid == sid:
                    if last:
                        qid = qids[-1]
                        triplets.append((sid, qid, self.data[sid][qid]))
                    else:
                        for qid in qids:
                            triplets.append((sid, qid, self.data[sid][qid]))
            return TrainDataset(triplets, self.concept_map,
                               self.num_students, self.num_questions, self.num_concepts)
