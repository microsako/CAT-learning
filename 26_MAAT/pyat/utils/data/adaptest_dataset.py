from collections import defaultdict, deque

import torch

from ._dataset import _Dataset
from .train_dataset import TrainDataset


class AdapTestDataset(_Dataset):
    """自适应测试数据集:在 _Dataset 基础上维护每个学生的已测/未测题目集合。
    "未测集合"= 该学生在数据里真实作答过的全部题目(选中后才能查到真实对错)
    """

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        Args:
            data: list,[(学生id, 题目id, 对错)]
            concept_map: dict,{题目id: 知识点id列表}
            num_students: int,学生总数
            num_questions: int,题目总数
            num_concepts: int,知识点总数

        Requirements:
            学生、题目、知识点的 id 都须已重编号为从 0 起的连续整数
        """
        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

        # 初始化已测/未测集合
        self._tested = None
        self._untested = None
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        """把一道未测题移入已测集合(即"该学生作答了这道题")

        Args:
            student_idx: int
            question_idx: int
        """
        assert question_idx in self._untested[student_idx], \
            'Selected question not allowed'
        self._untested[student_idx].remove(question_idx)
        self._tested[student_idx].append(question_idx)

    def reset(self):
        """重置:全部题目回到未测状态(换策略重跑前调用)"""
        self._tested = defaultdict(deque)
        self._untested = defaultdict(set)
        for sid in self._data:
            self._untested[sid] = set(self._data[sid].keys())

    @property
    def tested(self):
        return self._tested

    @property
    def untested(self):
        return self._untested

    def get_tested_dataset(self, last=False):
        """把已测题目转成可训练的 TrainDataset

        Args:
            last: bool,True 则只取每个学生最近作答的一道题(增量更新用)
        """
        triplets = []
        for sid, qids in self._tested.items():
            if last:
                qid = qids[-1]
                triplets.append((sid, qid, self.data[sid][qid]))
            else:
                for qid in qids:
                    triplets.append((sid, qid, self.data[sid][qid]))
        return TrainDataset(triplets, self.concept_map,
                            self.num_students, self.num_questions, self.num_concepts)
