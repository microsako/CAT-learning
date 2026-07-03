from torch.utils import data
from ._dataset import _Dataset


class TrainDataset(_Dataset, data.dataset.Dataset):
    """可直接喂给 PyTorch DataLoader 的训练数据集"""

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

    def __getitem__(self, item):
        # 注意:不返回知识点列表——不同题的知识点数不同,默认 collate 无法拼批,
        # 且训练循环根本不用它(原仓库这里返回 concepts,在新版 torch 下会直接报错)
        sid, qid, score = self._raw_data[item]
        return sid, qid, score

    def __len__(self):
        return len(self._raw_data)
