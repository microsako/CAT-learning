from collections import defaultdict, deque


class _Dataset(object):
    """数据集基类:把三元组列表整理成 {学生id: {题目id: 对错}} 的嵌套字典"""

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
        self._raw_data = data
        self._concept_map = concept_map
        # 重组为嵌套字典,方便按学生查询
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts
        # 校验 id 已重编号(最大 id 必须小于总数)
        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        concept_ids = set(sum(concept_map.values(), []))
        assert max(student_ids) < num_students, \
            'Require student ids renumbered'
        assert max(question_ids) < num_questions, \
            'Require student ids renumbered'
        assert max(concept_ids) < num_concepts, \
            'Require student ids renumbered'

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map
