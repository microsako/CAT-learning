"""
基础数据集类 - 定义数据集的基本结构和属性

本模块定义了CAT系统中所有数据集的基类，包含：
- 学生数量
- 题目数量
- 知识点数量
- 原始数据
- 知识点映射关系

数据集格式说明：
- data: list, [(学生ID, 题目ID, 得分)] - 三元组格式的原始数据
- concept_map: dict, {题目ID: [知识点ID列表]} - 题目与知识点的映射关系
"""

from collections import defaultdict, deque


class Dataset(object):
    """
    数据集基类
    
    存储和管理教育数据，包括学生作答记录和题目-知识点映射。
    
    数据结构：
    - _raw_data: 原始三元组数据 [(sid, qid, correct), ...]
    - _data: 按学生组织的数据 {sid: {qid: correct, ...}, ...}
    - _concept_map: 题目-知识点映射 {qid: [cid, ...], ...}
    """

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        初始化数据集
        
        Args:
            data: list - 原始数据，格式为 [(学生ID, 题目ID, 得分), ...]
            concept_map: dict - 题目-知识点映射，格式为 {题目ID: [知识点ID列表], ...}
            num_students: int - 学生总数
            num_questions: int - 题目总数
            num_concepts: int - 知识点总数
        """
        self._raw_data = data  # 原始数据
        self._concept_map = concept_map  # 知识点映射
        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts
        
        # 重组数据集：按学生组织
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        # 提取ID范围用于验证
        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        concept_ids = set(sum(concept_map.values(), []))

        # 验证ID是否有效（需要从0开始连续编号）
        assert max(student_ids) < num_students, \
            '学生ID需要重新编号'
        assert max(question_ids) < num_questions, \
            '题目ID需要重新编号'
        assert max(concept_ids) < num_concepts, \
            '知识点ID需要重新编号'

    @property
    def num_students(self):
        """学生总数"""
        return self.n_students

    @property
    def num_questions(self):
        """题目总数"""
        return self.n_questions

    @property
    def num_concepts(self):
        """知识点总数"""
        return self.n_concepts

    @property
    def raw_data(self):
        """原始数据 [(学生ID, 题目ID, 得分), ...]"""
        return self._raw_data

    @property
    def data(self):
        """按学生组织的数据 {学生ID: {题目ID: 得分, ...}, ...}"""
        return self._data

    @property
    def concept_map(self):
        """题目-知识点映射 {题目ID: [知识点ID列表], ...}"""
        return self._concept_map
