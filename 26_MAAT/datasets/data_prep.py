import random
import pandas as pd
from collections import defaultdict


class DataPrep(object):
    """自适应测试数据预处理器:一组静态工具方法"""

    @staticmethod
    def deduplicate_data(data, policy):
        """去除重复作答记录(原仓库未实现)

        Args:
            data: pandas DataFrame,至少含 'student_id', 'question_id', 'correct' 三列
            policy: str,取值 ('keep_first', 'keep_last', 'average')

        Returns:
            去重后的数据: pandas DataFrame
        """
        # TODO 原仓库遗留的未实现方法
        raise NotImplementedError

    @staticmethod
    def parse_data(data):
        """把三元组列表整理成按学生/按题目索引的两个嵌套字典

        Args:
            data: list,[(学生id, 题目id, 得分)]

        Returns:
            按学生索引的数据: defaultdict {学生id: {题目id: 得分}}
            按题目索引的数据: defaultdict {题目id: {学生id: 得分}}
        """
        stu_data = defaultdict(lambda: defaultdict(dict))
        ques_data = defaultdict(lambda: defaultdict(dict))
        for sid, qid, correct in data:
            stu_data[sid][qid] = correct
            ques_data[qid][sid] = correct
        return stu_data, ques_data

    @staticmethod
    def prep_data(data, **kwargs):
        """数据预处理入口(原仓库未实现)

        Args:
            data: list,[(学生id, 题目id, 得分)]

        Returns:
            处理后的数据: list,[(学生id, 题目id, 得分)]
        """
        # TODO 原仓库遗留的未实现方法
        raise NotImplementedError

    @staticmethod
    def split_data_by_student(data, test_size=0.2, least_test_length=None):
        """按学生划分训练/测试集(测试集学生的记录全部留作模拟考试)

        Args:
            data: list,[(学生id, 题目id, 得分)]
            test_size: float 或 int,测试集大小(比例或人数)
            least_test_length: int > 0,测试集学生至少要有的作答记录数

        Returns:
            train_data: list,[(学生id, 题目id, 得分)]
            test_data: list,[(学生id, 题目id, 得分)]
        """
        stu_data, ques_data = DataPrep.parse_data(data)
        n_students = len(stu_data)
        if isinstance(test_size, float):
            test_size = int(n_students * test_size)
        train_size = n_students - test_size
        assert(train_size > 0 and test_size > 0)
        students = list(range(n_students))
        random.shuffle(students)
        # 记录太少的学生撑不满一场考试,不能进测试集
        if least_test_length is not None:
            student_lens = defaultdict(int)
            for t in data:
                student_lens[t[0]] += 1
            students = [student for student in students
                        if student_lens[student] >= least_test_length]
        test_students = set(students[:test_size])
        train_data = [record for record in data if record[0] not in test_students]
        test_data = [record for record in data if record[0] in test_students]
        train_data = DataPrep.renumber_student_id(train_data)
        test_data = DataPrep.renumber_student_id(test_data)
        return train_data, test_data

    @staticmethod
    def save_to_csv(data, path):
        """保存三元组到 csv

        Args:
            data: list,[(学生id, 题目id, 对错)]
            path: str,保存路径
        """
        pd.DataFrame.from_records(sorted(data), columns=['student_id', 'question_id', 'correct']).to_csv(path, index=False)

    @staticmethod
    def renumber_student_id(data):
        """把学生 id 重编号为从 0 起的连续整数

        Args:
            data: list,[(学生id, 题目id, 得分)]

        Returns:
            重编号后的数据: list,[(学生id, 题目id, 得分)]
        """
        student_ids = sorted(set(t[0] for t in data))
        renumber_map = {sid: i for i, sid in enumerate(student_ids)}
        data = [(renumber_map[t[0]], t[1], t[2]) for t in data]
        return data
